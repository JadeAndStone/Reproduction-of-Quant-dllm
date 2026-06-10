#!/usr/bin/env python
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


KEY_RE = re.compile(r"^(?P<weight>.+)::r(?P<rs>\d+)-(?P<re>\d+)::c(?P<cs>\d+)-(?P<ce>\d+)$")


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def parse_unit_key(key):
    match = KEY_RE.match(key)
    if match is None:
        return key, None
    groups = match.groupdict()
    return groups["weight"], (
        int(groups["rs"]),
        int(groups["re"]),
        int(groups["cs"]),
        int(groups["ce"]),
    )


def make_unit_key(weight_name, row_start, row_end, col_start, col_end):
    return f"{weight_name}::r{row_start}-{row_end}::c{col_start}-{col_end}"


def parse_case(spec):
    parts = spec.split("=", 3)
    if len(parts) != 4:
        raise ValueError(
            "--case must be label=quant_output=z_cache=daq_granularity, "
            f"got: {spec}"
        )
    label, quant_output, z_cache, daq_granularity = parts
    return {
        "label": label,
        "quant_output": Path(quant_output),
        "z_cache": Path(z_cache),
        "daq_granularity": daq_granularity,
    }


def load_weight_map(model_path):
    return load_json(Path(model_path) / "model.safetensors.index.json")["weight_map"]


def load_safetensor_tensor(root, weight_map, tensor_name):
    shard_path = Path(root) / weight_map[tensor_name]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name).to(torch.float32)


def load_block_tensor(root, block_idx, tensor_name):
    path = Path(root) / f"block_{block_idx:02d}.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name).to(torch.float32)


def block_idx_from_weight(weight_name):
    parts = weight_name.split(".")
    if len(parts) >= 4 and parts[:3] == ["model", "transformer", "blocks"]:
        return int(parts[3])
    raise ValueError(f"Cannot infer transformer block from {weight_name}")


def selected_weight_names(weight_map, blocks, modules):
    names = []
    for block_idx in blocks:
        for module_name in modules:
            weight_name = f"model.transformer.blocks.{block_idx}.{module_name}.weight"
            if weight_name in weight_map:
                names.append(weight_name)
    return names


def infer_precision_units(precision_map):
    units_by_weight = defaultdict(list)
    direct = {}
    for key, bits in precision_map.items():
        if key.startswith("__"):
            continue
        weight_name, unit = parse_unit_key(key)
        if unit is None:
            direct[weight_name] = int(bits)
        else:
            units_by_weight[weight_name].append((unit, int(bits), key))
    for units in units_by_weight.values():
        units.sort(key=lambda item: (item[0][0], item[0][2], item[0][1], item[0][3]))
    return dict(units_by_weight), direct


def slice_from_unit(unit):
    row_start, row_end, col_start, col_end = unit
    return slice(row_start, row_end), slice(col_start, col_end)


def unit_numel(unit):
    row_start, row_end, col_start, col_end = unit
    return (row_end - row_start) * (col_end - col_start)


def unit_score_sum(score_map, key, fallback_z=None):
    if key in score_map:
        return float(score_map[key])
    if fallback_z is not None:
        return float(fallback_z.sum().item())
    return 0.0


def summarize_precision_map(case):
    precision_map = load_json(case["quant_output"] / "precision_map.json")
    score_path = case["quant_output"] / "importance_scores.json"
    score_map = load_json(score_path) if score_path.exists() else {}
    units_by_weight, direct = infer_precision_units(precision_map)

    bit_group_count = defaultdict(int)
    bit_element_count = defaultdict(int)
    bit_score_sum = defaultdict(float)
    total_score = 0.0
    total_groups = 0
    total_elements = 0

    for weight_name, bits in direct.items():
        bit_group_count[bits] += 1
        total_groups += 1
        if weight_name in score_map:
            score = float(score_map[weight_name])
            bit_score_sum[bits] += score
            total_score += score

    for weight_name, units in units_by_weight.items():
        for unit, bits, key in units:
            numel = unit_numel(unit)
            score = float(score_map.get(key, 0.0))
            bit_group_count[bits] += 1
            bit_element_count[bits] += numel
            bit_score_sum[bits] += score
            total_groups += 1
            total_elements += numel
            total_score += score

    bits = sorted(set(bit_group_count) | set(bit_element_count) | set(bit_score_sum))
    summary = {
        "num_weights_with_unit_precision": len(units_by_weight),
        "num_weights_with_direct_precision": len(direct),
        "total_groups": total_groups,
        "total_elements_from_units": total_elements,
        "total_score": total_score,
        "by_bit": {},
    }
    for bit in bits:
        summary["by_bit"][str(bit)] = {
            "groups": bit_group_count[bit],
            "group_share": safe_div(bit_group_count[bit], total_groups),
            "elements": bit_element_count[bit],
            "element_share": safe_div(bit_element_count[bit], total_elements),
            "score_sum": bit_score_sum[bit],
            "score_share": safe_div(bit_score_sum[bit], total_score),
            "score_per_group": safe_div(bit_score_sum[bit], bit_group_count[bit]),
        }
    return summary, precision_map, score_map, units_by_weight, direct


def safe_div(num, den):
    if den == 0:
        return 0.0
    return float(num) / float(den)


def stats_init():
    return {
        "groups": 0,
        "elements": 0,
        "mask": 0,
        "mask_z_sum": 0.0,
        "z_sum": 0.0,
        "w2": 0.0,
        "err2": 0.0,
        "zw2": 0.0,
        "zerr2": 0.0,
    }


def add_tensor_stats(stats, weight, quant, z):
    err = quant - weight
    stats["w2"] += float(torch.sum(weight * weight).item())
    stats["err2"] += float(torch.sum(err * err).item())
    stats["zw2"] += float(torch.sum(z * weight * weight).item())
    stats["zerr2"] += float(torch.sum(z * err * err).item())


def add_mask_stats(stats, z, mask):
    stats["elements"] += int(z.numel())
    mask_count = int(mask.sum().item())
    stats["mask"] += mask_count
    stats["z_sum"] += float(z.sum().item())
    if mask_count:
        stats["mask_z_sum"] += float(z[mask].sum().item())


def finalize_stats(raw):
    total_mask = sum(item["mask"] for item in raw.values())
    total_mask_z = sum(item["mask_z_sum"] for item in raw.values())
    total_err2 = sum(item["err2"] for item in raw.values())
    total_zerr2 = sum(item["zerr2"] for item in raw.values())
    out = {}
    for bit, item in sorted(raw.items(), key=lambda kv: int(kv[0])):
        w2 = item["w2"]
        err2 = item["err2"]
        zw2 = item["zw2"]
        zerr2 = item["zerr2"]
        out[str(bit)] = {
            "groups": item["groups"],
            "elements": item["elements"],
            "mask_rate": safe_div(item["mask"], item["elements"]),
            "masked_z_share_within_bit": safe_div(item["mask_z_sum"], item["z_sum"]),
            "masked_element_share_global": safe_div(item["mask"], total_mask),
            "masked_z_share_global": safe_div(item["mask_z_sum"], total_mask_z),
            "rel_l2_error": math.sqrt(safe_div(err2, w2)) if w2 > 0 else 0.0,
            "rel_z_weighted_error": math.sqrt(safe_div(zerr2, zw2)) if zw2 > 0 else 0.0,
            "err2_share": safe_div(err2, total_err2),
            "zerr2_share": safe_div(zerr2, total_zerr2),
        }
    return out


def build_mask(z):
    mean = torch.mean(z)
    std = torch.std(z)
    return (z - mean).abs() > 3 * std


def bit_for_tile(weight_name, unit, units, direct_bits):
    if weight_name in direct_bits:
        return direct_bits[weight_name]
    row_start, row_end, col_start, col_end = unit
    for map_unit, bits, _ in units:
        rs, re, cs, ce = map_unit
        if row_start >= rs and row_end <= re and col_start >= cs and col_end <= ce:
            return bits
    raise KeyError(f"No precision bit covers {weight_name} unit {unit}")


def iter_tiles(shape, block_size):
    rows, cols = shape
    for row_start in range(0, rows, block_size):
        row_end = min(row_start + block_size, rows)
        for col_start in range(0, cols, block_size):
            col_end = min(col_start + block_size, cols)
            yield row_start, row_end, col_start, col_end


def summarize_selected_case(case, fp_model_path, selected_names, block_size):
    precision_summary, precision_map, score_map, units_by_weight, direct_bits = summarize_precision_map(case)
    fp_weight_map = load_weight_map(fp_model_path)
    quant_weight_map = load_weight_map(case["quant_output"])

    detail = {
        "label": case["label"],
        "quant_output": str(case["quant_output"]),
        "z_cache": str(case["z_cache"]),
        "daq_granularity": case["daq_granularity"],
        "precision_summary": precision_summary,
        "selected_weights": selected_names,
        "selected_by_bit": {
            "precision_unit_scope_mask": {},
            "daq_tile_scope_mask": {},
            "weight_scope_mask": {},
        },
        "per_weight": {},
    }

    selected_raw = {
        "precision_unit_scope_mask": defaultdict(stats_init),
        "daq_tile_scope_mask": defaultdict(stats_init),
        "weight_scope_mask": defaultdict(stats_init),
    }

    for idx, weight_name in enumerate(selected_names, start=1):
        print(f"[{case['label']}] {idx}/{len(selected_names)} {weight_name}", flush=True)
        block_idx = block_idx_from_weight(weight_name)
        weight = load_safetensor_tensor(fp_model_path, fp_weight_map, weight_name)
        quant = load_safetensor_tensor(case["quant_output"], quant_weight_map, weight_name)
        z = load_block_tensor(case["z_cache"], block_idx, weight_name)
        if weight.shape != quant.shape or weight.shape != z.shape:
            raise ValueError((weight_name, weight.shape, quant.shape, z.shape))

        units = units_by_weight.get(weight_name, [])
        if not units and weight_name in direct_bits:
            rows, cols = weight.shape
            units = [((0, rows, 0, cols), direct_bits[weight_name], weight_name)]

        per_weight = {
            "shape": list(weight.shape),
            "precision_units": len(units),
            "score_total": sum(unit_score_sum(score_map, key) for _, _, key in units),
            "by_bit": {},
        }
        local_raw = {
            "precision_unit_scope_mask": defaultdict(stats_init),
            "daq_tile_scope_mask": defaultdict(stats_init),
            "weight_scope_mask": defaultdict(stats_init),
        }

        weight_mask = build_mask(z)

        for unit, bits, key in units:
            row_slice, col_slice = slice_from_unit(unit)
            z_unit = z[row_slice, col_slice]
            w_unit = weight[row_slice, col_slice]
            q_unit = quant[row_slice, col_slice]

            for raw in (selected_raw["precision_unit_scope_mask"], local_raw["precision_unit_scope_mask"]):
                raw[bits]["groups"] += 1
                add_mask_stats(raw[bits], z_unit, build_mask(z_unit))
                add_tensor_stats(raw[bits], w_unit, q_unit, z_unit)

            for raw in (selected_raw["weight_scope_mask"], local_raw["weight_scope_mask"]):
                raw[bits]["groups"] += 1
                add_mask_stats(raw[bits], z_unit, weight_mask[row_slice, col_slice])
                add_tensor_stats(raw[bits], w_unit, q_unit, z_unit)

        for tile_unit in iter_tiles(weight.shape, block_size):
            row_slice, col_slice = slice_from_unit(tile_unit)
            bits = bit_for_tile(weight_name, tile_unit, units, direct_bits)
            z_tile = z[row_slice, col_slice]
            w_tile = weight[row_slice, col_slice]
            q_tile = quant[row_slice, col_slice]
            for raw in (selected_raw["daq_tile_scope_mask"], local_raw["daq_tile_scope_mask"]):
                raw[bits]["groups"] += 1
                add_mask_stats(raw[bits], z_tile, build_mask(z_tile))
                add_tensor_stats(raw[bits], w_tile, q_tile, z_tile)

        for scope, raw in local_raw.items():
            per_weight[scope] = finalize_stats(raw)
        detail["per_weight"][weight_name] = per_weight

        del weight, quant, z, weight_mask

    for scope, raw in selected_raw.items():
        detail["selected_by_bit"][scope] = finalize_stats(raw)
    return detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-model", required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="label=quant_output=z_cache=daq_granularity",
    )
    parser.add_argument("--blocks", default="0,15,31")
    parser.add_argument(
        "--modules",
        default="q_proj,k_proj,v_proj,attn_out,ff_proj,up_proj,ff_out",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fp_weight_map = load_weight_map(args.fp_model)
    blocks = [int(item) for item in args.blocks.split(",") if item]
    modules = [item for item in args.modules.split(",") if item]
    selected_names = selected_weight_names(fp_weight_map, blocks, modules)
    cases = [parse_case(spec) for spec in args.case]

    output = {
        "fp_model": args.fp_model,
        "blocks": blocks,
        "modules": modules,
        "selected_weights": selected_names,
        "cases": {},
    }
    for case in cases:
        output["cases"][case["label"]] = summarize_selected_case(
            case,
            args.fp_model,
            selected_names,
            args.block_size,
        )

    save_json(output, args.output)
    print(f"Saved diagnostics to {args.output}")


if __name__ == "__main__":
    main()
