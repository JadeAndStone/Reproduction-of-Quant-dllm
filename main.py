import argparse
import gc
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from DAQ.quantizer import DAQ, build_imp_mask
from utils.generate_imp_mat import (
    build_imp_mat_from_mcs,
    get_mcs_mat,
    load_weight_from_checkpoint,
    load_weight_map,
)

MODEL_PATH = "/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base"
C4_TRAIN_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
TARGET_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "attn_out.weight",
    "ff_proj.weight",
    "up_proj.weight",
    "ff_out.weight",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--output_dir", default="/root/data-fs/Quant-dllm/outputs/llada_daq_fake_quant")
    parser.add_argument("--source", default=C4_TRAIN_URL, help="C4 .json.gz URL or local path.")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--group_size",type=int,default=128)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--allocate_ratio", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument(
        "--mcs_normalization",
        choices=["tokens", "mcs_steps"],
        default="tokens",
        help="Normalize MCS/SMCS by all observed tokens or only by MCS timestep count.",
    )
    parser.add_argument("--inverse_diag_floor", type=float, default=1e-12)
    parser.add_argument("--daq_steps", type=int, default=10)
    parser.add_argument("--imp_lamda", type=float, default=2.0)
    parser.add_argument(
        "--dor_mask_scope",
        choices=["unit", "weight"],
        default="unit",
        help="Scope used to compute the DOR 3-sigma importance mask: per DAQ unit or full Linear weight.",
    )
    parser.add_argument("--daq_component", choices=["custom", "baseline", "rsr_no_dor", "rsr_w_dor"], default="custom")
    parser.add_argument("--daq_row_center", action="store_true", help="Apply per-weight-row centering before DAQ and add row mean back after quantization.")
    parser.add_argument("--z_row_center", action="store_true", help="Build Z from row-centered weights using the same granularity as DAQ.")
    parser.add_argument("--start_block", type=int, default=0)
    parser.add_argument("--end_block", type=int, default=None)
    parser.add_argument("--skip_abmp", action="store_true")
    parser.add_argument("--default_bits", type=int, default=2)
    parser.add_argument("--max_mcs_samples", type=int, default=None)
    parser.add_argument(
        "--mcs_no_full_visible",
        action="store_true",
        help="Legacy ablation: exclude the fully-visible endpoint while keeping num_steps MCS states.",
    )
    parser.add_argument("--only_build_z", action="store_true", help="Build or validate the Z cache and exit before quantization.")
    parser.add_argument("--reuse_z", action="store_true")
    parser.add_argument("--z_cache_dir", default=None)
    parser.add_argument("--z_save_dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--save_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--gpu_memory", default="13GiB")
    parser.add_argument("--cpu_memory", default="80GiB")
    parser.add_argument("--offload_folder", default="/tmp/llada_offload")
    parser.add_argument("--daq_granularity", choices=["weight", "row", "column", "tile"], default="transformer_block")
    parser.add_argument("--abmp_granularity", choices=["weight", "row", "column", "tile"], default=None)
    parser.add_argument("--abmp_rank_scope", choices=["global", "transformer_block", "weight"], default="weight")
    parser.add_argument("--quant_block_size", type=int, default=128)
    parser.add_argument("--daq_device", default="cpu")
    parser.add_argument("--log_unit_interval", type=int, default=0)
    return parser.parse_args()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_target_weight_names(weight_map):
    names = []
    for name in weight_map:
        if not name.startswith("model.transformer.blocks."):
            continue
        if any(name.endswith(suffix) for suffix in TARGET_SUFFIXES):
            names.append(name)
    suffix_order = {suffix: idx for idx, suffix in enumerate(TARGET_SUFFIXES)}

    def sort_key(name):
        parts = name.split(".")
        block_idx = int(parts[3])
        suffix_idx = next(
            suffix_order[suffix]
            for suffix in TARGET_SUFFIXES
            if name.endswith(suffix)
        )
        return block_idx, suffix_idx

    return sorted(names, key=sort_key)


def infer_num_layers(target_weight_names):
    block_ids = []
    for name in target_weight_names:
        parts = name.split(".")
        if len(parts) > 3 and parts[:3] == ["model", "transformer", "blocks"]:
            block_ids.append(int(parts[3]))
    return max(block_ids) + 1 if block_ids else 0


def block_weight_names(target_weight_names, block_idx):
    prefix = f"model.transformer.blocks.{block_idx}."
    return [name for name in target_weight_names if name.startswith(prefix)]


def selected_block_range(args, target_weight_names):
    num_layers = infer_num_layers(target_weight_names)
    end_block = args.end_block if args.end_block is not None else num_layers
    if args.start_block < 0:
        raise ValueError("--start_block must be non-negative.")
    if end_block < args.start_block:
        raise ValueError("--end_block must be greater than or equal to --start_block.")
    return range(args.start_block, min(end_block, num_layers))


def selected_block_map(args, target_weight_names):
    block_map = {}
    for block_idx in selected_block_range(args, target_weight_names):
        names = block_weight_names(target_weight_names, block_idx)
        if names:
            block_map[block_idx] = names
    return block_map


def score_to_precision(scores, allocate_ratio):
    if not 0 <= allocate_ratio <= 0.5:
        raise ValueError("--allocate_ratio must be in [0, 0.5].")
    sorted_items = sorted(scores.items(), key=lambda item: item[1])
    num_k = int(len(sorted_items) * allocate_ratio)
    precision_map = {}

    for idx, (name, _) in enumerate(sorted_items):
        if idx < num_k:
            precision_map[name] = 1
        elif idx >= len(sorted_items) - num_k:
            precision_map[name] = 3
        else:
            precision_map[name] = 2
    return precision_map


def score_to_precision_grouped(scores, score_groups, allocate_ratio):
    if not 0 <= allocate_ratio <= 0.5:
        raise ValueError("--allocate_ratio must be in [0, 0.5].")

    grouped_items = {}
    for unit_key, score in scores.items():
        group_key = score_groups[unit_key]
        grouped_items.setdefault(group_key, []).append((unit_key, score))

    precision_map = {}
    for group_key, items in grouped_items.items():
        sorted_items = sorted(items, key=lambda item: item[1])
        num_k = int(len(sorted_items) * allocate_ratio)
        if num_k == 0:
            print(f"  ABMP group {group_key}: {len(sorted_items)} units, no 1/3-bit units at ratio={allocate_ratio}")

        for idx, (unit_key, _) in enumerate(sorted_items):
            if idx < num_k:
                precision_map[unit_key] = 1
            elif idx >= len(sorted_items) - num_k:
                precision_map[unit_key] = 3
            else:
                precision_map[unit_key] = 2

    return precision_map


def transformer_block_key(weight_name):
    parts = weight_name.split(".")
    if len(parts) >= 4 and parts[:3] == ["model", "transformer", "blocks"]:
        return ".".join(parts[:4])
    return "__unknown_transformer_block__"


def score_group_key(weight_name, rank_scope):
    if rank_scope == "global":
        return "__global__"
    if rank_scope == "transformer_block":
        return transformer_block_key(weight_name)
    if rank_scope == "weight":
        return weight_name
    raise ValueError(f"Unsupported ABMP rank scope: {rank_scope}")


def make_unit_key(weight_name, row_start, row_end, col_start, col_end):
    return f"{weight_name}::r{row_start}-{row_end}::c{col_start}-{col_end}"


def iter_quant_units(weight_name, shape, granularity, block_size):
    rows, cols = shape
    if granularity == "weight":
        yield weight_name, slice(0, rows), slice(0, cols)
        return

    if block_size <= 0:
        raise ValueError("--quant_block_size must be positive.")

    if granularity == "column":
        for col_start in range(0, cols, block_size):
            col_end = min(col_start + block_size, cols)
            unit_key = make_unit_key(weight_name, 0, rows, col_start, col_end)
            yield unit_key, slice(0, rows), slice(col_start, col_end)
        return

    if granularity == "row":
        for row_start in range(0, rows, block_size):
            row_end = min(row_start + block_size, rows)
            unit_key = make_unit_key(weight_name, row_start, row_end, 0, cols)
            yield unit_key, slice(row_start, row_end), slice(0, cols)
        return

    if granularity == "tile":
        for row_start in range(0, rows, block_size):
            row_end = min(row_start + block_size, rows)
            for col_start in range(0, cols, block_size):
                col_end = min(col_start + block_size, cols)
                unit_key = make_unit_key(weight_name, row_start, row_end, col_start, col_end)
                yield unit_key, slice(row_start, row_end), slice(col_start, col_end)
        return

    raise ValueError(f"Unsupported granularity: {granularity}")


def validate_granularity_args(args):
    if args.abmp_granularity is None:
        args.abmp_granularity = args.daq_granularity
    if args.daq_granularity != "weight" and args.quant_block_size <= 0:
        raise ValueError("--quant_block_size must be positive for block-level DAQ.")
    if args.abmp_granularity != "weight" and args.quant_block_size <= 0:
        raise ValueError("--quant_block_size must be positive for block-level ABMP.")


def precision_lookup_key_for_unit(weight_name, shape, row_slice, col_slice, abmp_granularity):
    rows, cols = shape
    row_start = 0 if row_slice.start is None else row_slice.start
    row_end = rows if row_slice.stop is None else row_slice.stop
    col_start = 0 if col_slice.start is None else col_slice.start
    col_end = cols if col_slice.stop is None else col_slice.stop

    if abmp_granularity == "weight":
        return weight_name
    if abmp_granularity == "column":
        return make_unit_key(weight_name, 0, rows, col_start, col_end)
    if abmp_granularity == "row":
        return make_unit_key(weight_name, row_start, row_end, 0, cols)
    if abmp_granularity == "tile":
        return make_unit_key(weight_name, row_start, row_end, col_start, col_end)
    raise ValueError(f"Unsupported ABMP granularity: {abmp_granularity}")


def build_default_precision_map(args):
    if args.daq_granularity == "weight":
        return None
    return {
        "__granularity__": args.daq_granularity,
        "__default_bits__": args.default_bits,
        "__quant_block_size__": args.quant_block_size,
    }


def get_precision_bits(precision_map, weight_name, unit_key, default_bits):
    if unit_key in precision_map:
        return int(precision_map[unit_key])
    if weight_name in precision_map:
        return int(precision_map[weight_name])
    if "__default_bits__" in precision_map:
        return int(precision_map["__default_bits__"])
    return int(default_bits)


def resolve_z_cache_dir(args):
    if args.z_cache_dir is not None:
        return Path(args.z_cache_dir)
    return Path(args.output_dir) / "z_cache"


def build_max_memory(args):
    if torch.cuda.is_available():
        max_memory = {gpu_idx: args.gpu_memory for gpu_idx in range(torch.cuda.device_count())}
        max_memory["cpu"] = args.cpu_memory
        return max_memory
    return {"cpu": args.cpu_memory}


def z_block_path(z_cache_dir, block_idx):
    return Path(z_cache_dir) / f"block_{block_idx:02d}.safetensors"


def is_block_cache_ready(z_block_file, names):
    if not z_block_file.exists():
        return False
    with safe_open(str(z_block_file), framework="pt", device="cpu") as f:
        keys = set(f.keys())
    return all(name in keys for name in names)


def build_z_cache(args, model, weight_map, block_map, z_cache_dir):
    z_cache_dir = Path(z_cache_dir)
    z_cache_dir.mkdir(parents=True, exist_ok=True)

    missing_block_map = {}
    for block_idx, names in block_map.items():
        block_file = z_block_path(z_cache_dir, block_idx)
        if args.reuse_z and is_block_cache_ready(block_file, names):
            continue
        missing_block_map[block_idx] = names

    if not missing_block_map:
        print(f"Using existing Z cache at {z_cache_dir}")
        return

    missing_names = [name for names in missing_block_map.values() for name in names]
    print(f"Building Z cache for {len(missing_names)} weights across {len(missing_block_map)} blocks")

    mcs_mats = get_mcs_mat(
        model=model,
        model_path=args.model_path,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        num_steps=args.num_steps,
        gamma=args.gamma,
        group_size=args.group_size,
        schedule=args.schedule,
        max_docs=args.max_docs,
        source=args.source,
        target_weight_names=missing_names,
        max_mcs_samples=args.max_mcs_samples,
        include_full_visible=not args.mcs_no_full_visible,
        mcs_normalization=args.mcs_normalization,
    )

    for block_idx, names in missing_block_map.items():
        print(f"Saving Z cache for block {block_idx}: {len(names)} weights")
        block_tensors = {}
        for name in names:
            z_tensor = build_imp_mat_from_mcs(
                name,
                mcs_mats[name],
                args.model_path,
                args.damping,
                weight_map=weight_map,
                row_center=args.z_row_center,
                row_center_granularity=args.daq_granularity,
                row_center_block_size=args.quant_block_size,
                inverse_diag_floor=args.inverse_diag_floor,
            )
            block_tensors[name] = z_tensor.to(quant_dtype(args.z_save_dtype)).contiguous()
            del z_tensor
        save_file(block_tensors, str(z_block_path(z_cache_dir, block_idx)), metadata={"format": "pt"})
        del block_tensors
        gc.collect()

    del mcs_mats
    gc.collect()


def compute_abmp_scores_from_cache(block_map, z_cache_dir):
    scores = {}
    for block_idx, names in block_map.items():
        block_file = z_block_path(z_cache_dir, block_idx)
        print(f"Scoring block {block_idx} from cache")
        with safe_open(str(block_file), framework="pt", device="cpu") as f:
            for name in names:
                scores[name] = f.get_tensor(name).to(torch.float32).sum().item()
    return scores


def compute_unit_abmp_scores_from_cache(args, block_map, z_cache_dir):
    scores = {}
    score_groups = {}
    granularity = args.abmp_granularity

    for block_idx, names in block_map.items():
        block_file = z_block_path(z_cache_dir, block_idx)
        print(f"Scoring transformer block {block_idx} from cache at granularity={granularity}")
        with safe_open(str(block_file), framework="pt", device="cpu") as f:
            for weight_name in names:
                z_tensor = f.get_tensor(weight_name).to(torch.float32)
                unit_count = 0
                for unit_key, row_slice, col_slice in iter_quant_units(
                    weight_name,
                    z_tensor.shape,
                    granularity,
                    args.quant_block_size,
                ):
                    scores[unit_key] = z_tensor[row_slice, col_slice].sum().item()
                    score_groups[unit_key] = score_group_key(weight_name, args.abmp_rank_scope)
                    unit_count += 1
                print(f"  scored {weight_name}: {unit_count} units")
                del z_tensor
        gc.collect()

    return scores, score_groups


def quant_dtype(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported save dtype: {name}")


def resolve_daq_component_params(args):
    if args.daq_component == "baseline":
        return 0, 1.0
    if args.daq_component == "rsr_no_dor":
        return args.daq_steps, 1.0
    if args.daq_component == "rsr_w_dor":
        return args.daq_steps, args.imp_lamda
    return args.daq_steps, args.imp_lamda


def run_daq_unit(weight, imp_mat, bits, args, apply_row_center=True, imp_mask=None):
    device = torch.device(args.daq_device)
    daq_steps, imp_lamda = resolve_daq_component_params(args)
    weight_device = weight.to(device)
    imp_device = imp_mat.to(device)
    mask_device = imp_mask.to(device) if imp_mask is not None else None

    row_mean = None
    daq_weight = weight_device
    if apply_row_center and args.daq_row_center:
        row_mean = weight_device.mean(dim=1, keepdim=True)
        daq_weight = weight_device - row_mean

    quantized = DAQ(
        weight=daq_weight,
        imp_mat=imp_device,
        bits=bits,
        daq_steps=daq_steps,
        lamda=imp_lamda,
        imp_mask=mask_device,
    )
    if row_mean is not None:
        quantized = quantized + row_mean

    quantized_cpu = quantized.detach().cpu()
    del weight_device, imp_device, mask_device, daq_weight, row_mean, quantized
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return quantized_cpu


def precision_bit_counts(args, units, precision_map, weight_name, shape, default_bits):
    counts = {}
    for unit_key, row_slice, col_slice in units:
        lookup_key = precision_lookup_key_for_unit(
            weight_name,
            shape,
            row_slice,
            col_slice,
            args.abmp_granularity,
        )
        bits = get_precision_bits(precision_map, weight_name, lookup_key, default_bits)
        counts[bits] = counts.get(bits, 0) + 1
    return counts


def quantize_blocks_from_cache(args, weight_map, block_map, precision_map, z_cache_dir):
    quant_dir = Path(args.output_dir) / "quantized_blocks"
    quant_dir.mkdir(parents=True, exist_ok=True)
    quantized_tensor_map = {}
    save_dtype = quant_dtype(args.save_dtype)
    granularity = args.daq_granularity

    for block_idx, names in block_map.items():
        print(f"Quantizing block {block_idx}: {len(names)} weights")
        quantized_block = {}
        block_file = z_block_path(z_cache_dir, block_idx)
        with safe_open(str(block_file), framework="pt", device="cpu") as z_src:
            for weight_name in names:
                W = load_weight_from_checkpoint(weight_name, weight_map, args.model_path)
                Z = z_src.get_tensor(weight_name).to(dtype=torch.float32)
                assert W.shape == Z.shape, (weight_name, W.shape, Z.shape)
                if args.dor_mask_scope == "weight":
                    _, imp_lamda = resolve_daq_component_params(args)
                    weight_imp_mask = build_imp_mask(Z, imp_lamda)
                else:
                    weight_imp_mask = None

                units = list(
                    iter_quant_units(
                        weight_name,
                        W.shape,
                        granularity,
                        args.quant_block_size,
                    )
                )
                bit_counts = precision_bit_counts(args, units, precision_map, weight_name, W.shape, args.default_bits)
                print(
                    f"  DAQ {weight_name}, granularity={granularity}, "
                    f"units={len(units)}, bits={bit_counts}, shape={tuple(W.shape)}"
                )

                if granularity == "weight":
                    bits = get_precision_bits(precision_map, weight_name, weight_name, args.default_bits)
                    Q = run_daq_unit(W, Z, bits, args, imp_mask=weight_imp_mask)
                else:
                    if args.daq_row_center:
                        row_mean = W.mean(dim=1, keepdim=True)
                        daq_source = W - row_mean
                    else:
                        row_mean = None
                        daq_source = W

                    Q = torch.empty_like(W)
                    for unit_idx, (unit_key, row_slice, col_slice) in enumerate(units, start=1):
                        precision_lookup_key = precision_lookup_key_for_unit(
                            weight_name,
                            W.shape,
                            row_slice,
                            col_slice,
                            args.abmp_granularity,
                        )
                        bits = get_precision_bits(
                            precision_map,
                            weight_name,
                            precision_lookup_key,
                            args.default_bits,
                        )
                        unit_imp_mask = (
                            weight_imp_mask[row_slice, col_slice].contiguous()
                            if weight_imp_mask is not None
                            else None
                        )
                        Q[row_slice, col_slice] = run_daq_unit(
                            daq_source[row_slice, col_slice].contiguous(),
                            Z[row_slice, col_slice].contiguous(),
                            bits,
                            args,
                            apply_row_center=False,
                            imp_mask=unit_imp_mask,
                        )
                        if args.log_unit_interval > 0 and unit_idx % args.log_unit_interval == 0:
                            print(f"    finished {unit_idx}/{len(units)} units for {weight_name}")

                    if row_mean is not None:
                        Q += row_mean
                    del daq_source, row_mean

                assert Q.shape == W.shape
                assert torch.isfinite(Q).all()
                err = torch.norm(W - Q).item()
                rel_err = (torch.norm(W - Q) / (torch.norm(W) + 1e-12)).item()
                print(f"  error_norm={err:.6f}, rel_error={rel_err:.6f}")
                quantized_block[weight_name] = Q.to(save_dtype).contiguous()
                del W, Z, Q, units, weight_imp_mask

        block_file = quant_dir / f"block_{block_idx:02d}.safetensors"
        save_file(quantized_block, str(block_file), metadata={"format": "pt"})
        for name in quantized_block:
            quantized_tensor_map[name] = str(block_file)
        del quantized_block
        gc.collect()

    return quantized_tensor_map


def copy_model_assets(model_path, output_dir):
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in model_path.iterdir():
        if not src.is_file():
            continue
        if src.name == "model.safetensors.index.json":
            continue
        if src.suffix == ".safetensors":
            continue
        shutil.copy2(src, output_dir / src.name)


def rewrite_full_checkpoint(model_path, output_dir, quantized_tensor_map):
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_to_names = {}
    for name, shard in weight_map.items():
        shard_to_names.setdefault(shard, []).append(name)

    total_size = 0
    for shard_name, names in shard_to_names.items():
        print(f"Writing checkpoint shard {shard_name}")
        shard_tensors = {}
        with safe_open(str(model_path / shard_name), framework="pt", device="cpu") as src:
            with ExitStack() as stack:
                quant_cache = {}
                for name in names:
                    if name in quantized_tensor_map:
                        q_path = quantized_tensor_map[name]
                        if q_path not in quant_cache:
                            quant_cache[q_path] = stack.enter_context(
                                safe_open(q_path, framework="pt", device="cpu")
                            )
                        tensor = quant_cache[q_path].get_tensor(name)
                    else:
                        tensor = src.get_tensor(name)
                    shard_tensors[name] = tensor.contiguous()
                    total_size += tensor.numel() * tensor.element_size()

        save_file(shard_tensors, str(output_dir / shard_name), metadata={"format": "pt"})
        del shard_tensors
        gc.collect()

    new_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)


def main():
    args = parse_args()
    validate_granularity_args(args)
    effective_daq_steps, effective_imp_lamda = resolve_daq_component_params(args)
    print(
        f"DAQ component: {args.daq_component} "
        f"(effective_daq_steps={effective_daq_steps}, "
        f"effective_imp_lamda={effective_imp_lamda}, "
        f"dor_mask_scope={args.dor_mask_scope}, "
        f"daq_row_center={args.daq_row_center}, "
        f"z_row_center={args.z_row_center})"
    )
    print(
        "MCS states: "
        + (
            "num_steps endpoint-inclusive states excluding the fully-visible endpoint"
            if args.mcs_no_full_visible
            else "num_steps endpoint-inclusive states over [0, 1]"
        )
    )
    output_dir = Path(args.output_dir)
    if output_dir.resolve() == Path(args.model_path).resolve():
        raise ValueError("--output_dir must not be the same as --model_path.")
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_map = load_weight_map(args.model_path)
    all_target_weight_names = get_target_weight_names(weight_map)
    block_map = selected_block_map(args, all_target_weight_names)
    target_weight_names = [name for names in block_map.values() for name in names]
    print(f"Target weights: {len(target_weight_names)}")

    z_cache_dir = resolve_z_cache_dir(args)
    need_model = True
    if args.reuse_z:
        need_model = any(
            not is_block_cache_ready(z_block_path(z_cache_dir, block_idx), names)
            for block_idx, names in block_map.items()
        )

    model = None
    if need_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=build_max_memory(args),
            offload_folder=args.offload_folder,
        )
        build_z_cache(args, model, weight_map, block_map, z_cache_dir)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"Reusing cached Z tensors from {z_cache_dir}")

    if args.only_build_z:
        print(f"Z cache ready at {z_cache_dir}")
        return

    if args.skip_abmp:
        precision_map = {name: args.default_bits for name in target_weight_names}
    else:
        if args.abmp_granularity == "weight":
            scores = compute_abmp_scores_from_cache(block_map, z_cache_dir)
            precision_map = score_to_precision(scores, args.allocate_ratio)
        else:
            scores, score_groups = compute_unit_abmp_scores_from_cache(args, block_map, z_cache_dir)
            precision_map = score_to_precision_grouped(scores, score_groups, args.allocate_ratio)
        save_json(scores, output_dir / "importance_scores.json")

    save_json(precision_map, output_dir / "precision_map.json")

    quantized_tensor_map = quantize_blocks_from_cache(
        args,
        weight_map,
        block_map,
        precision_map,
        z_cache_dir,
    )

    save_json(quantized_tensor_map, output_dir / "quantized_tensor_map.json")
    copy_model_assets(args.model_path, output_dir)
    rewrite_full_checkpoint(args.model_path, output_dir, quantized_tensor_map)
    print(f"Saved fake-quantized model to {output_dir}")


if __name__ == '__main__':
    main()
