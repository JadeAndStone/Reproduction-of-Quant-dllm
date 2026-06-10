#!/usr/bin/env python
import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MCS import load_dataset as mcs
from main import TARGET_SUFFIXES, build_max_memory, get_target_weight_names
from utils.generate_imp_mat import inverse_diag_adaptive, load_weight_map, load_weight_from_checkpoint


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def block_idx_from_weight(weight_name):
    parts = weight_name.split(".")
    if len(parts) >= 4 and parts[:3] == ["model", "transformer", "blocks"]:
        return int(parts[3])
    raise ValueError(f"Cannot infer block index from {weight_name}")


def z_block_path(z_cache_dir, block_idx):
    return Path(z_cache_dir) / f"block_{block_idx:02d}.safetensors"


def load_z_from_cache(z_cache_dir, weight_name):
    with safe_open(str(z_block_path(z_cache_dir, block_idx_from_weight(weight_name))), framework="pt", device="cpu") as handle:
        return handle.get_tensor(weight_name).to(torch.float32)


def selected_weight_names(weight_map, blocks, modules, explicit_targets):
    if explicit_targets:
        return [target for target in explicit_targets if target in weight_map]
    names = []
    for block_idx in blocks:
        for module_name in modules:
            weight_name = f"model.transformer.blocks.{block_idx}.{module_name}.weight"
            if weight_name in weight_map:
                names.append(weight_name)
    return names


def module_lookup_name(weight_name):
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected a weight tensor name, got {weight_name}")
    return weight_name[: -len(".weight")]


def register_full_smcs_hooks(model, target_weight_names, max_tokens_per_state, seed):
    target_modules = {module_lookup_name(name): name for name in target_weight_names}
    covs = {}
    token_counts = {}
    handles = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def make_hook(weight_name):
        def hook(module, inputs, output):
            hidden = inputs[0].detach()
            hidden = hidden.reshape(-1, hidden.shape[-1])
            total_tokens = hidden.shape[0]
            if max_tokens_per_state and total_tokens > max_tokens_per_state:
                sample_idx = torch.randperm(total_tokens, generator=generator)[:max_tokens_per_state]
                hidden = hidden[sample_idx.to(hidden.device)]
                scale = float(total_tokens) / float(max_tokens_per_state)
            else:
                scale = 1.0

            hidden = hidden.to(torch.float32)
            device = hidden.device
            if weight_name not in covs:
                dim = hidden.shape[-1]
                covs[weight_name] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
                token_counts[weight_name] = 0
            covs[weight_name] += scale * (hidden.T @ hidden)
            token_counts[weight_name] += total_tokens

        return hook

    for name, module in model.named_modules():
        if name not in target_modules:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Target module {name} is not nn.Linear: {type(module)}")
        weight_name = target_modules[name]
        handles.append(module.register_forward_hook(make_hook(weight_name)))

    missing = set(target_modules) - {
        name for name, module in model.named_modules() if name in target_modules
    }
    if missing:
        raise KeyError(f"Missing target modules: {sorted(missing)}")
    return handles, covs, token_counts


def build_calibset(args):
    _, calibset, mcs_step_count = mcs.get_calib(
        args.model_path,
        args.nsamples,
        args.seed,
        args.seqlen,
        args.num_steps,
        args.gamma,
        args.schedule,
        args.max_docs,
        args.source,
        include_full_visible=not args.mcs_no_full_visible,
    )
    calibset = torch.concat(calibset, dim=0)
    if args.max_mcs_samples is not None and args.max_mcs_samples > 0:
        calibset = calibset[: args.max_mcs_samples]
        mcs_step_count = min(mcs_step_count, len(calibset))
    if len(calibset) == 0:
        raise ValueError("Calibration set is empty.")
    return calibset, mcs_step_count


def inverse_diag(cov, damping, device, min_diag=1e-12, context="", return_info=False):
    work = cov.to(device=device, dtype=torch.float32)
    diag, used_damping = inverse_diag_adaptive(
        work,
        damping,
        min_diag=min_diag,
        context=context or "full_smcs",
    )
    diag = diag.detach().to("cpu")
    del work
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if return_info:
        return diag, {"used_damping": used_damping}
    return diag


def safe_corr(x, y):
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if den.item() == 0:
        return 0.0
    return float(torch.dot(x, y).item() / den.item())


def rankdata(values):
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    return ranks


def top_overlap(values_a, values_b, ratio):
    count = max(1, int(len(values_a) * ratio))
    top_a = set(torch.topk(values_a, count).indices.tolist())
    top_b = set(torch.topk(values_b, count).indices.tolist())
    bottom_a = set(torch.topk(-values_a, count).indices.tolist())
    bottom_b = set(torch.topk(-values_b, count).indices.tolist())
    return {
        "count": count,
        "top_overlap": len(top_a & top_b) / count,
        "bottom_overlap": len(bottom_a & bottom_b) / count,
    }


def score_groups(z, granularity, block_size):
    rows, cols = z.shape
    scores = []
    if granularity == "column":
        for col_start in range(0, cols, block_size):
            col_end = min(col_start + block_size, cols)
            scores.append(float(z[:, col_start:col_end].sum().item()))
    elif granularity == "tile":
        for row_start in range(0, rows, block_size):
            row_end = min(row_start + block_size, rows)
            for col_start in range(0, cols, block_size):
                col_end = min(col_start + block_size, cols)
                scores.append(float(z[row_start:row_end, col_start:col_end].sum().item()))
    else:
        raise ValueError(f"Unsupported score granularity: {granularity}")
    return torch.tensor(scores, dtype=torch.float64)


def top_score_share(scores, ratio):
    count = max(1, int(len(scores) * ratio))
    total = float(scores.sum().item())
    top = float(torch.topk(scores, count).values.sum().item())
    bottom = float(torch.topk(-scores, count).values.neg().sum().item())
    return {
        "count": count,
        "top_score_share": top / total if total else 0.0,
        "bottom_score_share": bottom / total if total else 0.0,
    }


def build_mask(z):
    return (z - z.mean()).abs() > 3 * z.std()


def mask_jaccard(mask_a, mask_b):
    inter = torch.logical_and(mask_a, mask_b).sum().item()
    union = torch.logical_or(mask_a, mask_b).sum().item()
    return float(inter / union) if union else 0.0




def blockdiag_inverse_diag_from_cov(cov, damping, block_size, device, min_diag=1e-12):
    cov = cov.to(device=device, dtype=torch.float32)
    dim = cov.shape[0]
    pieces = []
    for start in range(0, dim, block_size):
        end = min(start + block_size, dim)
        block = cov[start:end, start:end]
        pieces.append(
            inverse_diag(
                block,
                damping,
                device,
                min_diag=min_diag,
                context=f"blockdiag[{start}:{end}]",
            )
        )
    del cov
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(pieces, dim=0)


def compare_z_pair(z_ref, z_other, block_size, allocate_ratio):
    out = {
        "z_sum_ref": float(z_ref.sum().item()),
        "z_sum_other": float(z_other.sum().item()),
        "z_element_pearson": safe_corr(z_ref.flatten(), z_other.flatten()),
        "z_element_spearman": safe_corr(rankdata(z_ref.flatten()), rankdata(z_other.flatten())),
        "mask_weight_jaccard": mask_jaccard(build_mask(z_ref), build_mask(z_other)),
        "granularity": {},
    }
    for granularity in ("column", "tile"):
        ref_scores = score_groups(z_ref, granularity, block_size)
        other_scores = score_groups(z_other, granularity, block_size)
        out["granularity"][granularity] = {
            "groups": int(len(ref_scores)),
            "score_pearson": safe_corr(ref_scores, other_scores),
            "score_spearman": safe_corr(rankdata(ref_scores), rankdata(other_scores)),
            "top_bottom_overlap": top_overlap(ref_scores, other_scores, allocate_ratio),
            "ref_top_score_share": top_score_share(ref_scores, allocate_ratio),
            "other_top_score_share": top_score_share(other_scores, allocate_ratio),
        }
    return out

def compare_weight(args, weight_name, full_cov, normalizer, weight_map):
    full_cov = full_cov / float(normalizer)
    diag_full, inverse_info = inverse_diag(
        full_cov,
        args.damping,
        args.inverse_device,
        min_diag=args.inverse_diag_floor,
        context=weight_name,
        return_info=True,
    )
    diag_block_same = blockdiag_inverse_diag_from_cov(
        full_cov,
        args.damping,
        args.quant_block_size,
        args.inverse_device,
        min_diag=args.inverse_diag_floor,
    )
    scale_full = 1.0 / diag_full.clamp_min(args.inverse_diag_floor)
    scale_block_same = 1.0 / diag_block_same.clamp_min(args.inverse_diag_floor)

    weight = load_weight_from_checkpoint(weight_name, weight_map, args.model_path)
    z_full = (weight * scale_full.unsqueeze(0)) ** 2
    z_block_same = (weight * scale_block_same.unsqueeze(0)) ** 2
    z_block_cache = load_z_from_cache(args.z_cache_dir, weight_name)

    result = {
        "shape": list(weight.shape),
        "diag_inv_full_min": float(diag_full.min().item()),
        "diag_inv_full_max": float(diag_full.max().item()),
        "diag_inv_block_same_min": float(diag_block_same.min().item()),
        "diag_inv_block_same_max": float(diag_block_same.max().item()),
        "inverse": inverse_info,
        "against_same_sample_blockdiag": compare_z_pair(
            z_full,
            z_block_same,
            args.quant_block_size,
            args.allocate_ratio,
        ),
        "against_cache_blockdiag": compare_z_pair(
            z_full,
            z_block_cache,
            args.quant_block_size,
            args.allocate_ratio,
        ),
    }

    del weight, z_full, z_block_same, z_block_cache, diag_full, diag_block_same, scale_full, scale_block_same
    gc.collect()
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--z_cache_dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks", default="0")
    parser.add_argument("--modules", default="q_proj,attn_out")
    parser.add_argument("--targets", default="")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-mcs-samples", type=int, default=4)
    parser.add_argument("--max-tokens-per-state", type=int, default=512)
    parser.add_argument("--mcs_no_full_visible", action="store_true")
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--mcs-normalization", choices=["tokens", "mcs_steps"], default="tokens")
    parser.add_argument("--inverse-diag-floor", type=float, default=1e-12)
    parser.add_argument("--allocate_ratio", type=float, default=0.05)
    parser.add_argument("--quant_block_size", type=int, default=128)
    parser.add_argument("--max-input-dim", type=int, default=4096)
    parser.add_argument("--inverse-device", default="cuda:0")
    parser.add_argument("--gpu_memory", default="15GiB")
    parser.add_argument("--cpu_memory", default="80GiB")
    parser.add_argument("--offload_folder", default="/root/data-tmp/offload/full_smcs_diag")
    return parser.parse_args()


def main():
    args = parse_args()
    weight_map = load_weight_map(args.model_path)
    all_target_names = get_target_weight_names(weight_map)
    valid_targets = set(all_target_names)
    blocks = [int(item) for item in args.blocks.split(",") if item]
    modules = [item for item in args.modules.split(",") if item]
    explicit_targets = [item for item in args.targets.split(",") if item]
    target_weight_names = selected_weight_names(weight_map, blocks, modules, explicit_targets)
    target_weight_names = [name for name in target_weight_names if name in valid_targets]
    if not target_weight_names:
        raise ValueError("No target weights selected.")

    print("Loading model for full SMCS diagnostic", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=build_max_memory(args),
        offload_folder=args.offload_folder,
    )

    module_dims = {}
    for module_name, module in model.named_modules():
        weight_name = module_name + ".weight"
        if weight_name in target_weight_names:
            module_dims[weight_name] = module.in_features
    target_weight_names = [
        name for name in target_weight_names if module_dims.get(name, 10**9) <= args.max_input_dim
    ]
    if not target_weight_names:
        raise ValueError(f"All selected targets exceed --max-input-dim={args.max_input_dim}.")
    print("Targets:", *target_weight_names, sep="\n  ", flush=True)

    handles, covs, token_counts = register_full_smcs_hooks(
        model,
        target_weight_names,
        args.max_tokens_per_state,
        args.seed,
    )
    calibset, mcs_step_count = build_calibset(args)
    input_device = model.model.device
    print(
        f"Running {len(calibset)} MCS states, mcs_steps={mcs_step_count}, "
        f"mcs_normalization={args.mcs_normalization}, "
        f"max_tokens_per_state={args.max_tokens_per_state}",
        flush=True,
    )
    for idx, batch in enumerate(calibset, start=1):
        print(f"  full SMCS forward {idx}/{len(calibset)}", flush=True)
        batch = batch.unsqueeze(0).to(input_device)
        with torch.no_grad():
            model.model(input_ids=batch, use_cache=False, last_logits_only=True)

    for handle in handles:
        handle.remove()
    del model, calibset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output = {
        "model_path": args.model_path,
        "z_cache_dir": args.z_cache_dir,
        "targets": target_weight_names,
        "config": {
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "num_steps": args.num_steps,
            "gamma": args.gamma,
            "schedule": args.schedule,
            "max_mcs_samples": args.max_mcs_samples,
            "max_tokens_per_state": args.max_tokens_per_state,
            "mcs_no_full_visible": args.mcs_no_full_visible,
            "damping": args.damping,
            "allocate_ratio": args.allocate_ratio,
            "quant_block_size": args.quant_block_size,
            "mcs_step_count": mcs_step_count,
            "mcs_normalization": args.mcs_normalization,
            "inverse_diag_floor": args.inverse_diag_floor,
        },
        "weights": {},
    }

    for weight_name in target_weight_names:
        if weight_name not in covs:
            raise KeyError(f"No covariance collected for {weight_name}")
        print(f"Comparing full SMCS vs blockdiag Z: {weight_name}", flush=True)
        tokens_seen = token_counts.get(weight_name, 0)
        normalizer = tokens_seen if args.mcs_normalization == "tokens" else mcs_step_count
        output["weights"][weight_name] = compare_weight(
            args,
            weight_name,
            covs[weight_name],
            normalizer,
            weight_map,
        )
        output["weights"][weight_name]["tokens_seen"] = tokens_seen
        output["weights"][weight_name]["normalizer"] = normalizer
        del covs[weight_name]
        gc.collect()

    save_json(output, args.output)
    print(f"Saved full SMCS diagnostic to {args.output}", flush=True)


if __name__ == "__main__":
    main()
