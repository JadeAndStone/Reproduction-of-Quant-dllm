#!/usr/bin/env python
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MCS import load_dataset as mcs
from main import (
    C4_TRAIN_URL,
    MODEL_PATH,
    block_weight_names,
    build_max_memory,
    compute_abmp_scores_from_cache,
    compute_unit_abmp_scores_from_cache,
    compute_z_and_inverse_factor,
    copy_model_assets,
    get_target_weight_names,
    infer_num_layers,
    quant_dtype,
    quantize_blocks_from_cache,
    rewrite_full_checkpoint,
    score_to_precision,
    should_apply_fulls_refine,
    score_to_precision_grouped,
    validate_granularity_args,
    z_block_path,
)
from tools.build_full_smcs_z_cache import compute_z_for_weight
from tools.diagnose_full_smcs import register_full_smcs_hooks
from utils.generate_imp_mat import load_weight_map


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Layer-wise Quant-dLLM PTQ pipeline: MCS -> current block SMCS/Z -> "
            "current block ABMP/DAQ -> quantized hidden propagation."
        )
    )
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source", default=C4_TRAIN_URL)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max_mcs_samples", type=int, default=None)
    parser.add_argument("--mcs_no_full_visible", action="store_true")
    parser.add_argument("--mcs_normalization", choices=["tokens", "mcs_steps"], default="tokens")
    parser.add_argument("--max_tokens_per_state", type=int, default=0)

    parser.add_argument("--allocate_ratio", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--damping_mode", choices=["absolute", "diag_mean"], default="absolute")
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--inverse_diag_floor", type=float, default=1e-12)
    parser.add_argument("--inverse_device", default="cuda:0")
    parser.add_argument("--daq_steps", type=int, default=10)
    parser.add_argument("--imp_lamda", type=float, default=2.0)
    parser.add_argument("--dor_mask_scope", choices=["unit", "weight"], default="unit")
    parser.add_argument("--daq_component", choices=["custom", "baseline", "rsr_no_dor", "rsr_w_dor"], default="rsr_w_dor")
    parser.add_argument("--daq_row_center", action="store_true")
    parser.add_argument("--daq_block_row_center", action="store_true")
    parser.add_argument("--z_row_center", action="store_true")
    parser.add_argument("--skip_abmp", action="store_true")
    parser.add_argument("--default_bits", type=int, default=2)
    parser.add_argument("--z_save_dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--save_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--daq_granularity", choices=["weight", "row", "column", "tile"], default="column")
    parser.add_argument("--abmp_granularity", choices=["weight", "row", "column", "tile"], default="column")
    parser.add_argument("--abmp_rank_scope", choices=["global", "transformer_block", "weight"], default="weight")
    parser.add_argument("--quant_block_size", type=int, default=128)
    parser.add_argument("--daq_device", default="cpu")
    parser.add_argument(
        "--daq_activation_diag",
        action="store_true",
        help=(
            "Diagnostic experiment: include activation covariance diagonal in DAQ "
            "alpha updates, minimizing diag(X^T X)-weighted weight error inside each unit."
        ),
    )
    parser.add_argument(
        "--daq_activation_diag_mode",
        choices=["activation", "dor_activation"],
        default="dor_activation",
        help="Solve weight for --daq_activation_diag: diag only or DOR mask squared times diag.",
    )
    parser.add_argument(
        "--gptq_compensation",
        action="store_true",
        help=(
            "Apply GPTQ/OBC-style block-wise compensation. Requires full SMCS "
            "covariances collected by this pipeline."
        ),
    )
    parser.add_argument("--gptq_compensation_device", default="cuda:0")
    parser.add_argument(
        "--daq_fulls_refine",
        action="store_true",
        help="Apply fixed-B full-S scale refinement after DAQ for selected Linear weights.",
    )
    parser.add_argument("--daq_fulls_device", default="cuda:0")
    parser.add_argument("--daq_fulls_steps", type=int, default=3)
    parser.add_argument("--daq_fulls_ridge", type=float, default=1e-5)
    parser.add_argument("--daq_fulls_target_modules", default="ff_proj,up_proj,ff_out")
    parser.add_argument("--daq_fulls_start_block", type=int, default=0)
    parser.add_argument("--daq_fulls_end_block", type=int, default=None)
    parser.add_argument("--log_unit_interval", type=int, default=0)

    parser.add_argument("--start_block", type=int, default=0)
    parser.add_argument("--end_block", type=int, default=None)
    parser.add_argument("--only_build_z", action="store_true")
    parser.add_argument("--z_cache_dir", default=None)
    parser.add_argument("--hidden_cache_dir", required=True)
    parser.add_argument("--hidden_chunk_size", type=int, default=1)
    parser.add_argument("--hidden_save_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--keep_hidden_cache", action="store_true")
    parser.add_argument("--reuse_initial_hidden", action="store_true")

    parser.add_argument("--gpu_memory", default="15GiB")
    parser.add_argument("--cpu_memory", default="120GiB")
    parser.add_argument("--offload_folder", default="/root/data-tmp/offload/layerwise_quant")
    return parser.parse_args()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


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
        raise RuntimeError("empty calibration set")
    return calibset, mcs_step_count


def module_device(module, fallback=None):
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        if tensor.device.type != "meta":
            return tensor.device
    if fallback is not None:
        return torch.device(fallback)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_block(model, block_idx):
    if getattr(model.model.config, "block_group_size", 1) != 1:
        raise NotImplementedError("Layer-wise pipeline currently requires block_group_size == 1.")
    return model.model.transformer.blocks[block_idx]


def chunk_file(cache_dir, chunk_idx):
    return Path(cache_dir) / f"chunk_{chunk_idx:06d}.safetensors"


def write_manifest(cache_dir, total_states, chunk_size, hidden_shape, dtype_name):
    save_json(
        {
            "total_states": int(total_states),
            "chunk_size": int(chunk_size),
            "hidden_shape": list(hidden_shape),
            "dtype": dtype_name,
        },
        Path(cache_dir) / "manifest.json",
    )


def list_chunk_files(cache_dir):
    return sorted(Path(cache_dir).glob("chunk_*.safetensors"))


def load_hidden(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor("hidden")


def save_hidden(path, hidden, dtype):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"hidden": hidden.detach().cpu().to(dtype).contiguous()}, str(path), metadata={"format": "pt"})


def embed_inputs(model, input_ids):
    core = model.model
    embed_device = module_device(core.transformer.wte, fallback=getattr(core, "device", None))
    input_ids = input_ids.to(embed_device)
    x = core.transformer.wte(input_ids)
    if core.config.input_emb_norm:
        x = x * (core.config.d_model**0.5)
    if not (core.config.alibi or core.config.rope):
        seq_len = input_ids.shape[1]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        x = core.transformer.wpe(pos) + x
    return core.transformer.emb_drop(x)


def build_initial_hidden_cache(args, model, calibset, cache_dir):
    cache_dir = Path(cache_dir)
    manifest = cache_dir / "manifest.json"
    if args.reuse_initial_hidden and manifest.exists() and list_chunk_files(cache_dir):
        print(f"Reusing initial hidden cache at {cache_dir}", flush=True)
        return
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    save_dtype = quant_dtype(args.hidden_save_dtype)
    total = len(calibset)
    print(f"Building initial hidden cache: states={total}, chunk_size={args.hidden_chunk_size}", flush=True)
    first_shape = None
    chunk_idx = 0
    model.eval()
    for start in range(0, total, args.hidden_chunk_size):
        end = min(start + args.hidden_chunk_size, total)
        input_ids = calibset[start:end]
        with torch.no_grad():
            hidden = embed_inputs(model, input_ids)
        if first_shape is None:
            first_shape = tuple(hidden.shape[1:])
        save_hidden(chunk_file(cache_dir, chunk_idx), hidden, save_dtype)
        if chunk_idx == 0 or end == total or (chunk_idx + 1) % 64 == 0:
            print(f"  embedded states {end}/{total}", flush=True)
        del hidden, input_ids
        chunk_idx += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_manifest(cache_dir, total, args.hidden_chunk_size, first_shape, args.hidden_save_dtype)


def forward_block(model, block_idx, hidden):
    block = get_block(model, block_idx)
    device = module_device(block, fallback=getattr(model.model, "device", None))
    hidden = hidden.to(device)
    output, _ = block(hidden, attention_bias=None, layer_past=None, use_cache=False)
    return output


def collect_block_covs(args, model, block_idx, names, current_cache_dir):
    print(f"Block {block_idx}: collecting layer-wise SMCS for {len(names)} weights", flush=True)
    handles, covs, token_counts = register_full_smcs_hooks(
        model,
        names,
        args.max_tokens_per_state,
        args.seed + block_idx * 100000,
    )
    files = list_chunk_files(current_cache_dir)
    try:
        with torch.no_grad():
            for idx, path in enumerate(files, start=1):
                hidden = load_hidden(path)
                output = forward_block(model, block_idx, hidden)
                del hidden, output
                if idx == 1 or idx == len(files) or idx % 64 == 0:
                    print(f"  block {block_idx}: collect chunk {idx}/{len(files)}", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
    return covs, token_counts


def save_block_z(args, block_idx, names, covs, token_counts, mcs_step_count, weight_map, z_cache_dir):
    z_cache_dir = Path(z_cache_dir)
    z_cache_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    reports = {}
    inverse_chol_map = {}
    activation_diag_map = {}
    fulls_cov_map = {}
    fulls_normalizer_map = {}
    for name in names:
        normalizer = token_counts[name] if args.mcs_normalization == "tokens" else mcs_step_count
        print(f"  block {block_idx}: inverse+Z {name}, normalizer={normalizer}", flush=True)
        if getattr(args, "daq_activation_diag", False):
            activation_diag_map[name] = (
                torch.diagonal(covs[name]).detach().cpu().to(torch.float32) / float(normalizer)
            ).contiguous()
        if should_apply_fulls_refine(args, name):
            fulls_cov_map[name] = covs[name].detach().cpu().to(torch.float32).contiguous()
            fulls_normalizer_map[name] = int(normalizer)
        if args.gptq_compensation:
            z_tensor, inverse_info, inverse_chol_upper = compute_z_and_inverse_factor(
                args,
                name,
                covs[name],
                normalizer,
                weight_map,
            )
            inverse_chol_map[name] = inverse_chol_upper
        else:
            z_tensor, inverse_info = compute_z_for_weight(args, name, covs[name], normalizer, weight_map)
        tensors[name] = z_tensor.to(quant_dtype(args.z_save_dtype)).contiguous()
        reports[name] = inverse_info
        del z_tensor, covs[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out_path = z_block_path(z_cache_dir, block_idx)
    save_file(tensors, str(out_path), metadata={"format": "pt"})
    save_json(reports, z_cache_dir / f"block_{block_idx:02d}_inverse_report.json")
    del tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path, inverse_chol_map, activation_diag_map, fulls_cov_map, fulls_normalizer_map


def precision_for_block(args, block_idx, names, z_cache_dir):
    block_map = {block_idx: names}
    if args.skip_abmp:
        return {name: args.default_bits for name in names}, {}
    if args.abmp_granularity == "weight":
        scores = compute_abmp_scores_from_cache(block_map, z_cache_dir)
        precision_map = score_to_precision(scores, args.allocate_ratio)
    else:
        scores, score_groups = compute_unit_abmp_scores_from_cache(args, block_map, z_cache_dir)
        precision_map = score_to_precision_grouped(scores, score_groups, args.allocate_ratio)
    return precision_map, scores


def apply_quantized_tensors(model, tensor_map):
    modules = dict(model.named_modules())
    for weight_name, tensor_path in tensor_map.items():
        module_name = weight_name[: -len(".weight")]
        if module_name not in modules:
            raise KeyError(f"Missing module for {weight_name}: {module_name}")
        module = modules[module_name]
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(weight_name)
        if module.weight.device.type == "meta":
            raise RuntimeError(f"Cannot update meta tensor for {weight_name}; use a device map that keeps blocks materialized.")
        module.weight.data.copy_(tensor.to(device=module.weight.device, dtype=module.weight.dtype))
        del tensor


def propagate_block(args, model, block_idx, current_cache_dir, next_cache_dir):
    current_cache_dir = Path(current_cache_dir)
    next_cache_dir = Path(next_cache_dir)
    if next_cache_dir.exists():
        shutil.rmtree(next_cache_dir)
    next_cache_dir.mkdir(parents=True, exist_ok=True)
    files = list_chunk_files(current_cache_dir)
    save_dtype = quant_dtype(args.hidden_save_dtype)
    first_shape = None
    print(f"Block {block_idx}: propagating quantized hidden cache", flush=True)
    with torch.no_grad():
        for idx, path in enumerate(files):
            hidden = load_hidden(path)
            output = forward_block(model, block_idx, hidden)
            if first_shape is None:
                first_shape = tuple(output.shape[1:])
            save_hidden(chunk_file(next_cache_dir, idx), output, save_dtype)
            if idx == 0 or idx + 1 == len(files) or (idx + 1) % 64 == 0:
                print(f"  block {block_idx}: propagate chunk {idx + 1}/{len(files)}", flush=True)
            del hidden, output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    total_states = json.load(open(current_cache_dir / "manifest.json"))["total_states"]
    write_manifest(next_cache_dir, total_states, args.hidden_chunk_size, first_shape, args.hidden_save_dtype)


def main():
    args = parse_args()
    validate_granularity_args(args)
    if args.start_block != 0:
        raise NotImplementedError("Layer-wise quantized propagation currently requires --start_block 0.")
    if args.hidden_chunk_size <= 0:
        raise ValueError("--hidden_chunk_size must be positive.")

    output_dir = Path(args.output_dir)
    if output_dir.resolve() == Path(args.model_path).resolve():
        raise ValueError("--output_dir must not equal --model_path.")
    output_dir.mkdir(parents=True, exist_ok=True)
    z_cache_dir = Path(args.z_cache_dir) if args.z_cache_dir else output_dir / "z_cache"
    hidden_root = Path(args.hidden_cache_dir)
    hidden_root.mkdir(parents=True, exist_ok=True)

    metadata = vars(args).copy()
    save_json(metadata, output_dir / "layerwise_args.json")

    weight_map = load_weight_map(args.model_path)
    target_weight_names = get_target_weight_names(weight_map)
    num_layers = infer_num_layers(target_weight_names)
    end_block = args.end_block if args.end_block is not None else num_layers
    selected_blocks = list(range(args.start_block, min(end_block, num_layers)))
    block_map = {idx: block_weight_names(target_weight_names, idx) for idx in selected_blocks}
    print(f"Layer-wise target blocks: {selected_blocks}", flush=True)
    print(f"Target weights: {sum(len(v) for v in block_map.values())}", flush=True)

    print("Loading model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=build_max_memory(args),
        offload_folder=args.offload_folder,
    )
    model.eval()

    calibset, mcs_step_count = build_calibset(args)
    print(
        f"Calibration states={len(calibset)}, mcs_step_count={mcs_step_count}, "
        f"mcs_normalization={args.mcs_normalization}",
        flush=True,
    )
    current_cache = hidden_root / "block_00"
    build_initial_hidden_cache(args, model, calibset, current_cache)
    del calibset
    gc.collect()

    all_precision_map = {}
    all_scores = {}
    quantized_tensor_map = {}

    for pos, block_idx in enumerate(selected_blocks):
        names = block_map[block_idx]
        print(f"===== Layer-wise block {block_idx} ({pos + 1}/{len(selected_blocks)}) =====", flush=True)
        covs, token_counts = collect_block_covs(args, model, block_idx, names, current_cache)
        (
            z_path,
            block_inverse_chol_map,
            block_activation_diag_map,
            block_fulls_cov_map,
            block_fulls_normalizer_map,
        ) = save_block_z(
            args,
            block_idx,
            names,
            covs,
            token_counts,
            mcs_step_count,
            weight_map,
            z_cache_dir,
        )
        print(f"Block {block_idx}: saved Z to {z_path}", flush=True)
        del covs, token_counts
        gc.collect()

        if args.only_build_z:
            block_precision_map = {name: args.default_bits for name in names}
            scores = {}
        else:
            block_precision_map, scores = precision_for_block(args, block_idx, names, z_cache_dir)
            all_precision_map.update(block_precision_map)
            all_scores.update(scores)
            save_json(all_precision_map, output_dir / "precision_map.json")
            if scores:
                save_json(all_scores, output_dir / "importance_scores.json")

            block_quantized_map = quantize_blocks_from_cache(
                args,
                weight_map,
                {block_idx: names},
                block_precision_map,
                z_cache_dir,
                cov_map=block_fulls_cov_map,
                normalizer_map=block_fulls_normalizer_map,
                inverse_chol_map=block_inverse_chol_map,
                activation_diag_map=block_activation_diag_map,
            )
            quantized_tensor_map.update(block_quantized_map)
            save_json(quantized_tensor_map, output_dir / "quantized_tensor_map.json")
            apply_quantized_tensors(model, block_quantized_map)

        del block_inverse_chol_map, block_activation_diag_map, block_fulls_cov_map, block_fulls_normalizer_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        is_last = pos == len(selected_blocks) - 1
        if not is_last:
            next_cache = hidden_root / f"block_{block_idx + 1:02d}"
            # In normal PTQ mode this propagates with the just-quantized block.
            # In --only_build_z mode the block remains FP, so the cache is FP-propagated.
            propagate_block(args, model, block_idx, current_cache, next_cache)
            if not args.keep_hidden_cache:
                shutil.rmtree(current_cache)
            current_cache = next_cache

    if args.only_build_z:
        print(f"Layer-wise Z cache ready at {z_cache_dir}", flush=True)
        return

    save_json(all_precision_map, output_dir / "precision_map.json")
    save_json(quantized_tensor_map, output_dir / "quantized_tensor_map.json")
    copy_model_assets(args.model_path, output_dir)
    rewrite_full_checkpoint(args.model_path, output_dir, quantized_tensor_map)
    if not args.keep_hidden_cache and current_cache.exists():
        shutil.rmtree(current_cache)
    print(f"Saved layer-wise fake-quantized model to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
