#!/usr/bin/env python
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    block_weight_names,
    build_imp_mask,
    build_max_memory,
    compute_z_and_inverse_factor,
    copy_model_assets,
    get_precision_bits,
    get_target_weight_names,
    infer_num_layers,
    iter_quant_units,
    precision_lookup_key_for_unit,
    quant_dtype,
    quantize_weight_tensor,
    resolve_daq_component_params,
    rewrite_full_checkpoint,
    run_daq_unit,
    score_group_key,
    score_to_precision,
    score_to_precision_grouped,
    should_apply_fulls_refine,
    validate_granularity_args,
    z_block_path,
)
from tools.pipeline_common import (
    build_calibset,
    build_initial_hidden_cache,
    get_block,
    list_chunk_files,
    load_hidden,
    module_device,
    parse_args,
    propagate_block,
    save_json,
)
from transformers import AutoModel
from utils.smcs import compute_z_for_weight, register_full_smcs_hooks
from utils.generate_imp_mat import load_weight_from_checkpoint, load_weight_map


LLADA_STAGES = (
    ("qkv", ("q_proj.weight", "k_proj.weight", "v_proj.weight")),
    ("attn_out", ("attn_out.weight",)),
    ("ff_in", ("ff_proj.weight", "up_proj.weight")),
    ("ff_out", ("ff_out.weight",)),
)
DREAM_STAGES = (
    ("qkv", ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")),
    ("attn_out", ("self_attn.o_proj.weight",)),
    ("ff_in", ("mlp.gate_proj.weight", "mlp.up_proj.weight")),
    ("ff_out", ("mlp.down_proj.weight",)),
)
def block_family(block):
    if all(hasattr(block, attr) for attr in ("q_proj", "k_proj", "v_proj", "ff_proj", "up_proj")):
        return "llada"
    if hasattr(block, "self_attn") and hasattr(block, "mlp"):
        return "dream"
    raise NotImplementedError(f"Unsupported transformer block: {type(block)}")


def stage_definitions(block, requested_stages=""):
    definitions = LLADA_STAGES if block_family(block) == "llada" else DREAM_STAGES
    requested = {item.strip() for item in requested_stages.split(",") if item.strip()}
    if not requested:
        return definitions
    valid = {name for name, _ in definitions}
    unknown = requested - valid
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}; valid stages are {sorted(valid)}")
    return tuple(item for item in definitions if item[0] in requested)


def names_for_stage(block_names, suffixes):
    return [name for name in block_names if any(name.endswith(suffix) for suffix in suffixes)]


def to_block_device(block, hidden):
    return hidden.to(module_device(block))


def qkv_forward(block, hidden):
    x_normed = block.attn_norm(hidden)
    q = block.q_proj(x_normed)
    k = block.k_proj(x_normed)
    v = block.v_proj(x_normed)
    return q, k, v


def attention_output(block, hidden):
    q, k, v = qkv_forward(block, hidden)
    att, _ = block.attention(q, k, v, attention_bias=None, layer_past=None, use_cache=False)
    return att


def attention_residual(block, hidden):
    att = attention_output(block, hidden)
    return hidden + block.dropout(att)


def dream_position_ids(hidden):
    position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    if hidden.shape[0] > 1:
        position_ids = position_ids.expand(hidden.shape[0], -1)
    return position_ids


def dream_attention_output(block, hidden):
    normed = block.input_layernorm(hidden)
    return block.self_attn(
        hidden_states=normed,
        attention_mask=None,
        position_ids=dream_position_ids(hidden),
        use_cache=False,
    )[0]


def run_stage_forward(block, stage_name, hidden):
    hidden = to_block_device(block, hidden)
    family = block_family(block)
    if family == "dream":
        if stage_name == "qkv":
            normed = block.input_layernorm(hidden)
            q = block.self_attn.q_proj(normed)
            k = block.self_attn.k_proj(normed)
            v = block.self_attn.v_proj(normed)
            del normed, q, k, v
            return
        if stage_name == "attn_out":
            att = dream_attention_output(block, hidden)
            del att
            return
        mid = hidden + dream_attention_output(block, hidden)
        normed = block.post_attention_layernorm(mid)
        if stage_name == "ff_in":
            gate = block.mlp.gate_proj(normed)
            up = block.mlp.up_proj(normed)
            del mid, normed, gate, up
            return
        if stage_name == "ff_out":
            gate = block.mlp.gate_proj(normed)
            up = block.mlp.up_proj(normed)
            gated = block.mlp.act_fn(gate) * up
            out = block.mlp.down_proj(gated)
            del mid, normed, gate, up, gated, out
            return
        raise ValueError(f"Unsupported stage: {stage_name}")

    if stage_name == "qkv":
        q, k, v = qkv_forward(block, hidden)
        del q, k, v
        return
    if stage_name == "attn_out":
        att = attention_output(block, hidden)
        del att
        return
    if stage_name == "ff_in":
        mid = attention_residual(block, hidden)
        normed = block.ff_norm(mid)
        ff = block.ff_proj(normed)
        up = block.up_proj(normed)
        del mid, normed, ff, up
        return
    if stage_name == "ff_out":
        mid = attention_residual(block, hidden)
        normed = block.ff_norm(mid)
        ff = block.ff_proj(normed)
        up = block.up_proj(normed)
        gated = block.act(ff) * up
        out = block.ff_out(gated)
        del mid, normed, ff, up, gated, out
        return
    raise ValueError(f"Unsupported stage: {stage_name}")


def collect_stage_covs(args, model, block_idx, stage_name, names, current_cache_dir):
    if not names:
        return {}, {}
    block = get_block(model, block_idx)
    block_family(block)
    print(f"Block {block_idx} stage {stage_name}: collecting SMCS for {len(names)} weights", flush=True)
    handles, covs, token_counts = register_full_smcs_hooks(
        model,
        names,
        args.max_tokens_per_state,
        args.seed + block_idx * 100000 + len(stage_name),
    )
    files = list_chunk_files(current_cache_dir)
    try:
        with torch.no_grad():
            for idx, path in enumerate(files, start=1):
                hidden = load_hidden(path)
                run_stage_forward(block, stage_name, hidden)
                del hidden
                if idx == 1 or idx == len(files) or idx % 64 == 0:
                    print(f"  block {block_idx} stage {stage_name}: chunk {idx}/{len(files)}", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
    return covs, token_counts


def compute_stage_z(args, stage_name, names, covs, token_counts, mcs_step_count, weight_map):
    z_tensors = {}
    reports = {}
    inverse_chol_map = {}
    activation_diag_map = {}
    fulls_cov_map = {}
    fulls_normalizer_map = {}
    for name in names:
        normalizer = token_counts[name] if args.mcs_normalization == "tokens" else mcs_step_count
        print(f"  stage {stage_name}: inverse+Z {name}, normalizer={normalizer}", flush=True)
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
        z_tensors[name] = z_tensor
        reports[name] = inverse_info
        del covs[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return z_tensors, reports, inverse_chol_map, activation_diag_map, fulls_cov_map, fulls_normalizer_map


def precision_from_z_tensors(args, z_tensors):
    if args.skip_abmp:
        return {name: args.default_bits for name in z_tensors}, {}
    scores = {}
    if args.abmp_granularity == "weight":
        for name, z in z_tensors.items():
            scores[name] = z.to(torch.float32).sum().item()
        return score_to_precision(scores, args.allocate_ratio), scores

    score_groups = {}
    for name, z in z_tensors.items():
        z = z.to(torch.float32)
        for unit_key, row_slice, col_slice in iter_quant_units(
            name,
            z.shape,
            args.abmp_granularity,
            args.quant_block_size,
        ):
            scores[unit_key] = z[row_slice, col_slice].sum().item()
            score_groups[unit_key] = score_group_key(name, args.abmp_rank_scope)
    return score_to_precision_grouped(scores, score_groups, args.allocate_ratio), scores


def quantize_stage_tensors(
    args,
    weight_map,
    z_tensors,
    precision_map,
    out_file,
    cov_map=None,
    normalizer_map=None,
    inverse_chol_map=None,
    activation_diag_map=None,
):
    quantized = {}
    tensor_map = {}
    save_dtype = quant_dtype(args.save_dtype)
    for weight_name, z_tensor in z_tensors.items():
        W = load_weight_from_checkpoint(weight_name, weight_map, args.model_path)
        Z = z_tensor.to(dtype=torch.float32)
        Q, _ = quantize_weight_tensor(
            args,
            weight_name,
            W,
            Z,
            precision_map,
            cov=(cov_map or {}).get(weight_name),
            normalizer=(normalizer_map or {}).get(weight_name),
            inverse_chol_upper=(inverse_chol_map or {}).get(weight_name),
            activation_diag=(activation_diag_map or {}).get(weight_name),
        )
        quantized[weight_name] = Q.to(save_dtype).contiguous()
        tensor_map[weight_name] = str(out_file)
        del W, Z, Q
        gc.collect()
    save_file(quantized, str(out_file), metadata={"format": "pt"})
    return quantized, tensor_map


def apply_quantized_in_memory(model, quantized):
    modules = dict(model.named_modules())
    for weight_name, tensor in quantized.items():
        module_name = weight_name[: -len(".weight")]
        module = modules[module_name]
        if module.weight.device.type == "meta":
            raise RuntimeError(f"Cannot update meta tensor for {weight_name}")
        module.weight.data.copy_(tensor.to(device=module.weight.device, dtype=module.weight.dtype))


def stagewise_quantize_block(args, model, block_idx, block_names, current_cache, mcs_step_count, weight_map, z_cache_dir):
    z_cache_dir = Path(z_cache_dir)
    z_cache_dir.mkdir(parents=True, exist_ok=True)
    quant_dir = Path(args.output_dir) / "quantized_blocks"
    quant_dir.mkdir(parents=True, exist_ok=True)

    block_z_tensors = {}
    block_inverse_reports = {}
    block_precision_map = {}
    block_scores = {}
    block_tensor_map = {}

    block = get_block(model, block_idx)
    for stage_name, suffixes in stage_definitions(block, args.stages):
        names = names_for_stage(block_names, suffixes)
        covs, token_counts = collect_stage_covs(args, model, block_idx, stage_name, names, current_cache)
        (
            z_tensors,
            reports,
            inverse_chol_map,
            activation_diag_map,
            fulls_cov_map,
            fulls_normalizer_map,
        ) = compute_stage_z(
            args,
            stage_name,
            names,
            covs,
            token_counts,
            mcs_step_count,
            weight_map,
        )
        precision_map, scores = precision_from_z_tensors(args, z_tensors)
        out_file = quant_dir / f"block_{block_idx:02d}_{stage_name}.safetensors"
        quantized, tensor_map = quantize_stage_tensors(
            args,
            weight_map,
            z_tensors,
            precision_map,
            out_file,
            cov_map=fulls_cov_map,
            normalizer_map=fulls_normalizer_map,
            inverse_chol_map=inverse_chol_map,
            activation_diag_map=activation_diag_map,
        )
        apply_quantized_in_memory(model, quantized)

        block_z_tensors.update({name: tensor.to(quant_dtype(args.z_save_dtype)).contiguous() for name, tensor in z_tensors.items()})
        block_inverse_reports.update(reports)
        block_precision_map.update(precision_map)
        block_scores.update(scores)
        block_tensor_map.update(tensor_map)
        del covs, token_counts, z_tensors, quantized, inverse_chol_map, activation_diag_map, fulls_cov_map, fulls_normalizer_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_file(block_z_tensors, str(z_block_path(z_cache_dir, block_idx)), metadata={"format": "pt"})
    save_json(block_inverse_reports, z_cache_dir / f"block_{block_idx:02d}_inverse_report.json")
    return block_precision_map, block_scores, block_tensor_map


def main():
    args = parse_args()
    validate_granularity_args(args)
    if args.only_build_z:
        raise NotImplementedError("Stage-wise mode quantizes each stage before later-stage statistics; --only_build_z is not supported.")
    if args.start_block != 0 and not args.resume_stagewise:
        raise NotImplementedError("Use --resume_stagewise with --start_block > 0 and an existing hidden cache/output directory.")
    if args.resume_stagewise and args.start_block <= 0:
        raise ValueError("--resume_stagewise requires --start_block > 0.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    z_cache_dir = Path(args.z_cache_dir) if args.z_cache_dir else output_dir / "z_cache"
    hidden_root = Path(args.hidden_cache_dir)
    hidden_root.mkdir(parents=True, exist_ok=True)
    args_file = (
        output_dir / f"stagewise_resume_args_block_{args.start_block:02d}.json"
        if args.resume_stagewise
        else output_dir / "stagewise_args.json"
    )
    save_json(vars(args), args_file)

    weight_map = load_weight_map(args.model_path)
    target_weight_names = get_target_weight_names(weight_map)
    num_layers = infer_num_layers(target_weight_names)
    end_block = args.end_block if args.end_block is not None else num_layers
    selected_blocks = list(range(args.start_block, min(end_block, num_layers)))
    block_map = {idx: block_weight_names(target_weight_names, idx) for idx in selected_blocks}
    print(f"Stage-wise target blocks: {selected_blocks}", flush=True)

    print("Loading model", flush=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=build_max_memory(args),
        offload_folder=args.offload_folder,
    )
    model.eval()

    if args.resume_stagewise:
        current_cache = hidden_root / f"block_{args.start_block:02d}"
        manifest = current_cache / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Resume hidden cache is missing: {manifest}")
        total_states = json.load(open(manifest))["total_states"]
        mcs_step_count = args.num_steps
        print(
            f"Resuming stage-wise quantization at block {args.start_block}: "
            f"hidden_states={total_states}, mcs_step_count={mcs_step_count}",
            flush=True,
        )
        map_paths = {
            "precision": output_dir / "precision_map.json",
            "scores": output_dir / "importance_scores.json",
            "tensors": output_dir / "quantized_tensor_map.json",
        }
        missing = [str(path) for path in map_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Resume output maps are missing: {missing}")
        all_precision_map = json.load(open(map_paths["precision"]))
        all_scores = json.load(open(map_paths["scores"]))
        quantized_tensor_map = json.load(open(map_paths["tensors"]))
        print(
            f"Loaded resume maps: precision={len(all_precision_map)}, "
            f"scores={len(all_scores)}, tensors={len(quantized_tensor_map)}",
            flush=True,
        )
    else:
        calibset, mcs_step_count = build_calibset(args)
        print(f"Calibration states={len(calibset)}, mcs_step_count={mcs_step_count}", flush=True)
        current_cache = hidden_root / "block_00"
        build_initial_hidden_cache(args, model, calibset, current_cache)
        del calibset
        gc.collect()
        all_precision_map = {}
        all_scores = {}
        quantized_tensor_map = {}

    for pos, block_idx in enumerate(selected_blocks):
        names = block_map[block_idx]
        print(f"===== Stage-wise block {block_idx} ({pos + 1}/{len(selected_blocks)}) =====", flush=True)
        block_precision_map, block_scores, block_tensor_map = stagewise_quantize_block(
            args,
            model,
            block_idx,
            names,
            current_cache,
            mcs_step_count,
            weight_map,
            z_cache_dir,
        )
        all_precision_map.update(block_precision_map)
        all_scores.update(block_scores)
        quantized_tensor_map.update(block_tensor_map)
        save_json(all_precision_map, output_dir / "precision_map.json")
        save_json(all_scores, output_dir / "importance_scores.json")
        save_json(quantized_tensor_map, output_dir / "quantized_tensor_map.json")

        is_last = pos == len(selected_blocks) - 1
        if not is_last:
            next_cache = hidden_root / f"block_{block_idx + 1:02d}"
            propagate_block(args, model, block_idx, current_cache, next_cache)
            if not args.keep_hidden_cache:
                shutil.rmtree(current_cache)
            current_cache = next_cache

    copy_model_assets(args.model_path, output_dir)
    rewrite_full_checkpoint(args.model_path, output_dir, quantized_tensor_map)
    if not args.keep_hidden_cache and current_cache.exists():
        shutil.rmtree(current_cache)
    print(f"Saved stage-wise fake-quantized model to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
