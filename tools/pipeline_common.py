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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MCS import load_dataset as mcs
from main import C4_TRAIN_URL, MODEL_PATH, quant_dtype


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Shared arguments and hidden-state cache utilities for the stage-wise "
            "Quant-dLLM reproduction pipeline."
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
    parser.add_argument(
        "--resume_stagewise",
        action="store_true",
        help="Resume stage-wise quantization from --start_block using an existing hidden cache and output maps.",
    )
    parser.add_argument(
        "--stages",
        default="",
        help="Stage-wise only: comma-separated subset of qkv,attn_out,ff_in,ff_out.",
    )

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


def model_family(model):
    core = model.model
    if hasattr(core, "transformer") and hasattr(core.transformer, "blocks"):
        return "llada"
    if hasattr(core, "layers") and hasattr(core, "embed_tokens"):
        return "dream"
    raise NotImplementedError(f"Unsupported model core: {type(core)}")


def get_block(model, block_idx):
    core = model.model
    family = model_family(model)
    if family == "llada":
        if getattr(core.config, "block_group_size", 1) != 1:
            raise NotImplementedError("Layer-wise pipeline currently requires block_group_size == 1.")
        return core.transformer.blocks[block_idx]
    return core.layers[block_idx]


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
    family = model_family(model)
    if family == "dream":
        embed_device = module_device(core.embed_tokens, fallback=getattr(core, "device", None))
        return core.embed_tokens(input_ids.to(embed_device))

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
    if model_family(model) == "dream":
        position_ids = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
        if hidden.shape[0] > 1:
            position_ids = position_ids.expand(hidden.shape[0], -1)
        return block(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
        )[0]
    output, _ = block(hidden, attention_bias=None, layer_past=None, use_cache=False)
    return output


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


