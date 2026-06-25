#!/usr/bin/env python
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MCS import load_dataset as mcs
from main import build_max_memory, get_target_weight_names, block_weight_names
from tools.diagnose_full_smcs import register_full_smcs_hooks, inverse_diag
from utils.generate_imp_mat import load_weight_map, load_weight_from_checkpoint, resolve_damping_value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--z_cache_dir', required=True)
    parser.add_argument('--start_block', type=int, default=0)
    parser.add_argument('--end_block', type=int, default=None)
    parser.add_argument('--nsamples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--num-steps', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--schedule', choices=['linear', 'cosine'], default='linear')
    parser.add_argument('--max-docs', type=int, default=None)
    parser.add_argument('--max-mcs-samples', type=int, default=None)
    parser.add_argument('--max-tokens-per-state', type=int, default=128)
    parser.add_argument('--mcs_no_full_visible', action='store_true')
    parser.add_argument('--damping', type=float, default=0.01)
    parser.add_argument('--damping-mode', choices=['absolute', 'diag_mean'], default='absolute')
    parser.add_argument('--damp-percent', type=float, default=0.01)
    parser.add_argument('--mcs-normalization', choices=['tokens', 'mcs_steps'], default='tokens')
    parser.add_argument('--inverse-diag-floor', type=float, default=1e-12)
    parser.add_argument('--inverse-device', default='cuda:0')
    parser.add_argument('--gpu_memory', default='15GiB')
    parser.add_argument('--cpu_memory', default='80GiB')
    parser.add_argument('--offload_folder', default='/root/data-tmp/offload/build_full_smcs_z')
    parser.add_argument('--reuse_existing', action='store_true')
    return parser.parse_args()


def selected_block_range(args, target_weight_names):
    block_ids = sorted({int(name.split('.')[3]) for name in target_weight_names})
    max_block = max(block_ids) + 1
    end_block = args.end_block if args.end_block is not None else max_block
    return range(args.start_block, min(end_block, max_block))


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
        calibset = calibset[:args.max_mcs_samples]
        mcs_step_count = min(mcs_step_count, len(calibset))
    if len(calibset) == 0:
        raise RuntimeError('empty calibration set')
    return calibset, mcs_step_count


def z_block_path(z_cache_dir, block_idx):
    return Path(z_cache_dir) / f'block_{block_idx:02d}.safetensors'


def block_ready(path, names):
    if not path.exists():
        return False
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
    return all(name in keys for name in names)


def compute_z_for_weight(args, weight_name, cov, normalizer, weight_map):
    cov = cov / float(normalizer)
    requested_damping = resolve_damping_value(
        cov,
        args.damping,
        damping_mode=args.damping_mode,
        damp_percent=args.damp_percent,
    )
    diag, inverse_info = inverse_diag(
        cov,
        requested_damping,
        args.inverse_device,
        min_diag=args.inverse_diag_floor,
        context=weight_name,
        return_info=True,
    )
    inverse_info['requested_damping'] = float(requested_damping)
    inverse_info['damping_mode'] = args.damping_mode
    inverse_info['damp_percent'] = args.damp_percent
    used_damping = inverse_info['used_damping']
    if used_damping != float(requested_damping):
        print(
            f'  {weight_name}: adaptive damping '
            f'{float(requested_damping):.6e} -> {used_damping:.6e}',
            flush=True,
        )
    scale = 1.0 / diag.clamp_min(args.inverse_diag_floor)
    weight = load_weight_from_checkpoint(weight_name, weight_map, args.model_path)
    z = (weight * scale.unsqueeze(0)) ** 2
    if not torch.isfinite(z).all():
        bad = int((~torch.isfinite(z)).sum().item())
        raise FloatingPointError(f'{weight_name}: Z contains {bad} non-finite values after adaptive inverse.')
    del diag, scale, weight
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return z.to(torch.float32).contiguous(), inverse_info


def main():
    args = parse_args()
    z_cache_dir = Path(args.z_cache_dir)
    z_cache_dir.mkdir(parents=True, exist_ok=True)

    weight_map = load_weight_map(args.model_path)
    target_weight_names = get_target_weight_names(weight_map)
    blocks = list(selected_block_range(args, target_weight_names))
    calibset, mcs_step_count = build_calibset(args)

    metadata = {
        'model_path': args.model_path,
        'source': args.source,
        'nsamples': args.nsamples,
        'seqlen': args.seqlen,
        'num_steps': args.num_steps,
        'gamma': args.gamma,
        'schedule': args.schedule,
        'max_mcs_samples': args.max_mcs_samples,
        'max_tokens_per_state': args.max_tokens_per_state,
        'mcs_no_full_visible': args.mcs_no_full_visible,
        'damping': args.damping,
        'damping_mode': args.damping_mode,
        'damp_percent': args.damp_percent,
        'inverse_diag_floor': args.inverse_diag_floor,
        'mcs_step_count': mcs_step_count,
        'mcs_normalization': args.mcs_normalization,
        'blocks': blocks,
    }
    with open(z_cache_dir / 'full_smcs_config.json', 'w') as handle:
        json.dump(metadata, handle, indent=2)

    print('Loading model for sampled full-SMCS Z cache', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
        max_memory=build_max_memory(args),
        offload_folder=args.offload_folder,
    )
    input_device = model.model.device

    print(
        f'Calibration states={len(calibset)}, mcs_steps={mcs_step_count}, '
        f'mcs_normalization={args.mcs_normalization}, blocks={blocks}',
        flush=True,
    )
    for block_idx in blocks:
        names = block_weight_names(target_weight_names, block_idx)
        out_path = z_block_path(z_cache_dir, block_idx)
        if args.reuse_existing and block_ready(out_path, names):
            print(f'Block {block_idx}: reuse existing {out_path}', flush=True)
            continue

        print(f'Block {block_idx}: collecting full SMCS for {len(names)} weights', flush=True)
        handles, covs, token_counts = register_full_smcs_hooks(
            model,
            names,
            args.max_tokens_per_state,
            args.seed + block_idx * 100000,
        )
        for state_idx, batch in enumerate(calibset, start=1):
            if state_idx == 1 or state_idx == len(calibset) or state_idx % 8 == 0:
                print(f'  block {block_idx}: forward {state_idx}/{len(calibset)}', flush=True)
            batch = batch.unsqueeze(0).to(input_device)
            with torch.no_grad():
                model.model(input_ids=batch, use_cache=False, last_logits_only=True)
        for handle in handles:
            handle.remove()

        tensors = {}
        for name in names:
            print(f'  block {block_idx}: inverse+Z {name}', flush=True)
            normalizer = token_counts[name] if args.mcs_normalization == 'tokens' else mcs_step_count
            print(f'  block {block_idx}: normalizer={normalizer}', flush=True)
            tensors[name], inverse_info = compute_z_for_weight(args, name, covs[name], normalizer, weight_map)
            if inverse_info['used_damping'] != float(inverse_info['requested_damping']):
                print(
                    f'  block {block_idx}: {name} used_damping={inverse_info["used_damping"]:.6e}',
                    flush=True,
                )
            del covs[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        save_file(tensors, str(out_path), metadata={'format': 'pt'})
        print(f'Block {block_idx}: saved {out_path}', flush=True)
        del tensors, covs, token_counts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, calibset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'Full-SMCS Z cache ready at {z_cache_dir}', flush=True)


if __name__ == '__main__':
    main()
