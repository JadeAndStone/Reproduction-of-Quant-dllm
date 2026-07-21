import gc

import torch
import torch.nn as nn

from utils.generate_imp_mat import (
    inverse_diag_adaptive,
    load_weight_from_checkpoint,
    resolve_damping_value,
)


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


