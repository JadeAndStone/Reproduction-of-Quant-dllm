import torch
import torch.nn as nn
from MCS import load_dataset as mcs
from pathlib import Path
from safetensors import safe_open
import json


def load_weight_map(model_path):
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        return json.load(f)["weight_map"]

def load_weight_from_checkpoint(weight_name, weight_map, model_path):
    if weight_name not in weight_map:
        raise KeyError(f"{weight_name} not found in checkpoint index.")

    shard_path = Path(model_path) / weight_map[weight_name]

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        weight = f.get_tensor(weight_name)

    return weight.to(dtype=torch.float32)


def iter_weight_units(shape, granularity, block_size):
    rows, cols = shape
    if granularity == "weight":
        yield slice(0, rows), slice(0, cols)
        return

    if block_size <= 0:
        raise ValueError("block_size must be positive for row/column/tile row-centering.")

    if granularity == "column":
        for col_start in range(0, cols, block_size):
            col_end = min(col_start + block_size, cols)
            yield slice(0, rows), slice(col_start, col_end)
        return

    if granularity == "row":
        for row_start in range(0, rows, block_size):
            row_end = min(row_start + block_size, rows)
            yield slice(row_start, row_end), slice(0, cols)
        return

    if granularity == "tile":
        for row_start in range(0, rows, block_size):
            row_end = min(row_start + block_size, rows)
            for col_start in range(0, cols, block_size):
                col_end = min(col_start + block_size, cols)
                yield slice(row_start, row_end), slice(col_start, col_end)
        return

    raise ValueError(f"Unsupported row-center granularity: {granularity}")


def row_center_weight_by_units(weight, granularity, block_size):
    for row_slice, col_slice in iter_weight_units(weight.shape, granularity, block_size):
        unit = weight[row_slice, col_slice]
        weight[row_slice, col_slice] = unit - unit.mean(dim=1, keepdim=True)
    return weight


def _target_weight_set(target_weight_names):
    if target_weight_names is None:
        return None
    return set(target_weight_names)


def init_mcs_mat(model, group_size=128, device="cpu", target_weight_names=None):
    mat = {}
    target_weight_names = _target_weight_set(target_weight_names)

    for name,module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_name = name + ".weight"
            if target_weight_names is not None and weight_name not in target_weight_names:
                continue
            if module.in_features % group_size != 0:
                raise ValueError(
                    f"{weight_name}: in_features={module.in_features} is not divisible by group_size={group_size}"
                )
            mat[weight_name] = torch.zeros(
                module.in_features // group_size,
                group_size,
                group_size,
                dtype=torch.float32,
                device=device
            )
    return mat


def register_mat_hooks(model, mcs_mat_dict, group_size=128, device="cpu", target_weight_names=None):
    handles = []
    target_weight_names = _target_weight_set(target_weight_names)

    def make_hook(weight_name):
        def simulated_mcs_hook(module, input, output):
            X = input[0].reshape(-1, input[0].shape[-1])
            X = X.reshape(X.shape[0], -1, group_size)
            X = X.detach().to(torch.float32)
            S_X = torch.einsum("ngi,ngj->gij", X, X)
            mcs_mat_dict[weight_name] += S_X.cpu()
        return simulated_mcs_hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_name = name + ".weight"
            if target_weight_names is not None and weight_name not in target_weight_names:
                continue
            handle=module.register_forward_hook(make_hook(weight_name))
            handles.append(handle)
    return handles


def inverse_diag_adaptive(
    matrix,
    damping,
    min_diag=1e-12,
    max_retries=8,
    growth=10.0,
    context="",
):
    base = matrix.to(dtype=torch.float32)
    if not torch.isfinite(base).all():
        raise FloatingPointError(f"{context}: MCS matrix contains non-finite values.")

    base = 0.5 * (base + base.T)
    eye = torch.eye(base.shape[0], dtype=base.dtype, device=base.device)
    start_damping = float(damping)
    if start_damping < 0:
        raise ValueError(f"{context}: damping must be non-negative, got {damping}.")

    current_damping = start_damping
    last_error = None
    for attempt in range(max_retries + 1):
        work = base + current_damping * eye
        try:
            chol = torch.linalg.cholesky(work)
            inv = torch.cholesky_inverse(chol)
            diag = torch.diagonal(inv).detach()
            finite = torch.isfinite(diag).all()
            positive = bool((diag > min_diag).all().item()) if finite else False
            if finite and positive:
                return diag, current_damping
            diag_min = float(diag[torch.isfinite(diag)].min().item()) if torch.isfinite(diag).any() else float("nan")
            last_error = (
                f"diag check failed: finite={bool(finite.item())}, "
                f"min={diag_min:.6e}, floor={min_diag:.6e}"
            )
        except RuntimeError as exc:
            last_error = str(exc)
        finally:
            for var_name in ("work", "chol", "inv", "diag"):
                if var_name in locals():
                    del locals()[var_name]

        if attempt == max_retries:
            break
        current_damping = current_damping * growth if current_damping > 0 else min_diag

    raise FloatingPointError(
        f"{context}: failed to compute a finite positive inverse diagonal after "
        f"{max_retries + 1} attempts; last damping={current_damping:.6e}; {last_error}"
    )


def resolve_damping_value(
    matrix,
    damping,
    damping_mode="absolute",
    damp_percent=0.01,
):
    if damping_mode == "absolute":
        return float(damping)
    if damping_mode == "diag_mean":
        diag_mean = torch.diagonal(matrix.to(dtype=torch.float32)).mean().item()
        return float(damp_percent) * float(diag_mean)
    raise ValueError(f"Unsupported damping_mode: {damping_mode}")


def get_mcs_mat(model, model_path, nsamples,
                seed,seqlen,
                num_steps,gamma,
                group_size,schedule,
                max_docs,source,
                target_weight_names=None,
                max_mcs_samples=None,
                include_full_visible=True,
                mcs_normalization="tokens",
                inverse_diag_floor=1e-12,
                damping_mode="absolute",
                damp_percent=0.01):
    sample_cnt = 1

    input_device = model.model.device

    trainloader, calibset, mcs_step_count = mcs.get_calib(
        model_path,
        nsamples,
        seed,
        seqlen,
        num_steps,
        gamma,
        schedule,
        max_docs,
        source,
        include_full_visible=include_full_visible,
    )
    calibset = torch.concat(calibset, dim=0)
    if max_mcs_samples is not None and max_mcs_samples > 0:
        calibset = calibset[:max_mcs_samples]
        mcs_step_count = min(mcs_step_count, len(calibset))
    if len(calibset) == 0:
        raise ValueError("MCS calibration set is empty.")

    mcs_mats = init_mcs_mat(model, group_size, target_weight_names=target_weight_names)

    handles = register_mat_hooks(
        model,
        mcs_mats,
        group_size,
        target_weight_names=target_weight_names,
    )

    for batchset in calibset:
        model.eval()
        batchset = batchset.unsqueeze(0).to(input_device)
        print(f"Start Calculating {sample_cnt}th sample")
        sample_cnt += 1
        with torch.no_grad():
            model.model(
                input_ids=batchset,
                use_cache=False,
                last_logits_only=True,
            )

    for handle in handles:
        handle.remove()

    if mcs_normalization == "tokens":
        normalizer = int(calibset.numel())
    elif mcs_normalization == "mcs_steps":
        normalizer = int(mcs_step_count)
    else:
        raise ValueError(f"Unsupported mcs_normalization: {mcs_normalization}")
    if normalizer <= 0:
        raise ValueError(f"Invalid MCS normalizer: {normalizer}")
    print(
        f"MCS normalization={mcs_normalization}, divisor={normalizer}, "
        f"states={len(calibset)}, mcs_steps={mcs_step_count}"
    )

    for name in mcs_mats.keys():
        mcs_mats[name] /= normalizer

    return mcs_mats


def build_imp_mat_from_mcs(
    weight_name,
    mcs_mat,
    model_path,
    damping,
    weight_map=None,
    row_center=False,
    row_center_granularity="weight",
    row_center_block_size=128,
    inverse_diag_floor=1e-12,
    damping_mode="absolute",
    damp_percent=0.01,
):
    if weight_map is None:
        weight_map = load_weight_map(model_path)

    S_diag = []
    for group_idx, g in enumerate(mcs_mat):
        requested_damping = resolve_damping_value(
            g,
            damping,
            damping_mode=damping_mode,
            damp_percent=damp_percent,
        )
        diag, used_damping = inverse_diag_adaptive(
            g,
            requested_damping,
            min_diag=inverse_diag_floor,
            context=f"{weight_name}[group={group_idx}]",
        )
        if used_damping != float(requested_damping):
            print(
                f"{weight_name}[group={group_idx}]: adaptive damping "
                f"{float(requested_damping):.6e} -> {used_damping:.6e}"
            )
        S_diag.append(diag)
    S_diag = torch.cat(S_diag, dim=0)
    scale = 1 / S_diag
    assert len(scale.shape) == 1
    target_weight = load_weight_from_checkpoint(weight_name, weight_map, model_path)
    if row_center:
        target_weight = row_center_weight_by_units(
            target_weight,
            row_center_granularity,
            row_center_block_size,
        )
    return (target_weight.to("cpu") * scale.unsqueeze(0)) ** 2


def get_imp_mat(model, model_path, nsamples,
                seed,seqlen,
                num_steps,gamma,
                group_size,schedule,
                max_docs,source,damping,
                target_weight_names=None,
                max_mcs_samples=None,
                include_full_visible=True,
                mcs_normalization="tokens",
                inverse_diag_floor=1e-12):
    mcs_mats = get_mcs_mat(
        model,
        model_path,
        nsamples,
        seed,
        seqlen,
        num_steps,
        gamma,
        group_size,
        schedule,
        max_docs,
        source,
        target_weight_names=target_weight_names,
        max_mcs_samples=max_mcs_samples,
        include_full_visible=include_full_visible,
        mcs_normalization=mcs_normalization,
    )
    imp_mats = dict.fromkeys(mcs_mats.keys())
    weight_map = load_weight_map(model_path)
    for weight_name in imp_mats.keys():
        imp_mats[weight_name] = build_imp_mat_from_mcs(
            weight_name,
            mcs_mats[weight_name],
            model_path,
            damping,
            weight_map=weight_map,
            inverse_diag_floor=inverse_diag_floor,
            damping_mode=damping_mode,
            damp_percent=damp_percent,
        )
    return imp_mats


# for name, module in model.named_modules():
#     print("name=",name)
#     if isinstance(module, nn.Linear):
#         print(
#             "in_features=",module.in_features,
#             "out_features=",module.out_features,
#             "weight_shape=",tuple(module.weight.shape)
#         )
# print(model.named_modules())
# print(model.model.transformer.blocks[0].attn_norm.weight.shape)
# print(model.model.transformer.blocks[0].attn_norm.__dict__)
