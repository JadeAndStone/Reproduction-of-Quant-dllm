import torch


def binary_rc_init(X, eps=1e-7):
    alpha_r = X.abs().mean(dim=1, keepdim=True)
    alpha_c = (X.abs().T / (eps + alpha_r.reshape(1, -1))).mean(dim=1, keepdim=True)
    B = torch.where(
        X >= 0,
        torch.ones(X.shape, device=X.device, dtype=torch.int8),
        -torch.ones(X.shape, device=X.device, dtype=torch.int8),
    )
    return alpha_r, alpha_c, B


def update_alpha_r(X, B, alpha_c, imp_mask, eps=1e-7):
    u = ((imp_mask**2) * X * B) @ alpha_c
    v = (imp_mask**2) @ (alpha_c**2) + eps
    return u / v


def update_alpha_c(X, B, alpha_r, imp_mask, eps=1e-7):
    u = ((imp_mask**2) * X * B).T @ alpha_r
    v = (imp_mask**2).T @ (alpha_r**2) + eps
    return u / v


def update_alpha_r_weighted(X, B, alpha_c, solve_weight, eps=1e-7):
    u = (solve_weight * X * B) @ alpha_c
    v = solve_weight @ (alpha_c**2) + eps
    return u / v


def update_alpha_c_weighted(X, B, alpha_r, solve_weight, eps=1e-7):
    u = (solve_weight * X * B).T @ alpha_r
    v = solve_weight.T @ (alpha_r**2) + eps
    return u / v


def reconstruct(alpha_r_list, alpha_c_list, B_list, like):
    quantized = torch.zeros_like(like)
    for idx in range(len(alpha_r_list)):
        quantized += (alpha_r_list[idx] @ alpha_c_list[idx].T) * B_list[idx].to(like.dtype)
    return quantized


def _build_patterns(bits, device, dtype):
    pattern_ids = torch.arange(2**bits, device=device)
    bit_ids = torch.arange(bits, device=device)
    bit_matrix = (pattern_ids[:, None] >> bit_ids[None, :]) & 1
    return torch.where(
        bit_matrix == 0,
        torch.ones((), device=device, dtype=dtype),
        -torch.ones((), device=device, dtype=dtype),
    )


def update_B(W, alpha_r_list, alpha_c_list, bits):
    if bits < 1:
        raise ValueError("bits must be greater than 0.")

    n, m = W.shape
    device = W.device
    dtype = W.dtype
    patterns = _build_patterns(bits, device=device, dtype=dtype)

    S_stack = torch.stack(
        [alpha_r_list[b] @ alpha_c_list[b].T for b in range(bits)],
        dim=0,
    )
    candidate_stack = torch.einsum("pb,bnm->pnm", patterns, S_stack)
    err_mats = (candidate_stack - W.unsqueeze(0)).abs()
    opt_order = torch.argmin(err_mats, dim=0)
    B_stack = patterns[opt_order].permute(2, 0, 1).to(torch.int8).contiguous()

    return B_stack


def build_imp_mask(imp_mat, lamda):
    mean = torch.mean(imp_mat)
    std = torch.std(imp_mat)
    mask = (imp_mat - mean).abs() > 3 * std
    return torch.ones_like(imp_mat) + (lamda - 1) * mask


def DAQ_with_state(weight, imp_mat, bits, daq_steps, lamda, imp_mask=None, solve_weight=None):
    alpha_r_list, alpha_c_list, B_init_list, S_list = [], [], [], []
    if imp_mask is None:
        imp_mask = build_imp_mask(imp_mat, lamda)
    if solve_weight is not None:
        solve_weight = solve_weight.to(device=weight.device, dtype=weight.dtype)
    opt_weight = torch.zeros_like(weight, device=weight.device)

    for _ in range(bits):
        res_weight = weight - opt_weight
        alpha_r, alpha_c, B = binary_rc_init(res_weight)
        S_list.append(alpha_r @ alpha_c.T)
        opt_weight += S_list[-1] * B
        alpha_r_list.append(alpha_r)
        alpha_c_list.append(alpha_c)
        B_init_list.append(B)

    B_list = torch.stack(B_init_list, dim=0)

    for _ in range(daq_steps):
        for b in range(bits):
            weight_no_b = torch.zeros_like(weight)
            for t in range(bits):
                if t != b:
                    weight_no_b += S_list[t] * B_list[t]
            res_weight = weight - weight_no_b
            if solve_weight is None:
                alpha_r_list[b] = update_alpha_r(
                    res_weight,
                    B_list[b],
                    alpha_c_list[b],
                    imp_mask,
                )
                alpha_c_list[b] = update_alpha_c(
                    res_weight,
                    B_list[b],
                    alpha_r_list[b],
                    imp_mask,
                )
            else:
                alpha_r_list[b] = update_alpha_r_weighted(
                    res_weight,
                    B_list[b],
                    alpha_c_list[b],
                    solve_weight,
                )
                alpha_c_list[b] = update_alpha_c_weighted(
                    res_weight,
                    B_list[b],
                    alpha_r_list[b],
                    solve_weight,
                )
            S_list[b] = alpha_r_list[b] @ alpha_c_list[b].T

        B_list = update_B(
            weight,
            alpha_r_list,
            alpha_c_list,
            bits,
        )

        opt_weight = reconstruct(alpha_r_list, alpha_c_list, B_list, weight)

    state = {
        "alpha_r": alpha_r_list,
        "alpha_c": alpha_c_list,
        "binary": B_list,
    }
    return opt_weight, state


def update_alpha_r_fulls(residual, binary, alpha_c, hessian, eps=1e-7):
    pattern = binary.to(residual.dtype) * alpha_c.T
    residual_h = residual @ hessian
    pattern_h = pattern @ hessian
    numerator = (pattern * residual_h).sum(dim=1, keepdim=True)
    denominator = (pattern * pattern_h).sum(dim=1, keepdim=True).clamp_min(eps)
    return numerator / denominator


def update_alpha_c_fulls(residual, binary, alpha_r, hessian, ridge_ratio, eps=1e-7):
    binary_f = binary.to(residual.dtype)
    weighted_binary = alpha_r * binary_f
    gram = binary_f.T @ ((alpha_r.square()) * binary_f)
    system = hessian * gram
    rhs = (weighted_binary * (residual @ hessian)).sum(dim=0, keepdim=True).T
    diag_mean = torch.diagonal(system).abs().mean().clamp_min(eps)
    ridge = float(ridge_ratio) * diag_mean
    eye = torch.eye(system.shape[0], device=system.device, dtype=system.dtype)
    system = 0.5 * (system + system.T) + ridge * eye
    try:
        return torch.linalg.solve(system, rhs)
    except RuntimeError:
        return torch.linalg.lstsq(system, rhs).solution


def full_s_refine_fixed_b(weight, state, hessian, steps, ridge_ratio):
    if steps <= 0:
        return reconstruct(state["alpha_r"], state["alpha_c"], state["binary"], weight)

    hessian = hessian.to(device=weight.device, dtype=torch.float32).contiguous()
    hessian = 0.5 * (hessian + hessian.T)
    alpha_r_list = [item.to(dtype=torch.float32).contiguous() for item in state["alpha_r"]]
    alpha_c_list = [item.to(dtype=torch.float32).contiguous() for item in state["alpha_c"]]
    B_list = state["binary"].contiguous()

    for _ in range(steps):
        for bit_idx in range(len(alpha_r_list)):
            weight_no_bit = torch.zeros_like(weight)
            for other_idx in range(len(alpha_r_list)):
                if other_idx != bit_idx:
                    weight_no_bit += (
                        (alpha_r_list[other_idx] @ alpha_c_list[other_idx].T)
                        * B_list[other_idx].to(weight.dtype)
                    )
            residual = weight - weight_no_bit
            alpha_r_list[bit_idx] = update_alpha_r_fulls(
                residual,
                B_list[bit_idx],
                alpha_c_list[bit_idx],
                hessian,
            )
            alpha_c_list[bit_idx] = update_alpha_c_fulls(
                residual,
                B_list[bit_idx],
                alpha_r_list[bit_idx],
                hessian,
                ridge_ratio,
            )

    return reconstruct(alpha_r_list, alpha_c_list, B_list, weight)


def DAQ(weight, imp_mat, bits, daq_steps, lamda, imp_mask=None, solve_weight=None):
    quantized, _state = DAQ_with_state(
        weight,
        imp_mat,
        bits,
        daq_steps,
        lamda,
        imp_mask=imp_mask,
        solve_weight=solve_weight,
    )
    return quantized


def DAQ_full_s_refine_fixed_b(
    weight,
    imp_mat,
    bits,
    daq_steps,
    lamda,
    hessian,
    fulls_steps,
    fulls_ridge,
    imp_mask=None,
    solve_weight=None,
):
    quantized, state = DAQ_with_state(
        weight,
        imp_mat,
        bits,
        daq_steps,
        lamda,
        imp_mask=imp_mask,
        solve_weight=solve_weight,
    )
    del quantized
    return full_s_refine_fixed_b(
        weight,
        state,
        hessian,
        fulls_steps,
        fulls_ridge,
    )
