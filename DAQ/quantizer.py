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
    B_stack = torch.empty(bits, n, m, device=device, dtype=torch.int8)

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


def DAQ(weight, imp_mat, bits, daq_steps, lamda, imp_mask=None):
    alpha_r_list, alpha_c_list, B_init_list, S_list = [], [], [], []
    if imp_mask is None:
        imp_mask = build_imp_mask(imp_mat, lamda)
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
            S_list[b] = alpha_r_list[b] @ alpha_c_list[b].T

        B_list = update_B(
            weight,
            alpha_r_list,
            alpha_c_list,
            bits,
        )

        opt_weight = torch.zeros_like(weight)
        for b in range(bits):
            opt_weight += S_list[b] * B_list[b]

    return opt_weight
