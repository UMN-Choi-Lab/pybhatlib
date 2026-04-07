"""GPU-accelerated MNP gradient for K=2 models.

Implements the full analytic gradient computation on GPU using PyTorch
tensors. All operations (BVN CDF, gradient, chain-rule accumulation)
run on the GPU device. Only the final (nll, grad) scalars are
transferred back to CPU for scipy.optimize.

For N < ~5000, the CPU vectorized NumPy path is faster.
For N >= 5000, the GPU path provides significant speedup.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from pybhatlib.gradmvn._mvncd_torch import (
    bvn_cdf_torch,
    mvncd_grad_batch_k2_torch,
    mvncd_grad_batch_k2_perobs_torch,
    get_compiled_grad_perobs,
    get_compiled_grad_shared,
)
from pybhatlib.matgradient._spherical import grad_corr_theta, theta_to_corr
from pybhatlib.models.mnp._mnp_control import MNPControl


def mnp_gradient_gpu(
    theta: NDArray,
    X_gpu: torch.Tensor,
    y_gpu: torch.Tensor,
    n_alts: int,
    n_beta: int,
    control: MNPControl,
    ranvar_indices: list[int] | None = None,
    *,
    device: str = "cuda",
) -> tuple[float, NDArray]:
    """Compute MNP negative mean log-likelihood and gradient on GPU.

    Parameters
    ----------
    theta : ndarray, shape (n_params,) — CPU
        Parameter vector.
    X_gpu : Tensor, shape (N, n_alts, n_vars) — on GPU
        Design matrix (pre-transferred to GPU at model init).
    y_gpu : Tensor, shape (N,) — on GPU (int64)
        Chosen alternative indices.
    n_alts, n_beta : int
    control : MNPControl
    ranvar_indices : list of int or None
    device : str

    Returns
    -------
    nll : float — CPU scalar
    grad : ndarray, shape (n_params,) — CPU
    """
    if control.nseg > 1:
        return _mixture_gradient_gpu(
            theta, X_gpu, y_gpu, n_alts, n_beta, control,
            ranvar_indices, device=device,
        )

    N = X_gpu.shape[0]
    I = n_alts
    dim = I - 1
    n_params = len(theta)

    # --- Unpack parameters (CPU, cheap) ---
    idx = 0
    beta_np = theta[idx:idx + n_beta]
    idx += n_beta

    n_scale = 0
    n_corr = 0
    n_lambda = 0
    lambda_params = None
    if not control.iid:
        if control.heteronly:
            n_scale = dim
            n_lambda = n_scale
        else:
            n_scale = dim
            n_corr = dim * (dim - 1) // 2
            n_lambda = n_scale + n_corr
        lambda_params = theta[idx:idx + n_lambda]
        idx += n_lambda

    n_rand = 0
    n_omega = 0
    omega_params = None
    if control.mix and ranvar_indices is not None:
        n_rand = len(ranvar_indices)
        if control.randdiag:
            n_omega = n_rand
        else:
            n_omega = n_rand * (n_rand + 1) // 2
        omega_params = theta[idx:idx + n_omega]
        idx += n_omega

    # --- Build covariance components (CPU, reuse existing functions) ---
    from pybhatlib.models.mnp._mnp_grad_analytic import (
        _build_lambda_components,
        _build_omega_components,
        _adj_lambda_to_params,
        _adj_omega_to_params,
    )

    Lambda_np, scales, corr = _build_lambda_components(lambda_params, dim, control)

    corr_jac = None
    if not control.iid and not control.heteronly and n_corr > 0:
        corr_theta = lambda_params[n_scale:n_scale + n_corr]
        corr_jac = grad_corr_theta(corr_theta, dim)

    Omega_np = None
    Omega_L_np = None
    has_random = control.mix and omega_params is not None
    if has_random:
        Omega_L_np, Omega_np = _build_omega_components(omega_params, n_rand, control)

    Lambda_full_np = np.eye(I, dtype=np.float64)
    if not control.iid:
        Lambda_full_np[1:, 1:] = Lambda_np + np.eye(dim)

    need_sigma_chain = (not control.iid) or has_random

    # --- Transfer to GPU ---
    beta_gpu = torch.tensor(beta_np, dtype=torch.float64, device=device)
    Lambda_full_gpu = torch.tensor(Lambda_full_np, dtype=torch.float64, device=device)

    Omega_gpu = None
    if has_random:
        Omega_gpu = torch.tensor(Omega_np, dtype=torch.float64, device=device)

    # --- Compute utilities on GPU ---
    V_all = torch.einsum('nij,j->ni', X_gpu, beta_gpu)  # (N, I)

    # --- Build per-obs diff_V, X_diff, Lambda_diff in one pass ---
    # Pre-compute M matrices and Lambda_diff for each possible chosen alt
    M_all = {}  # c -> (dim, I) tensor
    Ld_all = {}  # c -> (dim, dim) Lambda_diff tensor
    for c in range(I):
        avail = [j for j in range(I) if j != c]
        M = torch.zeros((dim, I), dtype=torch.float64, device=device)
        for k, j in enumerate(avail):
            M[k, j] = 1.0
            M[k, c] = -1.0
        M_all[c] = M
        Ld = M @ Lambda_full_gpu @ M.T
        Ld_all[c] = 0.5 * (Ld + Ld.T)

    # Build diff_V and X_diff for ALL observations at once
    diff_V_all = torch.empty((N, dim), dtype=torch.float64, device=device)
    X_diff_all = torch.empty((N, dim, X_gpu.shape[2]), dtype=torch.float64, device=device)
    # Per-obs Lambda_diff: (N, dim, dim) — needed when has_random or different per chosen
    if has_random:
        Lambda_diff_all = torch.empty((N, dim, dim), dtype=torch.float64, device=device)
    else:
        Lambda_diff_all = torch.empty((N, dim, dim), dtype=torch.float64, device=device)

    for c in range(I):
        mask = y_gpu == c
        if not mask.any():
            continue
        idx_c = torch.where(mask)[0]
        M_c = M_all[c]
        avail = [j for j in range(I) if j != c]

        V_group = V_all[idx_c]
        X_group = X_gpu[idx_c]
        for k, j in enumerate(avail):
            diff_V_all[idx_c, k] = V_group[:, c] - V_group[:, j]
            X_diff_all[idx_c, k, :] = X_group[:, c, :] - X_group[:, j, :]

        if has_random:
            X_rand = X_group[:, :, ranvar_indices]
            Omega_tilde = torch.einsum('nir,rs,njs->nij', X_rand, Omega_gpu, X_rand)
            Xi_full = Omega_tilde + Lambda_full_gpu.unsqueeze(0)
            Ld_perobs = torch.einsum('di,nij,ej->nde', M_c, Xi_full, M_c)
            Lambda_diff_all[idx_c] = 0.5 * (Ld_perobs + Ld_perobs.transpose(1, 2))
        else:
            Lambda_diff_all[idx_c] = Ld_all[c].unsqueeze(0)

    # --- Single batched BVN gradient call for ALL N observations ---
    grad_fn = (
        get_compiled_grad_perobs() if control.torch_compile
        else mvncd_grad_batch_k2_perobs_torch
    )
    prob_all, grad_a_all, grad_sv_all = grad_fn(diff_V_all, Lambda_diff_all)

    total_ll = torch.log(prob_all).sum()
    inv_p_all = 1.0 / prob_all

    grad_gpu = torch.zeros(n_params, dtype=torch.float64, device=device)

    # Beta gradient (all observations at once)
    weighted_grad_a = inv_p_all.unsqueeze(1) * grad_a_all
    grad_gpu[:n_beta] = torch.einsum('nkv,nk->v', X_diff_all, weighted_grad_a)

    # Sigma gradient chain-rule
    if need_sigma_chain:
        weighted_gsv = inv_p_all.unsqueeze(1) * grad_sv_all

        # Build per-obs adj_Lambda_diff
        adj_Ld = torch.zeros((N, dim, dim), dtype=torch.float64, device=device)
        adj_Ld[:, 0, 0] = weighted_gsv[:, 0]
        adj_Ld[:, 0, 1] = weighted_gsv[:, 1] / 2.0
        adj_Ld[:, 1, 0] = weighted_gsv[:, 1] / 2.0
        adj_Ld[:, 1, 1] = weighted_gsv[:, 2]

        # Per-obs adj_Xi_full = M_c.T @ adj_Ld @ M_c
        # Need per-chosen-alt M, so accumulate per group
        adj_Xi_sum = torch.zeros((I, I), dtype=torch.float64, device=device)
        for c in range(I):
            mask = y_gpu == c
            if not mask.any():
                continue
            idx_c = torch.where(mask)[0]
            M_c = M_all[c]
            adj_Xi_c = torch.einsum('di,nde,ej->nij', M_c, adj_Ld[idx_c], M_c)

            if has_random and n_omega > 0:
                X_rand = X_gpu[idx_c, :, ranvar_indices]
                adj_Omega_c = torch.einsum('nir,nij,njs->rs', X_rand, adj_Xi_c, X_rand)
                adj_op = _adj_omega_to_params(
                    adj_Omega_c.cpu().numpy(), Omega_L_np, omega_params,
                    n_rand, control,
                )
                grad_gpu[n_beta + n_lambda:n_beta + n_lambda + n_omega] += torch.tensor(
                    adj_op, dtype=torch.float64, device=device,
                )

            adj_Xi_sum += adj_Xi_c.sum(dim=0)

        if not control.iid and n_lambda > 0:
            adj_Lambda = adj_Xi_sum[1:, 1:].cpu().numpy()
            adj_lp = _adj_lambda_to_params(
                adj_Lambda, scales, corr, corr_jac,
                dim, n_scale, n_corr, control,
            )
            grad_gpu[n_beta:n_beta + n_lambda] += torch.tensor(
                adj_lp, dtype=torch.float64, device=device,
            )

    # --- Transfer back to CPU ---
    nll = float(-total_ll / N)
    grad = (-grad_gpu / N).cpu().numpy()
    return nll, grad


def _mixture_gradient_gpu(
    theta: NDArray,
    X_gpu: torch.Tensor,
    y_gpu: torch.Tensor,
    n_alts: int,
    n_beta: int,
    control: MNPControl,
    ranvar_indices: list[int] | None = None,
    *,
    device: str = "cuda",
) -> tuple[float, NDArray]:
    """GPU gradient for mixture-of-normals MNP (nseg > 1), K=2."""
    N = X_gpu.shape[0]
    I = n_alts
    dim = I - 1
    nseg = control.nseg
    n_params = len(theta)

    from pybhatlib.models.mnp._mnp_grad_analytic import (
        _build_lambda_components,
        _adj_lambda_to_params,
    )

    # --- Unpack parameters (CPU) ---
    n_scale = 0
    n_corr = 0
    n_lambda = 0
    if not control.iid:
        if control.heteronly:
            n_scale = dim
            n_lambda = n_scale
        else:
            n_scale = dim
            n_corr = dim * (dim - 1) // 2
            n_lambda = n_scale + n_corr

    idx = 0
    betas_np = []
    beta_1 = theta[idx:idx + n_beta]; idx += n_beta
    betas_np.append(beta_1)
    lambda_params = None
    if not control.iid:
        lambda_params = theta[idx:idx + n_lambda]; idx += n_lambda
    n_seg_params = nseg - 1
    seg_params_start = idx
    segment_params = theta[idx:idx + n_seg_params]; idx += n_seg_params

    seg_extra_starts = []
    for h in range(1, nseg):
        h_beta_start = idx; idx += n_beta
        betas_np.append(theta[h_beta_start:h_beta_start + n_beta])
        seg_extra_starts.append(h_beta_start)

    # Segment probabilities (softmax)
    raw = np.concatenate([[0.0], segment_params])
    raw_max = raw.max()
    exp_raw = np.exp(raw - raw_max)
    pi_h = exp_raw / exp_raw.sum()

    # Build Lambda (CPU)
    Lambda_np, scales, corr = _build_lambda_components(lambda_params, dim, control)
    corr_jac = None
    if not control.iid and not control.heteronly and n_corr > 0:
        corr_theta = lambda_params[n_scale:n_scale + n_corr]
        corr_jac = grad_corr_theta(corr_theta, dim)

    Lambda_full_np = np.eye(I, dtype=np.float64)
    if not control.iid:
        Lambda_full_np[1:, 1:] = Lambda_np + np.eye(dim)

    need_sigma_chain = not control.iid

    # --- Transfer to GPU ---
    Lambda_full_gpu = torch.tensor(Lambda_full_np, dtype=torch.float64, device=device)
    pi_h_gpu = torch.tensor(pi_h, dtype=torch.float64, device=device)
    betas_gpu = [torch.tensor(b, dtype=torch.float64, device=device) for b in betas_np]

    unique_chosen = torch.unique(y_gpu)

    # Per-segment probabilities and beta gradients
    P_q_h_all = torch.empty((N, nseg), dtype=torch.float64, device=device)
    seg_grad_beta_all = []
    grad_sv_cache = {}

    for h in range(nseg):
        V_all_h = torch.einsum('nij,j->ni', X_gpu, betas_gpu[h])
        grad_beta_h = torch.zeros((N, n_beta), dtype=torch.float64, device=device)

        for c_tensor in unique_chosen:
            c = int(c_tensor.item())
            mask = y_gpu == c
            obs_indices = torch.where(mask)[0]
            N_c = obs_indices.shape[0]
            if N_c == 0:
                continue

            avail_alts_c = [j for j in range(I) if j != c]
            M_c = torch.zeros((dim, I), dtype=torch.float64, device=device)
            for k, j in enumerate(avail_alts_c):
                M_c[k, j] = 1.0
                M_c[k, c] = -1.0

            Lambda_diff_c = M_c @ Lambda_full_gpu @ M_c.T
            Lambda_diff_c = 0.5 * (Lambda_diff_c + Lambda_diff_c.T)

            V_group = V_all_h[obs_indices]
            X_group = X_gpu[obs_indices]
            diff_V_group = torch.empty((N_c, dim), dtype=torch.float64, device=device)
            X_diff_group = torch.empty((N_c, dim, X_gpu.shape[2]), dtype=torch.float64, device=device)
            for k, j in enumerate(avail_alts_c):
                diff_V_group[:, k] = V_group[:, c] - V_group[:, j]
                X_diff_group[:, k, :] = X_group[:, c, :] - X_group[:, j, :]

            grad_fn = (
                get_compiled_grad_shared() if control.torch_compile
                else mvncd_grad_batch_k2_torch
            )
            prob_all, grad_a_all, grad_sv_all = grad_fn(
                diff_V_group, Lambda_diff_c,
            )
            P_q_h_all[obs_indices, h] = prob_all
            grad_beta_h[obs_indices] = torch.einsum('nkv,nk->nv', X_diff_group, grad_a_all)

            if need_sigma_chain:
                grad_sv_cache[(h, c)] = (obs_indices, grad_sv_all, M_c)

        seg_grad_beta_all.append(grad_beta_h)

    # Mixture probabilities
    P_q_all = P_q_h_all @ pi_h_gpu
    P_q_all = P_q_all.clamp(min=1e-300)
    total_ll = torch.log(P_q_all).sum()
    inv_P_q = 1.0 / P_q_all

    grad_gpu = torch.zeros(n_params, dtype=torch.float64, device=device)

    # Beta gradients per segment
    for h in range(nseg):
        weights = inv_P_q * pi_h_gpu[h]
        weighted_grad = weights.unsqueeze(1) * seg_grad_beta_all[h]
        grad_beta_h = weighted_grad.sum(dim=0)
        if h == 0:
            grad_gpu[:n_beta] += grad_beta_h
        else:
            start = seg_extra_starts[h - 1]
            grad_gpu[start:start + n_beta] += grad_beta_h

    # Lambda gradient
    if need_sigma_chain and n_lambda > 0:
        for h in range(nseg):
            for c_tensor in unique_chosen:
                c = int(c_tensor.item())
                key = (h, c)
                if key not in grad_sv_cache:
                    continue
                obs_indices, grad_sv_all, M_c = grad_sv_cache[key]
                weights = inv_P_q[obs_indices] * pi_h_gpu[h]
                weighted_gsv = weights.unsqueeze(1) * grad_sv_all
                sum_gsv = weighted_gsv.sum(dim=0)

                adj_Ld = torch.zeros((dim, dim), dtype=torch.float64, device=device)
                adj_Ld[0, 0] = sum_gsv[0]
                adj_Ld[0, 1] = sum_gsv[1] / 2.0
                adj_Ld[1, 0] = sum_gsv[1] / 2.0
                adj_Ld[1, 1] = sum_gsv[2]

                adj_Xi_full = M_c.T @ adj_Ld @ M_c
                if not control.iid:
                    adj_Lambda = adj_Xi_full[1:, 1:].cpu().numpy()
                    adj_lp = _adj_lambda_to_params(
                        adj_Lambda, scales, corr, corr_jac,
                        dim, n_scale, n_corr, control,
                    )
                    lambda_start = n_beta
                    grad_gpu[lambda_start:lambda_start + n_lambda] += torch.tensor(
                        adj_lp, dtype=torch.float64, device=device,
                    )

    # Segment probability gradients
    for k_idx in range(n_seg_params):
        k = k_idx + 1
        softmax_jac = pi_h_gpu * (torch.eye(nseg, device=device, dtype=torch.float64)[k] - pi_h_gpu[k])
        dPq_dsk = P_q_h_all @ softmax_jac
        grad_gpu[seg_params_start + k_idx] += (inv_P_q * dPq_dsk).sum()

    nll = float(-total_ll / N)
    grad = (-grad_gpu / N).cpu().numpy()
    return nll, grad
