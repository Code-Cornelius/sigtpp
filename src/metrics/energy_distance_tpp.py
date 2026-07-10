import torch

from src.metrics.tppnorm import pairwise_tpp_area_distance


@torch.no_grad()
def energy_distance_tpp(
    eta: torch.Tensor,  # (N_eta, L, D)  first sample of processes, padded with NaNs
    rho: torch.Tensor,  # (N_rho, L, D)  second sample of processes, padded with NaNs (same L,D)
    T,  # scalar: observation horizon, used to normalise
    chunk_size: int = 512,  # rows of eta processed per iteration to cap peak memory
) -> torch.Tensor:
    """
    Unbiased two-sample Energy Distance between TPP samples, normalised by T².
    ED(P,Q) = 2 E[d(eta,rho)] - E[d(eta,eta')] - E[d(rho,rho')]
    Uses upper-triangular pairs for within-sample terms (avoids self-distance bias).
    Returns: scalar float64 tensor.


    Requirements:
      - Sx >= 2 and Sy >= 2 (unbiased U-statistic)
      - X and Y share the same (L, D).
    """
    assert eta.ndim == 3 and rho.ndim == 3, f"eta,rho must be (N,L,D); got {eta.shape},{rho.shape}"
    assert eta.shape[1:] == rho.shape[1:], f"(L,D) mismatch: {eta.shape[1:]} vs {rho.shape[1:]}"
    N_eta = eta.shape[0]
    N_rho = rho.shape[0]
    assert N_eta >= 2 and N_rho >= 2, f"N_eta,N_rho must be >=2 for unbiased estimator; got {N_eta},{N_rho}"

    tensor_T = torch.as_tensor(T, device=eta.device)

    # 1. Cross term: full (N_eta, N_rho) cost matrix, averaged over all pairs.
    C_cross = pairwise_tpp_area_distance(eta, rho, tensor_T, chunk_size=chunk_size)  # (N_eta, N_rho)
    term_cross = 2.0 * C_cross.mean()

    # 2. Within-eta: upper-triangular pairs only (unbiased: excludes self-distances).
    C_ee = pairwise_tpp_area_distance(eta, eta, tensor_T, chunk_size=chunk_size)  # (N_eta, N_eta)
    ie_i, ie_j = torch.triu_indices(N_eta, N_eta, offset=1, device=eta.device)
    term_ee = C_ee[ie_i, ie_j].mean()

    # 3. Within-rho: upper-triangular pairs only (unbiased: excludes self-distances).
    C_rr = pairwise_tpp_area_distance(rho, rho, tensor_T, chunk_size=chunk_size)  # (N_rho, N_rho)
    ir_i, ir_j = torch.triu_indices(N_rho, N_rho, offset=1, device=rho.device)
    term_rr = C_rr[ir_i, ir_j].mean()

    # 4. Combine and normalise by T^2 so the result is scale-invariant.
    ed = term_cross - term_ee - term_rr
    return ed.to(dtype=eta.dtype) / (T * T)
