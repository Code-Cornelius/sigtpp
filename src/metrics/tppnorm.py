import torch


def tpp_area_distance_mean(
    xi: torch.Tensor,  # (N, L, D)  first process, padded with NaNs along L
    rho: torch.Tensor,  # (N, L, D)  second process, padded with NaNs along L (same L,D)
    T: float,  # scalar or tensor broadcastable to (N,1,D): horizon used to fill NaNs
) -> torch.Tensor:
    """
    Pairwise L1 area distance between N pairs of TPP sample-path step functions.
    NaN-padded entries are treated as the constant T (the observation horizon).
    Returns: (N,) float64: mean over D of the L1 area (sum of |xi - rho| over L steps).
    """
    # 1. Validate shapes: assertions ensure callers pass the right format.
    #    Tensors can be any dtype; outputs are always accumulated in float64.
    assert xi.ndim == 3 and rho.ndim == 3, f"xi,rho must be 3D; got {xi.shape},{rho.shape}"
    assert xi.shape[0] == rho.shape[0], f"Batch mismatch: {xi.shape[0]} vs {rho.shape[0]}"
    assert (
        xi.shape[1] == rho.shape[1] and xi.shape[2] == rho.shape[2]
    ), f"L/D mismatch: xi {(xi.shape[1], xi.shape[2])} vs rho {(rho.shape[1], rho.shape[2])}"

    # 2. Fill NaN-padded positions with T so they contribute zero relative area
    #    when both sequences are padded, and |T - t_k| when only one is.
    N, _, D = xi.shape
    Tpad = torch.ones((N, 1, D), device=xi.device, dtype=xi.dtype) * T  # (N,1,D) broadcasts along L
    xi_f = torch.where(torch.isnan(xi), Tpad, xi)  # (N, L, D)
    rho_f = torch.where(torch.isnan(rho), Tpad, rho)  # (N, L, D)

    # 3. Compute L1 area: sum absolute differences over L, then average over D.
    diff = (xi_f - rho_f).abs()  # (N, L, D)
    per_dim = torch.sum(diff, dim=1, dtype=torch.float64)  # (N, D)  float64
    out = torch.mean(per_dim, dim=-1, dtype=torch.float64)  # (N,)    float64
    return out


def pairwise_tpp_area_distance(
    eta: torch.Tensor,  # (Nf, L, D)  first set of processes
    rho: torch.Tensor,  # (Nt, L, D)  second set of processes (same L,D as eta)
    T,  # scalar or broadcastable to (1,1,D): horizon
    chunk_size: int = 512,  # number of eta rows processed per iteration to cap memory use
) -> torch.Tensor:
    """
    Full (Nf, Nt) pairwise cost matrix where C[i,j] = tpp_area_distance_mean(eta[i], rho[j], T).
    Processes eta in chunks of `chunk_size` rows to avoid materialising all Nf*Nt pairs at once.
    Returns: (Nf, Nt) float64 tensor.
    """
    assert eta.ndim == 3 and rho.ndim == 3
    assert eta.shape[1:] == rho.shape[1:]
    Nf = eta.shape[0]
    Nt = rho.shape[0]

    C = torch.empty(Nf, Nt, dtype=torch.float64, device=eta.device)  # (Nf, Nt) output
    for i in range(0, Nf, chunk_size):
        # 1. Slice the current chunk of eta rows.
        eta_i = eta[i : i + chunk_size]  # (cs, L, D)
        cs = eta_i.shape[0]

        # 2. Tile eta_i and rho to enumerate all cs*Nt pairs in a single batched call.
        eta_rep = eta_i.repeat_interleave(Nt, dim=0)  # (cs*Nt, L, D)
        rho_tile = rho.repeat((cs, 1, 1))  # (cs*Nt, L, D)

        # 3. Compute distances and reshape back into the cost-matrix rows.
        d = tpp_area_distance_mean(eta_rep, rho_tile, T)  # (cs*Nt,)
        C[i : i + cs] = d.view(cs, Nt)
    return C
