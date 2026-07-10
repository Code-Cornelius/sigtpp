"""
Sliced Wasserstein-1 Distance for Temporal Point Processes.

Projects TPP sequences (NaN-padded cumulative times) onto random 1D directions,
then computes the exact 1D W1 distance via sorted CDF integration.
Complexity: O(K * (N + N log N)) instead of O(N^2) for standard W1.
"""

from typing import Optional

import torch


def _w1_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Exact 1D Wasserstein-1 distance between two empirical distributions.

    W1(mu_n, nu_m) = integral |F_X(t) - F_Y(t)| dt

    Computed via sorted merge + prefix CDF sums in O((nx+ny) log(nx+ny)).
    Handles unequal sample sizes.
    """
    x_sorted = x.sort().values.to(torch.float64)
    y_sorted = y.sort().values.to(torch.float64)
    nx, ny = x_sorted.shape[0], y_sorted.shape[0]

    # All CDF breakpoints (merged sorted array)
    pts, _ = torch.cat([x_sorted, y_sorted]).sort()

    # Empirical CDF values just after each breakpoint
    fx = torch.searchsorted(x_sorted, pts, right=True).to(torch.float64) / nx
    fy = torch.searchsorted(y_sorted, pts, right=True).to(torch.float64) / ny

    # Interval widths between consecutive breakpoints (last width = 0)
    widths = torch.zeros_like(pts)
    widths[:-1] = pts[1:] - pts[:-1]

    return ((fx - fy).abs() * widths).sum()


@torch.no_grad()
def sliced_wasserstein_tpp(
    eta: torch.Tensor,  # (N_eta, L, D) cumulative times, NaN-padded
    rho: torch.Tensor,  # (N_rho, L, D) cumulative times, NaN-padded
    T: float,
    num_projections: int = 200,
    generator: Optional[torch.Generator] = None,
) -> float:
    """
    Sliced Wasserstein-1 Distance for TPP sequences.

    Algorithm:
      1. Replace NaN with T (same convention as tpp_area_distance_mean)
      2. Flatten each sequence to R^(L*D)
      3. Project onto `num_projections` random unit vectors
      4. Compute exact 1D W1 for each projection via sorted CDF integral
      5. Average across projections

    Returns scalar float, normalized by T^2 for scale invariance.

    Note: Unlike the unbiased ED estimator, SWD between two independent
    samples from the *same* distribution does NOT converge to zero at finite N
    (finite-sample OT bias). Use within-vs-cross comparisons for discrimination,
    not absolute proximity to zero.
    """
    assert eta.ndim == 3 and rho.ndim == 3
    assert eta.shape[1:] == rho.shape[1:]
    N_eta, L, D = eta.shape
    N_rho = rho.shape[0]
    assert N_eta >= 1 and N_rho >= 1

    device = eta.device
    T_tensor = torch.tensor(T, device=device, dtype=eta.dtype)

    # 1. Replace NaN with T (same convention as tpp_area_distance_mean).
    eta_filled = torch.where(torch.isnan(eta), T_tensor, eta)
    rho_filled = torch.where(torch.isnan(rho), T_tensor, rho)

    # 2. Flatten each sequence to R^(L*D) and project onto random unit vectors.
    eta_flat = eta_filled.reshape(N_eta, L * D).to(torch.float64)
    rho_flat = rho_filled.reshape(N_rho, L * D).to(torch.float64)

    dim = L * D
    projs = torch.randn(num_projections, dim, dtype=torch.float64, device=device, generator=generator)
    projs = projs / projs.norm(dim=1, keepdim=True)

    eta_proj = projs @ eta_flat.T  # (K, N_eta)
    rho_proj = projs @ rho_flat.T  # (K, N_rho)

    # 3. Compute exact 1D W1 for each projection and average.
    total_w1 = torch.tensor(0.0, dtype=torch.float64, device=device)
    for k in range(num_projections):
        total_w1 += _w1_1d(eta_proj[k], rho_proj[k])

    return (total_w1 / num_projections).item() / (T * T)
