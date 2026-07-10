"""
Sliced Energy Distance for Temporal Point Processes.

Projects TPP sequences (NaN-padded cumulative times) onto random 1D directions,
then computes 1D energy distance analytically via sorting.
Complexity: O(K * N log N) instead of O(N^2) for standard ED.
"""

from typing import Optional

import torch


def _mean_abs_diff_sorted(s: torch.Tensor) -> torch.Tensor:
    """
    Compute E|S - S'| (unbiased) for 1D sorted samples s of size n.
    Uses: sum_{i<j} (s_j - s_i) = sum_k s_k * (2k - n + 1)  (0-indexed)
    """
    n = s.shape[0]
    if n < 2:
        return torch.tensor(0.0, dtype=torch.float64, device=s.device)
    idx = torch.arange(n, dtype=torch.float64, device=s.device)
    weights = 2.0 * idx - (n - 1.0)
    return (s.to(torch.float64) * weights).sum() / (n * (n - 1.0))


def _mean_abs_cross_sorted(sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
    """
    Compute E|X - Y| for sorted 1D samples sx (nx,) and sy (ny,).
    Uses searchsorted + prefix sums: O((nx+ny) log(ny)).
    """
    nx = sx.shape[0]
    ny = sy.shape[0]
    sx_f = sx.to(torch.float64)
    sy_f = sy.to(torch.float64)

    ranks = torch.searchsorted(sy_f, sx_f)  # (nx,) number of y values < x_i

    prefix_y = torch.zeros(ny + 1, dtype=torch.float64, device=sy.device)
    prefix_y[1:] = sy_f.cumsum(0)
    total_y = prefix_y[ny]

    ranks_f = ranks.to(torch.float64)
    per_x = sx_f * ranks_f - prefix_y[ranks] + (total_y - prefix_y[ranks]) - sx_f * (ny - ranks_f)
    return per_x.sum() / (nx * ny)


def _energy_distance_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Unbiased 1D energy distance: ED = 2*E|X-Y| - E|X-X'| - E|Y-Y'|.
    O(n log n) via sorting + prefix sums.
    """
    x_sorted = x.sort().values
    y_sorted = y.sort().values
    exy = _mean_abs_cross_sorted(x_sorted, y_sorted)
    exx = _mean_abs_diff_sorted(x_sorted)
    eyy = _mean_abs_diff_sorted(y_sorted)
    return 2.0 * exy - exx - eyy


@torch.no_grad()
def sliced_energy_distance_tpp(
    eta: torch.Tensor,  # (N_eta, L, D) cumulative times, NaN-padded
    rho: torch.Tensor,  # (N_rho, L, D) cumulative times, NaN-padded
    T: float,
    num_projections: int = 200,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Sliced Energy Distance for TPP sequences, normalised by T²."""
    assert eta.ndim == 3 and rho.ndim == 3
    assert eta.shape[1:] == rho.shape[1:]
    N_eta, L, D = eta.shape
    N_rho = rho.shape[0]
    assert N_eta >= 2 and N_rho >= 2

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

    # 3. Compute 1D ED for each projection and average.
    total_ed = torch.tensor(0.0, dtype=torch.float64, device=device)
    for k in range(num_projections):
        total_ed += _energy_distance_1d(eta_proj[k], rho_proj[k])

    return (total_ed / num_projections).item() / (T * T)
