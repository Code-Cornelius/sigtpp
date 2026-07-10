import logging

import torch

logger = logging.getLogger(__name__)


from src.metrics.gpu_safe_wrapper import gpu_memory_safe
from src.metrics.reduce_weighted import reduce_weighted_per_num_samples

from src.data_transformations.statscompute import nanmean


def get_crps_loss_fine_grain_SORTED(samples: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    CRPS for samples [N, L, S] and targets [N, L], using the sorted-samples trick.
    NaNs in `samples` are ignored along the S dimension (variable # of valid draws).
    If all S are NaN for a timestep, the result is NaN for that [N, L] entry.
    """
    targets = targets.unsqueeze(-1)  # [N, L, 1]

    # Term 1: E[|X - y|] over valid samples (NaN-aware mean along S)
    term1 = nanmean(torch.abs(samples - targets), dim=-1)  # [N, L]

    # ---- Term 2: 0.5 * E|X - X'| via identity ----
    # Identity: sum_{i,j} |x_i - x_j| = 2 * sum_{k=1..m} (2k - m - 1) * x_(k)
    # => term2 = (1 / m^2) * sum_{k=1..m} (2k - m - 1) * x_(k)
    # where m = number of valid (non-NaN) samples for that [N, L].

    valid = ~torch.isnan(samples)  # [N, L, S]
    m = valid.sum(dim=-1)  # [N, L], integer counts of valid samples

    # Push NaNs to the end for sorting by replacing with +inf, then sort ascending
    x_filled = torch.where(valid, samples, float('inf'))
    x_sorted, _ = torch.sort(x_filled, dim=-1)  # [N, L, S], +inf at the end

    S = samples.shape[-1]
    k = torch.arange(1, S + 1, device=samples.device, dtype=samples.dtype).view(1, 1, S)  # 1..S
    m_f = m.to(samples.dtype).unsqueeze(-1)  # [N, L, 1]

    # Only the first m positions are valid after sorting (the rest are +inf placeholders).
    mask = k <= m_f  # [N, L, S], True for k = 1..m

    # Zero-out the invalid tail to avoid 0 * inf issues and to exclude from sums
    x_sorted = torch.where(mask, x_sorted, torch.zeros((), device=samples.device, dtype=samples.dtype))

    # Weights: (2k - m - 1) with m specific to each [N, L]
    w = torch.where(mask, 2 * k - m_f - 1, torch.zeros((), device=samples.device, dtype=samples.dtype))

    numerator = (w * x_sorted).sum(dim=-1)  # [N, L]
    denom = (m_f.squeeze(-1) ** 2).clamp_min(1)  # avoid div-by-zero for m=0
    term2 = numerator / denom
    term2 = term2.masked_fill(m == 0, float('nan'))  # if no valid samples, define as NaN

    return term1 - term2  # [N, L]


def get_crps_loss_fine_grain(samples: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the Continuous Ranked Probability Score (CRPS), modified to work
    with `samples` of shape [N, L, S] and handle NaNs for varying sequence lengths.

    Parameters:
    - samples: torch.Tensor of shape [N, L, S] (Sampled forecasts)
    - targets: torch.Tensor of shape [N, L] (Observed values)

    Returns:
    - crps: torch.Tensor of shape [N, L] (CRPS per sequence and timestep)
    """
    # Expand targets to match sample shape [N, L, S]
    targets = targets.unsqueeze(-1)  # [N, L, 1]

    # Compute first term: E[|X - y|]
    term1 = nanmean(torch.abs(samples - targets), dim=-1)

    # Compute second term: 0.5 * E[|X - X'|]
    samples_i = samples.unsqueeze(-2)  # [N, L, 1, S]
    samples_j = samples.unsqueeze(-1)  # [N, L, S, 1]
    term2 = 0.5 * torch.abs(samples_i - samples_j).sum(dim=(-1, -2)) / samples_i.shape[-1] / samples_i.shape[-1]

    return term1 - term2  # [N, L]


def get_crps_loss(samples: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for a set of forecast samples and targets.

    This function computes CRPS using a fine-grained method and then **performs a weighted average**
    over all available timesteps and sequences. The weighting ensures that sequences with missing
    values (or sequences with NaNs) do not contribute disproportionately to the final CRPS score.

    Parameters:
    - samples: torch.Tensor of shape [N, L, S]
        - Forecasted samples for `N` sequences, each of length `L`, with `S` Monte Carlo samples.
    - targets: torch.Tensor of shape [N, L]
        - Ground-truth values corresponding to each sequence and timestep.

    Returns:
    - crps: torch.Tensor (scalar)
        - The computed CRPS, averaged over all sequences and timesteps.
    """
    # Compute fine-grained CRPS per sequence and per timestep.
    # Extra precision because sometimes we may see instabilities there.
    samples = samples.double()
    targets = targets.double()
    # crps_loss = get_crps_loss_fine_grain(samples, targets)  # Shape: [N, L]
    crps_loss = get_crps_loss_fine_grain_SORTED(samples, targets)  # Shape: [N, L]

    # Compute the number of valid timesteps per sequence:
    # to avoid mistakenly excluding near-zero CRPS values due to numerical precision.
    sample_sizes = (crps_loss.abs() > 1e-6).sum(dim=0)  # Shape: [L]. Works despite NaNs.

    ########### This procedure might be wrong because we downsample twice. It is not correct and we should
    # use the other implementation.
    # Compute the **weighted average** over timesteps:
    # - Each timestep contributes proportionally based on how many valid sequences exist at that step.
    # - The denominator `sample_sizes.sum()` ensures normalization across all available data.
    weighted_crps = (crps_loss * sample_sizes / sample_sizes.sum()).nansum(dim=1).mean(0)
    return weighted_crps


# ---------- CRPS wrapper ----------
@gpu_memory_safe
def get_crps_loss_weighted_by_targets(
    samples: torch.Tensor,  # [N, L, S]
    targets: torch.Tensor,  # [N, L]
) -> torch.Tensor:
    """
    Reduce CRPS per-element loss [N, L] by mean over L per sample, then mean over N.
    Assumes you have a function `get_crps_loss_fine_grain_SORTED(samples, targets) -> [N, L]`.
    """
    samples = samples.double()
    targets = targets.double()
    crps_elem = get_crps_loss_fine_grain_SORTED(samples, targets)  # [N, L]
    return reduce_weighted_per_num_samples(crps_elem.unsqueeze(-1), targets=targets.unsqueeze(-1))
