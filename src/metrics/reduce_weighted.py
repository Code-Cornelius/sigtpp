import logging

import torch

logger = logging.getLogger(__name__)


def reduce_weighted_per_num_samples(
    loss_elem: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Reduce an elementwise loss to a scalar:
      1) per-sample masked mean along `count_dims` (default: (1, 2))
      2) average these per-sample means across axis 0 (samples) for samples with any valid data

    Assumptions:
      - axis 0 is batch/sample dimension
      - targets define validity (non-NaN => valid)
      - loss_elem has the same shape as targets (or broadcasts to it)
    """
    assert loss_elem.ndim == 3, f"loss_elem must have 3 dimensions (N, L, D) but has shape {loss_elem.shape}."
    assert loss_elem.shape[1:] == targets.shape[1:], f"Shapes must match: {loss_elem.shape} vs {targets.shape}."

    # Mask that both entries are finite (non-NaN, non-inf).
    valid_entries = ~(torch.isnan(targets) | torch.isnan(loss_elem))
    # Sum and count over the requested axes
    count_valid_p_feat = valid_entries.sum(dim=0)
    # Per-sample means; avoid div-by-zero by clamping denominator
    mean_per_feat = torch.nansum(loss_elem, dim=0) / count_valid_p_feat.clamp_min(1.0)

    # Dimensions per axis 1 and 2 where there is at least one valid data point
    has_data_p_feat = count_valid_p_feat > 0
    if has_data_p_feat.any():
        count_valid = count_valid_p_feat[has_data_p_feat]
        return (mean_per_feat[has_data_p_feat] * count_valid).sum() / count_valid.sum()
    else:
        logger.error("Error in the computation of the loss: no valid entries. Returning inf.")
        return torch.tensor(float("inf"), dtype=loss_elem.dtype, device=loss_elem.device)
