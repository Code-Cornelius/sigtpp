import logging

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

from src.data_transformations.statscompute import nanmean


def corr_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise Pearson correlation matrix of a 2D or 3D torch tensor,
    while safely handling missing (NaN) values via pairwise deletion.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape (N, L, D) or (N, M). If 3D, the tensor is reshaped to (N, L*D),
        where N is the number of samples and L*D or M is the number of features per sample.
        Missing values (NaNs) are preserved and handled appropriately.

    Returns:
    --------
    torch.Tensor
        A correlation matrix of shape (F, F), where F is the flattened feature dimension.
        Correlation is computed using pairwise deletion: for each pair of features,
        the correlation is calculated using only the samples where both features are observed.
        If fewer than 2 valid samples exist for a pair, the resulting correlation is NaN.

    Behavior:
    ---------
    - Automatically detects and handles NaNs.
    - Uses `np.ma.corrcoef` to compute the correlation matrix with masked values.
    - Issues a warning if any feature has fewer than 10 valid (non-NaN) samples,
      since such correlations may be statistically unreliable.

    Limitations:
    ------------
    - Relies on NumPy masked arrays and runs on CPU.
    - Assumes all sequences are equally shaped (padding must be NaN).
    """
    # torch.cov available since PyTorch 1.11, but numpy handles NaN masking more cleanly.
    x = x.reshape(x.shape[0], -1)
    # Move tensor to CPU and convert to a masked NumPy array, masking NaN values
    x_np = np.ma.masked_invalid(x.detach().cpu().numpy())
    # Compute correlation ignoring NaNs
    corr = np.ma.corrcoef(x_np, rowvar=False).filled(np.nan)

    # Count valid (non-NaN) observations per feature
    valid_counts = np.sum(~x_np.mask, axis=0)  # Shape: (F,)

    # Log a warning if any feature has < 10 valid samples
    if np.any(valid_counts < 10):
        logger.warning(
            "Not enough samples in the following columns to get a precise estimate of the correlation. Be careful!"
        )
        low_sample_indices = np.where(valid_counts < 10)[0]
        logger.warning(f"Feature indices with <10 samples: {low_sample_indices.tolist()}")
        logger.debug(f"Valid sample counts: {valid_counts}")

    # Return the result as a Torch tensor
    return torch.from_numpy(corr).to(x.device)


class CorrLoss(nn.Module):
    @staticmethod
    def compute(seqs: torch.Tensor) -> torch.Tensor:
        return corr_torch(seqs)

    # Computes the correlation matrix of the data and compares true and sampled data.
    # Done on CPU because torch does not support cov.
    # nn.Module because it has a parameter to register.
    def __init__(self, target_seqs: torch.Tensor) -> None:
        super().__init__()
        limit_t = L = target_seqs.shape[1]

        # MIN_AMOUNT_SAMPLES is arbitrary, so the statistics is computed properly.
        MIN_AMOUNT_SAMPLES = 50
        # Find first timestep where any feature has <MIN_AMOUNT_SAMPLES valid values
        mask = ~torch.isnan(target_seqs)
        per_timestep_counts = mask.sum(dim=0)
        counts_per_timestep = per_timestep_counts.cpu().numpy()
        for t in range(L):
            if np.any(counts_per_timestep[t] < MIN_AMOUNT_SAMPLES):
                limit_t = t
                break

        self.register_buffer('slice_t', torch.tensor(limit_t), persistent=False)
        if limit_t != L:
            logger.debug(
                f"For CorrLoss, truncating sequences to first {limit_t} timesteps due to not enough data in last sequences (less than {MIN_AMOUNT_SAMPLES} samples)."
            )

        # Slice sequences and compute correlation
        truncated = target_seqs[:, :limit_t, :]
        self.register_buffer('target_correlation', CorrLoss.compute(truncated), persistent=False)

        logger.log(
            5, f"Target correlation - shape {self.target_correlation.shape}, loss: {self.target_correlation.data}"
        )
        return

    def __call__(self, seqs: torch.Tensor) -> torch.Tensor:
        seqs = seqs[:, : self.slice_t, :]
        assert seqs.shape[1] * seqs.shape[2] == self.target_correlation.shape[0], (
            f"Expected identical shapes (times and features), but got as input to the computation: {seqs.shape}, "
            f"whereas the target is a matrix of shape: {self.target_correlation.shape}"
        )
        return torch.abs(CorrLoss.compute(seqs) - self.target_correlation)

    def loss(self, seqs: torch.Tensor):
        return nanmean(self(seqs))
