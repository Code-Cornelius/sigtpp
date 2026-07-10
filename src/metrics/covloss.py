import logging

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


def cov_torch(x: torch.Tensor) -> torch.Tensor:
    # torch.cov available since PyTorch 1.11, but numpy handles NaN masking more cleanly.
    x = x.reshape(x.shape[0], -1)
    # Move tensor to CPU and convert to a masked NumPy array, masking NaN values
    x_np = np.ma.masked_invalid(x.detach().cpu().numpy())
    # Compute covariance ignoring NaNs
    cov = np.ma.cov(x_np, rowvar=False).filled(np.nan)
    # Return the result as a Torch tensor
    return torch.from_numpy(cov).to(x.device)


class CovLoss(nn.Module):

    @staticmethod
    def compute(seqs: torch.Tensor) -> torch.Tensor:
        return cov_torch(seqs)

    # Computes the covariance matrix of the data and compares true and sampled data.
    # Done on CPU because torch does not support cov.
    # nn.Module because it has a parameter to register.
    def __init__(self, target_seqs: torch.Tensor) -> None:
        super().__init__()
        if target_seqs.shape[1] > 100:
            logger.warning(
                f"Target sequences are of shape {target_seqs.shape}. This might be too large for the correlation computation."
            )
        self.register_buffer('target_covariance', CovLoss.compute(target_seqs), persistent=False)
        return

    def __call__(self, seqs: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Target covariance loss: {self.target_covariance}")
        return torch.abs(CovLoss.compute(seqs) - self.target_covariance)
