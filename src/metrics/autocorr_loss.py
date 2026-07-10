import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

from src.data_transformations.statscompute import nanmean
from src.metrics.crosscor import autocorr, autocorr_nonstationary


class AutoCorrLoss(nn.Module):
    """
    Compute the autocorrelation difference between the real and fake data.
    It is defined as:
    .. math::
        \\mathcal{L}_{\\text{CrossCor}}(x, \\hat{x}) = \\mathbb{E}_{x, \\hat{x}} \\left[ \\left| \\frac{1}{n} \\sum_{t=1}^{n} x_t \\hat{x}_{t + \\tau} \\right| \\right]

    nn.Module because it has a parameter to register.
    """

    @staticmethod
    def compute(seqs: torch.Tensor, max_lag: int, stationary: bool) -> torch.Tensor:
        if stationary:
            return autocorr(seqs, max_lag)
        return autocorr_nonstationary(seqs, max_lag)

    def __init__(self, target_seqs: torch.Tensor, max_lag: int, stationary: bool):
        super().__init__()

        self.max_lag = max_lag
        self.stationary = stationary
        self.register_buffer(
            'target_autocorr',
            AutoCorrLoss.compute(target_seqs, self.max_lag, self.stationary),
            persistent=False,
        )
        logger.debug(f"Target autocorrelation loss: {self.target_autocorr.view(-1)}")

    def __call__(self, seqs: torch.Tensor):
        return torch.abs(AutoCorrLoss.compute(seqs, self.max_lag, self.stationary) - self.target_autocorr)

    def loss(self, seqs: torch.Tensor):
        return nanmean(self(seqs))


class AutoCorrLossTPPSeqs(nn.Module):
    # A wrapper for simple replacement where we compute the average AutoCorrLoss over the I.T. and the Cum. Times.
    def __init__(self, target_seqs: torch.Tensor, max_lag: int, stationary: bool):
        super().__init__()

        self.max_lag = max_lag
        self.stationary = stationary
        self.register_buffer(
            'target_autocorr',
            AutoCorrLoss.compute(target_seqs, self.max_lag, self.stationary),
            persistent=False,
        )

        self.loss_it = AutoCorrLoss(target_seqs, max_lag, stationary)
        self.loss_cum = AutoCorrLoss(target_seqs.cumsum(dim=1), max_lag, stationary)

    def __call__(self, seqs: torch.Tensor):
        return self.loss_it(seqs) + self.loss_cum(seqs.cumsum(dim=1))

    def loss(self, seqs: torch.Tensor):
        return nanmean(self(seqs))
