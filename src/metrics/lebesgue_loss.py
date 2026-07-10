import torch

from src.data_transformations.statscompute import nanmean
from src.metrics.gpu_safe_wrapper import gpu_memory_safe
from src.metrics.reduce_weighted import reduce_weighted_per_num_samples


def get_perc_L1loss(tensor1, tensor2):
    # To exclude values from the difference, replace with nans.
    return nanmean(torch.abs(tensor1 - tensor2) / (torch.abs(tensor2) + 1e-6))


def get_L1loss(tensor1, tensor2):
    # To exclude values from the difference, replace with nans.
    return nanmean(torch.abs(tensor1 - tensor2))


@gpu_memory_safe
def get_L2loss_weighted_by_targets(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
) -> torch.Tensor:
    elem = (tensor1 - tensor2) ** 2
    return reduce_weighted_per_num_samples(elem, targets=tensor2)


@gpu_memory_safe
def get_perc_L1loss_weighted_by_targets(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
) -> torch.Tensor:
    denom = torch.abs(tensor2) + 1e-6
    elem = torch.abs(tensor1 - tensor2) / denom
    return reduce_weighted_per_num_samples(elem, targets=tensor2)


@gpu_memory_safe
def get_L1loss_weighted_by_targets(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
) -> torch.Tensor:
    elem = torch.abs(tensor1 - tensor2)
    return reduce_weighted_per_num_samples(elem, targets=tensor2)


@gpu_memory_safe
def get_L1loss_conditional_weighted_by_targets(
    samples: torch.Tensor,  # (N, S, L)
    targets: torch.Tensor,  # (N, L)
) -> torch.Tensor:
    """MAE from conditional samples: mean over S of |sample_s - target|, then weighted reduce."""
    # (N, S, L) - (N, 1, L) → mean over S → (N, L)
    elem = torch.abs(samples - targets.unsqueeze(1)).mean(dim=1)  # (N, L)
    return reduce_weighted_per_num_samples(elem.unsqueeze(-1), targets=targets.unsqueeze(-1))
