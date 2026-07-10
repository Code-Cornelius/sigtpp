import typing

import torch


def compute_all_moment_losses(
    gen_seqs: torch.Tensor,
    autocorr_loss,
    corr_loss,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    return corr_loss(gen_seqs), autocorr_loss(gen_seqs)
