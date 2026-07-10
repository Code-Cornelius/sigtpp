import logging

import torch

logger = logging.getLogger(__name__)
from src.nn.architectures.mark_prediction_utils import MARK_IGNORE_INDEX


@torch.no_grad()
def top_k_accuracy(
    mark_logits: torch.Tensor,
    mark_targets: torch.Tensor,
    k: int,
) -> float:
    """Compute top-k accuracy for mark prediction, ignoring padded positions.

    Args:
        mark_logits: (N, L, M) raw logits over mark types.
        mark_targets: (N, L) integer ground-truth marks.
        k: Number of top predictions to consider (1 for top-1, 3 for top-3, etc.).
    Returns:
        Accuracy as a float in [0, 1]. Returns 0.0 if no valid positions exist.
    """
    mask = mark_targets != MARK_IGNORE_INDEX  # (N, L)
    if not mask.any():
        return 0.0

    # Clamp k so top-k works even when k > number of classes.
    num_classes = mark_logits.shape[-1]
    k_eff = min(k, num_classes)
    top_k_preds = mark_logits.topk(k_eff, dim=-1).indices

    # Expand targets for broadcasting: (N, L, 1)
    targets_expanded = mark_targets.unsqueeze(-1)

    # A position is correct if the true label appears among the top-k predictions
    correct = (top_k_preds == targets_expanded).any(dim=-1)  # (N, L)

    return (correct & mask).sum().item() / mask.sum().item()
