import torch
import torch.nn as nn


class MarkPredictor(nn.Module):
    """Linear head for mark classification. Eq. 26-27 from GNTPP.

    Given history encoding h, predicts a distribution over mark types via
    a linear projection followed by softmax (softmax applied externally).
    """

    def __init__(self, hidden_size: int, num_mark_types: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_mark_types)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, L, hidden_size) history encoding
        Returns:
            (N, L, num_mark_types) logits over mark types (apply softmax / CE loss externally)
        """
        return self.linear(h)
