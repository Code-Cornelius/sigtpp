import torch
import torch.nn as nn


class MarkEmbedding(nn.Module):
    """Learnable embedding for M discrete event types. Eq. 4 from GNTPP."""

    def __init__(self, num_mark_types: int, mark_emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_mark_types, mark_emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, marks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            marks: (N, L) integer tensor of mark types in [0, num_mark_types)
        Returns:
            (N, L, mark_emb_size) embedding tensor
        """
        return self.embedding(marks)
