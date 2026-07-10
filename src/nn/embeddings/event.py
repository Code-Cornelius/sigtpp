import torch
import torch.nn as nn

from src.nn.embeddings.mark import MarkEmbedding
from src.nn.embeddings.time import TimeEmbedding


class EventEmbedding(nn.Module):
    """Concatenates time and mark embeddings: e_j = [omega(tau_j); E_m^T m_j]. Eq. 4 from GNTPP."""

    def __init__(self, time_emb: TimeEmbedding, mark_emb: MarkEmbedding):
        super().__init__()
        self.time_emb = time_emb
        self.mark_emb = mark_emb

    @property
    def embed_size(self) -> int:
        return self.time_emb.embed_size + self.mark_emb.embedding.embedding_dim

    def forward(self, times: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            times: (N, L, 1) scaled inter-arrival times
            marks: (N, L) integer mark types
        Returns:
            (N, L, time_emb_size + mark_emb_size) concatenated embedding
        """
        t_emb = self.time_emb(times)  # (N, L, time_emb_size)
        m_emb = self.mark_emb(marks)  # (N, L, mark_emb_size)
        return torch.cat([t_emb, m_emb], dim=-1)
