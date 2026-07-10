import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size: int):
        assert embed_size > 0, f"Embedding size must be positive but got {embed_size:d}."
        super().__init__()
        self.embed_size = embed_size


class TrigoTimeEmbedding(TimeEmbedding):
    def __init__(self, embed_size: int, min_time: float = 0.0, max_time: float = 1.0, learnable_weights=True):
        # Min and max time for scaling the time embeddings. If you want it to have no impact, set it to 0 and 1.
        super().__init__(embed_size=embed_size)
        assert (
            embed_size % 2 == 0
        ), f"Embedding size must be even but got {embed_size:d}, half used with cos, the other with sin."
        assert min_time < max_time, f"Min time must be less than max time but got {min_time:.2f} and {max_time:.2f}."

        self.min_time = min_time
        self.max_time = max_time

        self.learnable_weights = learnable_weights

        if self.learnable_weights:
            self.embedding_weights: nn.Module = nn.Linear(1, embed_size // 2, bias=False)
            torch.nn.init.xavier_uniform_(self.embedding_weights.weight, gain=nn.init.calculate_gain('tanh'))
        else:
            # Parameter to be moved on device.
            self.embedding_weights: nn.Parameter = nn.Parameter(
                self._build_embedding_weights(embed_size), requires_grad=False
            )

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        # phi shape: (batch_size, Linput, embed_size // 2)
        times_rescaled = (times - self.min_time) / (
            self.max_time - self.min_time
        )  # normalise the time to be between 0 and 1
        if self.learnable_weights:
            # noinspection PyCallingNonCallable
            phi = self.embedding_weights(times_rescaled)
        else:
            phi = torch.matmul(times_rescaled, self.embedding_weights)
        pe_sin = torch.sin(phi * 2.0 * math.pi)
        pe_cos = torch.cos(phi * 2.0 * math.pi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        # Scaling to ensure that the magnitude of the embeddings does not change with the embedd size, as Transformers.
        return pe / math.sqrt(self.embed_size)

    def set_extrema_times(self, min_time: float, max_time: float):
        self.min_time = min_time
        self.max_time = max_time
        return

    @staticmethod
    def _build_embedding_weights(embed_size):
        return torch.pow(10.0, torch.arange(0, embed_size, 2, dtype=torch.float32) * (2.0 / embed_size)).unsqueeze(0)


if __name__ == "__main__":
    torch.manual_seed(0)
    embed_size = 6
    times = torch.tensor([[0.2], [0.5], [0.8]])

    embedding = TrigoTimeEmbedding(embed_size)
    embeddings = embedding(times)

    print("Generated embeddings:", embeddings)
