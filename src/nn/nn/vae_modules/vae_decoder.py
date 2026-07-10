import torch
import torch.nn as nn

from src.nn.nn.basic_nn import BasicNN


class VAEDecoder(nn.Module):
    """Decoder p_θ(τ̂ | z, h_{i-1}) with deterministic output (MSE/Gaussian likelihood).

    Two-branch architecture mirroring GANDecoderBaseline:
      - branch 1: z  (N, L, D)  →  (N, L, H)
      - branch 2: h  (N, L, H)  →  (N, L, H)
    Branches are summed, then passed through a multi-layer MLP to output τ̂ ∈ R.
    """

    def __init__(self, latent_dim: int, hidden_size: int, layer_num: int = 3, h_dropout: float = 0.5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.network_latent = BasicNN(latent_dim, [], hidden_size, [True], [], 0.0)
        self.network_hidden = BasicNN(hidden_size, [hidden_size * 2], hidden_size, [True, True], [nn.GELU()], 0.0)
        self.h_dropout = nn.Dropout(p=h_dropout)
        self.non_linearity = nn.GELU()
        self.network = BasicNN(
            hidden_size,
            [hidden_size] * layer_num,
            1,
            [True] * (layer_num + 1),
            [nn.GELU()] * layer_num,
            0.0,
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (N, L, D)  latent vectors
            h: (N, L, H)  history encodings h_{i-1}
        Returns:
            tau_hat: (N, L, 1)
        """
        z_emb = self.network_latent(z)  # (N, L, H)
        h_emb = self.network_hidden(self.h_dropout(h))  # (N, L, H)
        # Divide by sqrt(2) ≈ 1.44 to preserve unit variance when summing two
        # independent unit-variance vectors.
        emb = self.non_linearity((z_emb + h_emb) / 1.44)
        return self.network(emb)  # (N, L, 1)

    def sample(self, h: torch.Tensor) -> torch.Tensor:
        """Sample τ̂ by drawing z ~ N(0,I) and decoding.

        Args:
            h: (N, L, H) history encodings
        Returns:
            tau_hat: (N, L, 1)
        """
        z = torch.randn(*h.shape[:-1], self.latent_dim, device=h.device, dtype=h.dtype)
        return self.forward(z, h)
