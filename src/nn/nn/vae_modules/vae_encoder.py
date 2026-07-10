import torch
import torch.nn as nn

from src.nn.nn.basic_nn import BasicNN


class VAEEncoder(nn.Module):
    """Variational encoder q_ξ(z | τ_i, h_{i-1})  →  (μ, log_var).

    Two-branch architecture (matching GANDecoderBaseline pattern):
      - branch 1: τ_i  (N, L, 1)  →  (N, L, H)
      - branch 2: h_{i-1}  (N, L, H)  →  (N, L, H)
    Branches are summed and passed through GELU, then split into μ and log_var heads.

    Note: the output is log-variance (log σ²), not log-std. The reparameterization
    trick uses sigma = exp(0.5 * log_var), i.e. std = sqrt(exp(log_var)).
    The history branch uses a [H → 2H → H] MLP for additional capacity.
    """

    def __init__(self, hidden_size: int, latent_dim: int) -> None:
        super().__init__()
        self.network_tau = BasicNN(1, [], hidden_size, [True], [], 0.0)
        self.network_hidden = BasicNN(hidden_size, [hidden_size * 2], hidden_size, [True, True], [nn.GELU()], 0.0)
        self.non_linearity = nn.GELU()
        self.mu_head = BasicNN(hidden_size, [], latent_dim, [True], [], 0.0)
        self.log_var_head = BasicNN(hidden_size, [], latent_dim, [True], [], 0.0)

    def forward(self, tau_i: torch.Tensor, h: torch.Tensor):
        """
        Args:
            tau_i: (N, L, 1)  scaled log inter-arrival times (observations)
            h:     (N, L, H)  history encodings h_{i-1}
        Returns:
            mu:      (N, L, D)  posterior mean
            log_var: (N, L, D)  posterior log-variance (log σ²)
        """
        t_emb = self.network_tau(tau_i)  # (N, L, H)
        h_emb = self.network_hidden(h)  # (N, L, H)
        # Divide by sqrt(2) ≈ 1.44 to preserve unit variance when summing two
        # independent unit-variance vectors.
        emb = self.non_linearity((t_emb + h_emb) / 1.44)
        return self.mu_head(emb), self.log_var_head(emb)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + exp(0.5·log_var) · ε, ε ~ N(0,I).

        log_var is the log-variance (log σ²), so exp(0.5·log_var) = σ (std).
        """
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(mu)
