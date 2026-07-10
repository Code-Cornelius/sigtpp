import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from src.nn.nn.basic_nn import BasicNN

from config import EPSILON_STABILITY


class Decoder4GANFullInput(nn.Module):
    MODE_POISSON_OUTPUT = False
    # Remove as well the log, below, in the scaling and in the expscaler.

    def __init__(
        self,
        hidden_layer_input_dec,
        hidden_layer_dec,
        layer_num,
        hsize_latent_embedding,
        deterministic_model: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_layer_input_dec = hidden_layer_input_dec
        self.hidden_layer_dec = hidden_layer_dec
        self.layer_num = layer_num
        self.hsize_latent_embedding = hsize_latent_embedding
        self.deterministic_model = deterministic_model

        # ---- Choose the basic-interval generator once (no branching in forward) ----
        if deterministic_model:
            # E[log(Exp(1))] = -gamma ≈ -0.57721566  (good constant proxy; eps has negligible effect)
            self.register_buffer("_basic_interval_const", torch.tensor(-0.5772156649015329, dtype=torch.float32))

            def _gen_const(size: torch.Size, *, device: torch.device, dtype: torch.dtype):
                # broadcast the constant without allocating large tensors repeatedly
                return self._basic_interval_const.to(device=device, dtype=dtype).expand(size)

            self.generate_basic_interval = _gen_const
        else:

            def _gen_rand(size: torch.Size, *, device: torch.device, dtype: torch.dtype):
                # torch-only fast path (avoid numpy); sample Exp(1), add eps, then log
                return torch.empty(size, device=device, dtype=dtype).exponential_(1.0).add_(EPSILON_STABILITY).log_()

            self.generate_basic_interval = _gen_rand
        # ---------------------------------------------------------------------------

        if Decoder4GANFullInput.MODE_POISSON_OUTPUT:
            self.network_time = BasicNN(
                1,
                [],
                1,
                [False],
                [],
                0.0,
            )
        else:
            # depends on embedding size.
            self.network_time1 = BasicNN(
                1,
                [self.hidden_layer_input_dec],
                self.hidden_layer_input_dec,
                [True] * (2),
                [nn.GELU()],
                0.0,
            )
            self.network_time2 = BasicNN(
                1,
                [self.hidden_layer_input_dec],
                self.hidden_layer_input_dec,
                [True] * (2),
                [nn.GELU()],
                0.0,
            )
            self.network_time3 = BasicNN(
                64,
                [self.hidden_layer_input_dec],
                self.hidden_layer_input_dec,
                [True] * (2),
                [nn.GELU()],
                0.0,
            )

            self.network_hidden = BasicNN(
                self.hsize_latent_embedding,
                [self.hidden_layer_dec * 2],
                self.hidden_layer_input_dec,
                [True, True],
                [nn.GELU()],
                0.0,
            )
            self.network = BasicNN(
                self.hidden_layer_input_dec,
                [self.hidden_layer_dec],
                1,
                [True] * (2),
                [nn.GELU()],
                0.0,
            )
            self.non_linearity = nn.GELU()
        return

    def forward(self, prev_values_scaled, history_embedding, cum_time, num_samples_per_seq=1):
        size = (num_samples_per_seq,) + history_embedding.shape[:-1]

        ### This is needed when we work with num_samples_per_seq > 1. We don't need that here for now.
        # history_embedding = history_embedding.unsqueeze(dim=0)

        # Use the instance-bound generator (constant in deterministic mode; random otherwise)
        basic_t = self.generate_basic_interval(
            size=size,
            device=history_embedding.device,
            dtype=history_embedding.dtype,
        )

        assert num_samples_per_seq == 1, "Only one sample per sequence is supported."
        # If we want to change smthg, we would need to adapt the shape of cum_time, prev_values_scaled. Basic_t is reshaped below for case = 1.
        basic_t = basic_t.squeeze(dim=0)
        try:
            t_emb1 = self.network_time1(basic_t.unsqueeze(dim=-1))
            # Gen( h_i, tau_wn, x_{i-1}, cum_x_i ) := x_i
            t_emb2 = self.network_time2(cum_time)
            # t_emb3 = self.network_time3(prev_values_scaled)
            h_emb = self.network_hidden(history_embedding)
            # This works
            emb = (t_emb1 + t_emb2 + h_emb) / 2.0
            # emb = (t_emb1 + t_emb2 + t_emb3 + h_emb) / 2.0
            # emb = (t_emb1 + h_emb) / 1.44
            # emb = (t_emb1 + t_emb2) / 1.44
            emb = self.non_linearity(emb)
            out = self.network(emb)
            # samples are of shape (N, L, num_samples_per_seq) after transposing.
            # Transpose needed for the case num_samples_per_seq > 1.
            samples = out  # .transpose(0, 1).transpose(1, 2)
        except AttributeError as e:
            if Decoder4GANFullInput.MODE_POISSON_OUTPUT:
                # the output is X ~ log(Poisson(1)) + \phi. So after rescaling: exp(X) ~ Poisson(1) * exp(\phi).
                # samples = self.network_time(torch.zeros_like(basic_t.unsqueeze(dim=-1))) / 10.0 + basic_t.unsqueeze(dim=-1)
                # Alternatively, remove the logs.
                samples = self.network_time(basic_t.unsqueeze(dim=-1))
            else:
                logger.error(f"Failed to generate samples: {e}")
                raise
        # To put back if we want to support num_samples_per_seq > 1.
        # if num_samples_per_seq == 1:
        #     samples = samples.squeeze(dim=1)
        #     return samples

        # Remove time component which is 1. It could be done prior.
        return samples.squeeze(1)
