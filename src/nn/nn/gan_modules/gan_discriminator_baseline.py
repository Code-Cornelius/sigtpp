# Code from https://github.com/EDAPINENUT/GNTPP

import logging

import torch
import torch.nn as nn

from src.nn.nn.basic_nn import BasicNN
from src.utils.fix_seq_ends import set_seq_to_zero_from_index

logger = logging.getLogger(__name__)


class GANDiscriminatorBaseline(nn.Module):

    @staticmethod
    def lipschitz_loss(
        sample_t: torch.Tensor,
        true_t: torch.Tensor,
        out_sample_emb: torch.Tensor,
        out_true_emb: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the Lipschitz loss between embeddings and their corresponding timestamps.

        This function calculates the Lipschitz constant as the ratio of differences in embeddings
        and differences in corresponding time values, ensuring that the Lipschitz condition holds.

        Args:
            sample_t (Tensor): Tensor of shape (batch_size, seq_length, num_feat) representing sampled time values.
            true_t (Tensor): Tensor of shape (batch_size, seq_length, num_feat) representing true time values.
            out_sample_emb (Tensor): Tensor of shape (batch_size, seq_length, embedding_dim) representing embeddings of the sampled data.
            out_true_emb (Tensor): Tensor of shape (batch_size, seq_length, embedding_dim) representing embeddings of the true data.
            lengths (Tensor): Tensor of shape (batch_size,) representing the lengths of the sequences. Used for masking.

        Returns:
            Tensor: Scalar tensor representing the summed absolute deviation of the Lipschitz constant from 1.

        Formula:
            .. math::
                L = \sum \left| \frac{\|out\_sample\_emb - out\_true\_emb\|}{|sample\_t - true\_t| + \epsilon} - 1 \right|
        """
        assert (
            sample_t.shape == true_t.shape
        ), f"sample_t {sample_t.shape} and true_t {true_t.shape} must have the same shape."
        assert (
            out_sample_emb.shape == out_true_emb.shape
        ), f"out_sample_emb {out_sample_emb.shape} and out_true_emb {out_true_emb.shape} must have the same shape."
        assert sample_t.dim() == 3, f"sample_t {sample_t.shape} must be a 3-dimensional tensor."
        assert out_sample_emb.dim() == 3, f"out_sample_emb {out_sample_emb.shape} must be a 3-dimensional tensor."
        assert (
            sample_t.shape[:] == out_sample_emb.shape[:]
        ), f"The batch and sequence dimensions of sample_t {sample_t.shape} and out_sample_emb {out_sample_emb.shape} must match."

        # Compute the Lipschitz constant. Shape: (batch_size, seq_length, embedding_dim).
        # Here we set to zero when we are out of bounds. Because we sample with the same history,
        # the length is identical and we do not use the trick of bounding at time Tmax.
        lip = set_seq_to_zero_from_index(
            (out_sample_emb - out_true_emb).abs() / ((sample_t.cumsum(1) - true_t.cumsum(1)).abs() + 1e-12), lengths
        )

        # Return the Lipschitz loss as the sum of absolute differences from 1
        return (lip - 1.0).abs().mean()

    def __init__(self, hidden_layer_disc, layer_num, hsize_latent_embedding):
        super().__init__()

        self.layer_num = layer_num
        self.hidden_layer_disc = hidden_layer_disc
        self.hsize_latent_embedding = hsize_latent_embedding

        self.network_time = BasicNN(1, [], self.hidden_layer_disc, [True], [], 0.0)
        self.network_hidden = BasicNN(
            self.hsize_latent_embedding,
            [self.hidden_layer_disc * 2],
            self.hidden_layer_disc,
            [True, True],
            [nn.GELU()],
            0.0,
        )
        self.non_linearity = nn.GELU()
        self.network = BasicNN(
            self.hidden_layer_disc,
            [self.hidden_layer_disc] * self.layer_num,
            1,
            [True] * (self.layer_num + 1),
            [nn.GELU()] * self.layer_num,
            0.0,
        )
        return

    def mlp_transform(self, time, h):
        t_emb = self.network_time(time)
        h_emb = self.network_hidden(h)
        emb = (t_emb + h_emb) / 1.44  # sqrt(2)
        emb = self.non_linearity(emb)
        return self.network(emb)

    def forward(
        self,
        fake_samples: torch.Tensor,
        true_samples: torch.Tensor,
        lengths: torch.Tensor,
        hist_embedding: torch.Tensor,
        nu: float = 1.0,
        mode: str = "g",
    ):
        """
        Condition on mode (g for generator, d for discriminator)
        Passing the hist_embedding for discrimination capability of the discriminator.

        lengths should be the length of the fake/true samples. That means, for a sequence
        [1, 2, 3, 3, 3] where 3 repeated means the sequence is of length 3.
        """

        # Expand dimensions if necessary, when multiple samples have been made for the fake samples.
        if len(fake_samples.shape) - len(true_samples.shape) == 1:
            true_samples = true_samples[:, None, ...].expand_as(fake_samples)
            hist_embedding = hist_embedding[:, None, ...].expand(fake_samples.shape + (-1,))

        # Compute embeddings for both sample and true values
        out_sample_emb = self.mlp_transform(fake_samples, hist_embedding)
        out_true_emb = self.mlp_transform(true_samples, hist_embedding)

        # Generator loss
        cum_difference = out_sample_emb - out_true_emb  # .cumsum(1) ## cumsum redundant/unnecessary.
        # - 1 to convert lengths to indices.
        masked_difference = set_seq_to_zero_from_index(cum_difference, lengths - 1)
        g_loss = masked_difference.mean()
        # No need to clone because it is a partition of cases.
        d_loss = g_loss
        if mode == "d" and nu > 1e-8:
            d_loss = d_loss - nu * GANDiscriminatorBaseline.lipschitz_loss(
                fake_samples, true_samples, out_sample_emb, out_true_emb, lengths
            )
        return g_loss if mode == "g" else -d_loss
