import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

from src.nn.nn.basic_nn import BasicNN
from src.nn.nn.basic_nn_with_residual import BasicNNWithResiduals

from src.nn.embeddings.time import TrigoTimeEmbedding


class Decoder4ScoreMatching(nn.Module):

    def __init__(
        self, hidden_layer_input_dec, hidden_layer_dec, layer_num, hsize_latent_embedding, dim_timeseries_embed
    ) -> None:
        super().__init__()
        self.time_embed_dim = 32
        self.trigotime_embed = TrigoTimeEmbedding(self.time_embed_dim, 0, 10)

        self.hidden_layer_input_dec = hidden_layer_input_dec
        self.hidden_layer_dec = hidden_layer_dec
        self.layer_num = layer_num
        self.hsize_latent_embedding = hsize_latent_embedding

        # For timestamp
        self.network_time1 = BasicNN(
            self.time_embed_dim,
            [self.hidden_layer_input_dec],
            self.hidden_layer_input_dec,
            [True] * (2),
            [nn.GELU()],
            0.0,
        )
        # cum time
        self.network_time2 = BasicNN(
            1,
            [self.hidden_layer_input_dec],
            self.hidden_layer_input_dec,
            [True] * (2),
            [nn.GELU()],
            0.0,
        )
        # actual noising and denoising ts
        self.network_time3 = BasicNN(
            1,  # we could use dim_timeseries_embed, but it makes it slightly more complicated as we would need to unembed once denoised.
            [self.hidden_layer_input_dec],
            self.hidden_layer_input_dec,
            [True] * (2),
            [nn.GELU()],
            0.0,
        )
        # hidden state
        self.network_hidden = BasicNN(
            self.hsize_latent_embedding,
            [self.hidden_layer_input_dec * 2],
            self.hidden_layer_input_dec,
            [True, True],
            [nn.GELU()],
            0.0,
        )
        ### NOT WORKING YET
        self.network = BasicNNWithResiduals(
            self.hidden_layer_input_dec,
            [self.hidden_layer_dec],
            1,
            [True] * (2),
            [nn.GELU()],
            0.0,
        )
        self.non_linearity = nn.GELU()
        return

    def forward(self, prev_values_scaled, history_embedding, cum_time, noisy_input, time_stamps, num_samples_per_seq=1):
        t_emb = self.trigotime_embed(time_stamps)

        assert num_samples_per_seq == 1, "Only one sample per sequence is supported."
        try:
            t_emb1 = self.network_time1(t_emb).expand(cum_time.shape[0], cum_time.shape[1], -1)
            t_emb2 = self.network_time2(cum_time)
            t_emb3 = self.network_time3(noisy_input)
            h_emb = self.network_hidden(history_embedding)

            emb = (t_emb1 + t_emb2 + t_emb3 + h_emb) / 2.0
            emb = self.non_linearity(emb)
            out = self.network(emb)
        except Exception as e:
            logger.error(f"Failed to generate samples: {e}")
            raise

        return out
