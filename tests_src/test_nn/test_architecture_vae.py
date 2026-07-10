import pytest

pytest.importorskip("signatory")

import torch
import torch.nn as nn


# -----------------------------------------------------------------------
# Task 4: ELBO loss
# -----------------------------------------------------------------------
import sys
from unittest.mock import MagicMock
from src.nn.architectures.architecture_vae import Architecture_VAE

from src.nn.nn.vae_modules.vae_encoder import VAEEncoder
from src.nn.nn.vae_modules.vae_decoder import VAEDecoder
from src.nn.rnn.recurrent_nn import Recurrent_nn, RNNType
from src.nn.embeddings.time import TrigoTimeEmbedding


def test_architecture_vae_is_importable():
    """Smoke-test: the class exists and has the right abstract methods."""
    assert hasattr(Architecture_VAE, 'training_step')
    assert hasattr(Architecture_VAE, 'validation_step')
    assert hasattr(Architecture_VAE, 'sample')
    assert hasattr(Architecture_VAE, 'filter_patho_seqs')


def _make_vae_stub():
    """Create Architecture_VAE without calling TPPArchitecture.__init__."""
    obj = object.__new__(Architecture_VAE)
    # Must initialise nn.Module before assigning sub-modules
    nn.Module.__init__(obj)
    # _device is required by the LightningModule.device property
    object.__setattr__(obj, '_device', torch.device('cpu'))
    H, D = 16, 8
    obj.hid_size_rep = H
    obj.latent_dim = D
    obj.time_emb_size = 64
    obj.num_marks = 1
    obj.use_marks = False

    obj.time_emb = TrigoTimeEmbedding(obj.time_emb_size, min_time=0.0, max_time=1.0)

    obj.enc_rnn = Recurrent_nn(obj.time_emb_size, H, 1, False, 0.0, sys.maxsize, RNNType.LSTM, True)
    obj.vae_encoder = VAEEncoder(H, D)
    obj.vae_decoder = VAEDecoder(D, H)
    obj.mse_loss = torch.nn.MSELoss(reduction='none')
    obj.free_bits = 0.0
    obj.recon_weight = 1.0
    return obj


def test_elbo_loss_returns_scalar():
    obj = _make_vae_stub()
    N, L = 4, 10
    log_its = torch.randn(N, L, 1)
    lengths = torch.tensor([L, L - 1, L - 2, L])
    total_loss, _, _, _, time_elbo, _ = obj._compute_elbo_loss(log_its, lengths)
    assert total_loss.shape == torch.Size([]), "loss must be a scalar"
    assert total_loss.item() > 0, "ELBO loss must be positive"
    assert time_elbo.shape == torch.Size([]), "time_elbo must be a scalar"


def test_elbo_loss_backward():
    obj = _make_vae_stub()
    N, L = 4, 10
    log_its = torch.randn(N, L, 1)
    lengths = torch.full((N,), L, dtype=torch.long)
    total_loss, _, _, _, _, _ = obj._compute_elbo_loss(log_its, lengths)
    loss = total_loss
    loss.backward()  # must not raise


# -----------------------------------------------------------------------
# Task 5: Conditional sampling
# -----------------------------------------------------------------------


def test_sample_conditional_shapes():
    obj = _make_vae_stub()
    N, L = 4, 10
    starting_times = torch.zeros(N, 1, 1)
    log_inter_arr_times = torch.randn(N, L, 1)

    obj.MIN_SCALED_DATA = torch.tensor(-5.0)
    obj.MAX_SCALED_DATA = torch.tensor(5.0)

    samples, h_all, gen_marks = obj.sample(
        starting_times=starting_times,
        log_inter_arr_times=log_inter_arr_times,
    )
    assert samples.shape == (N, L, 1), f"Expected ({N},{L},1), got {samples.shape}"
    assert h_all.shape == (N, L, obj.hid_size_rep)
    assert gen_marks is None, "Conditional sampling should return None for gen_marks"


# -----------------------------------------------------------------------
# Task 6: Unconditional sampling
# -----------------------------------------------------------------------


def test_sample_unconditional_shapes():
    obj = _make_vae_stub()
    obj.MIN_SCALED_DATA = torch.tensor(-5.0)
    obj.MAX_SCALED_DATA = torch.tensor(5.0)

    N_gen = 8
    seq_template = torch.randn(32, 15, 1)
    obj.data_train_dts = seq_template
    obj.num_dim_seqs = 1
    obj.first_value_ts_sampler = MagicMock()
    obj.first_value_ts_sampler.sample = MagicMock(
        side_effect=lambda cats, return_indices=False: (
            (torch.zeros(N_gen, 1, 1), torch.zeros(N_gen, dtype=torch.long))
            if return_indices
            else torch.zeros(N_gen, 1, 1)
        )
    )
    obj.anchor_times_sampler = MagicMock()
    obj.anchor_times_sampler.sample = MagicMock(return_value=torch.zeros(N_gen, 1, 1))
    obj.train_marks = torch.zeros(32, 16, dtype=torch.long)  # (N, L+1) marks
    obj.seq_len = seq_template.shape[1]

    samples, h_all, gen_marks = obj.sample(num_seq=N_gen)

    L_train = seq_template.shape[1]
    assert samples.shape == (N_gen, L_train + 1, 1)
    assert h_all.shape == (N_gen, L_train, obj.hid_size_rep)
    assert gen_marks is None, "Non-mark model should return None for gen_marks"
