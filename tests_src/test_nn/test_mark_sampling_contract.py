"""Tests for the mark sampling return contract (plan phases C–G).

Covers:
- All architectures return 3-tuple from sample()
- Unconditional marked samplers return gen_marks with shape (N, L)
- Non-mark / baseline architectures return None
- Conditional sample() returns None for gen_marks
- sample_and_fix_seqs() strips first mark in lockstep with τ₁
- Pathological-sequence filtering keeps gen_marks aligned
- forward(include_first_it) preserves mark alignment
- _sample_first_event co-samples from valid first-event marks
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built
import torch
import torch.nn as nn

from src.nn.architectures.tpp_architecture import TPPArchitecture


# ---------------------------------------------------------------------------
# Stub subclass
# ---------------------------------------------------------------------------


class _StubTPP(TPPArchitecture):
    """Minimal concrete subclass for testing base-class plumbing."""

    @staticmethod
    def filter_patho_seqs(tensor1, lens, tensor2=None):
        # Actually filter pathological sequences (len <= 1) to match real behavior.
        mask = lens > 1
        tensor1 = tensor1[mask]
        lens = lens[mask]
        if tensor2 is not None:
            tensor2 = tensor2[mask]
        return tensor1, lens, tensor2

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def sample(self, *, num_seq=None, starting_times=None, log_inter_arr_times=None, marks=None):
        pass


def _make_stub(num_marks=1, time_max=10.0):
    obj = object.__new__(_StubTPP)
    nn.Module.__init__(obj)
    object.__setattr__(obj, "_device", torch.device("cpu"))
    obj.time_max = time_max

    class IdentityScaler:
        def unscale(self, x):
            return x

        def __call__(self, x):
            return x

    obj.scaler_exp = IdentityScaler()
    obj.num_marks = num_marks
    obj.use_marks = num_marks > 1
    obj.train_marks = None
    return obj


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
N, L, D = 4, 8, 1


# ===================================================================
# Test: unconditional marked sample returns gen_marks (N, L_full)
# ===================================================================


class TestUnconditionalMarkedSampleShape:

    def test_score_unconditional_returns_gen_marks(self):
        """Architecture_DDPM unconditional sample returns gen_marks aligned with times."""
        from src.nn.architectures.architecture_ddpm import Architecture_DDPM

        obj = object.__new__(Architecture_DDPM)
        nn.Module.__init__(obj)
        object.__setattr__(obj, "_device", torch.device("cpu"))
        obj.num_marks = 3
        obj.use_marks = True
        obj.num_dim_seqs = D

        # Real relationship: full_data_train_dts[:, 1:] == data_train_dts
        L_full = 8
        obj.full_data_train_dts = torch.randn(10, L_full, D)
        obj.data_train_dts = torch.randn(10, L_full - 1, D)

        obj.anchor_times_sampler = MagicMock()
        obj.anchor_times_sampler.sample = MagicMock(return_value=torch.zeros(N, 1, D))
        obj.first_value_ts_sampler = MagicMock()
        obj.first_value_ts_sampler.sample = MagicMock(
            return_value=(torch.zeros(N, 1, D), torch.zeros(N, dtype=torch.long))
        )
        obj.train_marks = torch.zeros(10, L_full + 1, dtype=torch.long)
        obj.enc_rnn = MagicMock()
        obj.enc_rnn.hidden_size = 16

        def fake_all_step(samples, starting_times, latent_rep, gen_marks):
            return samples, latent_rep, gen_marks

        obj._all_step_scorenet_all_iter_with_marks = fake_all_step

        samples, history, gen_marks = obj.sample(num_seq=N)

        assert gen_marks is not None, "Marked unconditional sample must return gen_marks"
        # gen_marks must be aligned with samples: same (N, L_full) shape
        assert gen_marks.shape == (N, L_full), f"Expected ({N}, {L_full}), got {gen_marks.shape}"
        assert gen_marks.shape[0] == samples.shape[0]
        assert gen_marks.shape[1] == samples.shape[1]
        assert gen_marks.dtype == torch.long
        # First mark should be seeded (not -1)
        assert (gen_marks[:, 0] == 0).all()

    def test_vae_unconditional_returns_gen_marks(self):
        """Architecture_VAE unconditional sample returns gen_marks aligned with times."""
        from src.nn.architectures.architecture_vae import Architecture_VAE

        obj = object.__new__(Architecture_VAE)
        nn.Module.__init__(obj)
        object.__setattr__(obj, "_device", torch.device("cpu"))
        obj.num_marks = 3
        obj.use_marks = True
        obj.num_dim_seqs = D
        obj.MIN_SCALED_DATA = torch.tensor(-5.0)
        obj.MAX_SCALED_DATA = torch.tensor(5.0)

        L_train = 7
        obj.data_train_dts = torch.randn(10, L_train, D)
        obj.train_marks = torch.zeros(10, L_train + 1, dtype=torch.long)
        L_full = L_train + 1  # VAE uses L_train + 1

        obj.first_value_ts_sampler = MagicMock()
        obj.first_value_ts_sampler.sample = MagicMock(
            return_value=(torch.zeros(N, 1, D), torch.zeros(N, dtype=torch.long))
        )

        # Mock encoder/decoder for the autoregressive loop
        H = 16
        obj.enc_rnn = MagicMock()
        obj.enc_rnn.hidden_size = H
        obj.enc_rnn.get_first_hidden_state = MagicMock(return_value=(torch.zeros(1, N, H), torch.zeros(1, N, H)))
        obj.enc_rnn.return_value = (
            torch.zeros(N, 1, H),
            (torch.zeros(1, N, H), torch.zeros(1, N, H)),
        )
        obj.event_emb = MagicMock(return_value=torch.randn(N, 1, 64))
        obj.vae_decoder = MagicMock()
        obj.vae_decoder.sample = MagicMock(return_value=torch.randn(N, 1, 1))
        obj.mark_predictor = MagicMock(return_value=torch.randn(N, 1, 3))

        samples, history, gen_marks = obj.sample(num_seq=N)

        assert gen_marks is not None, "Marked unconditional VAE sample must return gen_marks"
        assert gen_marks.shape == (N, L_full), f"Expected ({N}, {L_full}), got {gen_marks.shape}"
        assert gen_marks.shape[1] == samples.shape[1]
        assert gen_marks.dtype == torch.long
        assert (gen_marks[:, 0] == 0).all(), "First mark must be seeded"

    def test_sigwgan_generator_uses_seeded_first_mark_for_first_step(self):
        """RNNSamplingGeneratorTPP must condition the first recurrent step on gen_marks[:, 0]."""
        from src.nn.nn.sigwgan_modules.rnn_sampling_generator_tpp import RNNSamplingGeneratorTPP

        generator = object.__new__(RNNSamplingGeneratorTPP)
        nn.Module.__init__(generator)

        N_test, L_full, D_test, H = 2, 4, 1, 3
        first_marks = torch.tensor([2, 1], dtype=torch.long)
        latent_step = torch.zeros(N_test, 1, H)
        hidden_state = (torch.zeros(1, N_test, H), torch.zeros(1, N_test, H))

        captured = {}

        def fake_event_emb(times, marks):
            captured.setdefault("marks", []).append(marks.detach().clone())
            return torch.zeros(N_test, 1, 5)

        generator.event_emb = MagicMock(side_effect=fake_event_emb)
        generator.mark_predictor = MagicMock(return_value=torch.tensor([[[0.0, 10.0, 0.0]], [[0.0, 0.0, 10.0]]]))
        generator.recurrent_unit = MagicMock()
        generator.recurrent_unit.get_first_hidden_state = MagicMock(return_value=hidden_state)
        generator.recurrent_unit.return_value = (latent_step, hidden_state)
        generator.decoder = MagicMock(return_value=torch.zeros(N_test, 1))
        generator.time_emb = MagicMock()

        scaling_output = SimpleNamespace(unscale=lambda x: x)
        scaling_cumsum = lambda x: x

        sequences = torch.zeros(N_test, L_full, D_test)
        latent_rep_history = torch.zeros(N_test, L_full - 1, H)
        gen_marks = torch.full((N_test, L_full), -1, dtype=torch.long)
        gen_marks[:, 0] = first_marks

        generator.generate(
            starting_time_sequences=torch.zeros(N_test, 1, D_test),
            initial_intertimes=torch.zeros(N_test, 1, D_test),
            sequences=sequences,
            latent_rep_history=latent_rep_history,
            min_value=torch.tensor(-5.0),
            max_value=torch.tensor(5.0),
            scaling_output=scaling_output,
            scaling_cumsum=scaling_cumsum,
            gen_marks=gen_marks,
        )

        assert "marks" in captured
        assert torch.equal(captured["marks"][0].squeeze(1), first_marks)

    def test_sigwgan_generator_rejects_marks_and_gen_marks_together(self):
        """Conditional and unconditional mark inputs should remain mutually exclusive."""
        from src.nn.nn.sigwgan_modules.rnn_sampling_generator_tpp import RNNSamplingGeneratorTPP

        generator = object.__new__(RNNSamplingGeneratorTPP)
        nn.Module.__init__(generator)

        generator.event_emb = MagicMock()
        generator.mark_predictor = MagicMock()
        generator.recurrent_unit = MagicMock()
        generator.recurrent_unit.get_first_hidden_state = MagicMock(
            return_value=(torch.zeros(1, N, 3), torch.zeros(1, N, 3))
        )
        generator.decoder = MagicMock()
        generator.time_emb = MagicMock()

        with pytest.raises(
            AssertionError, match="either conditional history marks or unconditional generated-mark storage"
        ):
            generator.generate(
                starting_time_sequences=torch.zeros(N, 1, D),
                initial_intertimes=torch.zeros(N, 1, D),
                sequences=torch.zeros(N, L, D),
                latent_rep_history=torch.zeros(N, L - 1, 3),
                min_value=torch.tensor(-5.0),
                max_value=torch.tensor(5.0),
                scaling_output=SimpleNamespace(unscale=lambda x: x),
                scaling_cumsum=lambda x: x,
                marks=torch.zeros(N, L, dtype=torch.long),
                gen_marks=torch.zeros(N, L, dtype=torch.long),
            )


# ===================================================================
# Test: non-mark and baseline architectures return None
# ===================================================================


class TestBaselineArchitecturesReturnNone:

    def test_deter_returns_none_gen_marks(self):
        from src.nn.architectures.architecture_deter import ArchitectureDeter

        obj = object.__new__(ArchitectureDeter)
        nn.Module.__init__(obj)
        object.__setattr__(obj, "_device", torch.device("cpu"))
        obj.num_dim_seqs = D
        obj.MIN_SCALED_DATA = torch.tensor(-5.0)
        obj.MAX_SCALED_DATA = torch.tensor(5.0)

        fake_gen = MagicMock()
        fake_gen.generate = MagicMock(return_value=(torch.randn(N, L, D), torch.randn(N, L - 1, 16), None))
        obj.generator = fake_gen
        obj.data_train_dts = torch.randn(10, L - 1, D)
        obj.hid_size_rep = 16
        obj.scaler_exp = MagicMock()
        obj.scaler_exp.unscale = lambda x: x
        obj.scaler_cumsum_value_for_generator = lambda x: x
        obj.anchor_times_sampler = MagicMock()
        obj.anchor_times_sampler.sample = MagicMock(return_value=torch.zeros(N, 1, D))
        obj.first_value_ts_sampler = MagicMock()
        obj.first_value_ts_sampler.sample = MagicMock(return_value=torch.zeros(N, 1, D))

        _, _, gen_marks = obj.sample(num_seq=N)
        assert gen_marks is None

    def test_gamma_returns_none_gen_marks(self):
        from src.nn.architectures.architecture_gamma import ArchitectureGamma

        obj = object.__new__(ArchitectureGamma)
        nn.Module.__init__(obj)
        object.__setattr__(obj, "_device", torch.device("cpu"))
        # Gamma needs log_k, log_theta, scaling_factor as nn.Parameters
        obj.log_k = nn.Parameter(torch.tensor(0.5))
        obj.log_theta = nn.Parameter(torch.tensor(0.0))
        obj.scaling_factor = nn.Parameter(torch.tensor(1.0))
        obj.data_train_dts = torch.randn(10, L - 1, D)
        obj.num_dim_seqs = D

        _, _, gen_marks = obj.sample(num_seq=N)
        assert gen_marks is None

# ===================================================================
# Test: conditional sample returns None for gen_marks
# ===================================================================


class TestConditionalReturnsNoneGenMarks:

    def test_score_conditional_returns_none(self):
        from src.nn.architectures.architecture_ddpm import Architecture_DDPM

        obj = object.__new__(Architecture_DDPM)
        nn.Module.__init__(obj)
        object.__setattr__(obj, "_device", torch.device("cpu"))
        obj.num_marks = 3
        obj.use_marks = True
        obj.all_step_scorenet_one_iter = MagicMock(return_value=(torch.randn(N, L, D), torch.randn(N, L - 1, 16)))

        starting = torch.zeros(N, 1, D)
        log_its = torch.randn(N, L, D)
        marks = torch.zeros(N, L, dtype=torch.long)
        _, _, gen_marks = obj.sample(starting_times=starting, log_inter_arr_times=log_its, marks=marks)
        assert gen_marks is None

    def test_vae_conditional_returns_none(self):
        from src.nn.architectures.architecture_vae import Architecture_VAE

        obj = object.__new__(Architecture_VAE)
        nn.Module.__init__(obj)
        object.__setattr__(obj, "_device", torch.device("cpu"))
        obj._sample_conditional = MagicMock(return_value=(torch.randn(N, L, D), torch.randn(N, L, 16)))
        obj.num_marks = 3
        obj.use_marks = True

        starting = torch.zeros(N, 1, D)
        log_its = torch.randn(N, L, D)
        _, _, gen_marks = obj.sample(starting_times=starting, log_inter_arr_times=log_its)
        assert gen_marks is None


# ===================================================================
# Test: sample_and_fix_seqs strips first mark in lockstep with τ₁
# ===================================================================


class TestSampleAndFixSeqsMarkStripping:

    def test_gen_marks_stripped_in_lockstep_with_tau1(self):
        obj = _make_stub(num_marks=3)
        N_test, L_test = 3, 6

        dt = obj.time_max / (L_test + 1)
        fake_samples = torch.full((N_test, L_test, D), dt)
        fake_history = torch.randn(N_test, L_test - 1, 8)
        fake_marks = torch.arange(L_test).unsqueeze(0).expand(N_test, -1).clone()  # [0,1,2,3,4,5]

        obj.sample = MagicMock(return_value=(fake_samples, fake_history, fake_marks))

        result = obj.sample_and_fix_seqs(num_seq=N_test)

        assert result.gen_marks is not None
        assert result.gen_marks.shape[1] == fake_marks.shape[1] - 1
        # First mark in result should be mark "1" (originally at position 1)
        assert (result.gen_marks[:, 0] == 1).all()

    def test_gen_marks_none_when_sample_returns_none(self):
        obj = _make_stub(num_marks=1)
        N_test, L_test = 2, 5

        dt = obj.time_max / (L_test + 1)
        fake_samples = torch.full((N_test, L_test, D), dt)
        fake_history = torch.randn(N_test, L_test - 1, 8)
        obj.sample = MagicMock(return_value=(fake_samples, fake_history, None))

        result = obj.sample_and_fix_seqs(num_seq=N_test)
        assert result.gen_marks is None

    def test_gen_marks_tail_is_padded_with_minus_one_after_truncation(self):
        obj = _make_stub(num_marks=3)
        fake_samples = torch.tensor(
            [
                [[1.0], [1.0], [1.0], [20.0], [20.0], [20.0]],
                [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
            ]
        )
        fake_history = torch.randn(2, 5, 8)
        fake_marks = torch.tensor(
            [
                [0, 1, 2, 2, 2, 2],
                [0, 1, 2, 0, 1, 2],
            ],
            dtype=torch.long,
        )

        obj.sample = MagicMock(return_value=(fake_samples, fake_history, fake_marks))

        result = obj.sample_and_fix_seqs(num_seq=2)

        assert result.gen_marks is not None
        # First sequence has valid post-tau1 length 2, so positions >=2 must be padded with -1.
        assert result.seq_lens[0].item() == 2
        assert torch.equal(result.gen_marks[0], torch.tensor([1, 2, -1, -1, -1]))
        # Second sequence remains fully valid after stripping tau1, so no -1 padding is introduced.
        assert (result.gen_marks[1] >= 0).all()


# ===================================================================
# Test: pathological-sequence filtering keeps gen_marks aligned
# ===================================================================


class TestPathoFilteringAlignedWithMarks:

    def test_filtering_removes_mark_rows_in_lockstep(self):
        """When some sequences are pathological (len<=1), their marks must be removed too."""
        obj = _make_stub(num_marks=3)
        N_test, L_test = 5, 6

        dt = obj.time_max / (L_test + 1)
        fake_samples = torch.full((N_test, L_test, D), dt)
        # Make sequences 1 and 3 have huge inter-arrival times so they
        # hit time_max after 1 step → to_cst_val_gr gives len=1 → pathological.
        fake_samples[1, :, :] = obj.time_max * 2
        fake_samples[3, :, :] = obj.time_max * 2

        fake_history = torch.randn(N_test, L_test - 1, 8)
        fake_marks = torch.tensor(
            [
                [0, 1, 2, 0, 1, 2],
                [10, 11, 12, 13, 14, 15],  # pathological
                [20, 21, 22, 23, 24, 25],
                [30, 31, 32, 33, 34, 35],  # pathological
                [40, 41, 42, 43, 44, 45],
            ],
            dtype=torch.long,
        )

        obj.sample = MagicMock(return_value=(fake_samples, fake_history, fake_marks))

        result = obj.sample_and_fix_seqs(num_seq=N_test)

        # 3 survivors (sequences 0, 2, 4)
        assert result.gen_marks is not None
        n_survivors = result.its_scaled_cst.shape[0]
        assert result.gen_marks.shape[0] == n_survivors, (
            f"gen_marks rows ({result.gen_marks.shape[0]}) must match " f"surviving sequences ({n_survivors})"
        )
        # After stripping τ₁, first column should be position 1 of the original
        assert result.gen_marks[0, 0].item() == 1  # seq 0, pos 1
        assert result.gen_marks[1, 0].item() == 21  # seq 2, pos 1
        assert result.gen_marks[2, 0].item() == 41  # seq 4, pos 1


# ===================================================================
# Test: forward(include_first_it) preserves mark alignment
# ===================================================================


class TestForwardMarkAlignment:

    def _setup_stub_with_marks(self):
        obj = _make_stub(num_marks=3)
        N_test, L_test = 3, 6

        dt = obj.time_max / (L_test + 1)
        fake_samples = torch.full((N_test, L_test, D), dt)
        fake_history = torch.randn(N_test, L_test - 1, 8)
        fake_marks = torch.arange(L_test).unsqueeze(0).expand(N_test, -1).clone()

        obj.sample = MagicMock(return_value=(fake_samples, fake_history, fake_marks))
        return obj, N_test, L_test

    def test_include_first_it_true_keeps_all_marks(self):
        obj, N_test, L_test = self._setup_stub_with_marks()
        samples, lens, gen_marks = obj.forward(N_test, include_first_it=True)

        assert gen_marks is not None
        assert gen_marks.shape[0] == samples.shape[0]
        assert gen_marks.shape[1] == samples.shape[1]

    def test_include_first_it_false_strips_first_mark(self):
        obj, N_test, L_test = self._setup_stub_with_marks()
        samples, lens, gen_marks = obj.forward(N_test, include_first_it=False)

        assert gen_marks is not None
        assert gen_marks.shape[0] == samples.shape[0]
        assert gen_marks.shape[1] == samples.shape[1]
        # First mark in result is originally position 1
        assert (gen_marks[:, 0] == 1).all()

    def test_include_first_it_true_pads_tail_marks_with_minus_one(self):
        obj = _make_stub(num_marks=3)
        fake_samples = torch.tensor(
            [
                [[1.0], [1.0], [1.0], [20.0], [20.0], [20.0]],
                [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
            ]
        )
        fake_history = torch.randn(2, 5, 8)
        fake_marks = torch.tensor(
            [
                [0, 1, 2, 2, 2, 2],
                [0, 1, 2, 0, 1, 2],
            ],
            dtype=torch.long,
        )
        obj.sample = MagicMock(return_value=(fake_samples, fake_history, fake_marks))

        samples, lens, gen_marks = obj.forward(2, include_first_it=True)

        assert gen_marks is not None
        assert lens[0].item() == 3
        assert torch.equal(gen_marks[0], torch.tensor([0, 1, 2, -1, -1, -1]))
        assert (gen_marks[1] >= 0).all()

    def test_include_first_it_false_pads_tail_marks_with_minus_one(self):
        obj = _make_stub(num_marks=3)
        fake_samples = torch.tensor(
            [
                [[1.0], [1.0], [1.0], [20.0], [20.0], [20.0]],
                [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
            ]
        )
        fake_history = torch.randn(2, 5, 8)
        fake_marks = torch.tensor(
            [
                [0, 1, 2, 2, 2, 2],
                [0, 1, 2, 0, 1, 2],
            ],
            dtype=torch.long,
        )
        obj.sample = MagicMock(return_value=(fake_samples, fake_history, fake_marks))

        samples, lens, gen_marks = obj.forward(2, include_first_it=False)

        assert gen_marks is not None
        assert lens[0].item() == 2
        assert torch.equal(gen_marks[0], torch.tensor([1, 2, -1, -1, -1]))
        assert (gen_marks[1] >= 0).all()


# ===================================================================
# Test: _sample_first_event co-samples valid marks
# ===================================================================


class TestSampleFirstEvent:

    def test_co_samples_mark_from_correct_column(self):
        obj = _make_stub(num_marks=3)
        N_test = 5
        # train_marks: column 0 is anchor, column 1 is first real event mark
        obj.train_marks = torch.tensor(
            [
                [99, 0, 1, 2],
                [99, 1, 2, 0],
                [99, 2, 0, 1],
            ],
            dtype=torch.long,
        )

        indices = torch.tensor([0, 1, 2, 0, 2])
        obj.first_value_ts_sampler = MagicMock()
        obj.first_value_ts_sampler.sample = MagicMock(return_value=(torch.zeros(N_test, 1, D), indices))

        first_it, first_mark = obj._sample_first_event(N_test)

        assert first_mark.shape == (N_test,)
        expected = obj.train_marks[:, 1][indices]
        assert torch.equal(first_mark, expected)
        # Should never be 99 (anchor column)
        assert (first_mark != 99).all()


# ===================================================================
# Test: tail-corruption verification for sample_and_fix_seqs
# ===================================================================


class TestTailPaddingContract:
    """Deterministic tail-corruption test per docs/marks/04-05-01_BUGF_mark_padding_verification.md."""

    def test_sample_and_fix_seqs_freezes_corrupt_tail(self):
        """Sentinel values in the tail must not survive post-processing."""
        torch.manual_seed(0)
        obj = _make_stub(num_marks=3, time_max=10.0)
        L_full = 8
        SENTINEL_DT = 777.0
        SENTINEL_MARK = 99

        # Seq 0: no padding — all ITs small, cumsum stays < time_max.
        # Seq 1: moderate padding — first 3 ITs valid, rest sentinel.
        # Seq 2: heavy padding — first 1 IT valid, rest sentinel.
        fake_samples = torch.full((3, L_full, D), SENTINEL_DT)
        dt_small = 1.0  # cumsum of 8 * 1.0 = 8.0 < 10.0
        fake_samples[0, :, :] = dt_small
        fake_samples[1, :3, :] = dt_small
        fake_samples[2, :1, :] = dt_small

        fake_history = torch.randn(3, L_full - 1, 8)
        fake_marks = torch.full((3, L_full), SENTINEL_MARK, dtype=torch.long)
        fake_marks[0, :] = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        fake_marks[1, :3] = torch.tensor([0, 1, 2])
        fake_marks[2, :1] = torch.tensor([0])

        obj.sample = MagicMock(return_value=(fake_samples, fake_history, fake_marks))
        result = obj.sample_and_fix_seqs(num_seq=3)

        # Shape agreement.
        assert result.gen_marks.shape[0] == result.its_scaled_cst.shape[0]
        assert result.gen_marks.shape[1] == result.its_scaled_cst.shape[1]

        for n in range(result.seq_lens.shape[0]):
            slen = result.seq_lens[n].item()
            tail_start = slen

            # NaN-masked times: tail must be all NaN.
            assert torch.isnan(
                result.its_scaled_nan[n, tail_start:]
            ).all(), f"Seq {n}: its_scaled_nan tail not NaN from position {tail_start}"
            assert torch.isnan(
                result.cum_rel_nan[n, tail_start:]
            ).all(), f"Seq {n}: cum_rel_nan tail not NaN from position {tail_start}"

            # Marks: tail must be -1, valid prefix must not be -1.
            if tail_start < result.gen_marks.shape[1]:
                assert (
                    result.gen_marks[n, tail_start:] == -1
                ).all(), f"Seq {n}: gen_marks tail not -1 from position {tail_start}"
            if slen > 0:
                assert (result.gen_marks[n, :slen] >= 0).all(), f"Seq {n}: gen_marks valid prefix contains -1"

        # Global: sentinel mark 99 must not appear anywhere in gen_marks.
        assert (result.gen_marks != SENTINEL_MARK).all(), "Sentinel mark 99 survived post-processing"

    def test_runtime_assertion_catches_bad_marks(self):
        """_check_tail_contract raises when marks tail is not -1."""
        from src.nn.architectures.tpp_architecture import TPPArchitecture

        its_nan = torch.tensor([[[float("nan")], [float("nan")]]])
        cum_nan = torch.tensor([[[float("nan")], [float("nan")]]])
        bad_marks = torch.tensor([[5, 7]])  # should be -1
        seq_lens = torch.tensor([0])

        with pytest.raises(AssertionError, match="gen_marks"):
            TPPArchitecture._check_tail_contract(its_nan, cum_nan, bad_marks, seq_lens)

    def test_runtime_assertion_passes_correct_data(self):
        """_check_tail_contract returns True for correctly padded data."""
        from src.nn.architectures.tpp_architecture import TPPArchitecture

        its_nan = torch.tensor([[[1.0], [float("nan")]]])
        cum_nan = torch.tensor([[[2.0], [float("nan")]]])
        marks = torch.tensor([[1, -1]])
        seq_lens = torch.tensor([1])

        assert TPPArchitecture._check_tail_contract(its_nan, cum_nan, marks, seq_lens)


# ===================================================================
# Test: _sample_first_event co-samples valid marks (continued)
# ===================================================================


class TestSampleFirstEventEdgeCases:

    def test_returns_zeros_when_no_marks(self):
        obj = _make_stub(num_marks=1)
        N_test = 4
        obj.train_marks = None
        obj.first_value_ts_sampler = MagicMock()
        obj.first_value_ts_sampler.sample = MagicMock(
            return_value=(torch.zeros(N_test, 1, D), torch.zeros(N_test, dtype=torch.long))
        )

        first_it, first_mark = obj._sample_first_event(N_test)
        assert first_mark.shape == (N_test,)
        assert (first_mark == 0).all()

    def test_returns_zeros_when_marks_too_short(self):
        """If train_marks has only 1 column (anchor only), fall back to zeros."""
        obj = _make_stub(num_marks=3)
        N_test = 3
        obj.train_marks = torch.zeros(5, 1, dtype=torch.long)
        obj.first_value_ts_sampler = MagicMock()
        obj.first_value_ts_sampler.sample = MagicMock(
            return_value=(torch.zeros(N_test, 1, D), torch.zeros(N_test, dtype=torch.long))
        )

        first_it, first_mark = obj._sample_first_event(N_test)
        assert (first_mark == 0).all()
