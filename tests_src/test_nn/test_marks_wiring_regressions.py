import pytest

pytest.importorskip("signatory")

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from src.nn.architectures.mark_prediction_utils import prepare_next_mark_targets
from src.nn.architectures.architecture_ddpm import Architecture_DDPM
from src.nn.architectures.architecture_types import Architectures
from src.nn.architectures.architecture_one_to_one import ArchitectureOneToOne
from src.nn.architectures.architecture_vae import Architecture_VAE
from src.nn.architectures.architecture_wgan_baseline import Architecture_wgan_baseline
import test.paper_experiments.settings.earthquake as earthquake_settings
import test.paper_experiments.settings.poisson as poisson_settings
import test.paper_experiments.settings.stackoverflow as stackoverflow_settings
import test.paper_experiments.settings.taobao as taobao_settings


def _make_module_stub(module_cls):
    obj = object.__new__(module_cls)
    nn.Module.__init__(obj)
    object.__setattr__(obj, "_device", torch.device("cpu"))
    return obj


def _make_fake_marked_data():
    zeros = torch.zeros(2, 4, 1)
    mark_zeros = torch.zeros(2, 5, dtype=torch.long)
    lens = torch.tensor([4, 3], dtype=torch.long)
    return SimpleNamespace(
        train_in=zeros,
        train_in_len=lens,
        val_in=zeros,
        val_in_len=lens,
        time_max=1.0,
        num_marks=3,
        train_marks=mark_zeros,
        val_marks=mark_zeros,
    )


def test_score_prepare_mark_targets_masks_and_aligns():
    marks = torch.tensor(
        [
            [1, 2, 0, 1],
            [2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([4, 2], dtype=torch.long)

    targets = prepare_next_mark_targets(marks, lengths)

    expected = torch.tensor(
        [
            [2, 0, 1],
            [1, -1, -1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(targets, expected)


def test_score_embed_history_uses_event_embeddings_for_marked_model():
    obj = _make_module_stub(Architecture_DDPM)
    obj.num_marks = 3
    obj.use_marks = True
    obj.time_emb = MagicMock()
    obj.event_emb = MagicMock(return_value=torch.randn(2, 4, 5))
    log_inter_arr_times = torch.randn(2, 4, 1)
    marks = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)

    result = obj._embed_history(log_inter_arr_times, marks)

    obj.event_emb.assert_called_once_with(log_inter_arr_times, marks)
    obj.time_emb.assert_not_called()
    assert result.shape == (2, 4, 5)


def test_score_sample_conditional_forwards_marks():
    obj = _make_module_stub(Architecture_DDPM)
    obj.all_step_scorenet_one_iter = MagicMock(return_value=("samples", "history"))
    starting_times = torch.zeros(2, 1, 1)
    log_inter_arr_times = torch.randn(2, 4, 1)
    marks = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)

    result = obj.sample(starting_times=starting_times, log_inter_arr_times=log_inter_arr_times, marks=marks)

    assert result == ("samples", "history", None)
    obj.all_step_scorenet_one_iter.assert_called_once_with(starting_times, log_inter_arr_times, marks=marks)


def test_vae_sample_conditional_forwards_marks():
    obj = _make_module_stub(Architecture_VAE)
    obj._sample_conditional = MagicMock(return_value=("samples", "history"))
    starting_times = torch.zeros(2, 1, 1)
    log_inter_arr_times = torch.randn(2, 4, 1)
    marks = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)

    result = obj.sample(starting_times=starting_times, log_inter_arr_times=log_inter_arr_times, marks=marks)

    assert result == ("samples", "history", None)
    obj._sample_conditional.assert_called_once_with(starting_times, log_inter_arr_times, marks=marks)


def test_vae_mark_logits_encode_history_without_sampling():
    obj = _make_module_stub(Architecture_VAE)
    obj.use_marks = True
    obj.scaler_exp = MagicMock(side_effect=lambda x: x)
    h_all = torch.randn(2, 4, 5)
    obj._encode_history = MagicMock(return_value=h_all)
    obj.mark_predictor = MagicMock(return_value=torch.randn(2, 3, 3))
    obj.sample = MagicMock(side_effect=AssertionError("sample() should not be used for mark logits"))

    dts = torch.randn(2, 4, 1)
    marks_full = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)

    logits = obj._compute_mark_logits(
        marks_with_anchor=torch.cat([torch.zeros(2, 1, dtype=torch.long), marks_full], dim=1),
        marks_full=marks_full,
        dts=dts,
        dt_lens=torch.tensor([4, 4], dtype=torch.long),
        current_targets=torch.tensor([[1, 2, 0], [2, 0, 1]], dtype=torch.long),
    )

    obj._encode_history.assert_called_once_with(dts, marks_full)
    obj.mark_predictor.assert_called_once()
    (pred_args, _) = obj.mark_predictor.call_args
    assert torch.equal(pred_args[0], h_all[:, :-1, :])
    assert logits.shape == (2, 3, 3)


def test_sigwgan_mark_logits_encode_history_without_sampling():
    obj = _make_module_stub(ArchitectureOneToOne)
    obj.use_marks = True
    obj.scaler_exp = MagicMock(side_effect=lambda x: x)
    obj.generator = MagicMock()
    latent_rep_history = torch.randn(2, 3, 5)
    obj.generator.encode_history = MagicMock(return_value=latent_rep_history)
    obj.mark_predictor = MagicMock(return_value=torch.randn(2, 3, 3))
    obj.sample = MagicMock(side_effect=AssertionError("sample() should not be used for mark logits"))

    dts = torch.randn(2, 4, 1)
    marks_full = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)

    logits = obj._compute_mark_logits(
        marks_with_anchor=torch.cat([torch.zeros(2, 1, dtype=torch.long), marks_full], dim=1),
        marks_full=marks_full,
        dts=dts,
        dt_lens=torch.tensor([4, 4], dtype=torch.long),
        current_targets=torch.tensor([[1, 2, 0], [2, 0, 1]], dtype=torch.long),
    )

    encode_args, _ = obj.generator.encode_history.call_args
    assert encode_args[0].shape == (2, 3, 1)
    assert torch.equal(encode_args[1], marks_full[:, :-1])
    obj.mark_predictor.assert_called_once_with(latent_rep_history)
    assert logits.shape == (2, 3, 3)


def test_wgan_mark_logits_encode_history_without_sampling():
    obj = _make_module_stub(Architecture_wgan_baseline)
    obj.use_marks = True
    obj.scaler_exp = MagicMock(side_effect=lambda x: x)
    latent_rep_history = torch.randn(2, 3, 5)
    obj._encode_history = MagicMock(return_value=latent_rep_history)
    obj.mark_predictor = MagicMock(return_value=torch.randn(2, 3, 3))
    obj.sample = MagicMock(side_effect=AssertionError("sample() should not be used for mark logits"))

    dts = torch.randn(2, 4, 1)
    marks_full = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)

    logits = obj._compute_mark_logits(
        marks_with_anchor=torch.cat([torch.zeros(2, 1, dtype=torch.long), marks_full], dim=1),
        marks_full=marks_full,
        dts=dts,
        dt_lens=torch.tensor([4, 4], dtype=torch.long),
        current_targets=torch.tensor([[1, 2, 0], [2, 0, 1]], dtype=torch.long),
    )

    obj._encode_history.assert_called_once()
    obj.mark_predictor.assert_called_once_with(latent_rep_history)
    assert logits.shape == (2, 3, 3)


def test_learnable_mark_metrics_include_top3_when_available():
    metrics = Architectures.get_metrics(Architectures.DDPM, num_marks=3)

    assert "train_mark_ce" in metrics
    assert "val_mark_ce" in metrics
    assert "val_top1_mark_acc" in metrics
    assert "val_top3_mark_acc" in metrics


def test_marked_wgan_train_lip_loss_excludes_mark_ce(monkeypatch):
    obj = _make_module_stub(Architecture_wgan_baseline)
    obj.use_marks = True
    obj.mark_loss_weight = 1.0
    obj.optimizers = MagicMock(return_value=(MagicMock(), MagicMock()))
    obj.discriminator = MagicMock()
    obj.scaler_exp = SimpleNamespace(unscale=lambda x: x)

    dts_scaled = torch.tensor([[[0.1], [0.2], [0.3]]], dtype=torch.float32)
    lengths = torch.tensor([4], dtype=torch.long)
    ce_loss = torch.tensor(3.0)
    gen_loss = torch.tensor(2.0)  # pure generator loss, excluding mark CE
    disc_loss = torch.tensor(11.0)

    obj._training_step_gen = MagicMock(return_value=(gen_loss, ce_loss))
    obj._training_step_disc = MagicMock(return_value=disc_loss)
    captured = {}
    obj._log_all_metrics = lambda metrics, prefix: captured.update({"metrics": metrics, "prefix": prefix})

    monkeypatch.setattr(
        "src.nn.architectures.architecture_wgan_baseline.tpp_utils.cum_times_to_log_inter_times",
        lambda batch, scaler: (lengths, dts_scaled),
    )

    batch = (
        torch.tensor([[[0.0], [0.1], [0.3], [0.6]]], dtype=torch.float32),
        lengths,
        torch.tensor([[0, 1, 2, 1]], dtype=torch.long),
    )

    obj.training_step(batch, batch_nb=0)

    assert captured["prefix"] == "train_"
    assert torch.isclose(captured["metrics"]["wasserstein"], torch.tensor(2.0))
    assert torch.isclose(captured["metrics"]["lip_loss"], torch.tensor(13.0))
    assert torch.isclose(captured["metrics"]["mark_ce"], ce_loss)


def test_validation_mark_diagnostics_save_required_plot_files(monkeypatch, tmp_path):
    obj = _make_module_stub(Architecture_DDPM)
    obj.use_marks = True
    obj.output_dir = str(tmp_path).replace("\\", "/") + "/"
    obj.val_marks = torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 2, 0, 1],
        ],
        dtype=torch.long,
    )
    obj.full_data_val_dts = torch.zeros(2, 3, 1, dtype=torch.float32)
    obj.full_data_val_dt_lens = torch.tensor([3, 2], dtype=torch.long)
    obj.hist_fig, obj.hist_ax = plt.subplots(2, 2)
    obj.acf_fig, obj.acf_ax = plt.subplots(1, 2)
    obj.intensity_fig, obj.intensity_ax = plt.subplots(1, 2)
    obj.cov_err_fig, obj.cov_err_ax = plt.subplots()
    obj.temporal_plot_fig, obj.temporal_plot_ax = plt.subplots()
    obj.mark_marginal_fig, obj.mark_marginal_ax = plt.subplots()
    obj.mark_conditional_fig, obj.mark_conditional_axes = plt.subplots(1, 2, sharey=True)
    obj.plot_diffusion_fig, obj.plot_diffusion_axes = plt.subplots(2, 2)
    obj._compute_mark_logits = MagicMock(
        return_value=torch.tensor(
            [
                [[4.0, 1.0, 0.0], [0.0, 5.0, 1.0]],
                [[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
    )

    saved_paths = []

    def fake_savefig(_fig, path):
        saved_paths.append(path)

    monkeypatch.setattr("src.nn.architectures.tpp_architecture.savefig", fake_savefig)

    obj._plot_validation_mark_diagnostics()
    obj._save_eval_plots("5", include_mark_plots=True)
    obj._compute_mark_logits.assert_called_once()

    assert saved_paths[-2:] == [
        f"{obj.output_dir}mk_marg_5.png",
        f"{obj.output_dir}mk_cond_5.png",
    ]

    plt.close(obj.mark_marginal_fig)
    plt.close(obj.mark_conditional_fig)
    plt.close(obj.hist_fig)
    plt.close(obj.acf_fig)
    plt.close(obj.intensity_fig)
    plt.close(obj.cov_err_fig)
    plt.close(obj.temporal_plot_fig)
    plt.close(obj.plot_diffusion_fig)


def test_unmarked_on_test_end_skips_mark_metrics(monkeypatch):
    obj = _make_module_stub(Architecture_DDPM)
    obj.num_marks = 1
    obj.use_marks = False
    obj.output_dir = None
    obj.val_marks = torch.zeros(1, 4, dtype=torch.long)
    obj.full_data_val_dts = torch.zeros(1, 3, 1, dtype=torch.float32)
    obj.full_data_val_dt_lens = torch.tensor([3], dtype=torch.long)
    obj.sample_and_plot = MagicMock()
    obj._compute_mark_logits = MagicMock(return_value=None)
    obj._get_fake_real_samples = MagicMock(
        return_value=(
            torch.zeros(1, 3, 1, dtype=torch.float32),
            torch.tensor([3], dtype=torch.long),
            None,
            None,
            None,
        )
    )
    obj.metrics_test = {}
    obj._test_batch_cache = {
        "data": torch.tensor([[[0.0], [0.5], [1.0], [1.5]]], dtype=torch.float32),
        "data_lens": torch.tensor([4], dtype=torch.long),
        "marks": torch.zeros(1, 4, dtype=torch.long),
    }

    monkeypatch.setattr("src.nn.architectures.tpp_architecture.plt.pause", lambda *_args, **_kwargs: None)

    obj.on_test_end()

    assert "top1_mark_acc" not in obj.metrics_test
    assert "top3_mark_acc" not in obj.metrics_test
    assert "mark_ce" not in obj.metrics_test
    obj._compute_mark_logits.assert_not_called()


def test_missing_use_marks_fails_loudly_in_shared_mark_eval_path():
    obj = _make_module_stub(Architecture_DDPM)
    obj.val_marks = torch.zeros(1, 4, dtype=torch.long)
    obj.full_data_val_dts = torch.zeros(1, 3, 1, dtype=torch.float32)
    obj.full_data_val_dt_lens = torch.tensor([3], dtype=torch.long)

    with pytest.raises(AttributeError):
        obj._build_mark_tensors_for_validation()


def test_mark_ce_is_computed_from_logits_and_targets(monkeypatch):
    """Mark CE and top-1 accuracy are computed from the architecture-supplied logits.

    After the bootstrap refactor, mark CE is computed inside
    `_run_bootstrap_metrics` (via `_compute_mark_metrics_from_inputs`) rather than
    `on_test_end`. This test exercises that unit directly.
    """
    obj = _make_module_stub(Architecture_DDPM)
    obj.num_marks = 3
    obj.use_marks = True
    # marks: [0, 1, 2, 1] → current_targets = marks[:, 2:] = [2, 1]
    data = torch.tensor([[[0.0], [0.5], [1.0], [1.5]]], dtype=torch.float32)
    data_lens = torch.tensor([4], dtype=torch.long)
    marks = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
    # Logits that correctly predict targets [2, 1]: argmax → [2, 1]
    obj._compute_mark_logits = MagicMock(
        return_value=torch.tensor(
            [[[0.0, 1.0, 5.0], [0.0, 5.0, 1.0]]],
            dtype=torch.float32,
        )
    )

    mark_metrics = obj._compute_mark_metrics_from_inputs(
        dts=data.diff(dim=1),
        dt_lens=data_lens - 1,
        marks=marks,
        include_ce=True,
    )

    assert mark_metrics is not None
    assert "mark_ce" in mark_metrics
    assert float(mark_metrics["mark_ce"]) > 0
    assert float(mark_metrics["top1_mark_acc"]) == 1.0


def test_on_test_end_invokes_sample_and_plot(monkeypatch):
    """`on_test_end` still drives the diagnostic plotting/sampling pass."""
    obj = _make_module_stub(Architecture_DDPM)
    obj.num_marks = 3
    obj.use_marks = True
    obj.output_dir = None
    obj.sample_and_plot = MagicMock()
    obj.metrics_test = {}
    obj._get_fake_real_samples = MagicMock(
        return_value=(
            torch.zeros(1, 3, 1, dtype=torch.float32),
            torch.tensor([3], dtype=torch.long),
            None,
            None,
            None,
        )
    )
    obj._test_batch_cache = {
        "data": torch.tensor([[[0.0], [0.5], [1.0], [1.5]]], dtype=torch.float32),
        "data_lens": torch.tensor([4], dtype=torch.long),
        "marks": torch.tensor([[0, 1, 2, 1]], dtype=torch.long),
    }
    monkeypatch.setattr("src.nn.architectures.tpp_architecture.plt.pause", lambda *_args, **_kwargs: None)

    obj.on_test_end()

    obj.sample_and_plot.assert_called_once()


def test_residual_train_metric_forwards_float64_flag(monkeypatch):
    """Regression: the terminal-anchor (e.g. RESIDUAL) training pipeline rebuilds
    sigw1metric_train directly instead of via create_and_get_signature_metrics.
    It must forward use_float64_signature, otherwise an f64-configured run would
    silently train with an f32 signature loss (and disagree with the f64 val metric)."""
    from src.data_types.sigw_loss_data_props import SigWLossDataProps
    from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode

    obj = _make_module_stub(ArchitectureOneToOne)
    obj._terminal_anchor_mode_train = TerminalAnchorMode.RESIDUAL
    obj.train_seq_cap = None  # __init__ sets this; the stub bypasses __init__ so provide it
    obj.scaler_exp = MagicMock()
    obj.time_max = 1.0
    obj.full_data_train_dt_lens = torch.tensor([4, 3], dtype=torch.long)
    obj.sigw_loss_properties = SigWLossDataProps(
        sig_degree=2,
        scale_high_degrees=False,
        standardise_sig=True,
        use_float64_signature=True,
    )

    # Stub the heavy preprocessing so the method reaches the metric construction.
    obj._preprocess_dataset_for_metrics = MagicMock(return_value=(torch.zeros(2, 3, 1), torch.zeros(2, 3, 1), None))
    obj._scale_paths_pre_sig_train = MagicMock(return_value=torch.randn(2, 4, 2))
    obj._compute_approx_errors = MagicMock(return_value=(0.0, 0.0))

    strat = MagicMock()
    strat.append = MagicMock(return_value=torch.zeros(2, 4, 2))
    strat.terminal_anchor_extra_len = MagicMock(return_value=0)
    monkeypatch.setattr(
        "src.nn.architectures.architecture_one_to_one.make_anchor_strategy",
        lambda *a, **k: strat,
    )
    # Patch where the names are looked up (architecture_one_to_one imports them
    # via ``from ... import ...``), not where they are defined, or the patches
    # bind too late and never take effect.
    monkeypatch.setattr(
        "src.nn.architectures.architecture_one_to_one.variable_len_standard_stats",
        lambda *a, **k: (torch.zeros(2), torch.ones(2)),
    )
    monkeypatch.setattr(
        "src.nn.architectures.architecture_one_to_one.StandardScaler",
        lambda **k: MagicMock(side_effect=lambda x: x),
    )
    monkeypatch.setattr("src.nn.architectures.architecture_one_to_one.total_var", lambda x: torch.tensor([1.0]))

    captured = {}

    class FakeMetric:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr("src.nn.architectures.architecture_one_to_one.SigW1MetricExp", FakeMetric)

    obj._setup_training_anchor_pipeline(torch.zeros(2, 4, 1), torch.tensor([4, 3], dtype=torch.long))

    assert captured["kwargs"].get("use_float64_signature") is True


def test_poisson_three_marks_defaults_to_uniform_mark_probs(monkeypatch):
    captured = {}

    class FakePoissonDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(poisson_settings, "PoissonDataModule", FakePoissonDataModule)
    factories = poisson_settings._make_poisson_factories(default_num_marks=3, namer_prefix="hp_three_marks")
    cfg = {
        "seed": 7,
        "parameter_sets": {"batch_size": 32},
    }

    factories["data_factory"](cfg)

    assert np.allclose(captured["mark_probs"], np.array([1 / 3, 1 / 3, 1 / 3]))


@pytest.mark.parametrize(
    ("settings_module", "register_fn_name"),
    [
        (stackoverflow_settings, "register_stackoverflow_factories"),
        (earthquake_settings, "register_earthquake_factories"),
        (taobao_settings, "register_taobao_factories"),
    ],
)
def test_marked_vae_settings_forward_vae_hyperparameters(monkeypatch, settings_module, register_fn_name):
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(settings_module.Architectures, "get_model_class", staticmethod(lambda _: FakeModel))
    factories = getattr(settings_module, register_fn_name)()
    cfg = {
        "version": "vae",
        "epochs": 5,
        "server_training": False,
        "n_bootstraps": 1,
        "parameter_sets": {
            "lr": 1.0e-3,
            "hid_size_rnn": 8,
            "concentration_factor": 1.0,
            "latent_dim": 16,
            "kl_anneal_epochs": 7,
            "free_bits": 0.2,
            "recon_weight": 3.0,
        },
    }

    factories["model_factory"](
        cfg,
        data=_make_fake_marked_data(),
        period_plot_val=1,
        datamodel_path="out",
        logger_custom=None,
        checkpoint=None,
    )

    assert captured["kl_anneal_epochs"] == 7
    assert captured["free_bits"] == 0.2
    assert captured["recon_weight"] == 3.0
