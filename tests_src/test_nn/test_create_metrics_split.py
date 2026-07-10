"""Tests for TPPArchitecture._create_metrics_from_batch split awareness.

The diagnostic metric builder must be usable for both the validation split
(cross-config ranking) and the test split (final reporting). It must:
  1. default to ``DatasetSplitType.TEST`` so existing test/bootstrap callers
     keep their behaviour;
  2. tag the returned ``TPPMetrics`` with the requested split so validation
     diagnostics and test diagnostics never get confused downstream.
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

from unittest.mock import MagicMock

import torch

from src.data_types.tppmetrics import DatasetSplitType, TPPMetricsConfig
from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.metrics.anchors.terminal_anchor_strategy import make_anchor_strategy
from src.nn.architectures.tpp_architecture import TPPArchitecture


class _StubTPP(TPPArchitecture):
    @staticmethod
    def filter_patho_seqs(tensor1, lens_for_masking, tensor2=None):
        return tensor1, lens_for_masking, tensor2

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def sample(self, *, num_seq=None, starting_times=None, log_inter_arr_times=None):
        pass


def _make_stub() -> _StubTPP:
    obj = object.__new__(_StubTPP)
    torch.nn.Module.__init__(obj)
    obj.terminal_anchor_mode = TerminalAnchorMode.FREE_ENDPOINT
    obj.time_max = 20.0
    obj.use_marks = False

    class _Scaler:
        def __call__(self, x):
            return x * 2

        def unscale(self, x):
            return x / 2

    obj.scaler_exp = _Scaler()
    obj._anchor_strategy = make_anchor_strategy(obj.terminal_anchor_mode, scaler_exp=obj.scaler_exp)
    obj._metrics_config = TPPMetricsConfig(time_max=20.0)
    # Identity stand-in for signature path scaling; accepts the optional seq_lens kwarg.
    obj.scale_paths_pre_sig = lambda x, seq_lens=None: x
    # LightningModule.device reads this; object.__new__ skips the LightningModule
    # __init__ that normally sets it.
    obj._device = torch.device("cpu")
    return obj


def _make_data(n=5, l=7, d=1, seed=42):
    torch.manual_seed(seed)
    its = torch.rand(n, l, d) + 0.5
    cum = torch.cat([torch.zeros(n, 1, d), its.cumsum(dim=1)], dim=1)
    lens = torch.randint(3, l + 1, (n,))
    return cum, lens


def test_create_metrics_from_batch_tags_val_split():
    obj = _make_stub()
    data, lens = _make_data()

    metrics = obj._create_metrics_from_batch(data, lens, split=DatasetSplitType.VAL)

    assert metrics.split is DatasetSplitType.VAL


def test_create_metrics_from_batch_defaults_to_test_split():
    obj = _make_stub()
    data, lens = _make_data()

    metrics = obj._create_metrics_from_batch(data, lens)

    assert metrics.split is DatasetSplitType.TEST


def test_evaluate_split_threads_split_and_sets_metrics():
    """evaluate_split tags the bootstrap with the requested split and stores metrics_test."""
    obj = _make_stub()
    obj.sample_and_fix_seqs = MagicMock(return_value=MagicMock())
    obj.sample_for_a_fixed_batch_and_fix = MagicMock(return_value=MagicMock())

    captured = {}

    def fake_bootstrap(*, data, data_lens, marks, uncond_result, cond_result, split):
        captured["split"] = split
        return {"ED_mean": 1.0}

    obj._run_bootstrap_metrics = fake_bootstrap

    data, lens = _make_data()
    out = obj.evaluate_split(data, lens, None, split=DatasetSplitType.VAL)

    assert captured["split"] is DatasetSplitType.VAL
    assert out == {"ED_mean": 1.0}
    # The split it was called with is the one threaded through; no hidden state.
    assert obj.metrics_test == {"ED_mean": 1.0}


def test_evaluate_split_no_grad_runs_eval_mode_no_grad_and_moves_inputs_to_device():
    """evaluate_split_no_grad wraps evaluate_split with eval(), no_grad(), and device moves."""
    obj = _make_stub()
    obj.train()  # start in training mode to verify it switches to eval

    captured = {}

    def fake_evaluate_split(data, data_lens, marks, *, split):
        captured["training"] = obj.training
        captured["grad_enabled"] = torch.is_grad_enabled()
        captured["data_device"] = data.device
        captured["data_lens_device"] = data_lens.device
        captured["split"] = split
        return {"ED_mean": 1.0}

    obj.evaluate_split = fake_evaluate_split

    data, lens = _make_data()
    out = obj.evaluate_split_no_grad(data, lens, None, split=DatasetSplitType.VAL)

    assert out == {"ED_mean": 1.0}
    assert captured["training"] is False
    assert captured["grad_enabled"] is False
    assert captured["data_device"] == obj.device
    assert captured["data_lens_device"] == obj.device
    assert captured["split"] is DatasetSplitType.VAL


def test_test_step_delegates_to_evaluate_split_with_test_split():
    obj = _make_stub()
    obj.evaluate_split = MagicMock()
    data, lens = _make_data()

    obj.test_step((data, lens, None), 0)

    obj.evaluate_split.assert_called_once()
    assert obj.evaluate_split.call_args.kwargs.get("split") is DatasetSplitType.TEST


def test_test_step_rejects_multi_batch():
    obj = _make_stub()
    obj.evaluate_split = MagicMock()
    with pytest.raises(RuntimeError):
        obj.test_step((None, None, None), 1)
    obj.evaluate_split.assert_not_called()


def test_run_bootstrap_metrics_threads_split_to_metric_factory():
    """The bootstrap diagnostic loop must tag its TPPMetrics with the requested split."""
    obj = _make_stub()
    obj._metrics_config = TPPMetricsConfig(time_max=20.0, n_bootstraps=1)

    captured = {}

    def fake_create(data, lens, split=DatasetSplitType.TEST):
        captured["split"] = split
        fake_metrics = MagicMock()
        fake_metrics.compute_all_metrics.return_value = {"ED": 1.0}
        return fake_metrics

    obj._create_metrics_from_batch = fake_create
    obj._compute_mark_metrics_from_inputs = lambda **kwargs: None

    data, lens = _make_data()
    n = data.shape[0]
    uncond_result = MagicMock()
    cond_result = MagicMock()
    cond_result.gen_its_tf_nan = torch.zeros(2, n, 3, 1)
    cond_result.ref_its_nan = torch.zeros(2, n, 3, 1)

    out = obj._run_bootstrap_metrics(
        data=data,
        data_lens=lens,
        marks=None,
        uncond_result=uncond_result,
        cond_result=cond_result,
        split=DatasetSplitType.VAL,
    )

    assert captured["split"] is DatasetSplitType.VAL
    assert "ED_mean" in out


def test_on_test_end_writes_artifacts_on_test_pass():
    """on_test_end runs only on the test pass now; it writes samples/plots and clears the cache.

    Validation never reaches this hook (it calls evaluate_split directly), so there
    is no split flag to check: a single, unconditional artifact path remains.
    """
    obj = _make_stub()
    obj.output_dir = None  # skip disk writes while still exercising the artifact path
    obj.metrics_test = {}
    obj._test_batch_cache = {
        "metrics": {},
        "data": torch.zeros(2, 4, 1),
        "data_lens": torch.tensor([4, 4]),
        "marks": None,
    }
    obj._get_fake_real_samples = MagicMock(return_value=(torch.zeros(2, 3, 1), torch.tensor([3, 3]), None, None, None))
    obj.sample_and_plot = MagicMock()
    obj.log_results_comparison = MagicMock()

    obj.on_test_end()

    obj._get_fake_real_samples.assert_called_once()
    obj.sample_and_plot.assert_called_once()
    obj.log_results_comparison.assert_called_once()
    assert not hasattr(obj, "_test_batch_cache")
