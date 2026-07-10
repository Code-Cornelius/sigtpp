"""Tests for consistency between set_scaler_paths_for_sig and scale_paths_pre_sig.

Invariant
---------
``set_scaler_paths_for_sig`` defines ``total_vars`` as the mean total variation
of the scaled training paths.  When ``scale_paths_pre_sig`` is applied to the
**same** training data, it divides those scaled paths by ``total_vars``, so the
resulting paths must have mean total variation == 1.0.

    total_var(scale_paths_pre_sig(training_data)).mean()  ≈  1.0

This tests that the two methods are mutually consistent (same transforms, same
order) without duplicating their implementation in the expected value.

Test cases
----------
1. baseline : use_lead_lag=False, terminal_anchor_mode=NONE, full-length seqs
2. lead_lag : use_lead_lag=True,  terminal_anchor_mode=NONE, full-length seqs
3. var_len  : use_lead_lag=False, terminal_anchor_mode=NONE, variable-length seqs
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import unittest

import torch

from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.metrics.anchors.terminal_anchor_strategy import make_anchor_strategy
from src.metrics.totalvar import total_var
from src.nn.architectures.tpp_architecture import TPPArchitecture


# ---------------------------------------------------------------------------
# Stub subclass :  same pattern as test_sample_for_fixed_batch.py
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_stub(anchor_mode: TerminalAnchorMode) -> _StubTPP:
    """Allocate a _StubTPP without calling __init__; wire only what the methods need.

    set_scaler_paths_for_sig assigns a StandardScaler (an nn.Module) to self.scaler_std,
    which requires PyTorch's Module.__setattr__ to find _modules already initialised.
    We call torch.nn.Module.__init__ directly to set up that internal bookkeeping without
    triggering TPPArchitecture.__init__ (which needs a full config object).
    """
    obj = object.__new__(_StubTPP)
    torch.nn.Module.__init__(obj)
    obj.terminal_anchor_mode = anchor_mode
    obj._anchor_strategy = make_anchor_strategy(anchor_mode)
    obj.time_max = 10.0
    obj.use_marks = False
    obj.scaler_exp = lambda x: x  # identity; unused by FREE_ENDPOINT but required by interface
    return obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScalePathsConsistency(unittest.TestCase):
    """Verify that set_scaler_paths_for_sig and scale_paths_pre_sig are consistent.

    The key invariant: after fitting on training data D, calling scale_paths_pre_sig(D)
    produces paths whose mean total variation equals 1.0, because total_vars was defined
    as that mean total variation over the same scaled paths.
    """

    def _assert_mean_tv_is_one(self, obj: _StubTPP, data: torch.Tensor, msg: str = "") -> None:
        scaled = obj.scale_paths_pre_sig(data)
        mean_tv = total_var(scaled).mean().item()
        self.assertAlmostEqual(
            mean_tv,
            1.0,
            places=5,
            msg=f"mean total variation of scale_paths_pre_sig output should be 1.0 {msg}, got {mean_tv:.6f}",
        )

    def test_baseline_no_lead_lag_no_anchor(self):
        """Baseline: no lead-lag, no anchor, full-length sequences."""
        obj = _make_stub(anchor_mode=TerminalAnchorMode.FREE_ENDPOINT)

        N, L, D = 4, 6, 2
        torch.manual_seed(0)
        data = torch.rand(N, L, D) + 0.1

        obj.set_scaler_paths_for_sig(data, target_seq_lens=None)

        self._assert_mean_tv_is_one(obj, data, "(no lead-lag, no anchor)")

    def test_variable_length_sequences(self):
        """Variable-length fit: scaler fitted on true-length rows must normalise correctly."""
        obj = _make_stub(anchor_mode=TerminalAnchorMode.FREE_ENDPOINT)

        N, L, D = 6, 8, 2
        torch.manual_seed(2)
        data = torch.rand(N, L, D) + 0.1

        lens = torch.tensor([3, 5, 4, 6, 2, 7])
        obj.set_scaler_paths_for_sig(data, target_seq_lens=lens)

        self._assert_mean_tv_is_one(obj, data, "(variable-length)")


if __name__ == "__main__":
    unittest.main()
