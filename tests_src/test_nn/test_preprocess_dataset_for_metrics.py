"""Tests for TPPArchitecture._preprocess_dataset_for_metrics.

The method must produce correct outputs for the two supported anchor modes:
  1. RESIDUAL     : cumsum without τ₁ shift
  2. FREE_ENDPOINT: same cumsum behavior
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import unittest

import torch

from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.metrics.anchors.terminal_anchor_strategy import make_anchor_strategy
from src.nn.architectures.tpp_architecture import TPPArchitecture
from src.utils.fix_seq_ends import set_seq_to_cst_val_from_index


# ---------------------------------------------------------------------------
# Stub subclass: same pattern as other test files
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


def _make_stub(anchor_mode: TerminalAnchorMode) -> _StubTPP:
    obj = object.__new__(_StubTPP)
    torch.nn.Module.__init__(obj)
    obj.terminal_anchor_mode = anchor_mode
    obj.time_max = 20.0
    obj.use_marks = False

    class _Scaler:
        def __call__(self, x):
            return x * 2  # non-trivial scaler to catch copy-paste errors

        def unscale(self, x):
            return x / 2

    obj.scaler_exp = _Scaler()
    obj._anchor_strategy = make_anchor_strategy(anchor_mode, scaler_exp=obj.scaler_exp)
    return obj


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference_preprocess(obj, data, data_lens):
    """Reproduce the expected preprocessing logic."""
    full_dts = data.diff(dim=1)
    dt_lens = data_lens - 1
    dts = full_dts[:, 1:, :]

    cum = dts.cumsum(dim=1)

    scaled_dts = obj.scaler_exp(dts)

    cum = set_seq_to_cst_val_from_index(cum, dt_lens - 2)
    scaled_dts = set_seq_to_cst_val_from_index(scaled_dts, dt_lens - 2)

    return scaled_dts, cum, dt_lens


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_data(n=5, l=7, d=1, seed=42):
    """Return cumulative-time data (N, L+1, D) and lengths (N,)."""
    torch.manual_seed(seed)
    its = torch.rand(n, l, d) + 0.5
    cum = torch.cat([torch.zeros(n, 1, d), its.cumsum(dim=1)], dim=1)
    lens = torch.randint(3, l + 1, (n,))
    return cum, lens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreprocessDatasetForMetrics(unittest.TestCase):
    """_preprocess_dataset_for_metrics must match the reference logic."""

    def _assert_match(self, anchor_mode: TerminalAnchorMode):
        obj = _make_stub(anchor_mode)
        data, data_lens = _make_data()

        expected = _reference_preprocess(obj, data, data_lens)
        result = obj._preprocess_dataset_for_metrics(data, data_lens)

        scaled_dts_exp, cum_exp, dt_lens_exp = expected
        scaled_dts_got, cum_got, dt_lens_got = result

        self.assertTrue(torch.allclose(scaled_dts_got, scaled_dts_exp), f"{anchor_mode}: scaled_dts mismatch")
        self.assertTrue(torch.allclose(cum_got, cum_exp), f"{anchor_mode}: cum mismatch")
        self.assertTrue(torch.equal(dt_lens_got, dt_lens_exp), f"{anchor_mode}: dt_lens mismatch")

    def test_residual(self):
        self._assert_match(TerminalAnchorMode.RESIDUAL)

    def test_free_endpoint(self):
        self._assert_match(TerminalAnchorMode.FREE_ENDPOINT)


if __name__ == "__main__":
    unittest.main()
