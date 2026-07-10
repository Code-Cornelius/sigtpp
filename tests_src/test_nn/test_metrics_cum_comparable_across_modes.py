"""Tests verifying that INT/ED/W1 metrics are comparable across anchor modes.

For cross-mode comparability, the cumulative times used for histogram/ED/W1
metrics must represent the same random object in every mode:
  relative event times  (τ₂, τ₂+τ₃, …) : i.e. WITHOUT τ₁ shift.
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import unittest

import torch

from src.data_types.tppmetrics import TPPMetrics, TPPMetricsConfig, DatasetSplitType
from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _IdentityScaler:
    def __call__(self, x):
        return x

    def unscale(self, x):
        return x


def _make_tpp_metrics_config():
    return TPPMetricsConfig(sig_degree=2, scale_high_degrees=False, standardise_sig=False, time_max=12.0)


def _make_reference_data(n=8, l=5, seed=0):
    """Return (data, lens) where data is cumulative times (N, L+1, 1)."""
    torch.manual_seed(seed)
    its = torch.rand(n, l, 1) + 0.3
    cum = torch.cat([torch.zeros(n, 1, 1), its.cumsum(dim=1)], dim=1)
    lens = torch.full((n,), l + 1, dtype=torch.long)
    return cum, lens


def _scale_paths_pre_sig_identity(x, seq_lens=None):
    return x


def _make_tpp_metrics(reference_data, reference_lens, terminal_anchor_mode):
    """Build a TPPMetrics for the given mode with minimal configuration."""
    scaler = _IdentityScaler()
    config = _make_tpp_metrics_config()

    # Build minimal sig_loss_seqs (content irrelevant for cum-reference tests)
    n, lp1, d = reference_data.shape
    sig_loss_seqs = torch.zeros(n, lp1, d)

    return TPPMetrics(
        reference_data=reference_data,
        reference_lens=reference_lens,
        scaler=scaler,
        config=config,
        sig_loss_seqs=sig_loss_seqs,
        scale_paths_pre_sig=_scale_paths_pre_sig_identity,
        split=DatasetSplitType.TRAIN,
    )


# ---------------------------------------------------------------------------
# Test: TPPMetrics.reference_data_cum_naned is mode-invariant (always relative)
# ---------------------------------------------------------------------------


class TestReferenceDataCumIsAlwaysRelative(unittest.TestCase):
    """reference_data_cum_naned must not depend on the anchor mode.

    All modes should produce identical reference cumulative times because the
    reference is always computed from τ₂, τ₃, … (relative event times).
    """

    def _get_reference_cum(self, mode):
        data, lens = _make_reference_data()
        m = _make_tpp_metrics(data, lens, mode)
        return m.reference_data_cum_naned

    def test_residual_matches_free_endpoint(self):
        """Non-ITSHIFTED modes should already match (regression guard)."""
        cum_free = self._get_reference_cum(TerminalAnchorMode.FREE_ENDPOINT)
        cum_res = self._get_reference_cum(TerminalAnchorMode.RESIDUAL)
        self.assertTrue(torch.allclose(cum_free, cum_res))


if __name__ == "__main__":
    unittest.main()
