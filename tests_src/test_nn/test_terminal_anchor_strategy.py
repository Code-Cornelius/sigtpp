import unittest

import torch

from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.metrics.anchors.terminal_anchor_strategy import (
    FreeEndpointStrategy,
    ResidualStrategy,
    make_anchor_strategy,
)

# ---------------------------------------------------------------------------
# Shared fixture  (docstring example from terminal_anchor_mode.py)
#   τ₁=2, τ₂=3, τ₃=4, T_max=12, seq_lens=2
# _replace_from_index replaces positions > seq_lens=2, i.e. position 3 only
# ---------------------------------------------------------------------------

_T_MAX = 12.0
_SEQ_LENS = torch.tensor([2])

# Non-ITSHIFTED paths: cumsum excludes τ₁
_PATHS = torch.tensor([[[0.0, 0.0], [3.0, 3.0], [4.0, 7.0], [0.0, 7.0]]])  # (1, 4, 2)


class IdentityScaler:
    def __call__(self, x):
        return x

    def unscale(self, x):
        return x


_SCALER = IdentityScaler()


# ---------------------------------------------------------------------------


class TestFreeEndpointStrategy(unittest.TestCase):
    def test_apply(self):
        result = FreeEndpointStrategy().apply(_PATHS, _T_MAX, _SEQ_LENS)
        desired = torch.tensor([[[0.0, 0.0], [3.0, 3.0], [4.0, 7.0], [0.0, 7.0]]])
        self.assertTrue(torch.allclose(result, desired))


class TestResidualStrategy(unittest.TestCase):
    def test_apply(self):
        # gap = 12 - 7 = 5  (biased: τ₁ missing from cum)
        result = ResidualStrategy(_SCALER).apply(_PATHS, _T_MAX, _SEQ_LENS)
        desired = torch.tensor([[[0.0, 0.0], [3.0, 3.0], [4.0, 7.0], [5.0, 12.0]]])
        self.assertTrue(torch.allclose(result, desired))


# ---------------------------------------------------------------------------


class TestMakeAnchorStrategy(unittest.TestCase):
    def test_factory_types(self):
        self.assertIsInstance(make_anchor_strategy(TerminalAnchorMode.FREE_ENDPOINT), FreeEndpointStrategy)
        self.assertIsInstance(make_anchor_strategy(TerminalAnchorMode.RESIDUAL, scaler_exp=_SCALER), ResidualStrategy)

    def test_residual_raises_without_scaler_exp(self):
        with self.assertRaises(ValueError):
            make_anchor_strategy(TerminalAnchorMode.RESIDUAL)


class TestTerminalAnchorMode(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(TerminalAnchorMode.FREE_ENDPOINT, "free_endpoint")
        self.assertEqual(TerminalAnchorMode.RESIDUAL, "residual")


if __name__ == "__main__":
    unittest.main()
