"""
Tests for dedup_consecutive_torch. The inline tests in dedup.py check for
ValueError/TypeError but the source uses assert (AssertionError), so we
write corrected versions here alongside the passing tests.
"""
import torch
import pytest
from src.utils.dedup import dedup_consecutive_torch


def assert_close_with_nan(actual, expected):
    torch.testing.assert_close(actual, expected, equal_nan=True)


class TestDedupBasicCases:
    def test_consecutive_duplicates_removed(self):
        inp = torch.tensor([[[1.0], [1.0], [2.0], [2.0], [3.0]]], dtype=torch.float64)
        exp = torch.tensor([[[1.0], [2.0], [3.0], [float('nan')], [float('nan')]]], dtype=torch.float64)
        assert_close_with_nan(dedup_consecutive_torch(inp), exp)

    def test_with_trailing_nan(self):
        inp = torch.tensor([[[5.0], [5.0], [6.0], [6.0], [float('nan')]]], dtype=torch.float32)
        exp = torch.tensor([[[5.0], [6.0], [float('nan')], [float('nan')], [float('nan')]]], dtype=torch.float32)
        assert_close_with_nan(dedup_consecutive_torch(inp), exp)

    def test_small_values(self):
        inp = torch.tensor([[[1e-12], [1e-12], [2e-12], [3e-12], [3e-12], [4e-12]]], dtype=torch.float32)
        exp = torch.tensor(
            [[[1e-12], [2e-12], [3e-12], [4e-12], [float('nan')], [float('nan')]]], dtype=torch.float32
        )
        assert_close_with_nan(dedup_consecutive_torch(inp), exp)


class TestDedupMultiBatch:
    def test_multi_batch(self):
        x = torch.tensor(
            [[[1.0], [1.0], [2.0]], [[3.0], [3.0], [3.0]]],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [[[1.0], [2.0], [float('nan')]], [[3.0], [float('nan')], [float('nan')]]],
            dtype=torch.float32,
        )
        assert_close_with_nan(dedup_consecutive_torch(x), expected)


class TestDedupEdgeCases:
    def test_invalid_shape_2d(self):
        with pytest.raises(AssertionError):
            dedup_consecutive_torch(torch.randn(2, 3))

    def test_invalid_shape_last_dim_not_1(self):
        with pytest.raises(AssertionError):
            dedup_consecutive_torch(torch.randn(2, 3, 2))

    def test_invalid_dtype(self):
        with pytest.raises(AssertionError):
            dedup_consecutive_torch(torch.randint(0, 5, (1, 4, 1), dtype=torch.int32))

    def test_no_duplicates_unchanged(self):
        inp = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32)
        exp = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32)
        assert_close_with_nan(dedup_consecutive_torch(inp), exp)

    def test_all_same_values(self):
        inp = torch.tensor([[[5.0], [5.0], [5.0], [5.0]]], dtype=torch.float32)
        exp = torch.tensor([[[5.0], [float('nan')], [float('nan')], [float('nan')]]], dtype=torch.float32)
        assert_close_with_nan(dedup_consecutive_torch(inp), exp)

    def test_interior_nan_not_treated_as_duplicate(self):
        """NaN values should not be considered duplicates of each other."""
        inp = torch.tensor([[[1.0], [float('nan')], [1.0], [1.0]]], dtype=torch.float32)
        # NaN is kept, then 1.0 appears twice -> second deduped
        exp = torch.tensor([[[1.0], [float('nan')], [1.0], [float('nan')]]], dtype=torch.float32)
        assert_close_with_nan(dedup_consecutive_torch(inp), exp)

    def test_single_element_sequence(self):
        inp = torch.tensor([[[7.0]]], dtype=torch.float32)
        exp = torch.tensor([[[7.0]]], dtype=torch.float32)
        assert_close_with_nan(dedup_consecutive_torch(inp), exp)
