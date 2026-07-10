"""
Unit tests for TPP utility functions in utils.tpp_utils.

Tests cover all 8 utility functions with various edge cases and error conditions.
"""

import os
import tempfile
from unittest import TestCase
from unittest.mock import patch

import pytest
pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import torch

from src.nn.architectures.tpp_architecture import TPPArchitecture
from src.utils import tpp_utils


class TestApplyMask(TestCase):
    """Tests for apply_mask function."""

    def test_apply_mask_filters_correctly(self):
        """Test that apply_mask correctly filters tensors based on boolean mask."""
        tensor1 = torch.randn(10, 5, 1)
        mask_valid = torch.tensor([True] * 8 + [False] * 2)

        filtered_t1, filtered_t2 = tpp_utils.apply_mask(tensor1, mask_valid)

        self.assertEqual(filtered_t1.shape, (8, 5, 1))
        self.assertIsNone(filtered_t2)

    def test_apply_mask_with_two_tensors(self):
        """Test that apply_mask filters two tensors in lockstep."""
        tensor1 = torch.randn(10, 5, 1)
        tensor2 = torch.randn(10, 3, 2)
        mask_valid = torch.tensor([True, False, True, True, False, True, True, True, False, True])

        filtered_t1, filtered_t2 = tpp_utils.apply_mask(tensor1, mask_valid, tensor2)

        self.assertEqual(filtered_t1.shape, (7, 5, 1))
        self.assertEqual(filtered_t2.shape, (7, 3, 2))

    def test_apply_mask_raises_on_empty(self):
        """Test that apply_mask raises ValueError if all sequences are filtered out."""
        tensor1 = torch.randn(5, 10, 1)
        mask_valid = torch.tensor([False] * 5)

        with self.assertRaises(ValueError) as context:
            tpp_utils.apply_mask(tensor1, mask_valid)

        self.assertIn("all sequences were removed", str(context.exception))

    def test_apply_mask_all_valid(self):
        """Test apply_mask when all sequences are valid."""
        tensor1 = torch.randn(10, 5, 1)
        mask_valid = torch.tensor([True] * 10)

        filtered_t1, filtered_t2 = tpp_utils.apply_mask(tensor1, mask_valid)

        self.assertEqual(filtered_t1.shape, tensor1.shape)
        self.assertTrue(torch.allclose(filtered_t1, tensor1))


class TestGetExtremaForClamping(TestCase):
    """Tests for _get_extrema_for_clamping static method in TPPArchitecture."""

    def test_get_extrema_10_percent_positive_data(self):
        """Test 10% margin with positive data."""
        data = torch.ones(100, 50, 1) * 5.0
        min_val, max_val = TPPArchitecture._get_extrema_for_clamping(data, use_10_or_30=True)

        self.assertAlmostEqual(min_val, 5.0 * 0.9, places=5)
        self.assertAlmostEqual(max_val, 5.0 * 1.1, places=5)

    def test_get_extrema_30_percent_positive_data(self):
        """Test 30% margin with positive data."""
        data = torch.ones(100, 50, 1) * 5.0
        min_val, max_val = TPPArchitecture._get_extrema_for_clamping(data, use_10_or_30=False)

        self.assertAlmostEqual(min_val, 5.0 * 0.7, places=5)
        self.assertAlmostEqual(max_val, 5.0 * 1.3, places=5)

    def test_get_extrema_negative_data(self):
        """Test margin calculation with negative data."""
        data = torch.ones(100, 50, 1) * -5.0
        min_val, max_val = TPPArchitecture._get_extrema_for_clamping(data, use_10_or_30=True)

        # For negative values, magnifying and reduction factors are swapped
        self.assertAlmostEqual(min_val, -5.0 * 1.1, places=5)
        self.assertAlmostEqual(max_val, -5.0 * 0.9, places=5)

    def test_get_extrema_mixed_data(self):
        """Test with data containing both positive and negative values."""
        data = torch.randn(100, 50, 1)
        data[0, 0, 0] = -10.0  # Set minimum
        data[1, 0, 0] = 10.0  # Set maximum

        min_val, max_val = TPPArchitecture._get_extrema_for_clamping(data, use_10_or_30=True)

        self.assertLess(min_val, -10.0)  # Should be magnified
        self.assertGreater(max_val, 10.0)  # Should be magnified

    def test_get_extrema_raises_on_wrong_shape(self):
        """Test that function raises assertion error for wrong input shape."""
        # 2D tensor (missing feature dimension)
        data_2d = torch.randn(100, 50)
        with self.assertRaises(AssertionError):
            TPPArchitecture._get_extrema_for_clamping(data_2d, use_10_or_30=True)

    def test_get_extrema_raises_on_multivariate(self):
        """Test that function raises assertion error for multivariate data."""
        # Multivariate data (D > 1)
        data_multi = torch.randn(100, 50, 3)
        with self.assertRaises(AssertionError):
            TPPArchitecture._get_extrema_for_clamping(data_multi, use_10_or_30=True)


class TestConcatTwoSamplesTogether(TestCase):
    """Tests for concat_two_samples_together function."""

    def test_concat_pads_with_secondary(self):
        """Test that function pads from secondary tensor when primary is short."""
        primary = [torch.randn(80, 10), torch.randn(90, 10)]
        secondary = [torch.randn(50, 10), torch.randn(50, 10)]
        target_length = 100

        result = tpp_utils.concat_two_samples_together(primary, secondary, target_length)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (100, 10))
        self.assertEqual(result[1].shape, (100, 10))

    def test_concat_truncates_when_too_long(self):
        """Test that function truncates when primary is longer than target."""
        primary = [torch.randn(120, 10), torch.randn(110, 10)]
        secondary = [torch.randn(50, 10), torch.randn(50, 10)]
        target_length = 100

        result = tpp_utils.concat_two_samples_together(primary, secondary, target_length)

        self.assertEqual(result[0].shape, (100, 10))
        self.assertEqual(result[1].shape, (100, 10))

    def test_concat_exact_length(self):
        """Test when primary is exactly target length."""
        primary = [torch.randn(100, 10)]
        secondary = [torch.randn(50, 10)]
        target_length = 100

        result = tpp_utils.concat_two_samples_together(primary, secondary, target_length)

        self.assertEqual(result[0].shape, (100, 10))

    def test_concat_none_secondary(self):
        """Test with None secondary list."""
        primary = [torch.randn(100, 10), torch.randn(100, 10)]
        target_length = 100

        result = tpp_utils.concat_two_samples_together(primary, None, target_length)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (100, 10))
        self.assertEqual(result[1].shape, (100, 10))

    def test_concat_raises_on_insufficient_samples(self):
        """Test that function raises ValueError when not enough samples available."""
        primary = [torch.randn(80, 10)]
        secondary = [torch.randn(10, 10)]  # Not enough to reach target
        target_length = 100

        with self.assertRaises(ValueError) as context:
            tpp_utils.concat_two_samples_together(primary, secondary, target_length)

        self.assertIn("Not enough values available", str(context.exception))


class TestInsertZeroBeg(TestCase):
    """Tests for insert_zero_beg function."""

    def test_insert_zero_prepends_correctly(self):
        """Test that insert_zero_beg prepends zeros to sequences."""
        sequences = torch.tensor([[[1.0], [2.0], [3.0]]])  # Shape (1, 3, 1)
        result = tpp_utils.insert_zero_beg(sequences)

        self.assertEqual(result.shape, (1, 4, 1))
        self.assertEqual(result[0, 0, 0].item(), 0.0)
        self.assertEqual(result[0, 1, 0].item(), 1.0)
        self.assertEqual(result[0, 2, 0].item(), 2.0)
        self.assertEqual(result[0, 3, 0].item(), 3.0)

    def test_insert_zero_batch(self):
        """Test insert_zero_beg with batch of sequences."""
        sequences = torch.randn(10, 50, 1)
        result = tpp_utils.insert_zero_beg(sequences)

        self.assertEqual(result.shape, (10, 51, 1))
        self.assertTrue(torch.allclose(result[:, 0, :], torch.zeros(10, 1)))
        self.assertTrue(torch.allclose(result[:, 1:, :], sequences))

    def test_insert_zero_multivariate(self):
        """Test insert_zero_beg with multivariate sequences."""
        sequences = torch.randn(5, 20, 3)
        result = tpp_utils.insert_zero_beg(sequences)

        self.assertEqual(result.shape, (5, 21, 3))
        self.assertTrue(torch.allclose(result[:, 0, :], torch.zeros(5, 3)))

    def test_insert_zero_preserves_device(self):
        """Test that insert_zero_beg preserves tensor device."""
        sequences = torch.randn(10, 50, 1)
        result = tpp_utils.insert_zero_beg(sequences)
        self.assertEqual(result.device.type, 'cpu')


class TestLogResultsComparison(TestCase):
    """Tests for log_results_comparison static method in TPPArchitecture."""

    def test_log_results_comparison_runs(self):
        """Test that log_results_comparison logs without error."""
        metrics = {
            'sigW_loword_notstd_mean': 0.5678,
            'hist_it_mean': 0.0123,
            'hist_int_mean': 0.0456,
            'MAE_proper_mean': 0.789,
            'MAE_mean': 0.812,
            'MSE_proper_mean': 1.234,
            'CRPS_mean': 0.567,
            'corr_mean': 0.890,
            'autocorr_mean': 0.123,
            'ED_mean': 0.456,
            'W1_mean': 0.789,
        }

        # Should not raise any exception
        TPPArchitecture.log_results_comparison(metrics)

    def test_log_results_comparison_with_various_values(self):
        """Test with various numeric values."""
        metrics = {
            'sigW_loword_notstd_mean': 1.0,
            'hist_it_mean': 0.0001,
            'hist_int_mean': 0.0,
            'MAE_proper_mean': 10.5,
            'MAE_mean': 9.8,
            'MSE_proper_mean': 100.0,
            'CRPS_mean': 0.5,
            'corr_mean': 0.9999,
            'autocorr_mean': 0.1111,
            'ED_mean': 0.4444,
            'W1_mean': 0.7777,
        }

        # Should not raise any exception
        TPPArchitecture.log_results_comparison(metrics)

    def test_log_results_comparison_includes_scalar_mark_metrics(self):
        """Scalar metrics computed outside bootstrap should still appear in the log table."""
        metrics = {
            'sigW_loword_notstd_mean': 0.5,
            'mark_ce': 1.25,
            'top1_mark_acc': 0.8,
            'top3_mark_acc': 0.95,
        }

        with patch('src.nn.architectures.tpp_architecture.logger.info') as mock_info:
            TPPArchitecture.log_results_comparison(metrics)

        logged_text = mock_info.call_args[0][0]
        self.assertIn('mark_ce', logged_text)
        self.assertIn('top1_mark_acc', logged_text)
        self.assertIn('top3_mark_acc', logged_text)


class TestCumTimesToLogInterTimes(TestCase):
    """Tests for cum_times_to_log_inter_times function."""

    def test_cum_times_to_log_inter_times_converts(self):
        """Test conversion from cumulative times to inter-arrival times."""
        # Create cumulative times starting from 0
        cum_times = torch.cumsum(torch.rand(10, 20, 1) + 0.1, dim=1)
        cum_with_anchor = torch.cat([torch.zeros(10, 1, 1), cum_times], dim=1)
        lens = torch.full((10,), 21)

        # Simple scaler (log transformation)
        scaler = lambda x: torch.log(x + 1e-8)

        new_lens, scaled_its = tpp_utils.cum_times_to_log_inter_times((cum_with_anchor, lens), scaler)

        self.assertEqual(new_lens.shape, (10,))
        self.assertTrue(torch.allclose(new_lens, torch.full((10,), 20)))
        self.assertEqual(scaled_its.shape, (10, 20, 1))

    def test_cum_times_diff_is_correct(self):
        """Test that diff operation produces correct inter-arrival times."""
        # Create known cumulative times
        cum_times = torch.tensor([[[0.0], [1.0], [3.0], [6.0], [10.0]]])  # Shape (1, 5, 1)
        lens = torch.tensor([5])

        # Identity scaler
        scaler = lambda x: x

        new_lens, its = tpp_utils.cum_times_to_log_inter_times((cum_times, lens), scaler)

        expected_its = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # Shape (1, 4, 1)

        self.assertEqual(new_lens.item(), 4)
        self.assertTrue(torch.allclose(its, expected_its))

    @pytest.mark.xfail(
        reason=(
            "cum_times_to_log_inter_times uses nan_to_num(nan=0.0) which does not handle "
            "-inf produced by log(0) on constant-padded zeros (diff=0 at padding positions). "
            "Padding positions are not zeroed out by a length mask. "
            "Padding positions should be zeroed out in cum_times_to_log_inter_times."
        ),
        strict=True,
    )
    def test_no_nan_with_constant_padded_input_shift_param_zero(self):
        """With constant-padded cumulative times (diff->0 at padding) and shift_param=0,
        the output must have no NaN and no inf -- padding positions must be exactly 0.0."""
        # 1 sequence, anchor + 2 events, padded to length 5 total -> diff length 4
        # Positions 2 and 3 (0-indexed) in the diff output are padding (inter-time = 0.0)
        cum_times = torch.tensor(
            [
                [[0.0], [0.5], [1.2], [1.2], [1.2]],  # constant-padded with last valid time 1.2
            ]
        )
        lens = torch.tensor([3])  # includes anchor; diff length = 2 valid, 2 padding

        scaler = lambda x: torch.log(x)  # shift_param=0 -- hardest case

        new_lens, scaled = tpp_utils.cum_times_to_log_inter_times((cum_times, lens), scaler)

        self.assertFalse(torch.any(torch.isnan(scaled)).item(), "Output must not contain NaN")
        self.assertFalse(torch.any(torch.isinf(scaled)).item(), "Output must not contain inf")
        # Padding positions must be zeroed out
        self.assertEqual(scaled[0, 2, 0].item(), 0.0)
        self.assertEqual(scaled[0, 3, 0].item(), 0.0)

    @pytest.mark.xfail(
        reason=(
            "cum_times_to_log_inter_times does not zero out padding positions by length mask. "
            "log(0+1e-8) at padding slots produces a large negative value, not 0.0. "
            "Padding positions should be zeroed out in cum_times_to_log_inter_times."
        ),
        strict=True,
    )
    def test_valid_positions_scaled_correctly(self):
        """Valid inter-arrival times must be correctly scaled after the fix."""
        cum_times = torch.tensor(
            [
                [[0.0], [1.0], [3.0], [3.0]],  # valid inter-times: 1.0, 2.0; one padding slot
            ]
        )
        lens = torch.tensor([3])  # anchor + 2 events

        scaler = lambda x: torch.log(x + 1e-8)

        new_lens, scaled = tpp_utils.cum_times_to_log_inter_times((cum_times, lens), scaler)

        import math

        self.assertAlmostEqual(scaled[0, 0, 0].item(), math.log(1.0 + 1e-8), places=5)
        self.assertAlmostEqual(scaled[0, 1, 0].item(), math.log(2.0 + 1e-8), places=5)
        self.assertEqual(scaled[0, 2, 0].item(), 0.0)  # padding zeroed


class TestSaveLoadSamples(TestCase):
    """Tests for save_samples and load_samples functions."""

    def test_save_and_load_roundtrip(self):
        """Test that save_samples and load_samples work together."""
        # Create test data
        inter_times = torch.randn(1000, 50, 1)
        lengths = torch.randint(10, 50, (1000,))

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_samples.pth")

            tpp_utils.save_samples(inter_times, lengths, path)
            loaded_its, loaded_lens = tpp_utils.load_samples(path)

        # Verify
        self.assertTrue(torch.allclose(loaded_its, inter_times))
        self.assertTrue(torch.allclose(loaded_lens.float(), lengths.float()))

    def test_save_creates_file(self):
        """Test that save_samples creates the file."""
        inter_times = torch.randn(10, 20, 1)
        lengths = torch.randint(5, 20, (10,))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pth")

            tpp_utils.save_samples(inter_times, lengths, path)

            self.assertTrue(os.path.exists(path))

    def test_load_maps_to_cpu(self):
        """Test that load_samples maps tensors to CPU."""
        inter_times = torch.randn(10, 20, 1)
        lengths = torch.randint(5, 20, (10,))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pth")

            tpp_utils.save_samples(inter_times, lengths, path)
            loaded_its, loaded_lens = tpp_utils.load_samples(path)

            self.assertEqual(loaded_its.device.type, 'cpu')
            self.assertEqual(loaded_lens.device.type, 'cpu')

    def test_load_preserves_shapes(self):
        """Test that save/load preserves tensor shapes."""
        inter_times = torch.randn(500, 100, 2)  # Multivariate
        lengths = torch.randint(50, 100, (500,))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pth")

            tpp_utils.save_samples(inter_times, lengths, path)
            loaded_its, loaded_lens = tpp_utils.load_samples(path)

            self.assertEqual(loaded_its.shape, (500, 100, 2))
            self.assertEqual(loaded_lens.shape, (500,))

    def test_save_and_load_roundtrip_with_marks(self):
        """Test that marks survive the save/load roundtrip."""
        inter_times = torch.randn(20, 10, 1)
        lengths = torch.randint(3, 10, (20,))
        marks = torch.randint(0, 5, (20, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "marked_samples.pth")
            tpp_utils.save_samples(inter_times, lengths, path, marks=marks)
            loaded_its, loaded_lens, loaded_marks = tpp_utils.load_samples(path)

        self.assertTrue(torch.equal(loaded_its, inter_times))
        self.assertTrue(torch.equal(loaded_lens, lengths))
        self.assertTrue(torch.equal(loaded_marks, marks))
        self.assertEqual(loaded_marks.shape, (20, 10))

    def test_load_old_file_without_marks_returns_two_tuple(self):
        """Old .pth files without marks must still load as a 2-tuple."""
        inter_times = torch.randn(10, 5, 1)
        lengths = torch.full((10,), 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old_format.pth")
            # Manually save without marks key to simulate old format.
            torch.save({"inter_times": inter_times, "lengths": lengths}, path)
            result = tpp_utils.load_samples(path)

        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], inter_times))
        self.assertTrue(torch.equal(result[1], lengths))

    def test_save_without_marks_then_load_returns_two_tuple(self):
        """save_samples called without marks produces a 2-tuple on load."""
        inter_times = torch.randn(10, 5, 1)
        lengths = torch.full((10,), 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "no_marks.pth")
            tpp_utils.save_samples(inter_times, lengths, path)
            result = tpp_utils.load_samples(path)

        self.assertEqual(len(result), 2)

    def test_save_samples_leaves_no_tmp_file(self):
        """atomic_torch_save must not leave a .tmp file behind on success."""
        torch.manual_seed(0)
        inter_times = torch.randn(5, 5, 1)
        lengths = torch.full((5,), 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "samples.pth")
            tpp_utils.save_samples(inter_times, lengths, path)

            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))


class TestSaveTestTargetsOnce(TestCase):
    """Tests for save_test_targets_once (experiment-level shared test targets)."""

    def test_writes_shared_file(self):
        torch.manual_seed(0)
        inter_times = torch.randn(10, 5, 1)
        lengths = torch.full((10,), 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            written_path = tpp_utils.save_test_targets_once(tmpdir, inter_times, lengths)

            expected_path = os.path.join(tmpdir, "test_targets.pth")
            self.assertEqual(written_path, expected_path)
            loaded_its, loaded_lens = tpp_utils.load_samples(expected_path)
            self.assertTrue(torch.equal(loaded_its, inter_times))
            self.assertTrue(torch.equal(loaded_lens, lengths))

    def test_second_call_with_same_data_skips_rewrite(self):
        torch.manual_seed(0)
        inter_times = torch.randn(10, 5, 1)
        lengths = torch.full((10,), 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = tpp_utils.save_test_targets_once(tmpdir, inter_times, lengths)
            first_mtime = os.path.getmtime(path)

            tpp_utils.save_test_targets_once(tmpdir, inter_times, lengths)
            second_mtime = os.path.getmtime(path)

        self.assertEqual(first_mtime, second_mtime)

    def test_call_with_different_data_overwrites(self):
        torch.manual_seed(0)
        inter_times = torch.randn(10, 5, 1)
        lengths = torch.full((10,), 5)
        other_inter_times = torch.randn(10, 5, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = tpp_utils.save_test_targets_once(tmpdir, inter_times, lengths)
            tpp_utils.save_test_targets_once(tmpdir, other_inter_times, lengths)

            loaded_its, _ = tpp_utils.load_samples(path)

        self.assertTrue(torch.equal(loaded_its, other_inter_times))
