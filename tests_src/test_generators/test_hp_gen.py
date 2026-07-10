"""Tests for the homogeneous Poisson process generator (src/generators/hp.py)."""

import numpy as np
import pytest

from src.generators.hp import gen


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
class TestHpGenValidation:
    def test_both_modes_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            gen(10, 1.0, time_series_max_time=5.0, num_elements_in_ts=5)

    def test_neither_mode_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            gen(10, 1.0)

    def test_num_seq_zero_raises(self):
        with pytest.raises(ValueError, match="num_seq"):
            gen(0, 1.0, num_elements_in_ts=5)

    def test_num_seq_negative_raises(self):
        with pytest.raises(ValueError, match="num_seq"):
            gen(-1, 1.0, num_elements_in_ts=5)

    def test_lambda_zero_raises(self):
        with pytest.raises(ValueError, match="poisson_lambda"):
            gen(10, 0.0, num_elements_in_ts=5)

    def test_lambda_negative_raises(self):
        with pytest.raises(ValueError, match="poisson_lambda"):
            gen(10, -1.0, num_elements_in_ts=5)

    def test_lambda_inf_raises(self):
        with pytest.raises(ValueError, match="poisson_lambda"):
            gen(10, float("inf"), num_elements_in_ts=5)

    def test_max_time_zero_raises(self):
        with pytest.raises(ValueError, match="time_series_max_time"):
            gen(10, 1.0, time_series_max_time=0.0)

    def test_max_time_negative_raises(self):
        with pytest.raises(ValueError, match="time_series_max_time"):
            gen(10, 1.0, time_series_max_time=-5.0)

    def test_num_elements_zero_raises(self):
        with pytest.raises(ValueError, match="num_elements_in_ts"):
            gen(10, 1.0, num_elements_in_ts=0)

    def test_mark_probs_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="mark_probs"):
            gen(10, 1.0, num_elements_in_ts=5, num_marks=3, mark_probs=np.array([0.5, 0.5]))

    def test_mark_probs_not_sum_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            gen(10, 1.0, num_elements_in_ts=5, num_marks=2, mark_probs=np.array([0.3, 0.3]))

    def test_mark_probs_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            gen(10, 1.0, num_elements_in_ts=5, num_marks=2, mark_probs=np.array([-0.5, 1.5]))


# ---------------------------------------------------------------------------
# Fixed-length mode
# ---------------------------------------------------------------------------
class TestHpGenFixedLength:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        result = gen(20, 2.0, num_elements_in_ts=10, rng=rng)
        assert result.shape == (20, 10, 1)

    def test_all_positive(self):
        rng = np.random.default_rng(42)
        result = gen(50, 3.0, num_elements_in_ts=15, rng=rng)
        assert (result > 0).all(), "All inter-arrival times must be strictly positive"

    def test_mean_interarrival_approximates_inverse_lambda(self):
        """Mean of exponential(1/λ) should be ≈ 1/λ."""
        rng = np.random.default_rng(0)
        lam = 5.0
        result = gen(10_000, lam, num_elements_in_ts=50, rng=rng)
        mean_iat = result.mean()
        assert abs(mean_iat - 1.0 / lam) < 0.01, f"Expected mean ≈ {1/lam}, got {mean_iat}"

    def test_different_lambdas_produce_different_means(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result_slow = gen(5000, 1.0, num_elements_in_ts=30, rng=rng1)
        result_fast = gen(5000, 10.0, num_elements_in_ts=30, rng=rng2)
        assert result_slow.mean() > result_fast.mean(), "Higher λ should produce smaller mean IATs"

    def test_rng_reproducibility(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        r1 = gen(10, 2.0, num_elements_in_ts=5, rng=rng1)
        r2 = gen(10, 2.0, num_elements_in_ts=5, rng=rng2)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Time-truncated mode
# ---------------------------------------------------------------------------
class TestHpGenTimeTruncated:
    def test_cumulative_times_below_max(self):
        rng = np.random.default_rng(42)
        T = 10.0
        result = gen(100, 2.0, time_series_max_time=T, rng=rng)
        cum_times = np.cumsum(result, axis=1)
        # All non-zero cumulative times must be < T
        valid = result[:, :, 0] > 0
        assert (cum_times[valid] < T).all(), "All valid cumulative times must be < T"

    def test_output_has_three_dims(self):
        rng = np.random.default_rng(42)
        result = gen(10, 1.0, time_series_max_time=5.0, rng=rng)
        assert result.ndim == 3
        assert result.shape[0] == 10
        assert result.shape[2] == 1

    def test_zero_padding_after_truncation(self):
        """After truncation, trailing positions should be 0.0."""
        rng = np.random.default_rng(42)
        result = gen(50, 1.0, time_series_max_time=3.0, rng=rng)
        for i in range(result.shape[0]):
            seq = result[i, :, 0]
            first_zero = np.argmax(seq == 0)
            if first_zero > 0 or seq[0] == 0:
                assert (seq[first_zero:] == 0).all(), f"Seq {i}: non-zero after first zero"


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------
class TestHpGenMarks:
    def test_marks_returned_when_num_marks_gt_1(self):
        rng = np.random.default_rng(42)
        result = gen(10, 1.0, num_elements_in_ts=5, num_marks=3, rng=rng)
        assert isinstance(result, tuple), "Should return (inter_times, marks) tuple"
        inter_times, marks = result
        assert inter_times.shape == (10, 5, 1)
        assert marks.shape == (10, 5, 1)

    def test_no_marks_when_num_marks_1(self):
        rng = np.random.default_rng(42)
        result = gen(10, 1.0, num_elements_in_ts=5, num_marks=1, rng=rng)
        assert isinstance(result, np.ndarray), "num_marks=1 should return array, not tuple"

    def test_marks_in_valid_range(self):
        rng = np.random.default_rng(42)
        K = 4
        _, marks = gen(100, 1.0, num_elements_in_ts=20, num_marks=K, rng=rng)
        assert marks.min() >= 0
        assert marks.max() < K

    def test_mark_distribution_follows_probs(self):
        """With enough samples, empirical mark distribution should match specified probs."""
        rng = np.random.default_rng(0)
        probs = np.array([0.7, 0.2, 0.1])
        _, marks = gen(5000, 2.0, num_elements_in_ts=50, num_marks=3, mark_probs=probs, rng=rng)
        counts = np.bincount(marks.ravel(), minlength=3)
        empirical = counts / counts.sum()
        np.testing.assert_allclose(empirical, probs, atol=0.02)

    def test_marks_with_time_truncation(self):
        rng = np.random.default_rng(42)
        result = gen(50, 2.0, time_series_max_time=5.0, num_marks=2, rng=rng)
        assert isinstance(result, tuple)
        inter_times, marks = result
        assert inter_times.shape[0] == marks.shape[0] == 50
        assert inter_times.shape[1] == marks.shape[1], "inter_times and marks must have same length"
        assert marks.dtype == np.int64

    def test_marks_uniform_when_no_probs_specified(self):
        rng = np.random.default_rng(0)
        K = 3
        _, marks = gen(5000, 2.0, num_elements_in_ts=50, num_marks=K, rng=rng)
        counts = np.bincount(marks.ravel(), minlength=K)
        empirical = counts / counts.sum()
        expected = np.ones(K) / K
        np.testing.assert_allclose(empirical, expected, atol=0.02)
