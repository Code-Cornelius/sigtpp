"""Tests for the inhomogeneous Poisson process generator (src/generators/ihp.py)."""

import numpy as np
import pytest

from src.generators.ihp import gen


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
class TestIhpGenValidation:
    def test_both_modes_raises(self):
        with pytest.raises(ValueError, match="exactly one|mutually exclusive"):
            gen(10, lambda t: 1.0, 1.0, time_series_max_time=5.0, num_elements_in_ts=5)

    def test_neither_mode_raises(self):
        with pytest.raises(ValueError, match="exactly one|mutually exclusive"):
            gen(10, lambda t: 1.0, 1.0)

    def test_max_val_zero_raises(self):
        with pytest.raises(ValueError, match="max_val_time_function"):
            gen(10, lambda t: 1.0, 0.0, num_elements_in_ts=5)

    def test_max_val_negative_raises(self):
        with pytest.raises(ValueError, match="max_val_time_function"):
            gen(10, lambda t: 1.0, -1.0, num_elements_in_ts=5)

    def test_max_time_zero_raises(self):
        with pytest.raises(ValueError, match="time_series_max_time"):
            gen(10, lambda t: 1.0, 1.0, time_series_max_time=0.0)

    def test_num_elements_zero_raises(self):
        with pytest.raises(ValueError, match="num_elements_in_ts"):
            gen(10, lambda t: 1.0, 1.0, num_elements_in_ts=0)

    def test_num_seq_negative_raises(self):
        with pytest.raises(ValueError, match="num_seq"):
            gen(-1, lambda t: 1.0, 1.0, num_elements_in_ts=5)

    def test_mark_probs_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="mark_probs"):
            gen(5, lambda t: 1.0, 1.0, num_elements_in_ts=3, num_marks=3, mark_probs=np.array([0.5, 0.5]))

    def test_mark_probs_not_sum_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            gen(5, lambda t: 1.0, 1.0, num_elements_in_ts=3, num_marks=2, mark_probs=np.array([0.3, 0.3]))


# ---------------------------------------------------------------------------
# Fixed-length mode
# ---------------------------------------------------------------------------
class TestIhpGenFixedLength:
    def test_output_shape(self):
        np.random.seed(42)
        result = gen(10, lambda t: 2.0, 2.0, num_elements_in_ts=8)
        assert result.shape == (10, 8, 1), f"Expected (10, 8, 1), got {result.shape}"

    def test_inter_arrivals_non_negative(self):
        np.random.seed(42)
        result = gen(20, lambda t: 1.5, 2.0, num_elements_in_ts=5)
        assert (result >= 0).all(), "Inter-arrival times must be non-negative"

    def test_constant_intensity_matches_hp_mean(self):
        """With constant intensity λ, IHP should behave like HP: mean IAT ≈ 1/λ."""
        np.random.seed(0)
        lam = 3.0
        result = gen(2000, lambda t: lam, lam, num_elements_in_ts=30)
        mean_iat = result[result > 0].mean()
        assert abs(mean_iat - 1.0 / lam) < 0.05, f"Expected mean ≈ {1/lam:.3f}, got {mean_iat:.3f}"


# ---------------------------------------------------------------------------
# Time-truncated mode
# ---------------------------------------------------------------------------
class TestIhpGenTimeTruncated:
    def test_output_shape_3d(self):
        np.random.seed(42)
        result = gen(10, lambda t: 1.0, 1.0, time_series_max_time=5.0)
        assert result.ndim == 3
        assert result.shape[0] == 10
        assert result.shape[2] == 1

    def test_cumulative_times_below_max(self):
        np.random.seed(42)
        T = 8.0
        result = gen(50, lambda t: 2.0, 2.0, time_series_max_time=T)
        cum_times = np.cumsum(result, axis=1)
        valid = result[:, :, 0] > 0
        assert (cum_times[valid] < T).all(), "All valid cumulative times must be < T"

    def test_zero_padding_after_truncation(self):
        np.random.seed(42)
        result = gen(30, lambda t: 1.0, 1.0, time_series_max_time=5.0)
        for i in range(result.shape[0]):
            seq = result[i, :, 0]
            first_zero = np.argmax(seq == 0)
            if first_zero > 0 or seq[0] == 0:
                assert (seq[first_zero:] == 0).all(), f"Seq {i}: non-zero values after first zero"


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------
class TestIhpGenMarks:
    def test_marks_returned_when_num_marks_gt_1(self):
        np.random.seed(42)
        rng = np.random.default_rng(42)
        result = gen(5, lambda t: 1.0, 1.0, num_elements_in_ts=4, num_marks=3, rng=rng)
        assert isinstance(result, tuple), "Should return (inter_times, marks) tuple"
        inter_times, marks = result
        assert inter_times.shape[:2] == marks.shape[:2]
        assert marks.dtype == np.int64

    def test_no_marks_when_num_marks_1(self):
        np.random.seed(42)
        result = gen(5, lambda t: 1.0, 1.0, num_elements_in_ts=4, num_marks=1)
        assert isinstance(result, np.ndarray), "num_marks=1 should return array, not tuple"

    def test_marks_in_valid_range(self):
        np.random.seed(42)
        rng = np.random.default_rng(42)
        K = 4
        _, marks = gen(20, lambda t: 1.0, 1.0, num_elements_in_ts=10, num_marks=K, rng=rng)
        assert marks.min() >= 0
        assert marks.max() < K

    def test_marks_with_time_truncation(self):
        np.random.seed(42)
        rng = np.random.default_rng(42)
        result = gen(20, lambda t: 2.0, 2.0, time_series_max_time=5.0, num_marks=2, rng=rng)
        assert isinstance(result, tuple)
        inter_times, marks = result
        assert inter_times.shape[0] == marks.shape[0] == 20
        assert inter_times.shape[1] == marks.shape[1], "inter_times and marks must have same seq length"


# ---------------------------------------------------------------------------
# Lewis thinning correctness
# ---------------------------------------------------------------------------
class TestIhpLewisThinning:
    def test_higher_intensity_produces_more_events(self):
        """Under time-truncated mode, higher intensity should yield more events."""
        np.random.seed(42)
        T = 10.0
        result_low = gen(100, lambda t: 0.5, 0.5, time_series_max_time=T)
        result_high = gen(100, lambda t: 3.0, 3.0, time_series_max_time=T)
        events_low = (result_low[:, :, 0] > 0).sum(axis=1).mean()
        events_high = (result_high[:, :, 0] > 0).sum(axis=1).mean()
        assert events_high > events_low, "Higher intensity must produce more events on average"

    def test_sinusoidal_intensity_produces_valid_output(self):
        """Sinusoidal intensity function should produce valid inter-arrival times."""
        np.random.seed(42)
        import math

        result = gen(20, lambda t: 1.5 + math.sin(t), 2.5, time_series_max_time=10.0)
        assert result.ndim == 3
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# Seeding reproducibility
# ---------------------------------------------------------------------------
class TestIhpSeedingReproducibility:
    def test_same_seed_same_event_times_despite_global_rng_perturbation(self):
        """IHP generation with the same rng seed must produce identical event times
        regardless of global np.random state.  Previously non_homo_lewis_sampling_method
        consumed from np.random.rand instead of the passed rng, so cache filenames that
        included the seed gave a false guarantee of reproducibility."""
        rng1 = np.random.default_rng(42)
        result1 = gen(10, lambda t: 1.0, 1.0, time_series_max_time=5.0, rng=rng1)

        # Perturb global RNG state before the second call.
        np.random.seed(999)
        np.random.rand(10_000)

        rng2 = np.random.default_rng(42)
        result2 = gen(10, lambda t: 1.0, 1.0, time_series_max_time=5.0, rng=rng2)

        np.testing.assert_array_equal(
            result1,
            result2,
            err_msg="IHP event times must be identical for the same rng seed regardless of global RNG state",
        )
