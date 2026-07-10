from unittest import TestCase

import torch

from src.data_transformations.expscaler import ExpScaler, ScalingStrategy


class TestExpScalerRoundTrip(TestCase):
    def test_round_trip_no_scaling(self):
        """unscale(forward(x)) ≈ x for NO_SCALING strategy."""
        x = torch.rand(4, 10, 3) + 0.1  # positive values
        scaler = ExpScaler(fitted_data=x, scaling_strategy=ScalingStrategy.NO_SCALING)
        reconstructed = scaler.unscale(scaler(x))
        torch.testing.assert_close(reconstructed, x, rtol=1e-5, atol=1e-5)

    def test_round_trip_naive_scaling(self):
        """unscale(forward(x)) ≈ x for NAIVE scaling strategy."""
        x = torch.rand(4, 10, 3) + 0.1
        scaler = ExpScaler(fitted_data=x, scaling_strategy=ScalingStrategy.NAIVE)
        reconstructed = scaler.unscale(scaler(x))
        torch.testing.assert_close(reconstructed, x, rtol=1e-4, atol=1e-4)

    def test_round_trip_with_shift(self):
        """Round-trip holds when shift_param > 0, including at x=0."""
        x = torch.rand(4, 10, 2)  # may contain values near 0
        scaler = ExpScaler(fitted_data=x, shift_param=1.0, scaling_strategy=ScalingStrategy.NO_SCALING)
        reconstructed = scaler.unscale(scaler(x))
        torch.testing.assert_close(reconstructed, x, rtol=1e-5, atol=1e-5)

    def test_round_trip_variable_lengths(self):
        """Round-trip holds when sequences have different lengths."""
        x = torch.rand(4, 8, 2) + 0.1
        lengths = torch.tensor([8, 5, 3, 7])
        scaler = ExpScaler(fitted_data=x, lengths=lengths, scaling_strategy=ScalingStrategy.NAIVE)
        reconstructed = scaler.unscale(scaler(x))
        torch.testing.assert_close(reconstructed, x, rtol=1e-4, atol=1e-4)


class TestExpScalerConcentrationFactor(TestCase):
    def test_forward_divides_by_concentration_factor(self):
        """forward output equals log(x) / concentration_factor (no standard scaler)."""
        x = torch.rand(2, 5, 1) + 0.1
        factor = 3.0
        scaler = ExpScaler(fitted_data=x, concentration_factor=factor, scaling_strategy=ScalingStrategy.NO_SCALING)
        expected = torch.log(x) / factor
        torch.testing.assert_close(scaler(x), expected, rtol=1e-5, atol=1e-5)

    def test_unscale_multiplies_by_concentration_factor(self):
        """unscale reverses the concentration division."""
        x = torch.rand(2, 5, 1) + 0.1
        factor = 3.0
        scaler = ExpScaler(fitted_data=x, concentration_factor=factor, scaling_strategy=ScalingStrategy.NO_SCALING)
        scaled = scaler(x)
        reconstructed = scaler.unscale(scaled)
        torch.testing.assert_close(reconstructed, x, rtol=1e-5, atol=1e-5)


class TestExpScalerShiftParam(TestCase):
    def test_shift_prevents_log_negative_inf_at_zero(self):
        """With shift_param=1, log(0 + 1) = 0, no -inf."""
        x = torch.zeros(2, 4, 1)
        scaler = ExpScaler(fitted_data=x + 0.1, shift_param=1.0, scaling_strategy=ScalingStrategy.NO_SCALING)
        result = scaler(x)
        self.assertFalse(torch.any(torch.isinf(result)).item())
        self.assertFalse(torch.any(torch.isnan(result)).item())

    def test_negative_shift_param_raises(self):
        """Negative shift_param should raise an AssertionError."""
        x = torch.rand(2, 4, 1) + 0.1
        with self.assertRaises(AssertionError):
            ExpScaler(fitted_data=x, shift_param=-0.1, scaling_strategy=ScalingStrategy.NO_SCALING)


class TestExpScalerInputValidation(TestCase):
    def test_non_3d_input_raises(self):
        """Passing 2D fitted_data should raise an AssertionError."""
        x = torch.rand(4, 10)
        with self.assertRaises(AssertionError):
            ExpScaler(fitted_data=x)

    def test_mismatched_lengths_raises(self):
        """lengths length must match batch size of fitted_data."""
        x = torch.rand(4, 10, 2) + 0.1
        lengths = torch.tensor([5, 5])  # wrong: only 2 entries for batch of 4
        with self.assertRaises(AssertionError):
            ExpScaler(fitted_data=x, lengths=lengths)
