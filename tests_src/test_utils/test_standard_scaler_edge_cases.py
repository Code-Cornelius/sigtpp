"""Edge-case tests for StandardScaler (src/data_transformations/standardscaler.py).

Complements the existing test_scaling.py which covers the basic functionality.
These tests focus on input validation, scalar parameters, constant dimensions,
and round-trip invertibility.
"""

import pytest
import torch

from src.data_transformations.standardscaler import StandardScaler, interleave_tensors


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestStandardScalerValidation:
    def test_nan_in_means_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            StandardScaler(means=torch.tensor([float("nan"), 1.0]), stds=torch.tensor([1.0, 1.0]))

    def test_nan_in_stds_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            StandardScaler(means=torch.tensor([0.0, 0.0]), stds=torch.tensor([1.0, float("nan")]))

    def test_inf_in_means_raises(self):
        with pytest.raises(ValueError, match="Infinite"):
            StandardScaler(means=torch.tensor([float("inf"), 0.0]), stds=torch.tensor([1.0, 1.0]))

    def test_inf_in_stds_raises(self):
        with pytest.raises(ValueError, match="Infinite"):
            StandardScaler(means=torch.tensor([0.0]), stds=torch.tensor([float("-inf")]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            StandardScaler(means=torch.tensor([0.0, 0.0]), stds=torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# Scalar parameters
# ---------------------------------------------------------------------------
class TestStandardScalerScalar:
    def test_scalar_params_accepted(self):
        scaler = StandardScaler(means=torch.tensor(2.0), stds=torch.tensor(3.0))
        x = torch.tensor([[[5.0]]])
        result = scaler(x)
        expected = (5.0 - 2.0) / 3.0
        assert torch.isclose(result.squeeze(), torch.tensor(expected))

    def test_scalar_inverse(self):
        scaler = StandardScaler(means=torch.tensor(2.0), stds=torch.tensor(3.0))
        x = torch.tensor([[[1.0]]])
        result = scaler.inverse_transform(x)
        expected = 1.0 * 3.0 + 2.0
        assert torch.isclose(result.squeeze(), torch.tensor(expected))


# ---------------------------------------------------------------------------
# Constant dimension handling
# ---------------------------------------------------------------------------
class TestStandardScalerConstantDim:
    def test_zero_std_set_to_one(self):
        """When std < 1e-8, it should be replaced with 1.0 to avoid div-by-zero."""
        scaler = StandardScaler(means=torch.tensor([5.0, 0.0]), stds=torch.tensor([0.0, 2.0]))
        assert scaler.std_paths[0].item() == 1.0
        assert scaler.std_paths[1].item() == 2.0

    def test_very_small_std_set_to_one(self):
        scaler = StandardScaler(means=torch.tensor([0.0]), stds=torch.tensor([1e-10]))
        assert scaler.std_paths[0].item() == 1.0

    def test_normal_std_preserved(self):
        scaler = StandardScaler(means=torch.tensor([0.0]), stds=torch.tensor([0.5]))
        assert scaler.std_paths[0].item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Round-trip invertibility
# ---------------------------------------------------------------------------
class TestStandardScalerRoundTrip:
    def test_round_trip_1d(self):
        torch.manual_seed(42)
        scaler = StandardScaler(means=torch.tensor([3.0]), stds=torch.tensor([2.0]))
        x = torch.randn(5, 10, 1)
        reconstructed = scaler.inverse_transform(scaler(x))
        assert torch.allclose(x, reconstructed, atol=1e-6)

    def test_round_trip_multidim(self):
        torch.manual_seed(42)
        means = torch.tensor([1.0, -2.0, 0.5])
        stds = torch.tensor([3.0, 0.1, 7.0])
        scaler = StandardScaler(means=means, stds=stds)
        x = torch.randn(8, 20, 3)
        reconstructed = scaler.inverse_transform(scaler(x))
        assert torch.allclose(x, reconstructed, atol=1e-5)

    def test_forward_then_inverse_identity(self):
        """Verify (x * std + mean) after ((x - mean) / std) = x."""
        torch.manual_seed(0)
        scaler = StandardScaler(means=torch.tensor([10.0]), stds=torch.tensor([5.0]))
        x = torch.tensor([[[25.0], [15.0], [10.0]]])
        scaled = scaler(x)
        assert torch.isclose(scaled[0, 0, 0], torch.tensor(3.0))  # (25-10)/5 = 3
        assert torch.isclose(scaled[0, 1, 0], torch.tensor(1.0))  # (15-10)/5 = 1
        assert torch.isclose(scaled[0, 2, 0], torch.tensor(0.0))  # (10-10)/5 = 0
        back = scaler.inverse_transform(scaled)
        assert torch.allclose(x, back, atol=1e-6)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------
class TestStandardScalerRepr:
    def test_repr_contains_class_name(self):
        scaler = StandardScaler(means=torch.tensor([0.0]), stds=torch.tensor([1.0]))
        assert "StandardScaler" in repr(scaler)


# ---------------------------------------------------------------------------
# interleave_tensors utility
# ---------------------------------------------------------------------------
class TestInterleaveTensors:
    def test_basic(self):
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([4.0, 5.0, 6.0])
        result = interleave_tensors(v1, v2)
        expected = [
            torch.tensor(1.0),
            torch.tensor(4.0),
            torch.tensor(2.0),
            torch.tensor(5.0),
            torch.tensor(3.0),
            torch.tensor(6.0),
        ]
        for r, e in zip(result, expected):
            assert torch.isclose(r, e)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            interleave_tensors(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))
