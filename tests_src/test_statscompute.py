import unittest

import torch

from src.data_transformations.statscompute import nanmean, nanstd


# Assumes nanmean and nanstd are already defined/imported in this module.
# from your_module import nanmean, nanstd


class TestNanMeanAndNanStd(unittest.TestCase):
    def test_tensor_with_nans(self):
        # Tensor with NaNs
        tensor_with_nan = torch.tensor([[1.0, 2.0, float("nan")], [4.0, float("nan"), 6.0]])

        # Test nanmean and nanstd on tensor with NaNs
        mean_with_nan = nanmean(tensor_with_nan, (0, 1))
        std_with_nan = nanstd(tensor_with_nan, (0, 1))
        print("Tensor with NaNs:")
        print(tensor_with_nan)
        print(f"Computed mean (with NaNs): {mean_with_nan}")
        print(f"Computed std (with NaNs): {std_with_nan}")

        # Theoretical calculations (ignoring NaNs)
        valid_values_with_nan = [1.0, 2.0, 4.0, 6.0]
        theoretical_mean_with_nan = sum(valid_values_with_nan) / len(valid_values_with_nan)
        theoretical_variance_with_nan = sum((x - theoretical_mean_with_nan) ** 2 for x in valid_values_with_nan) / (
            len(valid_values_with_nan) - 1
        )
        theoretical_std_with_nan = theoretical_variance_with_nan**0.5
        print(f"Theoretical mean (with NaNs): {theoretical_mean_with_nan}")
        print(f"Theoretical std (with NaNs): {theoretical_std_with_nan}")

        # Optional sanity checks (won't change your printed tests; just validates)
        self.assertTrue(torch.isfinite(mean_with_nan).item() if mean_with_nan.numel() == 1 else True)
        self.assertTrue(torch.isfinite(std_with_nan).item() if std_with_nan.numel() == 1 else True)
        self.assertAlmostEqual(mean_with_nan.item(), theoretical_mean_with_nan, places=6)
        self.assertAlmostEqual(std_with_nan.item(), theoretical_std_with_nan, places=6)

    def test_tensor_without_nans(self):
        # Tensor without NaNs
        tensor_without_nan = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Test nanmean and nanstd on tensor without NaNs
        mean_without_nan = nanmean(tensor_without_nan, (0, 1))
        std_without_nan = nanstd(tensor_without_nan, (0, 1))
        print("\nTensor without NaNs:")
        print(tensor_without_nan)
        print(f"Computed mean (without NaNs): {mean_without_nan}")
        print(f"Computed std (without NaNs): {std_without_nan}")

        # Theoretical calculations for tensor without NaNs
        valid_values_without_nan = tensor_without_nan.flatten().tolist()
        theoretical_mean_without_nan = sum(valid_values_without_nan) / len(valid_values_without_nan)
        theoretical_variance_without_nan = sum(
            (x - theoretical_mean_without_nan) ** 2 for x in valid_values_without_nan
        ) / (len(valid_values_without_nan) - 1)
        theoretical_std_without_nan = theoretical_variance_without_nan**0.5
        print(f"Theoretical mean (without NaNs): {theoretical_mean_without_nan}")
        print(f"Theoretical std (without NaNs): {theoretical_std_without_nan}")

        # Optional validation
        self.assertAlmostEqual(mean_without_nan.item(), theoretical_mean_without_nan, places=6)
        self.assertAlmostEqual(std_without_nan.item(), theoretical_std_without_nan, places=6)

    def test_nanmean_nanstd_output_shapes_along_each_axis(self):
        # Verification for different set of dims:
        tensor_without_nan = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        print("Verification for different set of dims:")
        print(tensor_without_nan.shape)
        print("Computed mean over axis 0: ", nanmean(tensor_without_nan, 0))
        print("Computed mean over axis 1: ", nanmean(tensor_without_nan, 1))
        print("Computed mean over all axes: ", nanmean(tensor_without_nan))
        print("Computed std over axis 0: ", nanstd(tensor_without_nan, 0))
        print("Computed std over axis 1: ", nanstd(tensor_without_nan, 1))
        print("Computed std over all axes: ", nanstd(tensor_without_nan))

        # Optional shape checks
        self.assertEqual(nanmean(tensor_without_nan, 0).shape, torch.Size([3]))
        self.assertEqual(nanmean(tensor_without_nan, 1).shape, torch.Size([2]))
        self.assertEqual(nanstd(tensor_without_nan, 0).shape, torch.Size([3]))
        self.assertEqual(nanstd(tensor_without_nan, 1).shape, torch.Size([2]))


class TestVariableLenStandardStats:

    def test_ignores_nan_padding_in_mean(self):
        """Padding NaNs beyond seq_len must not affect the computed mean."""
        from src.data_transformations.statscompute import variable_len_standard_stats

        # Two sequences: first has 3 real values, second has 2 real values + 1 NaN pad
        data = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [6.0], [float('nan')]]])
        lens = torch.tensor([3, 2])

        mean, std = variable_len_standard_stats(data, lens)

        # Unpadded values: 1,2,3,4,6  → mean = 16/5 = 3.2
        expected_mean = (1.0 + 2.0 + 3.0 + 4.0 + 6.0) / 5
        assert abs(mean.item() - expected_mean) < 1e-5, f"Expected mean {expected_mean}, got {mean.item()}"

    def test_single_sequence_full_length(self):
        """With a single full-length sequence the result matches plain mean/std."""
        from src.data_transformations.statscompute import variable_len_standard_stats

        data = torch.tensor([[[2.0], [4.0], [6.0]]])
        lens = torch.tensor([3])

        mean, std = variable_len_standard_stats(data, lens)

        assert abs(mean.item() - 4.0) < 1e-5

    def test_zero_valid_entries_returns_finite_neutral_statistics(self):
        """An empty feature must not introduce NaNs into scaler construction."""
        from src.data_transformations.statscompute import variable_len_standard_stats

        data = torch.full((2, 3, 1), float("nan"), dtype=torch.float32)
        lens = torch.zeros(2, dtype=torch.long)

        mean, std = variable_len_standard_stats(data, lens)

        assert mean.shape == torch.Size([1])
        assert std.shape == torch.Size([1])
        assert mean.dtype == data.dtype
        assert std.dtype == data.dtype
        assert torch.isclose(mean[0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(std[0], torch.tensor(0.0), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
