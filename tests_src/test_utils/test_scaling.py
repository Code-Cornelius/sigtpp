from unittest import TestCase

import numpy as np
import torch

from src.data_transformations.standardscaler import StandardScaler
from src.data_transformations.statscompute import variable_len_standard_stats


class TestVariableLengthScaler(TestCase):
    # Test 1: Simple case with no padding
    # Read as (N, L, D) where each pair is one coordinate, then each block is a path, and the total is the whole batch.

    def test_scaler_simple_case_no_padding(self):
        paths = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)
        lengths = torch.tensor([2, 2])

        mean_paths, std_paths = variable_len_standard_stats(paths, lengths)
        scaler = StandardScaler(means=mean_paths, stds=std_paths)
        expected_mean = torch.tensor([1.0 + 3.0 + 5.0 + 7.0, 2.0 + 4.0 + 6.0 + 8.0]) / 4.0
        expected_std = torch.sqrt(
            torch.tensor([(1.0 + 9.0 + 25.0 + 49.0) / 4.0, (4.0 + 16.0 + 36.0 + 64.0) / 4.0]) - expected_mean**2
        )
        np.testing.assert_allclose(scaler.mean_paths, expected_mean, rtol=10e-5)
        np.testing.assert_allclose(scaler.std_paths, expected_std, rtol=10e-5)

    def test_scaler_case_with_padding(self):
        paths = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], [[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32
        )
        lengths = torch.tensor([2, 1])

        mean_paths, std_paths = variable_len_standard_stats(paths, lengths)
        scaler = StandardScaler(means=mean_paths, stds=std_paths)
        expected_mean = torch.tensor([1.0 + 3.0 + 5.0, 2.0 + 4.0 + 6.0]) / 3.0
        expected_std = torch.sqrt(torch.tensor([1.0 + 9.0 + 25.0, 4.0 + 16.0 + 36.0]) / 3.0 - expected_mean**2)
        np.testing.assert_allclose(scaler.mean_paths, expected_mean, rtol=10e-5)
        np.testing.assert_allclose(scaler.std_paths, expected_std, rtol=10e-5)

    def test_scaler_different_lengths(self):
        paths = torch.tensor(
            [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], [[4.0, 40.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32
        )
        lengths = torch.tensor([3, 1])

        mean_paths, std_paths = variable_len_standard_stats(paths, lengths)
        scaler = StandardScaler(means=mean_paths, stds=std_paths)
        expected_mean = torch.tensor([1.0 + 2.0 + 3.0 + 4.0, 10.0 + 20.0 + 30.0 + 40.0]) / 4.0
        expected_std = torch.sqrt(
            torch.tensor([1.0 + 4.0 + 9.0 + 16.0, 100.0 + 400.0 + 900.0 + 1600.0]) / 4.0 - expected_mean**2
        )
        np.testing.assert_allclose(scaler.mean_paths, expected_mean, rtol=10e-5)
        np.testing.assert_allclose(scaler.std_paths, expected_std, rtol=10e-5)

    def test_scaler_empty_input(self):
        paths = torch.tensor([], dtype=torch.float32).reshape(0, 0, 0)
        lengths = torch.tensor([], dtype=torch.int64)

        mean_paths, std_paths = variable_len_standard_stats(paths, lengths)
        scaler = StandardScaler(means=mean_paths, stds=std_paths)
        self.assertTrue(scaler.mean_paths.numel() == 0)
        self.assertTrue(scaler.std_paths.numel() == 0)

    def test_scaler_single_element_input(self):
        paths = torch.tensor([[[1.0]]], dtype=torch.float32)
        lengths = torch.tensor([1])

        mean_paths, std_paths = variable_len_standard_stats(paths, lengths)
        scaler = StandardScaler(means=mean_paths, stds=std_paths)
        expected_mean = torch.tensor([1.0])
        expected_std = torch.tensor([1.0])  # std is set to 1 if variance is 0
        np.testing.assert_allclose(scaler.mean_paths, expected_mean, rtol=10e-5)
        np.testing.assert_allclose(scaler.std_paths, expected_std, rtol=10e-5)

    def test_zero_variance_handling(self):
        paths = torch.tensor([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]], dtype=torch.float32)
        lengths = torch.tensor([2, 2])

        mean_paths, std_paths = variable_len_standard_stats(paths, lengths)
        scaler = StandardScaler(means=mean_paths, stds=std_paths)
        expected_std = torch.tensor([1.0, 1.0])  # std should be set to 1 due to zero variance
        np.testing.assert_allclose(scaler.std_paths, expected_std, rtol=10e-5)
