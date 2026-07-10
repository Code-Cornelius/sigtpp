import logging
from unittest import TestCase

import numpy as np
import torch

logger = logging.getLogger(__name__)

from src.utils.fix_seq_ends import (
    set_seq_to_cst_val_when_zero_torch,
    get_masked_array_on_lengths,
)


class TestSetSeqToCstValWhenZero(TestCase):

    @staticmethod
    def call_method_and_assert_torch(input_to_change, expected_result, input_for_mask=None):
        input_to_change = torch.tensor(input_to_change.transpose(1, 2, 0))
        expected_result = torch.tensor(expected_result.transpose(1, 2, 0)).contiguous()

        if input_for_mask is not None:
            input_for_mask = torch.tensor(input_for_mask.transpose(1, 2, 0))
            output_array, _ = set_seq_to_cst_val_when_zero_torch(input_to_change, input_for_mask)
        else:
            output_array, _ = set_seq_to_cst_val_when_zero_torch(input_to_change)

        logger.debug("TORCH Input %s", input_to_change.numpy().transpose(2, 0, 1))
        logger.debug("TORCH Result %s", output_array.numpy().transpose(2, 0, 1))
        logger.debug("TORCH Expected %s", expected_result.numpy().transpose(2, 0, 1))

        # We use the numpy function because the torch function (below) checks for strides
        # which does not work because of the transpose at the first line of the method.
        np.testing.assert_allclose(output_array, expected_result)
        # torch.testing.assert_close(output_array.clone(), expected_result.clone(), equal_nan=True)

    def test_set_seq_basic_case_with_zeros(self):
        test_input = np.array([[[5.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]])
        expected_result = np.array([[[5.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_set_seq_zeros_after_one_value(self):
        test_input = np.array([[[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        expected_result = np.array([[[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_set_seq_no_zeros(self):
        test_input = np.array([[[5.0, 1.0, 2.0, 3.0, 4.0, 1.0, 3.0, 2.0, 10.0]]])
        expected_result = np.array([[[5.0, 1.0, 2.0, 3.0, 4.0, 1.0, 3.0, 2.0, 10.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_set_seq_with_negative_values(self):
        test_input = np.array([[[3.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        expected_result = np.array([[[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_set_seq_starting_with_zero(self):
        # Choice between keep the sequence constant at first value or keep the sequence as it was.
        test_input = np.array([[[0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 3.0, 1.0, 4.0]]])
        expected_result = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_set_seq_starting_with_negative_value_propagates_constant(self):
        # Choice between keep the sequence constant at first value or keep the sequence as it was.
        test_input = np.array([[[-5.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 1.0, 2.0]]])
        expected_result = np.array([[[-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_two_dims_alternative(self):
        test_input = np.array([[[5.0, 1.0, 0.0]], [[1.0, 2.0, 0.0]]])

        expected_result = np.array([[[5.0, 1.0, 1.0]], [[1.0, 2.0, 2.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_two_dim_with_zeros(self):
        test_input = np.array([[[3.0, 0.0, 0.0]], [[3.0, 2.0, 0.0]]])

        expected_result = np.array([[[3.0, 3.0, 3.0]], [[3.0, 3.0, 3.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_two_dim(self):
        test_input = np.array([[[5.0, 0.0, 2.0]], [[5.0, 1.0, 2.0]]])

        expected_result = np.array([[[5.0, 5.0, 5.0]], [[5.0, 5.0, 5.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_three_dim(self):
        test_input = np.array([[[5.0, 0.0, 2.0]], [[1.0, 1.0, 2.0]], [[0.0, 5.0, 2.0]]])

        expected_result = np.array([[[5.0, 5.0, 5.0]], [[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_single_element_array(self):
        test_input = np.array([[[5.0]]])
        expected_result = np.array([[[5.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_arrays_with_inf_and_nan(self):
        test_input = np.array([[[np.nan, np.inf, -np.inf], [1.0, 2.0, 3.0]]])
        expected_result = np.array([[[np.nan, np.inf, np.inf], [1.0, 2.0, 3.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(test_input, expected_result, None)

    def test_set_seq_different_arr_for_mask(self):
        input_to_change = np.array([[[5.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]])
        input_for_mask = np.array([[[5.0, -1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]]])
        expected_result = np.array([[[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]]])

        TestSetSeqToCstValWhenZero.call_method_and_assert_torch(input_to_change, expected_result, input_for_mask)

    def test_get_mask_np_lengths(self):
        array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        lengths = np.array([1, 2, 1, 3])

        masked_array = get_masked_array_on_lengths(array, lengths)
        expected_mask = np.array(
            [[False, True, True], [False, False, True], [False, True, True], [False, False, False]]
        )
        np.testing.assert_allclose(masked_array.mask, expected_mask, verbose=True)
