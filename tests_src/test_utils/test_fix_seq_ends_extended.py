import numpy as np
import torch
import pytest
from src.utils.fix_seq_ends import (
    set_seq_to_nan_from_index,
    set_seq_to_zero_from_index,
    set_seq_to_cst_val_from_index,
    get_masked_array_on_lengths,
    to_cst_val_gr,
)


class TestSetSeqToNanFromIndex:
    def test_basic(self):
        arr = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        indices = torch.tensor([1])  # Replace after index 1
        result = set_seq_to_nan_from_index(arr, indices)
        # Positions 0 and 1 should be unchanged, 2 and 3 should be NaN
        assert result[0, 0, 0].item() == 1.0
        assert result[0, 1, 0].item() == 3.0
        assert torch.isnan(result[0, 2, 0])
        assert torch.isnan(result[0, 3, 0])

    def test_not_in_place(self):
        arr = torch.tensor([[[1.0], [2.0], [3.0]]])
        original = arr.clone()
        _ = set_seq_to_nan_from_index(arr, torch.tensor([0]))
        torch.testing.assert_close(arr, original)

    def test_batch(self):
        arr = torch.tensor([
            [[1.0], [2.0], [3.0], [4.0]],
            [[5.0], [6.0], [7.0], [8.0]],
        ])
        indices = torch.tensor([1, 2])
        result = set_seq_to_nan_from_index(arr, indices)
        # First sequence: after index 1
        assert not torch.isnan(result[0, 1, 0])
        assert torch.isnan(result[0, 2, 0])
        # Second sequence: after index 2
        assert not torch.isnan(result[1, 2, 0])
        assert torch.isnan(result[1, 3, 0])


class TestSetSeqToZeroFromIndex:
    def test_basic(self):
        arr = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        indices = torch.tensor([0])  # Replace after index 0
        result = set_seq_to_zero_from_index(arr, indices)
        assert result[0, 0, 0].item() == 1.0
        assert result[0, 1, 0].item() == 0.0
        assert result[0, 2, 0].item() == 0.0

    def test_not_in_place(self):
        arr = torch.tensor([[[1.0], [2.0], [3.0]]])
        original = arr.clone()
        _ = set_seq_to_zero_from_index(arr, torch.tensor([0]))
        torch.testing.assert_close(arr, original)


class TestSetSeqToCstValFromIndex:
    def test_basic(self):
        arr = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        indices = torch.tensor([1])  # Fix at value at index 1 (=2.0) from index 2 onward
        result = set_seq_to_cst_val_from_index(arr, indices)
        assert result[0, 0, 0].item() == 1.0
        assert result[0, 1, 0].item() == 2.0
        assert result[0, 2, 0].item() == 2.0
        assert result[0, 3, 0].item() == 2.0

    def test_not_in_place(self):
        arr = torch.tensor([[[1.0], [2.0], [3.0]]])
        original = arr.clone()
        _ = set_seq_to_cst_val_from_index(arr, torch.tensor([0]))
        torch.testing.assert_close(arr, original)


class TestGetMaskedArrayOnLengths:
    def test_basic(self):
        array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        lengths = np.array([2, 3])
        result = get_masked_array_on_lengths(array, lengths)
        # First row: positions 0,1 valid (length=2), positions 2,3 masked
        assert not result.mask[0, 0]
        assert not result.mask[0, 1]
        assert result.mask[0, 2]
        assert result.mask[0, 3]
        # Second row: positions 0,1,2 valid (length=3), position 3 masked
        assert not result.mask[1, 2]
        assert result.mask[1, 3]

    def test_full_length_nothing_masked(self):
        array = np.array([[1, 2, 3]])
        lengths = np.array([3])
        result = get_masked_array_on_lengths(array, lengths)
        assert not np.any(result.mask)

    def test_zero_length_all_masked(self):
        array = np.array([[1, 2, 3]])
        lengths = np.array([0])
        result = get_masked_array_on_lengths(array, lengths)
        assert np.all(result.mask)

    def test_invalid_shapes_raise(self):
        with pytest.raises(AssertionError):
            get_masked_array_on_lengths(np.array([1, 2, 3]), np.array([2]))
        with pytest.raises(AssertionError):
            get_masked_array_on_lengths(np.array([[1, 2]]), np.array([[2]]))

    def test_copy_not_modify_original(self):
        array = np.array([[10, 20, 30]])
        lengths = np.array([1])
        result = get_masked_array_on_lengths(array, lengths)
        result[0, 0] = 999
        assert array[0, 0] == 10


class TestToCstValGr:
    def test_basic(self):
        """Intertimes should be fixed to constant when cumtimes exceed timeseries_time_max."""
        intertimes = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        cumtimes = torch.tensor([[[1.0], [3.0], [6.0], [10.0], [15.0]]])
        time_max = 5.0
        result, lengths = to_cst_val_gr(intertimes, cumtimes, time_max)
        # First val > 5 in cumtimes is at index 2 (value 6)
        # index_before = 1, so values from index 2 onward should be fixed
        assert result[0, 0, 0].item() == 1.0
        assert result[0, 1, 0].item() == 2.0
        assert result[0, 2, 0].item() == 2.0
        assert result[0, 3, 0].item() == 2.0
        # lengths should be index_before + 1 = 2
        assert lengths[0].item() == 2

    def test_no_cumtime_exceeds_max(self):
        """When no cumtime exceeds the max, the entire sequence stays unchanged."""
        intertimes = torch.tensor([[[1.0], [2.0], [3.0]]])
        cumtimes = torch.tensor([[[1.0], [3.0], [4.0]]])
        time_max = 100.0
        result, lengths = to_cst_val_gr(intertimes, cumtimes, time_max)
        # No value exceeds 100 -> pathological case, index is 0 -> cdt check
        # cumtimes[0,0,-1] = 1.0 < 100 -> cdt True -> set to max_index = 2
        assert lengths[0].item() == 3  # max_index + 1

    def test_first_cumtime_already_exceeds(self):
        """When the first cumtime already exceeds time_max."""
        intertimes = torch.tensor([[[10.0], [2.0], [3.0]]])
        cumtimes = torch.tensor([[[200.0], [300.0], [400.0]]])
        time_max = 5.0
        result, lengths = to_cst_val_gr(intertimes, cumtimes, time_max)
        # cumtimes[0,0,-1] = 200 >= 5 -> cdt is False -> index set to 0
        assert lengths[0].item() == 1

    def test_not_in_place(self):
        intertimes = torch.tensor([[[1.0], [2.0], [3.0]]])
        cumtimes = torch.tensor([[[1.0], [3.0], [6.0]]])
        original = intertimes.clone()
        _ = to_cst_val_gr(intertimes, cumtimes, 2.0)
        torch.testing.assert_close(intertimes, original)
