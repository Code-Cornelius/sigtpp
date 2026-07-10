import torch

from src.utils.get_index import (
    index_first_zero_torch,
    index_before_first_zero_torch,
    index_first_val_gr_torch,
    set_neg_indices2max_index_if_cdt,
    set_zero_len2len,
)


class TestIndexFirstZeroTorch:
    def test_basic(self):
        arr = torch.tensor([[[5.0], [3.0], [0.0], [0.0]]])
        assert index_first_zero_torch(arr).item() == 2

    def test_no_zero_returns_0(self):
        # Documented pathological case: no zero -> returns 0
        arr = torch.tensor([[[1.0], [2.0], [3.0]]])
        assert index_first_zero_torch(arr).item() == 0

    def test_first_value_zero_returns_0(self):
        # Pathological case: first value is zero -> argmax returns 0
        arr = torch.tensor([[[0.0], [1.0], [2.0]]])
        assert index_first_zero_torch(arr).item() == 0

    def test_batch(self):
        arr = torch.tensor(
            [
                [[5.0], [3.0], [0.0], [0.0]],
                [[1.0], [0.0], [0.0], [0.0]],
            ]
        )
        result = index_first_zero_torch(arr)
        assert result[0].item() == 2
        assert result[1].item() == 1

    def test_last_element_zero(self):
        arr = torch.tensor([[[3.0], [2.0], [1.0], [0.0]]])
        assert index_first_zero_torch(arr).item() == 3


class TestIndexBeforeFirstZeroTorch:
    def test_basic(self):
        arr = torch.tensor([[[5.0], [3.0], [0.0], [0.0]]])
        assert index_before_first_zero_torch(arr).item() == 1

    def test_no_zero_returns_minus_one(self):
        # No zero: index_first_zero returns 0, so "before" is -1
        arr = torch.tensor([[[1.0], [2.0], [3.0]]])
        assert index_before_first_zero_torch(arr).item() == -1

    def test_batch(self):
        arr = torch.tensor(
            [
                [[5.0], [3.0], [0.0], [0.0]],
                [[1.0], [0.0], [0.0], [0.0]],
            ]
        )
        result = index_before_first_zero_torch(arr)
        assert result[0].item() == 1
        assert result[1].item() == 0


class TestIndexFirstValGrTorch:
    def test_basic(self):
        arr = torch.tensor([[[1.0], [2.0], [5.0], [8.0]]])
        assert index_first_val_gr_torch(arr, 4.0).item() == 2

    def test_no_match_returns_0(self):
        arr = torch.tensor([[[1.0], [2.0], [3.0]]])
        assert index_first_val_gr_torch(arr, 10.0).item() == 0

    def test_first_element_matches_returns_0(self):
        arr = torch.tensor([[[10.0], [1.0], [2.0]]])
        assert index_first_val_gr_torch(arr, 5.0).item() == 0

    def test_batch(self):
        arr = torch.tensor(
            [
                [[1.0], [2.0], [5.0]],
                [[0.0], [6.0], [3.0]],
            ]
        )
        result = index_first_val_gr_torch(arr, 4.0)
        assert result[0].item() == 2
        assert result[1].item() == 1


class TestSetNegIndices2MaxIndexIfCdt:
    def test_condition_true_sets_to_max(self):
        indices = torch.tensor([-1, 2])
        cdt = torch.tensor([True, False])
        result = set_neg_indices2max_index_if_cdt(indices, max_index=5, cdt=cdt)
        assert result[0].item() == 5
        assert result[1].item() == 2

    def test_condition_false_sets_neg_to_zero(self):
        indices = torch.tensor([-1, -1])
        cdt = torch.tensor([False, False])
        result = set_neg_indices2max_index_if_cdt(indices, max_index=5, cdt=cdt)
        assert result[0].item() == 0
        assert result[1].item() == 0

    def test_mixed(self):
        indices = torch.tensor([-1, -1, 3])
        cdt = torch.tensor([True, False, False])
        result = set_neg_indices2max_index_if_cdt(indices, max_index=10, cdt=cdt)
        assert result[0].item() == 10
        assert result[1].item() == 0
        assert result[2].item() == 3

    def test_no_negatives_unchanged(self):
        indices = torch.tensor([2, 4, 6])
        cdt = torch.tensor([True, True, True])
        result = set_neg_indices2max_index_if_cdt(indices, max_index=9, cdt=cdt)
        assert result.tolist() == [2, 4, 6]


class TestSetZeroLen2Len:
    def test_zeros_replaced(self):
        indices = torch.tensor([0, 3, 0, 5])
        result = set_zero_len2len(indices, len=10)
        assert result[0].item() == 10
        assert result[1].item() == 3
        assert result[2].item() == 10
        assert result[3].item() == 5

    def test_no_zeros_unchanged(self):
        indices = torch.tensor([1, 2, 3])
        result = set_zero_len2len(indices, len=99)
        assert result.tolist() == [1, 2, 3]
