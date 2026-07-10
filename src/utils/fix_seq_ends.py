import typing

import numpy as np
import numpy.typing
import torch

from src.utils.get_index import (
    index_before_first_zero_torch,
    index_before_first_val_gr_torch,
    set_neg_indices2max_index_if_cdt,
)


def set_seq_to_cst_val_when_zero_torch(
    arr_to_change: torch.Tensor, arr_for_mask: typing.Optional[torch.Tensor] = None
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    if arr_for_mask is None:
        arr_for_mask = arr_to_change

    # The newly created tensors need to be excluded from the graph.
    with torch.no_grad():
        last_index_seqs = set_neg_indices2max_index_if_cdt(
            index_before_first_zero_torch(arr_for_mask), arr_for_mask.shape[1] - 1, arr_for_mask[:, 0, 0] > 0
        )

    return set_seq_to_cst_val_from_index(arr_to_change, last_index_seqs), last_index_seqs + 1


def set_seq_to_cst_val_gr(arr: torch.Tensor, timeseries_time_max: float) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    See set_seq_to_cst_val_when_zero where we do the same thing for when the sequence reaches cumulatively `timeseries_time_max`.
    The cumsum needs to appear as the last feature.
    """
    # The newly created tensors need to be excluded from the graph.
    with torch.no_grad():
        # Requires last dim to be the cumulative sum.
        last_index_seqs = set_neg_indices2max_index_if_cdt(
            index_before_first_val_gr_torch(arr[:, :, -1:], timeseries_time_max),
            arr.shape[1] - 1,
            arr[:, 0, -1] < timeseries_time_max,
        )
    return set_seq_to_cst_val_from_index(arr, last_index_seqs), last_index_seqs + 1


def set_seq_to_cst_val_from_index(arr: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Replace all subsequent values in a sequence with the last non-zero value once a negative value is encountered.
    We assume that the first <= 0 value in the sequence indicates that all subsequent values are <= 0.
    If a sequence starts with a negative value, all values remain unchanged.
    The condition is taken over `array_for_mask` unless it is None. The operation is in place on `array_to_change`.
    Both arrays are expected to be of the shape (N,L,D).
    The condition is taken over the first entry of the last dimension.

    NOT IN PLACE
    """
    # Replace all subsequent values in a sequence starting at the index indices.
    with torch.no_grad():
        val_used_replace = arr[torch.arange(arr.shape[0]), indices].unsqueeze(1)
    arr = _replace_from_index_with_value_torch(arr, indices, val_used_replace)
    return arr


def _replace_from_index_with_value_torch(
    arr_to_change: torch.Tensor, arr_last_pos_index: torch.Tensor, val_used_replace: torch.Tensor
) -> torch.Tensor:
    """
    Replaces all values strictly after `arr_last_pos_index` with `val_used_replace`.
    Requires arr to have 3 dimensions.

    NOT in place.
    `torch.where(...)` (without an `out=` argument) **allocates a new tensor**. It does not modify `arr_to_change` in place.
    """
    assert len(arr_to_change.shape) == 3, f"Expected a 3D array, but got {list(arr_to_change.shape)}."
    _, seq_len, input_dim = arr_to_change.shape

    # (N, L, 1), torch.where broadcasts to (N, L, D)
    mask_val_to_be_changed = (
        torch.arange(seq_len, device=arr_last_pos_index.device) > arr_last_pos_index.unsqueeze(-1)
    ).unsqueeze(-1)

    # Create a tensor with the same shape as the original array.
    # val_repeated = val_used_replace.repeat(1, seq_len, 1) # not needed
    return torch.where(mask_val_to_be_changed, val_used_replace, arr_to_change)


def set_seq_to_nan_from_index(arr: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    As above but replacing with NaNs for baselines.
    Requires arr to have 3 dimensions.
    """
    return _replace_from_index_with_value_torch(arr, indices, arr.new_full((), float('nan')))


def set_seq_to_zero_from_index(arr: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    As above but replacing with zeros for baselines.
    """
    return _replace_from_index_with_value_torch(arr, indices, arr.new_zeros(()))


def get_masked_array_on_lengths(array: np.typing.ArrayLike, lengths: np.typing.ArrayLike):
    # Mask the array at their length. See test for an example.
    assert len(array.shape) == 2, f"Expected a 2D array, but got {list(array.shape)}."
    assert len(lengths.shape) == 1, f"Expected a 1D array, but got {list(lengths.shape)}."
    assert array.shape[0] == lengths.shape[0], (
        f"Expected the first dimension of array to be equal to the length of lengths, "
        f"but got {array.shape[0]} and {lengths.shape[0]}."
    )
    assert np.all(lengths >= 0), f"Expected lengths to be non-negative, but got {lengths}."

    # Create a range array with the same shape as the input array
    broadcast_lengths = np.arange(1, array.shape[1] + 1)
    return np.ma.masked_where(broadcast_lengths > lengths[:, None], array, copy=True)


def to_cst_val_gr(
    intertimes_to_fix: torch.Tensor, cumtimes_reference: torch.Tensor, timeseries_time_max: float
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    See set_seq_to_cst_val_when_zero where we do the same thing for when the sequence reaches cumulatively `timeseries_time_max`.
    The cumsum needs to appear as the last feature.

    Not in place.
    """
    assert len(intertimes_to_fix.shape) == 3, f"Expected a 3D array, but got {list(intertimes_to_fix.shape)}."
    assert len(cumtimes_reference.shape) == 3, f"Expected a 3D array, but got {list(cumtimes_reference.shape)}."
    assert intertimes_to_fix.shape[0] == cumtimes_reference.shape[0], (
        f"Expected the first dimension of intertimes_to_fix to be equal to the first dimension of cumtimes_reference, "
        f"but got {intertimes_to_fix.shape[0]} and {cumtimes_reference.shape[0]}."
    )

    # The newly created tensors need to be excluded from the graph.
    with torch.no_grad():
        # Requires last dim to be the cumulative sum.
        last_index_seqs = set_neg_indices2max_index_if_cdt(
            index_before_first_val_gr_torch(cumtimes_reference[:, :, -1:], timeseries_time_max),
            cumtimes_reference.shape[1] - 1,
            cumtimes_reference[:, 0, -1] < timeseries_time_max,
        )
    return set_seq_to_cst_val_from_index(intertimes_to_fix, last_index_seqs), last_index_seqs + 1
