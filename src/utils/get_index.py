import typing

import numpy as np
import numpy.typing
import torch

TOL_DIST_ZERO = 1e-10

"""
Two pathological cases do not return the expected result.
If the sequence does not have a zero, the index found is 0.
If the sequence starts with a value outside of the bounds (matching the condition), the index found is 0 as well.
The first case is handled by setting the indices at 0 to the length, representing that the sequence does not terminate.

For the second case, if the sequence is out of bound, we would like the index to stay at 0 
(in opposition to the never ending sequence). In the setting of signatures,
 this would mean that the sequence will be fixed as a constant hence making its signature zero.
This also means we would not penalise the model too much for that generated sequence.
"""


def index_first_zero(arr: np.typing.ArrayLike) -> np.typing.ArrayLike:
    # numpy version of below.
    return index_first_zero_torch(torch.tensor(arr)).numpy()


def index_before_first_zero_torch(arr: torch.Tensor) -> torch.Tensor:
    # Equivalent to index_first_zero_torch - 1.
    return index_before_first_val_gr_torch(-arr, -TOL_DIST_ZERO)


def index_first_zero_torch(arr: torch.Tensor) -> torch.Tensor:
    """
    Given an array (N,L,D), return the index of the first zero value in each row (second axis).
    If there are none, returns -1.
    If the first value is zero, returns -1.
    """
    return index_first_val_gr_torch(-arr, -TOL_DIST_ZERO)


def index_before_first_val_gr_torch(arr: torch.Tensor, value: float) -> torch.Tensor:
    # Return the first index where the value is greater than the given value, along the second axis.
    return index_first_val_gr_torch(arr, value) - 1


def index_first_val_gr_torch(arr: torch.Tensor, value: float) -> torch.Tensor:
    """
    Return the first index where the value is greater than the given value, along the 2nd dimension and 1st coordinate.

    Args:
        arr: Tensor of shape (N, L, D).
        value: Used in the condition `arr > value`.

    Returns:
        Tensor of shape (N,) with the first index per batch entry where `arr > value`.
        Returns 0 if no value satisfies the condition or if the first value satisfies it.
    """
    return torch.argmax((arr > value).to(dtype=torch.int), dim=1)[:, 0]


def set_neg_indices2max_index(
    indices: typing.Union[torch.Tensor, np.typing.ArrayLike], max_index: int
) -> typing.Union[torch.Tensor, np.typing.ArrayLike]:
    indices[indices < 0] = max_index
    return indices


def set_zero_len2len(
    indices: typing.Union[torch.Tensor, np.typing.ArrayLike], len: int
) -> typing.Union[torch.Tensor, np.typing.ArrayLike]:
    # ~indices does not work.
    indices[indices == 0] = len
    return indices


def set_neg_indices2max_index_if_cdt(indices: torch.Tensor, max_index: int, cdt: torch.Tensor) -> torch.Tensor:
    # Change values when the sequence's starting value is smaller than value_min and the index is -1.
    # (arr[:, 0, 0] <= value_min)
    indices[(indices < 0) & cdt] = max_index
    indices[(indices < 0)] = 0
    return indices
