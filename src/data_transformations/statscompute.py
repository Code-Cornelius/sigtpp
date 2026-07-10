import typing

import numpy as np
import torch


def variable_len_standard_stats(
    paths: torch.Tensor,
    lengths: torch.Tensor,
    compute_std: bool = True,
    ignore_infs: bool = False,
) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]:
    """
    Computes the mean and optionally the standard deviation of paths with variable lengths.

    Args:
        paths (torch.Tensor): A 3D tensor of shape `(N, L, D)` representing the input paths, where:
                              - `N` is the number of samples (batch size),
                              - `L` is the sequence length (can vary for each sample),
                              - `D` is the number of features.
        lengths (torch.Tensor): A 1D tensor of shape `(N,)` representing the length of each path (i.e., number of valid time steps).
        compute_std (bool, optional): Whether to compute the standard deviation in addition to the mean. Defaults to `True`.
        ignore_infs (bool, optional): If `True`, ignores infinities (i.e., treats them as NaNs). Defaults to `False`.

    Returns:
        typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]:
            - A tensor of the mean values of the paths across valid time steps.
            - A tensor of the standard deviations if `compute_std` is `True`, otherwise `None`.
    """
    assert len(paths.shape) == 3, f"Expected `paths` to be 3D, but got shape {list(paths.shape)}."
    assert len(lengths.shape) == 1, f"Expected `lengths` to be 1D, but got shape {list(lengths.shape)}."
    assert paths.shape[0] == lengths.shape[0], (
        f"Batch size mismatch: `paths` has shape {list(paths.shape)}, "
        f"but `lengths` has shape {list(lengths.shape)}."
    )

    # Create a mask to ignore time steps beyond the valid length for each sample.
    valid_length_mask = (torch.arange(paths.shape[1])[None, :] < lengths[:, None]).unsqueeze(-1)  # Shape: (N, L, 1)

    # If requested, ignore infinities in the input data.
    if ignore_infs:
        finite_mask = torch.isfinite(paths)
        mask = valid_length_mask & finite_mask
    else:
        mask = valid_length_mask

    # Calculate the masked mean across valid time steps.
    masked_paths = paths * mask
    total_paths = masked_paths.nansum(dim=(0, 1))  # Sum over batch and sequence length dimensions.
    total_lens = mask.sum(dim=(0, 1))  # Count valid (non-masked) values.
    safe_total_lens = total_lens.clamp(min=1)
    mean_paths = total_paths / safe_total_lens

    if not compute_std:
        return mean_paths, None

    # Calculate the masked standard deviation across valid time steps.
    std_paths = torch.sqrt((torch.pow(paths - mean_paths, 2) * mask).nansum(dim=(0, 1)) / safe_total_lens)

    return mean_paths, std_paths


def nanmean(
    tensor: torch.Tensor, dim: typing.Optional[typing.Union[int, typing.Tuple[int, ...]]] = None
) -> torch.Tensor:
    """
    Computes the mean of a tensor along the specified dimension(s), ignoring NaN values.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int, tuple, optional): The dimension(s) along which to compute the mean.
                                    If None, the mean is computed over all dimensions.

    Returns:
        torch.Tensor: The mean of the tensor along the specified dimension(s), ignoring NaN values.
    """
    # Count the number of non-NaN elements along the given dimension(s)
    if dim is None:
        # Flatten the tensor to compute the mean over all elements
        num_non_nan = torch.sum(~torch.isnan(tensor))
        # Compute the sum of the elements, ignoring NaN values
        nansum = torch.nansum(tensor)
    else:
        num_non_nan = torch.sum(~torch.isnan(tensor), dim=dim)
        nansum = torch.nansum(tensor, dim=dim)

    # Compute the mean by dividing the sum by the number of non-NaN elements.
    # Clamp required when dim is not None, can't use max.
    mean = nansum / torch.clamp(num_non_nan, min=1)
    return mean


def nanstd(
    tensor: torch.Tensor, dim: typing.Optional[typing.Union[int, typing.Tuple[int, ...]]] = None
) -> torch.Tensor:
    """
    Computes the standard deviation of a tensor along the specified dimension(s), ignoring NaN values.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int, tuple, optional): The dimension(s) along which to compute the standard deviation.
                                    If None, the std is computed over all dimensions.

    Returns:
        torch.Tensor: The standard deviation of the tensor along the specified dimension(s), ignoring NaN values.
    """
    # Compute the mean along the specified dimension(s), ignoring NaNs
    mean = nanmean(tensor, dim=dim)

    if dim is not None:
        # Ensure `mean` is broadcastable to `tensor` by unsqueezing appropriate dimensions
        if isinstance(dim, int):
            dim = (dim,)
        for d in sorted(dim):
            mean = mean.unsqueeze(d)

    # Compute the squared differences, ignoring NaNs
    squared_diff = (tensor - mean) ** 2

    if dim is None:
        # Compute over all elements
        num_non_nan = torch.sum(~torch.isnan(tensor))
        # Clamp required when dim is not None, can't use max.
        variance = torch.nansum(squared_diff) / torch.clamp(num_non_nan - 1, min=1)
    else:
        # Compute along specified dimension(s)
        num_non_nan = torch.sum(~torch.isnan(tensor), dim=dim)
        variance = torch.nansum(squared_diff, dim=dim) / torch.clamp(num_non_nan - 1, min=1)

    # Compute the standard deviation
    std = torch.sqrt(variance)
    return std


def nanmax(
    tensor: torch.Tensor, dim: typing.Optional[typing.Union[int, typing.Tuple[int, ...]]] = None
) -> torch.Tensor:
    """
    Computes the maximum of a tensor along the specified dimension(s), ignoring NaN values.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int, tuple, optional): The dimension(s) along which to compute the max.
                                    If None, the max is computed over all elements.

    Returns:
        torch.Tensor: The max of the tensor along the specified dimension(s), ignoring NaN values.
    """
    # Replace NaNs with -inf so they are never selected as the maximum
    tensor_no_nan = tensor.clone()
    tensor_no_nan[torch.isnan(tensor_no_nan)] = float("-inf")

    if dim is None:
        return tensor_no_nan.max()
    elif isinstance(dim, int):
        return tensor_no_nan.max(dim=dim).values
    else:
        result = tensor_no_nan
        for d in sorted(dim, reverse=True):
            result = result.max(dim=d).values
        return result


def nanmedian_numpy(tensor: torch.Tensor, dim: typing.Optional[int] = None) -> torch.Tensor:
    """
    Computes the median of a tensor along the specified dimension(s), ignoring NaN values.
    Uses NumPy's nanmedian internally (runs on CPU, breaks autograd).

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int, optional): The dimension along which to compute the median.
                             If None, computes over all elements.

    Returns:
        torch.Tensor: The median, ignoring NaNs, as a torch.Tensor on the original device.
    """
    # Move to CPU and convert to NumPy
    arr = tensor.detach().cpu().numpy()

    # Compute NaN-aware median with NumPy
    median_np = np.nanmedian(arr, axis=dim)

    # Convert back to torch.Tensor on original device/dtype
    return torch.as_tensor(median_np, dtype=tensor.dtype, device=tensor.device)
