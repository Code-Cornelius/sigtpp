from typing import Tuple

import torch

from src.data_transformations.statscompute import nanstd, nanmean


def crosscorr(seq1: torch.Tensor, seq2: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Compute the cross-correlation of two tensors.
    CrossCorrelation(X, Y, k) = sum_{t=1}^{T-k} (X_t+k - mu_X)(Y_t - mu_Y) / ((n-k) * std_X * std_Y)

    Args:
        seq1: Input tensor of shape [N, L, D].
        seq2: Input tensor of shape [N, L, D].
        max_lag: Specifies the number of lags to compute the cross-correlation for.

    Returns: Cross-correlation of seq1 and seq2, shape [max_lag+1, D].

    Idea:
        We could use crosscoef from Torch > 2.0
    """
    return _crosscorr_avg_param(seq1, seq2, max_lag, avg_res_dims=(0, 1))


def crosscorr_nonstationary(seq1: torch.Tensor, seq2: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Compute the cross-correlation of two tensors.
    CrossCorrelation(X, Y, k, t) =  (X_t+k - mu_X)(Y_t - mu_Y) / ((n-k) * std_X * std_Y)
    Compared to crosscorr, does not average out the result over the time dimension.

    Args:
        seq1: Input tensor of shape [N, L, D].
        seq2: Input tensor of shape [N, L, D].
        max_lag: Specifies the number of lags to compute the cross-correlation for.

    Returns: Cross-correlation of seq1 and seq2, shape [max_lag+1, L-max_lag, D].
    We return the same number of temporal points for all lags.

    Idea:
        We could use crosscoef from Torch > 2.0
    """
    return _crosscorr_avg_param(seq1, seq2, max_lag, avg_res_dims=(0,))


def _crosscorr_avg_param(
    seq1: torch.Tensor, seq2: torch.Tensor, max_lag: int, avg_res_dims: Tuple[int, ...]
) -> torch.Tensor:
    assert len(seq1.shape) == 3 and len(seq2.shape) == 3, (
        "The input tensors must be of shape [N, L, D] but are " f"{seq1.shape} and {seq2.shape}."
    )
    # Strict inequality (>) is mathematically necessary, not >=
    # Example: If L=2 and max_lag=2, at lag=2 we compute seq1[:, 2:] * seq2[:, :-2]
    # Both slices are EMPTY (length = L - max_lag = 0), giving undefined 0/0.
    # Minimum valid: L = max_lag + 1 gives exactly 1 pair to correlate at lag=max_lag.
    assert seq1.shape[1] > max_lag, (
        f"The sequence length must be strictly greater than the max_lag ({max_lag}), but has len " f"{seq1.shape[1]}."
    )

    seq1_ctr = seq1 - nanmean(seq1, dim=(0, 1))
    seq2_ctr = seq2 - nanmean(seq2, dim=(0, 1))
    std_seq1 = nanstd(seq1, dim=(0, 1))
    std_seq2 = nanstd(seq2, dim=(0, 1))

    # Roll and compute cross-correlation for all lags in a vectorized manner
    rmv_last_values_if_not_avg_over_time = slice(None, seq1.shape[1] - max_lag if 1 not in avg_res_dims else None)

    cross_corr = torch.stack(
        [nanmean(seq1_ctr * seq2_ctr, dim=avg_res_dims)[rmv_last_values_if_not_avg_over_time]] +
        # This step is the numerical implementation of the formula:
        # CrossCorrelation(X, Y, k) = sum_{t=1}^{T-k} X_t+k * Y_t / ((n-k) * std_X * std_Y)
        [
            nanmean(seq1_ctr[:, lag:] * seq2_ctr[:, :-lag], dim=avg_res_dims)[rmv_last_values_if_not_avg_over_time]
            for lag in range(1, max_lag + 1)
        ],
        dim=0,
    ) / (std_seq1 * std_seq2)

    return cross_corr


def autocorr(seqs: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Compute the autocorrelation of a tensor with formula:
    ACF(x, lag) = E[(x_t - mu)(x_{t+lag} - mu)] / var(x)
    The empirical (unbiased) estimator where var is also the empirical biased estimator of the variance:
    ACF(x, lag) = sum_{t=1}^{T-lag} (x_t - mu)(x_{t+lag} - mu) / ((n-k) * var(x)).

    Args:
        seqs:  Input tensor of shape [N, L, D].
        max_lag: Specifies the number of lags to compute the ACF for.

    Returns: Autocorrelation of x, shape [max_lag+1, D].

    """
    return crosscorr(seqs, seqs, max_lag)


def autocorr_nonstationary(seqs: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Compute the autocorrelation of a nonstationary tensor.

    Args:
        seqs:  Input tensor of shape [N, L, D].
        max_lag: Specifies the number of lags to compute the ACF for.

    Returns: Autocorrelation of seqs, shape [max_lag+1, L-max_lag, D].
    We return the same number of temporal points for all lags.
    """
    return crosscorr_nonstationary(seqs, seqs, max_lag)
