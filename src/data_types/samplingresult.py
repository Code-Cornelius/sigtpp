"""Result containers for TPP sampling operations."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class UnconditionalSamplingResult:
    """Result from sample_and_fix_seqs (unconditional generation).

    Attributes:
        its_scaled_cst: Scaled ITs, constant-padded beyond seq end (filtered).
            Shape: (N, L-1, D).
        cum_abs_cst: Unscaled cumulative time, absolute (includes τ₁ for ITSHIFTED),
            constant-padded → used for SIG path.
            Shape: (N, L-1, D).
        its_scaled_nan: Scaled ITs, NaN-masked beyond sequence length.
            Shape: (N, L-1, D).
        cum_rel_nan: Unscaled cumulative time, relative (always starts from τ₂),
            NaN-masked → used for INT/ED/W1.
            Shape: (N, L-1, D).
        its_scaled_raw: Scaled ITs, unfiltered raw (no padding or masking).
            Shape: (N, L-1, D).
        cond_its_scaled: Conditioning/input scaled inter-arrivals (None if unconditional).
            Shape: (N, L-1, D) or None.
        seq_lens: Valid sequence lengths, shape (N,).
    """

    its_scaled_cst: torch.Tensor
    cum_abs_cst: torch.Tensor
    its_scaled_nan: torch.Tensor
    cum_rel_nan: torch.Tensor
    its_scaled_raw: torch.Tensor
    cond_its_scaled: Optional[torch.Tensor]
    seq_lens: Optional[torch.Tensor] = None  # (N,) - per-sequence valid lengths after removing first value
    gen_marks: Optional[torch.Tensor] = None  # (N, L-1) generated marks aligned with times, or None


@dataclass
class ConditionalSamplingResult:
    """Result from sample_for_a_fixed_batch_and_fix (conditional generation).

    Attributes:
        its_scaled_cst: Scaled ITs, constant-padded (filtered).
            Shape: (N, L-1, D) for single sample, (S, N, L-1, D) for multi-sample.
        cum_abs_cst: Unscaled cumulative time, absolute, constant-padded.
            Shape: (N, L-1, D) for single sample, (S, N, L-1, D) for multi-sample.
        ref_its_nan: Unscaled real inter-arrival times, NaN-masked.
            Shape: (N, L-1, D) for single sample, (S, N, L-1, D) for multi-sample.
        gen_its_tf_nan: Unscaled generated inter-arrivals (teacher-forced), NaN-masked.
            Shape: (N, L-1, D) for single sample, (S, N, L-1, D) for multi-sample.
        seq_lens: Valid sequence lengths, shape (N,).
    """

    its_scaled_cst: torch.Tensor
    cum_abs_cst: torch.Tensor
    ref_its_nan: torch.Tensor
    gen_its_tf_nan: torch.Tensor
    seq_lens: Optional[torch.Tensor] = (
        None  # (N,) - per-sequence valid lengths after removing first value, based on condition!
    )
