import logging
import typing

import numpy as np

logger = logging.getLogger(__name__)


def cum_dist_fct_exp(x, lambda_param):
    # Reverse the log because uniform from numpy is on [0,1[.
    return -np.log(1.0 - x) / lambda_param


def non_homo_lewis_sampling_method(
    max_time_sampling: float,
    current_time: float,
    max_nu: float,
    nu_function: typing.Callable[[float], float],
    rng: np.random.Generator,
):
    """Method in order to get inter-arrivals times using Lewis' thinning algorithm.

    Args:
        max_time_sampling: We sample one value on [actual_time, max_time[.
        current_time: current time used to evaluate the exogenous rate.
        max_nu:  over the interval for thinning.
        nu_function: nu fct.
        rng: NumPy random number generator (must be provided for reproducibility).

    Returns:

    """
    arrival_time = 0
    while current_time + arrival_time < max_time_sampling:
        U = rng.random(1)
        arrival_time += cum_dist_fct_exp(U, max_nu)
        D = rng.random(1)
        if D <= nu_function(current_time + arrival_time) / max_nu:
            return arrival_time


def gen(
    num_seq: int,
    time_function_for_intensity: typing.Callable[[float], float],
    max_val_time_function: float,
    time_series_max_time: typing.Optional[float] = None,
    num_elements_in_ts: typing.Optional[int] = None,
    *,
    num_marks: int = 1,
    mark_probs: typing.Optional[np.ndarray] = None,
    rng: typing.Optional[np.random.Generator] = None,
) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray]]:
    """
    Generate inter-arrival time sequences using (non-homogeneous) Lewis thinning, optionally with marks.

    Exactly one of:
      - `time_series_max_time` (truncate when cumulative time reaches this), or
      - `num_elements_in_ts` (fixed-length output)

    must be provided.

    Args:
        num_seq: Number of sequences to generate.
        time_function_for_intensity: Time-dependent intensity function.
        max_val_time_function: Upper bound on intensity function for thinning.
        time_series_max_time: If provided, truncate when cumulative time reaches this.
        num_elements_in_ts: If provided, generate fixed-length sequences.
        num_marks: Number of mark categories (K). If 1, no marks are generated (default).
        mark_probs: Probability distribution over marks, shape (K,). If None, uniform distribution.
        rng: Optional NumPy random number generator (np.random.Generator).

    Returns
    -------
    If num_marks == 1:
        np.ndarray: Shape (num_seq, seq_len, 1) of non-negative inter-arrival times.
    If num_marks > 1:
        Tuple of:
            - inter_times: np.ndarray of shape (num_seq, seq_len, 1) of inter-arrival times.
            - marks: np.ndarray of shape (num_seq, seq_len, 1) of int mark indices (0 to K-1).
        For `time_series_max_time`, seq_len is the maximum valid length across sequences.
        For `num_elements_in_ts`, seq_len equals num_elements_in_ts exactly.
    """
    if not isinstance(num_seq, int) or num_seq < 0:
        raise ValueError(f"`num_seq` must be a non-negative int, got {num_seq!r}.")
    if max_val_time_function <= 0:
        raise ValueError(f"`max_val_time_function` must be > 0, got {max_val_time_function!r}.")
    if (time_series_max_time is None) == (num_elements_in_ts is None):
        raise ValueError(
            "Provide exactly one of `time_series_max_time` or `num_elements_in_ts` " "(they are mutually exclusive)."
        )
    if time_series_max_time is not None and time_series_max_time <= 0:
        raise ValueError(f"`time_series_max_time` must be > 0, got {time_series_max_time!r}.")
    if num_elements_in_ts is not None:
        if not isinstance(num_elements_in_ts, int) or num_elements_in_ts < 1:
            raise ValueError(f"`num_elements_in_ts` must be an int >= 1, got {num_elements_in_ts!r}.")

    # Validate mark parameters
    if not isinstance(num_marks, int) or num_marks < 1:
        raise ValueError(f"`num_marks` must be a positive int; got {num_marks!r}.")
    if mark_probs is not None:
        if not isinstance(mark_probs, np.ndarray) or mark_probs.shape != (num_marks,):
            raise ValueError(
                f"`mark_probs` must be a 1D array of shape ({num_marks},); got shape {mark_probs.shape if isinstance(mark_probs, np.ndarray) else type(mark_probs)}."
            )
        if not np.isclose(mark_probs.sum(), 1.0):
            raise ValueError(f"`mark_probs` must sum to 1.0; got {mark_probs.sum():.6f}.")
        if np.any(mark_probs < 0):
            raise ValueError("`mark_probs` must contain non-negative values.")

    if rng is None:
        rng = np.random.default_rng()

    # Set default uniform mark probabilities if not provided
    if mark_probs is None and num_marks > 1:
        mark_probs = np.ones(num_marks) / num_marks

    if num_elements_in_ts is not None:
        # Generate one extra cumulative time so that np.diff yields exactly
        # num_elements_in_ts inter-arrival times, matching the hp.gen contract.
        seqs_len: int = num_elements_in_ts + 1
    else:
        seqs_len = int(time_series_max_time * (max_val_time_function + 1.0) ** 2.0) + 1

    cum_time_seqs = np.zeros((num_seq, seqs_len, 1), dtype=float)

    # -------------------------
    # Generate cumulative times
    # -------------------------
    for n in range(num_seq):
        # Skip first time to have a zero.
        for i in range(1, seqs_len):
            t_prev = cum_time_seqs[n, i - 1, 0]

            dt = non_homo_lewis_sampling_method(
                10_000_000,
                t_prev,
                max_val_time_function,
                time_function_for_intensity,
                rng,
            )
            t_new = t_prev + dt
            cum_time_seqs[n, i, 0] = t_new

            if time_series_max_time is not None and t_new >= time_series_max_time:
                cum_time_seqs[n, i:, 0] = time_series_max_time
                break

    # -------------------------
    # Convert to inter-arrivals
    # -------------------------
    inter_time_seqs = np.diff(cum_time_seqs, axis=1)

    # -------------------------
    # Generate marks if num_marks > 1
    # -------------------------
    # NOTE: Marks are categorical integer indices (0 to K-1), not continuous probabilities.
    # Each event is assigned to one category sampled according to mark_probs distribution.
    # Marks are generated for inter-arrival times (transitions), so shape matches inter_time_seqs
    marks = None
    if num_marks > 1:
        marks = rng.choice(
            num_marks,
            size=inter_time_seqs.shape,
            p=mark_probs,
        ).astype(np.int64, copy=False)

    if time_series_max_time is not None:
        valid_mask = cum_time_seqs < time_series_max_time  # (N, L, 1) bool
        lengths = valid_mask[:, :, 0].sum(axis=1)
        seq_max_len = int(lengths.max(initial=0))

        # Inter-arrivals are defined for transitions, so usable diff length is (length - 1)
        usable_max = max(seq_max_len - 1, 0)

        # Ensure invalid parts of diff are zeroed (anything involving an invalid endpoint)
        valid_steps = valid_mask[:, 1:, 0] & valid_mask[:, :-1, 0]
        inter_time_seqs = np.where(valid_steps[:, :, None], inter_time_seqs, 0.0)

        # Apply same truncation to marks if present
        if marks is not None:
            marks = np.where(valid_steps[:, :, None], marks, 0)
            marks = marks[:, :usable_max, :]

        # If no sequence ever exceeded max_time, nothing was truncated (heuristic may be too short)
        if usable_max == inter_time_seqs.shape[1]:
            logger.error(
                "The generated sequences were not truncated because not enough samples have been generated. "
                "There is a risk that the sequences are too short."
            )

        inter_times_result = inter_time_seqs[:, :usable_max, :]

        if num_marks > 1:
            return inter_times_result, marks
        return inter_times_result

    else:
        # Fixed-length mode: keep your original behavior, but no truncation logic.
        # Remove numerical negatives if any (shouldn't happen, but safe).
        inter_time_seqs = np.where(inter_time_seqs > 0, inter_time_seqs, 0.0)

        if num_marks > 1:
            return inter_time_seqs, marks
        return inter_time_seqs
