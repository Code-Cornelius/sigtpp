import logging
import typing

import numpy as np

logger = logging.getLogger(__name__)


def gen(
    num_seq: int,
    poisson_lambda: float,
    time_series_max_time: typing.Optional[float] = None,
    num_elements_in_ts: typing.Optional[int] = None,
    *,
    num_marks: int = 1,
    mark_probs: typing.Optional[np.ndarray] = None,
    rng: typing.Optional[np.random.Generator] = None,
) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a batch of sequences of exponential inter-arrival times, optionally with marks.

    Exactly one of `time_series_max_time` or `num_elements_in_ts` must be provided.

    Args:
        num_seq: Number of sequences to generate (> 0).
        poisson_lambda: Rate parameter λ of the Poisson process (> 0).
        time_series_max_time: If provided, generate a sequence then truncate to times < T.
        num_elements_in_ts: If provided, generate exactly this many inter-arrival times (> 0).
        num_marks: Number of mark categories (K). If 1, no marks are generated (default).
        mark_probs: Probability distribution over marks, shape (K,). If None, uniform distribution.
        rng: Optional NumPy random number generator (np.random.Generator).

    Returns:
        If num_marks == 1:
            A NumPy array of shape (num_seq, L, 1) of float inter-arrival times.
        If num_marks > 1:
            A tuple of:
                - inter_times: NumPy array of shape (num_seq, L, 1) of float inter-arrival times.
                - marks: NumPy array of shape (num_seq, L, 1) of int mark indices (0 to K-1).
        If `time_series_max_time` is provided, L is the max (across sequences) number of
        events with cumulative time < T (can be 0).
    """
    if (time_series_max_time is None) == (num_elements_in_ts is None):
        raise ValueError("Exactly one of `time_series_max_time` or `num_elements_in_ts` must be provided.")

    if not isinstance(num_seq, int) or num_seq <= 0:
        raise ValueError(f"`num_seq` must be a positive int; got {num_seq!r}.")
    if not np.isfinite(poisson_lambda) or poisson_lambda <= 0:
        raise ValueError(f"`poisson_lambda` must be a finite positive number; got {poisson_lambda!r}.")
    if time_series_max_time is not None:
        if not np.isfinite(time_series_max_time) or time_series_max_time <= 0:
            raise ValueError(f"`time_series_max_time` must be a finite positive number; got {time_series_max_time!r}.")
    if num_elements_in_ts is not None:
        if not isinstance(num_elements_in_ts, int) or num_elements_in_ts <= 0:
            raise ValueError(f"`num_elements_in_ts` must be a positive int; got {num_elements_in_ts!r}.")

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

    if time_series_max_time is not None:
        # Generate too many exponential to truncate at a chosen time.
        # It would be possible to derive a maths formula such that the probability
        # we have enough points is very high, but we have not.
        sequences_len = int(time_series_max_time * (poisson_lambda + 1) ** 2.0) + 1
    else:
        sequences_len = int(num_elements_in_ts)  # type: ignore[arg-type]

    exp_rvs = rng.exponential(
        scale=1.0 / poisson_lambda,
        size=(num_seq, sequences_len, 1),
    ).astype(float, copy=False)

    # Generate marks if num_marks > 1
    # NOTE: Marks are categorical integer indices (0 to K-1), not continuous probabilities.
    # Each event is assigned to one category sampled according to mark_probs distribution.
    marks = None
    if num_marks > 1:
        marks = rng.choice(
            num_marks,
            size=(num_seq, sequences_len, 1),
            p=mark_probs,
        ).astype(np.int64, copy=False)

    if time_series_max_time is None:
        if num_marks > 1:
            return exp_rvs, marks
        return exp_rvs

    cum_times = np.cumsum(exp_rvs, axis=1)

    valid_mask = cum_times < time_series_max_time  # (N, L, 1) boolean
    lengths = valid_mask[:, :, 0].sum(axis=1)  # (N,) counts per sequence
    seq_max_len = int(lengths.max(initial=0))

    # Keep only inter-arrivals whose event time is < T (others set to 0.0)
    inter_times_trunc = np.where(valid_mask, exp_rvs, 0.0)

    # Apply same truncation to marks if present
    if marks is not None:
        # Set marks to 0 for invalid (out-of-time) events
        marks_trunc = np.where(valid_mask, marks, 0)
        marks = marks_trunc[:, :seq_max_len, :]

    # If the maximum valid length hits the generated cap, at least one sequence may
    # still not have crossed T, meaning we might have truncated nothing for that sequence.
    if seq_max_len == sequences_len:
        logger.error(
            "Not enough samples were generated to ensure truncation at `time_series_max_time` "
            "(some sequences may still be too short / not fully truncated). "
            "Increase the generation length heuristic."
        )

    inter_times_result = inter_times_trunc[:, :seq_max_len, :]

    if num_marks > 1:
        return inter_times_result, marks
    return inter_times_result
