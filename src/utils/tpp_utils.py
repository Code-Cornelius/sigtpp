"""
Utility functions for Temporal Point Process (TPP) operations.

This module provides standalone utility functions for tensor manipulation, I/O,
and formatting operations used across TPP architectures.

Extracted from TPPArchitecture to improve code organization and reusability.
"""

import hashlib
import logging
import os
from typing import Optional, Tuple, List, Callable, Union

import torch

logger = logging.getLogger(__name__)

from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.utils.fix_seq_ends import _replace_from_index_with_value_torch


def atomic_torch_save(payload, path: Union[str, os.PathLike]) -> None:
    """Write *payload* to *path* without ever leaving a half-written file there.

    ``torch.save`` writes incrementally, so a crash, kill, or Ctrl-C partway
    through leaves a truncated/corrupt file at ``path`` -- and if ``path`` was
    already tracked by git, that corruption can also get committed and later
    resurface via a `git checkout`/rollback of the "deletion".

    The fix is the standard atomic-replace trick: save to a temporary file in
    the same directory, then ``os.replace`` it onto the final path. ``os.replace``
    (and POSIX ``rename``) is atomic *within the same filesystem* -- the
    destination either still has its old complete contents or the new complete
    contents, never a partial write. Same-directory placement guarantees both
    paths are on the same filesystem.
    """
    tmp_path = os.fspath(path) + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return


def apply_mask(
    tensor1: torch.Tensor, mask_valid: torch.Tensor, tensor2: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Filter tensors based on a boolean mask, removing invalid sequences.

    This function is commonly used to filter out pathological sequences during training
    (e.g., sequences that became NaN or violate model constraints).

    Args:
        tensor1: Primary tensor to filter, typically shape (N, L, D).
        mask_valid: Boolean mask of shape (N,) where True indicates valid sequences to keep.
        tensor2: Optional secondary tensor to filter in lockstep with tensor1.

    Returns:
        Tuple of (filtered_tensor1, filtered_tensor2).
        If tensor2 is None, returns (filtered_tensor1, None).

    Raises:
        ValueError: If all sequences are filtered out (mask_valid is all False).

    Example:
        >>> sequences = torch.randn(100, 50, 1)
        >>> valid_mask = torch.tensor([True] * 95 + [False] * 5)
        >>> filtered_seqs, _ = apply_mask(sequences, valid_mask)
        >>> filtered_seqs.shape
        torch.Size([95, 50, 1])
    """
    filtered_indices = (~mask_valid).nonzero(as_tuple=True)[0]

    if filtered_indices.numel() > 0:
        logger.debug(
            f"Filtered sequences that were invalid for training. There were {filtered_indices.numel()} of them."
        )

    tensor1 = tensor1[mask_valid]
    if tensor2 is not None:
        tensor2 = tensor2[mask_valid]

    if tensor1.shape[0] == 0:
        logger.error("The training was unstable and all sequences were removed.")
        raise ValueError("The training was unstable and all sequences were removed.")

    return tensor1, tensor2


def concat_two_samples_together(
    list1: List[torch.Tensor], list2: Optional[List[torch.Tensor]], target_length: int
) -> List[torch.Tensor]:
    """
    Concatenate or truncate tensor lists to reach a target length.

    Used during exact sampling to ensure each sequence has exactly target_length
    samples by combining primary samples with secondary (resampled) samples or
    truncating if already sufficient.

    Args:
        list1: Primary list of tensors to use first.
        list2: Secondary list of tensors to draw from if list1 is insufficient.
               Can be None (will be treated as empty).
        target_length: Desired length for the first dimension of each tensor.

    Returns:
        List of tensors, each with shape (target_length, ...).

    Raises:
        ValueError: If combined list1 + list2 doesn't have enough samples.

    Example:
        >>> primary = [torch.randn(80, 10), torch.randn(90, 10)]
        >>> secondary = [torch.randn(50, 10), torch.randn(50, 10)]
        >>> result = concat_two_samples_together(primary, secondary, target_length=100)
        >>> result[0].shape, result[1].shape
        (torch.Size([100, 10]), torch.Size([100, 10]))
    """
    if list2 is None:
        list2 = [None] * len(list1)

    final_const_gen_out_list = []

    for primary_tensor, secondary_tensor in zip(list1, list2):
        current_length = primary_tensor.shape[0]

        if current_length < target_length:
            missing = target_length - current_length
            secondary_length = secondary_tensor.shape[0] if secondary_tensor is not None else 0

            # Take as many as needed from the secondary tensor
            take_n = min(missing, secondary_length)
            if take_n < missing:
                logger.error(f"Not enough values available. Needed: {missing}, available: {secondary_length}")
                raise ValueError(f"Not enough values available. Needed: {missing}, available: {secondary_length}")

            # Concatenate the missing part from secondary to primary if applicable
            if secondary_tensor is not None:
                primary_tensor = torch.cat([primary_tensor, secondary_tensor[:take_n]], dim=0)

        elif current_length > target_length:
            primary_tensor = primary_tensor[:target_length]

        final_const_gen_out_list.append(primary_tensor)

    return final_const_gen_out_list


def insert_zero_beg(paths: torch.Tensor) -> torch.Tensor:
    """
    Prepend a zero time-step to the beginning of sequences.

    Used for signature computation where sequences need an anchor point at t=0.
    Works with paths structured as [scaled inter-times; cumulative inter-times].

    Args:
        paths: Tensor of shape (N, L, D) containing inter-arrival times and/or cumulative times.

    Returns:
        Tensor of shape (N, L+1, D) with zeros prepended along the sequence dimension.

    Example:
        >>> sequences = torch.tensor([[[1.0], [2.0], [3.0]]])  # shape (1, 3, 1)
        >>> anchored = insert_zero_beg(sequences)
        >>> anchored.shape
        torch.Size([1, 4, 1])
        >>> anchored[0, 0, 0].item()
        0.0
    """
    # Prepend a zero to paths structured as [scaled inter-times; cumulative inter-times].
    # Inter-times can be any real value so prepending zero is valid;
    # cumulative times start at 0 by definition.
    return torch.cat([torch.zeros((paths.shape[0], 1, paths.shape[2]), device=paths.device), paths], dim=1)


def append_terminal_anchor(
    paths: torch.Tensor,
    mode: TerminalAnchorMode,
    time_max: float,
    seq_lens: Optional[torch.Tensor] = None,
    anchor_tau: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Append a terminal anchor point at T_max.

    Assumes paths[:, :, 0] contains scaled inter-arrival times
    and paths[:, :, -1] contains cumulative times (unscaled).
    The paths already have the zero anchor prepended.

    Args:
        paths: Tensor of shape (N, L, D).
        mode: Which terminal anchor mode to use.
        time_max: Maximum observation time T_max.
        seq_lens: Per-sequence lengths *before* insert_zero_beg, shape (N,).
            Required for RESIDUAL; ignored by FREE_ENDPOINT.
            After the zero prepend, position seq_lens[n] is the last real
            event for sequence n; positions beyond are constant-padded.
        anchor_tau: Optional pre-computed channel-0 value for the terminal
            anchor, shape (N,). When provided, used directly instead of
            computing ``time_max - paths[:, -1, -1]``. This allows callers
            to pass an exp-scaled gap so that channel 0 stays consistent.

    Returns:
        If mode is FREE_ENDPOINT, returns shape (N, L, D).
        Otherwise returns (N, L+1, D) with the terminal anchor appended.
    """
    if mode is TerminalAnchorMode.FREE_ENDPOINT:
        return paths

    N, L, D = paths.shape
    # Build per-sequence boundary values: (gap_tau, T_max).
    anchor = torch.zeros(N, 1, D, device=paths.device, dtype=paths.dtype)

    if seq_lens is None:
        raise ValueError(
            "RESIDUAL requires seq_lens (pre-insert_zero_beg lengths) "
            "to know where each sequence ends. Pass seq_lens to scale_paths_pre_sig."
        )

    assert (
        anchor_tau is not None
    ), "RESIDUAL mode requires anchor_tau to be provided to compute the gap value for channel 0."
    anchor[:, :, :1] = anchor_tau
    anchor[:, 0, -1] = time_max  # cumulative time channel
    # Overwrite positions strictly after seq_lens[n] (= first padding slot).
    return _replace_from_index_with_value_torch(paths, seq_lens.to(paths.device), anchor)


def cum_times_to_log_inter_times(
    batch: Tuple[torch.Tensor, torch.Tensor], scaler: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert cumulative times to scaled inter-arrival times.

    Takes a batch of cumulative time sequences and converts them to inter-arrival times
    (via diff), then applies a scaling function (typically exponential or log scaling).

    Args:
        batch: Tuple containing:
            - data: Cumulative times of shape (N, L+1, D)
            - data_lens: Sequence lengths of shape (N,)
        scaler: Callable that scales inter-arrival times (e.g., ExpScaler).

    Returns:
        Tuple of (data_dts_lens, data_dts_scaled):
            - data_dts_lens: Adjusted lengths (N,), decreased by 1 due to diff operation
            - data_dts_scaled: Scaled inter-arrival times (N, L, D)

    Example:
        >>> cum_times = torch.cumsum(torch.rand(10, 20, 1), dim=1)
        >>> cum_with_anchor = torch.cat([torch.zeros(10, 1, 1), cum_times], dim=1)
        >>> lens = torch.full((10,), 21)
        >>> scaler = lambda x: torch.log(x + 1e-8)
        >>> new_lens, scaled_its = cum_times_to_log_inter_times((cum_with_anchor, lens), scaler)
        >>> new_lens.shape, scaled_its.shape
        (torch.Size([10]), torch.Size([10, 20, 1]))
    """
    # Data in batch needs to be of shape (N, L+1, D).
    # Returns data of shape (N,) and (N, L, D).

    data, data_lens = batch[0], batch[1]
    data_dts = data.diff(dim=1)
    data_dts_scaled = scaler(data_dts)
    data_dts_lens = data_lens - 1
    return data_dts_lens, data_dts_scaled


def format_metrics_table(metrics: dict, metric_names: List[str], num_splits: int = 3) -> str:
    """
    Format a metrics dictionary as a horizontal ASCII table string.

    Args:
        metrics: Dictionary mapping metric names to float values.
        metric_names: Ordered list of keys to display.
        num_splits: Number of sub-tables to split columns across.

    Returns:
        Multi-line string with header, separator, and values rows.
    """
    metric_names = [name for name in metric_names if name in metrics]
    values = [
        f"{metrics[name]:.2e}" if abs(metrics[name]) < 0.00005 and metrics[name] != 0 else f"{metrics[name]:.4f}"
        for name in metric_names
    ]
    col_widths = [max(len(name), len(value)) for name, value in zip(metric_names, values)]

    def render_chunk(indices):
        header = " | ".join(f"{metric_names[i]:>{col_widths[i]}}" for i in indices)
        sep = "-+-".join('-' * col_widths[i] for i in indices)
        vals = " | ".join(f"{values[i]:>{col_widths[i]}}" for i in indices)
        return "\n".join([header, sep, vals])

    n = len(metric_names)
    if n == 0:
        return ""
    chunk_size = (n + num_splits - 1) // num_splits
    chunks = [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]
    return "\n\n".join(render_chunk(c) for c in chunks)


def save_samples(
    inter_times: torch.Tensor,
    lengths: torch.Tensor,
    path: str,
    marks: Optional[torch.Tensor] = None,
) -> None:
    """
    Save sampled TPP sequences to disk.

    Stores inter-arrival times and their corresponding lengths in a .pth file
    for later analysis or comparison.

    Convention: both generated (samples_generated.pth) and test (samples_test.pth)
    sequences include τ₁ (the first inter-arrival time). lengths[i] counts τ₁,
    so valid positions in inter_times[i] are 0 … lengths[i]-1.

    Args:
        inter_times: Inter-arrival time sequences of shape (N, L, D), τ₁ included.
        lengths: Sequence lengths of shape (N,), counting τ₁.
        path: File path to save to (should end in .pth).
        marks: Optional mark tensor of shape (N, L), aligned with inter_times.

    Example:
        >>> samples = torch.randn(1000, 50, 1)
        >>> lens = torch.randint(10, 50, (1000,))
        >>> save_samples(samples, lens, "output/samples.pth")
    """
    payload = {"inter_times": inter_times, "lengths": lengths}
    if marks is not None:
        payload["marks"] = marks
    atomic_torch_save(payload, path)
    return


def _fingerprint_samples(
    inter_times: torch.Tensor,
    lengths: torch.Tensor,
    marks: Optional[torch.Tensor] = None,
) -> str:
    """Content hash of a (inter_times, lengths, marks) sample triple, order- and dtype-sensitive."""
    hasher = hashlib.sha256()
    for tensor in (inter_times, lengths, marks):
        if tensor is None:
            continue
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def save_test_targets_once(
    experiment_dir: str,
    inter_times: torch.Tensor,
    lengths: torch.Tensor,
    marks: Optional[torch.Tensor] = None,
) -> str:
    """Save the test split's target samples once per experiment.

    Every model trained for a given experiment is evaluated against the same
    deterministically-split test set, so writing ``samples_tgt.pth`` into each
    model's own output directory produces byte-identical copies across models
    (and across seeds of the same experiment). This instead writes a single
    shared ``test_targets.pth`` under ``experiment_dir``, skipping the write
    entirely when a file with a matching content fingerprint is already there.

    If a file exists with a *different* fingerprint, it is overwritten (with a
    warning logged). This is the safety-valve path: the stale shared file must
    not silently linger when the test split changes. Normal multiseed runs are
    unaffected -- dataset generation uses a fixed ``data_seed`` (42 by default)
    that is independent of the training seed, so every seed in a sweep produces
    the same test split and the same fingerprint.

    Returns the path of the (verified or newly written) shared file.
    """
    os.makedirs(experiment_dir, exist_ok=True)
    path = os.path.join(experiment_dir, "test_targets.pth")
    fingerprint = _fingerprint_samples(inter_times, lengths, marks)

    if os.path.exists(path):
        try:
            existing = torch.load(path, map_location="cpu")
            if existing.get("fingerprint") == fingerprint:
                logger.debug("Shared test targets at %s already match this run's fingerprint; skipping write.", path)
                return path
            # Fingerprint mismatch: the on-disk test split differs from this run's.
            # Realistic triggers:
            #   - dataset regenerated (different data_seed, parameters, or cache rebuild)
            #   - split fractions, data_size, or preprocessing changed
            #   - mark configuration changed under the same experiment path
            #   - dtype changed (e.g. float32 vs float64)
            #   - file written by older code without a 'fingerprint' key (None != hash)
            #   - concurrent runs racing to write under the same experiment_dir
            # In all cases overwriting is correct; the warning surfaces the event.
            logger.warning(
                "Shared test targets at %s have fingerprint %s but this run's test split fingerprints as %s; "
                "overwriting. This is expected only if the test split itself changed.",
                path,
                existing.get("fingerprint"),
                fingerprint,
            )
        except Exception as e:
            logger.warning("Could not read existing shared test targets at %s (%s); overwriting.", path, e)

    payload = {"inter_times": inter_times, "lengths": lengths, "fingerprint": fingerprint}
    if marks is not None:
        payload["marks"] = marks
    atomic_torch_save(payload, path)
    logger.info("Saved shared test targets to %s", path)
    return path


def load_samples(
    path: str,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load sampled TPP sequences from disk.

    Reads a .pth file containing saved TPP samples and returns the inter-arrival
    times, their lengths, and optionally marks.

    Args:
        path: File path to load from (should be a .pth file saved by save_samples).

    Returns:
        Tuple of (inter_times, lengths) or (inter_times, lengths, marks):
            - inter_times: Inter-arrival time sequences of shape (N, L, D)
            - lengths: Sequence lengths of shape (N,)
            - marks: Mark tensor of shape (N, L), only present if saved with marks.

    Example:
        >>> inter_times, lengths = load_samples("output/samples.pth")
        >>> inter_times.shape, lengths.shape
        (torch.Size([1000, 50, 1]), torch.Size([1000]))
    """
    data = torch.load(path, map_location="cpu")
    if "marks" in data:
        return data["inter_times"], data["lengths"], data["marks"]
    return data["inter_times"], data["lengths"]


def get_num_needed_resample(
    num_samples_per_seq: int,
    oversampled_num_per_seq: int,
    oversampling_factor: float,
    sampled_sequences_per_batch: List[torch.Tensor],
) -> int:
    """Calculate how many additional samples are needed to ensure exact sampling.

    When sampling with filtering (removing pathological sequences), not all samples
    survive. This function estimates how many additional samples to generate to ensure
    each batch element gets exactly num_samples_per_seq valid samples.

    Args:
        num_samples_per_seq: Target number of valid samples per sequence.
        oversampled_num_per_seq: Number of samples generated in the first pass.
        oversampling_factor: Factor by which to oversample (e.g., 1.25).
        sampled_sequences_per_batch: List of tensors, one per batch element,
            containing the sequences that survived filtering.

    Returns:
        Number of additional samples to generate to reach the target.

    Raises:
        RuntimeError: If the survival rate is too low (<1e-8) to reliably recover.
    """
    num_survivors_per_batch = [tensor.shape[0] for tensor in sampled_sequences_per_batch]
    min_survivors = min(abs(count) for count in num_survivors_per_batch)

    survival_rate = min_survivors / oversampled_num_per_seq

    if survival_rate < 1e-8:
        raise RuntimeError(
            "All samples were filtered as pathological for at least one sequence. "
            "Resampling cannot recover; consider increasing num_samples_per_seq or check the model."
        )

    shortage = max(0, num_samples_per_seq - min_survivors)
    num_additional_samples_needed = int((shortage / survival_rate) * oversampling_factor)

    logger.debug(
        f"Resampling calculation: need {num_additional_samples_needed} additional samples "
        f"(min survivors: {min_survivors}/{num_samples_per_seq}, "
        f"survival rate: {survival_rate:.4f}, "
        f"oversampling factor: {oversampling_factor:.2f})"
    )

    return num_additional_samples_needed
