"""Tensor feature extraction utilities for temporal point process diagnostics.

Provides inter-arrival / event-time tensors, NaN masking, correlation-feature
preparation, and split metadata helpers. No matplotlib dependency.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import seaborn as sns
import torch

from src.utils.fix_seq_ends import set_seq_to_nan_from_index

logger = logging.getLogger(__name__)


DEFAULT_MAX_PATHS = 50
DEFAULT_MAX_LAG = 40
DEFAULT_ACF_DISPLAY_MAX_LAG = 10
DEFAULT_MAX_CORRELATION_HEATMAP_DIM = 30
DEFAULT_MIN_SAMPLES_FOR_CORR = 50
DEFAULT_LOG_SAMPLE_WINDOWS = 5
DEFAULT_LOG_SAMPLE_STRIDE = 10
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
SPLITS = ("train", "val", "test")
SPLIT_LABELS = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}
SPLIT_COLORS = {
    "train": sns.color_palette("flare")[5],
    "val": sns.color_palette("flare")[2],
    "test": sns.color_palette("flare")[0],
}
SPLIT_LINESTYLES = {
    "train": "-",
    "val": "--",
    "test": "-.",
}


def _dataset_report_slug(dm) -> str:
    """Lowercase identifier for a datamodule, used in report directories and filenames."""
    name = str(getattr(dm, "DATASET_NAME", "")).strip() or type(dm).__name__
    return name.lower()


def _split_attrs(dm, split: str) -> Dict[str, Any]:
    """Resolve a split name to (cum, lens, marks). marks may be None."""
    assert split in SPLITS, f"Unknown split '{split}'. Expected 'train', 'val', or 'test'."
    cum = getattr(dm, f"{split}_in")
    lens = getattr(dm, f"{split}_in_len")
    marks = getattr(dm, f"{split}_marks", None) if int(getattr(dm, "num_marks", 1)) > 1 else None
    return {"cum": cum, "lens": lens, "marks": marks}


def _split_label(split: str) -> str:
    return SPLIT_LABELS[split]


def _split_color(split: str):
    return SPLIT_COLORS[split]


def _overlay_alpha(position: int, total: int, max_alpha: float = 0.7, min_alpha: float = 0.3) -> float:
    if total <= 1:
        return max_alpha
    frac = position / (total - 1)
    return max_alpha - frac * (max_alpha - min_alpha)


def _resolve_indices(indices: Optional[Iterable[int]], upper_bound: int, max_paths: int) -> List[int]:
    if indices is None:
        return list(range(min(max_paths, upper_bound)))
    indices = list(indices)
    assert all(0 <= i < upper_bound for i in indices), "indices out of range"
    return indices


def _shared_max_paths(dm, max_paths: int) -> int:
    return min(max_paths, min(getattr(dm, f"{split}_in").shape[0] for split in SPLITS))


def _interarrivals_naned(cum: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Inter-arrivals from cumulative times, padding masked to NaN.

    Shape change: (N, L+1, 1) -> (N, L, 1).
    """
    dts = cum.diff(dim=1)
    return set_seq_to_nan_from_index(dts, lens - 1)


def _interarrivals_drop_first(cum: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Inter-arrivals with first tau dropped (matches training metric convention).

    Shape change: (N, L+1, 1) -> (N, L-1, 1).
    """
    dts = cum.diff(dim=1)[:, 1:, :]
    return set_seq_to_nan_from_index(dts, (lens - 2).clamp(min=0))


def _event_times_naned(cum: torch.Tensor, lens: torch.Tensor, align_to_anchor: bool = False) -> torch.Tensor:
    """Event times excluding the t0 anchor, padding masked to NaN.

    When ``align_to_anchor=True``, event times are shifted so each sequence
    starts at 0 before masking. This is required for windowed datasets whose
    anchor stores an absolute clock position rather than a synthetic zero.
    """
    event_times = cum[:, 1:, :]
    if align_to_anchor:
        event_times = event_times - cum[:, :1, :]
    return set_seq_to_nan_from_index(event_times, lens - 2)


def _truncate_for_min_samples(seqs_naned: torch.Tensor, min_samples: int, max_lag: int) -> torch.Tensor:
    """Drop sparse trailing steps and cap the retained lag horizon."""
    L = seqs_naned.shape[1]
    mask = ~torch.isnan(seqs_naned)
    counts = mask.sum(dim=0).cpu().numpy()  # (L, D)
    limit_t = min(L, max_lag)
    for t in range(limit_t):
        if np.any(counts[t] < min_samples):
            limit_t = t
            break
    return seqs_naned[:, :limit_t, :]


def _prepare_corr_features(
    seqs_naned: torch.Tensor,
    min_samples: int,
    max_lag: int = DEFAULT_MAX_LAG,
    variance_eps: float = 1e-12,
) -> torch.Tensor:
    """Truncate sparse tail steps and drop degenerate columns before correlation.

    Returns a 2D tensor of shape (N, F_kept), where each retained column has
    enough finite samples and non-negligible variance. This avoids undefined
    correlations that would otherwise show up as blank heatmap cells.
    """
    truncated = _truncate_for_min_samples(seqs_naned, min_samples, max_lag)
    if truncated.shape[1] == 0:
        return truncated.reshape(truncated.shape[0], 0)

    flat = truncated.reshape(truncated.shape[0], -1)
    flat_np = flat.detach().cpu().numpy()
    valid_counts = np.sum(np.isfinite(flat_np), axis=0)
    variances = np.nanvar(flat_np, axis=0)
    keep = (valid_counts >= max(min_samples, 2)) & np.isfinite(variances) & (variances > variance_eps)

    if not np.any(keep):
        logger.warning("_prepare_corr_features: all %d columns filtered (sparse or zero-variance).", flat.shape[1])
        return flat[:, :0]
    return flat[:, keep]


def _log_dataset_sample_windows(
    dm,
    num_windows: int = DEFAULT_LOG_SAMPLE_WINDOWS,
    stride: int = DEFAULT_LOG_SAMPLE_STRIDE,
) -> None:
    """Log rolling dataset windows so DataLogRecord exposes different sample prefixes."""
    logger.info(
        "Dataset sample windows for %s: %d windows with stride %d.",
        getattr(dm, "DATASET_NAME", type(dm).__name__),
        num_windows,
        stride,
    )
    for split in SPLITS:
        split_data = _split_attrs(dm, split)
        cum = split_data["cum"]
        lens = split_data["lens"]
        marks = split_data["marks"]
        total = int(cum.shape[0])
        logger.info("=" * 24 + " %s split " + "=" * 24, _split_label(split))
        logger.info("%s split dataset tensors: full cumulative %s", _split_label(split), cum)
        logger.info("%s split dataset tensors: full lengths %s", _split_label(split), lens)
        if marks is not None:
            logger.info("%s split dataset tensors: full marks %s", _split_label(split), marks)

        for window_idx in range(num_windows):
            start = window_idx * stride
            if start >= total:
                logger.info(
                    "%s split dataset window %d skipped: start=%d exceeds dataset size=%d.",
                    _split_label(split),
                    window_idx + 1,
                    start,
                    total,
                )
                break

            logger.info(
                "%s split dataset window %d/%d: cumulative tensor from row %d -> %s",
                _split_label(split),
                window_idx + 1,
                num_windows,
                start,
                cum[start:],
            )
            logger.info(
                "%s split dataset window %d/%d: lengths tensor from row %d -> %s",
                _split_label(split),
                window_idx + 1,
                num_windows,
                start,
                lens[start:],
            )
            if marks is not None:
                logger.info(
                    "%s split dataset window %d/%d: marks tensor from row %d -> %s",
                    _split_label(split),
                    window_idx + 1,
                    num_windows,
                    start,
                    marks[start:],
                )
        logger.info("=" * 22 + " end %s split " + "=" * 22, _split_label(split))
