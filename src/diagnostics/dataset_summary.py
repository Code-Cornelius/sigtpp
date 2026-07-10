"""Scalar dataset summary statistics for temporal point processes."""

from typing import Any, Dict

import numpy as np

from src.diagnostics._tpp_features import (
    SPLITS,
    QUANTILES,
    _interarrivals_naned,
    _split_attrs,
)


def _length_stats(lens) -> Dict[str, Any]:
    arr = lens.detach().cpu().numpy().astype(float)
    out = {
        "count": int(arr.size),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "median": float(np.median(arr)) if arr.size else None,
        "quantiles": {f"q{int(q * 100)}": float(np.quantile(arr, q)) for q in QUANTILES} if arr.size else {},
    }
    return out


def _inter_arrival_stats(its_naned) -> Dict[str, Any]:
    flat = its_naned.detach().cpu().numpy().reshape(-1)
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return {"count": 0}
    out = {
        "count": int(valid.size),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "quantiles": {f"q{int(q * 100)}": float(np.quantile(valid, q)) for q in QUANTILES},
        "zero_count": int(np.sum(valid == 0.0)),
        "near_zero_count_lt_1e_8": int(np.sum(valid < 1e-8)),
    }
    return out


def _format_stat_float(value: float) -> str:
    """Format scalar stats without fixed decimal rounding."""
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return np.format_float_scientific(value, unique=True, trim="-")
    return np.format_float_positional(value, unique=True, trim="-")


def _mark_frequency_stats(marks, lens, num_marks: int) -> Dict[str, Any]:
    marks_t = marks.squeeze(-1) if marks.ndim == 3 else marks
    counts = np.zeros(int(num_marks), dtype=np.int64)
    for i in range(marks_t.shape[0]):
        L = int(lens[i].item())
        m = marks_t[i, 1:L].detach().cpu().numpy().astype(int)
        for k in range(int(num_marks)):
            counts[k] += int(np.sum(m == k))
    total = int(counts.sum())
    return {
        "num_marks": int(num_marks),
        "total_events": total,
        "counts": counts.tolist(),
        "frequencies": (counts / max(total, 1)).tolist(),
    }


def get_dataset_summary(dm, split: str = "train") -> Dict[str, Any]:
    """Produce scalar stats for one dataset.

    The returned dict is JSON-serializable and covers train/val/test sequence
    counts, time_max, num_marks, length stats per split, inter-arrival stats
    on the chosen split, and mark frequencies for marked datasets.
    """
    summary: Dict[str, Any] = {
        "dataset_name": getattr(dm, "DATASET_NAME", type(dm).__name__),
        "datamodule_class": type(dm).__name__,
        "reference_split": split,
        "time_max": float(dm.time_max),
        "num_marks": int(dm.num_marks),
        "counts": {
            "train": int(dm.train_in.shape[0]),
            "val": int(dm.val_in.shape[0]),
            "test": int(dm.test_in.shape[0]),
        },
        "length_stats": {
            "train": _length_stats(dm.train_in_len),
            "val": _length_stats(dm.val_in_len),
            "test": _length_stats(dm.test_in_len),
        },
    }

    summary["inter_arrival_stats"] = {
        split_name: _inter_arrival_stats(
            _interarrivals_naned(getattr(dm, f"{split_name}_in"), getattr(dm, f"{split_name}_in_len"))
        )
        for split_name in SPLITS
    }

    if int(dm.num_marks) > 1:
        summary["mark_frequencies"] = {
            split_name: _mark_frequency_stats(
                getattr(dm, f"{split_name}_marks"),
                getattr(dm, f"{split_name}_in_len"),
                dm.num_marks,
            )
            for split_name in SPLITS
        }

    return summary
