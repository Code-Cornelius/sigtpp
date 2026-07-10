"""Shared numeric aggregation, ranking, and fixed-width report helpers."""

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


def coerce_float(value: Any) -> float:
    """Convert scalar-like values to ``float``; invalid values become NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarise_values(values: Iterable[Any]) -> Tuple[float, float, int]:
    """Return ``(mean, std, n_valid)`` after coercing values and dropping NaNs.

    Standard deviation uses the sample convention (``ddof=1``) when at least two
    valid values are present. A single valid value reports ``std=0.0``.
    """
    vals = np.asarray([coerce_float(value) for value in values], dtype=float)
    valid = vals[~np.isnan(vals)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    if valid.size == 1:
        return float(valid[0]), 0.0, 1
    return float(valid.mean()), float(valid.std(ddof=1)), int(valid.size)


def apply_competition_ranking(
    rows: List[Dict[str, Any]],
    ranking_metrics: Sequence[str],
    value_key_for_metric: Callable[[str], str],
    *,
    score_key: str = "norm_score",
    rank_key_for_metric: Optional[Callable[[str], str]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Apply NaN-aware Borda ranking in-place.

    Lower metric values rank better. Ties receive the same min/competition rank.
    NaN values are excluded from per-metric ranks and penalized with ``n + 1``
    in the aggregate score for each metric that participates.

    ``score_key`` names the aggregate-score key written on each row (default
    ``"norm_score"``). ``rank_key_for_metric`` maps a metric name to its
    per-metric rank key (default ``"rank_<metric>"``). Both are parameterised so
    val-split and test-split rankings can be stored side-by-side on the same row
    without clobbering each other (e.g. ``val_norm_score`` / ``rank_val_W1``).
    """
    if rank_key_for_metric is None:
        rank_key_for_metric = lambda metric: f"rank_{metric}"

    n = len(rows)
    ranked_metrics: List[str] = []

    for metric in ranking_metrics:
        rank_key = rank_key_for_metric(metric)
        values = [row["metrics"].get(value_key_for_metric(metric), np.nan) for row in rows]
        valid_count = sum(1 for value in values if not np.isnan(value))
        if valid_count < 1:
            for row in rows:
                row["metrics"][rank_key] = np.nan
            continue

        ranked_metrics.append(metric)
        sortable = [(value if not np.isnan(value) else np.inf, index) for index, value in enumerate(values)]
        sortable.sort(key=lambda item: item[0])

        rank = 1
        for pos, (value, index) in enumerate(sortable):
            if np.isinf(value):
                rows[index]["metrics"][rank_key] = np.nan
            else:
                if pos > 0 and value == sortable[pos - 1][0]:
                    rows[index]["metrics"][rank_key] = rows[sortable[pos - 1][1]]["metrics"][rank_key]
                else:
                    rows[index]["metrics"][rank_key] = rank
                rank += 1

    if not ranked_metrics:
        if logger is not None:
            logger.warning("normalize_and_rank: no ranking metrics had enough valid values to rank.")
        for row in rows:
            row["metrics"][score_key] = np.nan
        return

    for row in rows:
        score = 0.0
        for metric in ranked_metrics:
            rank_value = row["metrics"].get(rank_key_for_metric(metric), np.nan)
            score += rank_value if not np.isnan(rank_value) else (n + 1)
        row["metrics"][score_key] = score

    rows.sort(key=lambda row: row["metrics"].get(score_key, float("inf")))


def write_result_rows_fixed_width(
    rows: List[Dict[str, Any]],
    column_names: List[str],
    path: str,
    metric_names: List[str],
    *,
    include_seed: bool,
    align_error_header_left: bool = False,
    preformatted_metric_names: Optional[Iterable[str]] = None,
) -> None:
    """Write experiment-style row dicts as a fixed-width text table."""
    has_error = any("error" in row["metrics"] for row in rows)
    output_columns = list(column_names)
    if has_error:
        output_columns.append("ERROR")

    preformatted_metrics = set(preformatted_metric_names or ())
    widths = result_row_column_widths(
        rows,
        output_columns,
        metric_names,
        include_seed=include_seed,
        has_error=has_error,
        preformatted_metric_names=preformatted_metrics,
    )
    text_indices: Set[int] = {1 if include_seed else 0}
    integer_indices: Set[int] = {0} if include_seed else set()
    if has_error:
        text_indices.add(len(output_columns) - 1)

    metric_start_index = 2 if include_seed else 1
    preformatted_indices: Set[int] = {
        metric_start_index + index
        for index, metric in enumerate(metric_names)
        if metric in preformatted_metrics
    }

    header_text_indices: Set[int] = {0, 1} if include_seed else {0}
    if has_error and align_error_header_left:
        header_text_indices.add(len(output_columns) - 1)

    table_rows: List[List[Any]] = []
    for row in rows:
        values: List[Any] = []
        if include_seed:
            values.append(row["seed"])
        values.append(row["model_name"])
        values.extend(row["metrics"].get(metric, float("nan")) for metric in metric_names)
        if has_error:
            values.append(sanitize_error_text(row["metrics"].get("error", "")))
        table_rows.append(values)

    write_fixed_width_table(
        path,
        output_columns,
        table_rows,
        widths,
        text_indices=text_indices,
        integer_indices=integer_indices,
        header_text_indices=header_text_indices,
        preformatted_indices=preformatted_indices,
    )


def result_row_column_widths(
    rows: List[Dict[str, Any]],
    column_names: List[str],
    metric_names: List[str],
    *,
    include_seed: bool,
    has_error: bool,
    preformatted_metric_names: Optional[Set[str]] = None,
) -> List[int]:
    """Compute fixed-width columns for experiment-style row dicts."""
    preformatted_metrics = preformatted_metric_names or set()
    widths: List[int] = []
    if include_seed:
        max_seed_len = max([len("SEED")] + [len(str(row["seed"])) for row in rows])
        widths.append(max_seed_len)

    max_model_len = max([len("MODEL"), 50] + [len(row["model_name"]) for row in rows])
    widths.append(max_model_len)
    for name in metric_names:
        value_lengths = []
        if name in preformatted_metrics:
            value_lengths = [len(str(row["metrics"].get(name, ""))) for row in rows]
        widths.append(max([len(name), 15] + value_lengths))

    if has_error:
        error_values = [str(row["metrics"].get("error", "")) for row in rows]
        widths.append(max([len("ERROR"), 20] + [len(value) for value in error_values]))

    return [max(width, len(column_names[index])) for index, width in enumerate(widths)]


def write_fixed_width_table(
    path: str,
    column_names: Sequence[str],
    rows: Iterable[Sequence[Any]],
    widths: Sequence[int],
    *,
    text_indices: Iterable[int],
    integer_indices: Iterable[int] = (),
    header_text_indices: Optional[Iterable[int]] = None,
    preformatted_indices: Optional[Iterable[int]] = None,
) -> None:
    """Write already-shaped row values as a fixed-width text table."""
    header_indices = set(text_indices if header_text_indices is None else header_text_indices)
    with open(path, "w") as handle:
        handle.write(format_fixed_width_header(column_names, widths, header_indices) + "\n")
        for values in rows:
            handle.write(
                format_fixed_width_row(
                    values,
                    widths,
                    text_indices=set(text_indices),
                    integer_indices=set(integer_indices),
                    preformatted_indices=set(preformatted_indices or ()),
                )
                + "\n"
            )


def format_fixed_width_header(column_names: Sequence[str], widths: Sequence[int], text_indices: Set[int]) -> str:
    cells = [
        f"{name:<{width}}" if index in text_indices else f"{name:>{width}}"
        for index, (name, width) in enumerate(zip(column_names, widths))
    ]
    return "   ".join(cells)


def format_fixed_width_row(
    values: Sequence[Any],
    widths: Sequence[int],
    *,
    text_indices: Set[int],
    integer_indices: Set[int],
    preformatted_indices: Optional[Set[int]] = None,
) -> str:
    preformatted = preformatted_indices or set()
    cells: List[str] = []
    for index, (value, width) in enumerate(zip(values, widths)):
        if index in integer_indices:
            cells.append(f"{int(value):>{width}}")
        elif index in text_indices:
            cells.append(f"{str(value):<{width}}")
        elif index in preformatted:
            cells.append(f"{str(value):>{width}}")
        else:
            cells.append(f"{value:>{width}.5g}")
    return "   ".join(cells)


def sanitize_error_text(value: Any, *, max_length: int = 100) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    return (text[: max_length - 3] + "...") if len(text) > max_length else text
