"""Per-sig-degree bootstrap report rendering.

Pure-text, torch-free helpers that turn ``{sig_degree: winner_run_name}`` plus
bootstrap recompute rows into the same fixed-width raw-column schema used by the
main final-test reports. Bootstrap metrics are written as separate
``<metric>_mean`` / ``<metric>_std`` columns; metrics computed once outside the
bootstrap loop keep their plain column names.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


from src.utils.result_helpers import write_result_rows_fixed_width
from test.paper_experiments.experiment_results import ExperimentResults


# Metrics to show by default, matching the normal final-test report order.
REPORT_METRICS = tuple(ExperimentResults.DISPLAY_METRICS + ExperimentResults.EXTRA_METRICS)


def _report_metric_names(report_metrics: Sequence[str]) -> List[str]:
    """Expand base metrics to the same output columns as ``ExperimentResults``."""
    metric_names: List[str] = ["sig_degree"]
    for metric in report_metrics:
        if metric in ExperimentResults.NON_BOOTSTRAP_METRICS:
            metric_names.append(metric)
        else:
            metric_names.append(ExperimentResults._mean_key(metric))
            metric_names.append(ExperimentResults._std_key(metric))
    return metric_names


def _copy_metric_values(source: Dict[str, Any], target: Dict[str, Any], report_metrics: Sequence[str]) -> None:
    """Copy raw metric values into the final-test report column schema."""
    for metric in report_metrics:
        if metric in ExperimentResults.NON_BOOTSTRAP_METRICS:
            target[metric] = source.get(metric, source.get(ExperimentResults._mean_key(metric), float("nan")))
        else:
            target[ExperimentResults._mean_key(metric)] = source.get(ExperimentResults._mean_key(metric), float("nan"))
            target[ExperimentResults._std_key(metric)] = source.get(ExperimentResults._std_key(metric), float("nan"))


def _report_rows_from_bootstrap(
    winners: Dict[int, str],
    bootstrap_rows: Sequence[Dict[str, Any]],
    report_metrics: Sequence[str],
) -> List[Dict[str, Any]]:
    """Shape recompute rows as fixed-width report rows, preserving all degrees."""
    boot_by_run = {str(row.get("model_name")): row for row in bootstrap_rows}
    rows: List[Dict[str, Any]] = []

    for degree in sorted(winners):
        run_name = winners[degree]
        boot_row = boot_by_run.get(run_name)
        metrics: Dict[str, Any] = {"sig_degree": int(degree)}

        if boot_row is None:
            logger.warning("degree %d winner %r was not evaluated - rendering NaN", degree, run_name)
            rows.append({"model_name": run_name, "metrics": metrics})
            continue

        boot_metrics = dict(boot_row.get("metrics", {}))
        error_msg = boot_metrics.get("error")
        if error_msg:
            metrics["error"] = error_msg
            rows.append({"model_name": run_name, "metrics": metrics})
            continue

        _copy_metric_values(boot_metrics, metrics, report_metrics)
        rows.append({"model_name": run_name, "metrics": metrics})

    return rows


def write_report(
    winners: Dict[int, str],
    bootstrap_rows: Sequence[Dict[str, Any]],
    out_path: Path,
    report_metrics: Sequence[str] = REPORT_METRICS,
) -> Path:
    """Write the final per-degree bootstrap report with raw mean/std columns."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _report_rows_from_bootstrap(winners, bootstrap_rows, report_metrics)
    metric_names = _report_metric_names(report_metrics)
    write_result_rows_fixed_width(
        rows,
        ["MODEL"] + metric_names,
        str(out_path),
        metric_names,
        include_seed=False,
        align_error_header_left=True,
    )
    return out_path
