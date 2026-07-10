"""Orchestration for the signature-degree ablation on the test split.

Ties together validation-file selection (``sig_degree_selection``) and report
rendering (``sig_degree_report``) with a bootstrap test-evaluation step. The
test evaluation itself is injected (``recompute_one_row_fn`` / ``write_npz_fn``),
so this module stays torch-free and unit-testable; the training pipeline passes
the real ``recompute_bootstrap`` helpers.

This is invoked from ``TrainingManager`` right after the single-winner test pass
and before model pruning, so every per-degree winner's checkpoint is still on
disk when :func:`run_sig_degree_ablation_from_val_file` reloads it.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


from test.paper_experiments.sig_degree_report import REPORT_METRICS, write_report
from test.paper_experiments.sig_degree_selection import (
    select_winner_rows_by_sig_degree,
    winner_names_from_rows,
)


@dataclass
class SigDegreeAblationOutputs:
    """Paths and settings produced by one ablation run."""

    val_tuning_path: Path
    report_path: Path
    npz_path: Path
    n_bootstraps: int


def recompute_winners_on_test(
    winners: Dict[int, str],
    settings: Any,
    gpu_id: Optional[int],
    recompute_one_row_fn: Callable[..., Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluate each validation-selected winner on test, preserving failures as rows."""
    rows: List[Dict[str, Any]] = []
    total = len(winners)
    for index, degree in enumerate(sorted(winners), start=1):
        run_name = winners[degree]
        start = time.perf_counter()
        logger.info(
            "[%d/%d] Evaluating sig_degree=%d winner on test: %s",
            index,
            total,
            degree,
            run_name,
        )
        try:
            row = recompute_one_row_fn(run_name, settings, gpu_id=gpu_id)
        except Exception as exc:  # Defensive: keep the degree visible in the report.
            logger.exception("Unexpected bootstrap recompute failure for %s", run_name)
            row = {"model_name": run_name, "metrics": {"error": str(exc)}}

        row = dict(row)
        row["model_name"] = run_name
        row.setdefault("metrics", {})
        elapsed = time.perf_counter() - start
        error_msg = row.get("metrics", {}).get("error")
        if error_msg:
            logger.warning("[%d/%d] %s failed in %.1fs: %s", index, total, run_name, elapsed, error_msg)
        else:
            logger.info("[%d/%d] %s done in %.1fs", index, total, run_name, elapsed)
        rows.append(row)
    return rows


def run_sig_degree_ablation_from_val_file(
    val_tuning_path: Path,
    version: str,
    *,
    n_bootstraps: int,
    trainer_seed: int,
    gpu_id: Optional[int],
    recompute_one_row_fn: Callable[..., Dict[str, Any]],
    write_npz_fn: Callable[[List[Dict[str, Any]], str, int], None],
    settings_cls: Callable[..., Any],
    report_metrics: Sequence[str] = REPORT_METRICS,
    results_dir: Optional[Path] = None,
) -> SigDegreeAblationOutputs:
    """Select per-degree winners from ``val_tuning_path``, test them, write the report.

    ``val_tuning_path`` must be the exact validation tuning file produced by the
    run being ablated -- callers must not rediscover it by globbing a shared
    results directory for "the latest" file, since a concurrent run sharing the
    same ``version``/directory can write a newer file first and get silently
    selected instead (see ``ExperimentResults.save``). By default output
    text/NPZ files land next to ``val_tuning_path``, but callers can route the
    paired ablation artifacts to a dedicated folder. Selection reads only the
    validation tuning file (no test leakage); a degree whose winner fails or is
    missing is retained with NaN metric values.
    """
    val_tuning_path = Path(val_tuning_path)
    winner_rows = select_winner_rows_by_sig_degree(val_tuning_path)
    if not winner_rows:
        raise ValueError(
            f"No sig-degree winners in {val_tuning_path}: every row was dropped because its "
            "name has no `_sig_<d>`/`_rela<offset>` token (see the warning just above). Usually "
            "a fixed custom_file_name (a local 'test_debug_warning' run) collapsed the grid onto "
            "one name -- re-run with server_training: true for the ablation."
        )

    winners = winner_names_from_rows(winner_rows)
    settings = settings_cls(n_bootstraps=n_bootstraps, trainer_seed=trainer_seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(results_dir) if results_dir is not None else val_tuning_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"{version}_sig_degree_ablation_B{n_bootstraps}_{timestamp}.txt"
    npz_path = results_dir / f"{version}_sig_degree_ablation_B{n_bootstraps}_{timestamp}.npz"

    logger.info("Selected %d per-degree winner(s) from %s", len(winners), val_tuning_path)
    for degree in sorted(winners):
        logger.info("  sig_degree %d -> %s", degree, winners[degree])

    bootstrap_rows = recompute_winners_on_test(winners, settings, gpu_id, recompute_one_row_fn)
    write_report(winners, bootstrap_rows, report_path, report_metrics)
    write_npz_fn(list(bootstrap_rows), str(npz_path), n_bootstraps)

    logger.info("Sig-degree ablation report saved to: %s", report_path)
    logger.info("Sig-degree ablation per-replicate NPZ saved to: %s", npz_path)
    return SigDegreeAblationOutputs(
        val_tuning_path=val_tuning_path,
        report_path=report_path,
        npz_path=npz_path,
        n_bootstraps=n_bootstraps,
    )
