"""Helpers for config-driven multi-seed training runs and reports.

The regular :mod:`experiment_results` schema uses ``*_mean`` / ``*_std`` for
bootstrap summaries within a single trained model, and plain names for metrics
computed once outside the bootstrap loop. Multi-seed summaries use
``*_seed_mean`` / ``*_seed_std`` / ``*_seed_n_valid`` so the reported standard
deviation is unambiguously across independently trained seeds.
"""

import logging
import os
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


logger = logging.getLogger(__name__)

from test.paper_experiments.experiment_results import ExperimentResults
from src.utils.result_helpers import apply_competition_ranking, summarise_values, write_result_rows_fixed_width

SeedResults = Sequence[Tuple[int, ExperimentResults]]

# Multi-seed sub-runs append a ``_seed<N>`` tag to the model name (see
# ``build_seed_config`` and ``get_dir_name_from_params``/``get_model_name`` in
# ``training_helpers.py``). Both name forms end in exactly this token.
_SEED_SUFFIX_RE = re.compile(r"_seed\d+$")


def strip_seed_suffix(model_name: str) -> str:
    """Return ``model_name`` with a trailing ``_seed<digits>`` tag removed.

    The seed is baked into per-seed model names so checkpoints stay distinct on
    disk. Across-seed aggregation must group by the seed-independent config
    identity, so it peels this exact tag form (anchored to the end, digits only)
    before bucketing. A name that legitimately contains ``_seed<N>`` elsewhere is
    untouched; only the trailing tag is stripped.
    """
    return _SEED_SUFFIX_RE.sub("", model_name)


def seeds_from_config(cfg: Dict[str, Any]) -> List[int]:
    """Return the configured seed list."""
    assert "seeds" in cfg, f"Config must define 'seeds', e.g. seeds: [42]. Got keys {list(cfg)}."
    raw_seeds = cfg["seeds"]
    assert isinstance(
        raw_seeds, (list, tuple)
    ), f"'seeds' must be a list, e.g. seeds: [42]. Got {type(raw_seeds).__name__}."
    assert raw_seeds, f"'seeds' must contain at least one seed. Got {raw_seeds!r}."
    return [int(seed) for seed in raw_seeds]


def build_seed_config(base_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Copy ``base_cfg`` and make it safe to run for one seed.

    ``TrainingManager.run`` resolves the single-seed scalar via ``_resolve_single_seed``,
    which requires ``seeds`` to be a one-element list. Each per-seed sub-run therefore
    needs ``seeds=[seed]`` rather than ``seed=seed``.

    The ``_multiseed_seed_tag`` flag is read by each settings module's ``model_namer``
    (via ``get_model_name``) to append ``_seed<N>`` to the model directory name. This
    keeps per-seed checkpoints inside the shared ``out/<experiment>/models/`` tree
    while still distinguishing them.
    """
    cfg = deepcopy(base_cfg)
    cfg["seeds"] = [int(seed)]
    cfg["_multiseed_seed_tag"] = int(seed)
    original_b = int(base_cfg.get("n_bootstraps", 1))
    if original_b != 1:
        logger.error(
            "Multi-seed sub-run seed=%d: forcing n_bootstraps from %d to 1 so that "
            "<metric>_seed_std measures across-seed variance only (bootstrap variance "
            "would be silently dropped at aggregation otherwise).",
            seed,
            original_b,
        )
    cfg["n_bootstraps"] = 1
    # Multi-seed reports across-seed variance only. Per-seed bootstrap refinement
    # would silently produce one _refine_*.txt per seed sub-run and confuse the
    # output layout, so disable it inside the multi-seed expansion.
    cfg["refine_best_n_bootstraps"] = None
    return cfg


def write_multiseed_outputs(seed_results: SeedResults, version: str, results_dir: str) -> Tuple[str, str]:
    """Write the per-seed long file and the across-seed summary file."""
    assert seed_results, f"write_multiseed_outputs requires at least one seed result, got {seed_results!r}."

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    by_seed_path = os.path.join(results_dir, f"{version}_multiseed_per_seed_{timestamp}.txt")
    summary_path = os.path.join(results_dir, f"{version}_multiseed_summary_{timestamp}.txt")

    write_multiseed_by_seed_txt(seed_results, by_seed_path)
    summary_rows = aggregate_across_seeds(seed_results)
    write_multiseed_summary_txt(summary_rows, summary_path)

    logger.info("Multi-seed per-seed TXT saved to %s", by_seed_path)
    logger.info("Multi-seed seed-summary TXT saved to %s", summary_path)
    return by_seed_path, summary_path


def aggregate_across_seeds(seed_results: SeedResults) -> List[Dict[str, Any]]:
    """Aggregate per-seed metrics into explicit seed-summary columns.

    Contract: this function reads only ``val_<metric>_mean`` (validation
    diagnostics; ``<metric>_mean`` for split-independent metrics) from each
    per-seed row.
    Any per-seed ``*_std`` (bootstrap variance) is silently discarded. The
    multi-seed runner forces ``n_bootstraps=1`` so the dropped values are
    structurally zero; if a non-zero ``*_std`` is encountered here, the caller
    has bypassed that override and the resulting ``*_seed_std`` will be missing
    the within-seed bootstrap component of total variance. We emit a warning
    rather than raise so legacy callers keep working, but the output should
    not be treated as a valid total-variance estimate in that case.
    """
    # Group by the seed-independent config identity, not the seed-suffixed name:
    # per-seed rows are named ``..._seed<N>`` so the same config must have its tag
    # stripped to land in one bucket across seeds (otherwise every bucket is a
    # singleton and the across-seed std collapses to NaN).
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for _seed, result in seed_results:
        for row in result.rows:
            base_name = strip_seed_suffix(row["model_name"])
            if base_name not in grouped:
                grouped[base_name] = []
                order.append(base_name)
            grouped[base_name].append(row["metrics"])
    warn_if_bootstrap_std_would_be_dropped(grouped)

    rows: List[Dict[str, Any]] = []
    for model_name in order:
        metrics: Dict[str, Any] = {}
        seed_metrics = grouped[model_name]
        all_metrics = list(ExperimentResults.DISPLAY_METRICS) + list(ExperimentResults.EXTRA_METRICS)
        for metric in all_metrics:
            # Per-seed rows carry validation diagnostics (``val_`` namespace);
            # split-independent metrics such as train_time keep plain names.
            # ``_flat`` histograms are test-only artifacts and never aggregated.
            if metric.endswith("_flat"):
                continue
            base = metric if metric in ExperimentResults.SPLIT_INDEPENDENT_METRICS else f"val_{metric}"
            mean, std, n_valid = summarise_values(
                m.get(f"{base}_mean", m.get(base, float("nan"))) for m in seed_metrics
            )
            metrics[f"{base}_seed_n_valid"] = n_valid
            metrics[f"{base}_seed_mean"] = mean
            metrics[f"{base}_seed_std"] = std
        rows.append({"model_name": model_name, "metrics": metrics})

    apply_seed_ranking(rows)
    return rows


def write_multiseed_by_seed_txt(seed_results: SeedResults, path: str) -> None:
    # Per-seed rows are validation diagnostics: val_ columns, val_norm_score ranking.
    metric_names = ExperimentResults.build_column_names(prefix="val_")
    rows = [
        {"seed": seed, "model_name": row["model_name"], "metrics": row["metrics"]}
        for seed, result in seed_results
        for row in result.normalize_and_rank(prefix="val_").rows
    ]
    write_fixed_width_rows(rows, ["SEED", "MODEL"] + metric_names, path, metric_names, include_seed=True)


def aggregate_test_rows_across_seeds(test_rows: Sequence[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate one config's per-seed test rows into explicit seed-summary columns.

    Every row in ``test_rows`` must be the SAME config: the one chosen by
    ``aggregate_across_seeds``'s validation ranking, evaluated on test for
    every seed by the caller (see ``TrainingManager.evaluate_named_model_on_test``).
    Mirrors ``aggregate_across_seeds``'s per-metric reduction, but reads
    unprefixed test-metric names and never groups across configs -- there is
    exactly one config here, so no seed-suffix stripping or bucketing is
    needed.
    """
    metrics: Dict[str, Any] = {}
    seed_metrics = [row["metrics"] for _seed, row in test_rows]
    all_metrics = list(ExperimentResults.DISPLAY_METRICS) + list(ExperimentResults.EXTRA_METRICS)
    for metric in all_metrics:
        if metric.endswith("_flat"):
            continue
        mean, std, n_valid = summarise_values(
            m.get(f"{metric}_mean", m.get(metric, float("nan"))) for m in seed_metrics
        )
        metrics[f"{metric}_seed_n_valid"] = n_valid
        metrics[f"{metric}_seed_mean"] = mean
        metrics[f"{metric}_seed_std"] = std
    return metrics


def seed_test_summary_column_names() -> List[str]:
    names: List[str] = []
    for metric in list(ExperimentResults.DISPLAY_METRICS) + list(ExperimentResults.EXTRA_METRICS):
        if metric.endswith("_flat"):
            continue
        names.extend([f"{metric}_seed_mean", f"{metric}_seed_std", f"{metric}_seed_n_valid"])
    return names


def write_multiseed_test_by_seed_txt(
    test_rows: Sequence[Tuple[int, Dict[str, Any]]],
    version: str,
    results_dir: str,
) -> str:
    """Write per-seed test rows for the single cross-seed-selected config.

    Unlike the retired per-seed-winner test report, every row here is
    guaranteed to be the SAME model_name: the config chosen once by
    ``aggregate_across_seeds``'s validation ranking, evaluated on test for
    every seed. Test metrics keep unprefixed names, consistent with
    single-run final test reports.
    """
    assert test_rows, f"write_multiseed_test_by_seed_txt requires at least one row, got {test_rows!r}."

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(results_dir, f"{version}_multiseed_test_by_seed_{timestamp}.txt")

    metric_names = ExperimentResults.build_column_names()
    rows = [{"seed": seed, "model_name": row["model_name"], "metrics": row["metrics"]} for seed, row in test_rows]
    write_fixed_width_rows(rows, ["SEED", "MODEL"] + metric_names, path, metric_names, include_seed=True)
    logger.info("Multi-seed fixed-config test-by-seed TXT saved to %s", path)
    return path


def write_multiseed_test_summary_txt(
    winner_name: str,
    summary_metrics: Dict[str, Any],
    version: str,
    results_dir: str,
) -> str:
    """Write one row: the cross-seed-selected config's test mean/std/n_valid.

    This is the paper-level number: one config, chosen once on validation
    (averaged across seeds via ``aggregate_across_seeds``), evaluated on
    held-out test for every seed and reduced via
    ``aggregate_test_rows_across_seeds``.
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(results_dir, f"{version}_multiseed_test_summary_{timestamp}.txt")

    metric_names = seed_test_summary_column_names()
    rows = [{"model_name": winner_name, "metrics": summary_metrics}]
    write_fixed_width_rows(rows, ["MODEL"] + metric_names, path, metric_names, include_seed=False)
    logger.info("Multi-seed fixed-config test-summary TXT saved to %s", path)
    return path


def write_multiseed_summary_txt(rows: List[Dict[str, Any]], path: str) -> None:
    """Write one row per model with explicit across-seed aggregate columns."""
    metric_names = seed_summary_column_names()
    write_fixed_width_rows(rows, ["MODEL"] + metric_names, path, metric_names, include_seed=False)


def warn_if_bootstrap_std_would_be_dropped(
    grouped: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Log a warning if any per-seed row carries a non-zero bootstrap ``*_std``.

    Multi-seed aggregation reads only ``*_mean`` keys, so any non-zero
    ``<metric>_std`` would be silently dropped. The runner forces
    ``n_bootstraps=1`` to keep that drop harmless; this guard fires when
    that contract has been bypassed (e.g. ``aggregate_across_seeds`` invoked from
    a notebook on rows produced with ``B > 1``).
    """
    offenders: List[str] = []
    for model_name, seed_rows in grouped.items():
        for metrics in seed_rows:
            for key, value in metrics.items():
                if not key.endswith("_std") or key.endswith("_seed_std"):
                    continue
                fv = float(value)
                if not np.isnan(fv) and fv != 0.0:
                    offenders.append(f"{model_name}:{key}={fv:g}")
                    break  # one example per model is enough; keep the log short
            if offenders and offenders[-1].startswith(model_name + ":"):
                break
    if offenders:
        logger.warning(
            "aggregate_across_seeds: dropping non-zero bootstrap *_std on %d model(s) "
            "(%s%s). The resulting *_seed_std reports across-seed variance only; "
            "the within-seed bootstrap component is lost. Run with n_bootstraps=1 "
            "per seed to silence this warning.",
            len(offenders),
            ", ".join(offenders[:3]),
            "..." if len(offenders) > 3 else "",
        )


def seed_summary_column_names() -> List[str]:
    names: List[str] = []
    for metric in list(ExperimentResults.DISPLAY_METRICS) + list(ExperimentResults.EXTRA_METRICS):
        if metric.endswith("_flat"):
            continue
        base = metric if metric in ExperimentResults.SPLIT_INDEPENDENT_METRICS else f"val_{metric}"
        names.extend(
            [
                f"{base}_seed_mean",
                f"{base}_seed_std",
                f"{base}_seed_n_valid",
            ]
        )
    for metric in ExperimentResults.RANKING_METRICS:
        names.append(f"rank_val_{metric}")
    names.append("val_norm_score")
    return names


def apply_seed_ranking(rows: List[Dict[str, Any]]) -> None:
    """Rank seed summaries by validation seed means (hyperparameter selection)."""
    apply_competition_ranking(
        rows,
        ExperimentResults.RANKING_METRICS,
        lambda metric: f"val_{metric}_seed_mean",
        score_key="val_norm_score",
        rank_key_for_metric=lambda metric: f"rank_val_{metric}",
    )


def write_fixed_width_rows(
    rows: List[Dict[str, Any]],
    column_names: List[str],
    path: str,
    metric_names: List[str],
    *,
    include_seed: bool,
) -> None:
    write_result_rows_fixed_width(
        rows,
        column_names,
        path,
        metric_names,
        include_seed=include_seed,
    )
