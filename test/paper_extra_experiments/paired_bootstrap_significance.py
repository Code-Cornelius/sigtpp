"""Paired significance tests from per-replicate bootstrap ``.npz`` files.

The input files are written by ``test/paper_experiments/recompute_bootstrap.py``
or by the final winner test pass in ``trainingmanager.py``.  They contain
per-model, per-metric vectors over the same bootstrap replicate axis, which is
what makes paired tests valid.

Outputs:

* ``paired_tests.csv``: one row per dataset / metric / method pair.
* ``paired_outcomes.csv``: corrected-test win/tie/loss counts.
* ``paired_outcomes_latex.tex``: compact paper-ready table from the all-metric
  rows of ``paired_outcomes.csv``.

Usage:

    python -c "import sys; sys.path.insert(0, 'src'); exec(open('test/paper_extra_experiments/paired_bootstrap_significance.py').read())" \
        -- test/paper_experiments/out/results_bootstrap_YYYYMMDD_HHMMSS_gpu0.npz
"""

import argparse
import glob
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from test.paper_experiments.dataset_names import DATA_NAME_TO_EXPERIMENT


SCHEMA_VERSION = 1

DATASET_PREFIXES = sorted(DATA_NAME_TO_EXPERIMENT, key=len, reverse=True)

DATASET_DISPLAY_NAMES: Dict[str, str] = {
    "poisson_three_marks": "P3",
    "inh_poisson_three_marks": "IHP3",
    "hawkes": "H1",
    "hawkes_3x3": "H3",
    "taxi": "TX",
    "stackoverflow": "SO",
    "taobao": "TB",
    "earthquake": "EQ",
    "yelp_mississauga": "YLP",
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "deter": "Deter",
    "gamma": "Gamma",
    "ddpm": "DDPM",
    "vae": "VAE",
    "wgan": "WGAN",
    "sigtpp": "SigTPP",
    "ssm": "SSM",
}

MODEL_ALIASES: Dict[str, str] = {
    "score": "ddpm",
    "sigwgan": "sigtpp",
}

FILENAME_MODEL_TOKENS: Tuple[str, ...] = (
    "deter",
    "gamma",
    "score",
    "ddpm",
    "vae",
    "wgan",
    "sigwgan",
    "sigtpp",
    "ssm",
)

DEFAULT_BASELINES: Tuple[str, ...] = ("deter", "gamma", "ddpm", "vae", "wgan")

SYNTHETIC_DATASETS = frozenset(
    {
        "poisson_three_marks",
        "inh_poisson_three_marks",
        "hawkes",
        "hawkes_3x3",
    }
)
REAL_DATASETS = frozenset(
    {
        "earthquake",
        "stackoverflow",
        "taobao",
        "taxi",
        "yelp_mississauga",
    }
)

HIGHER_IS_BETTER = frozenset({"top1_mark_acc", "top3_mark_acc"})

ANALYSIS_METRICS: Tuple[str, ...] = (
    "ED",
    "W1",
    "sigW_loword_notstd",
    "CRPS",
    "hist_int_flat",
    "hist_it_flat",
    "corr",
    "corr_short",
    "autocorr",
    "autocorr_short",
    "MAE_proper",
    "MSE_proper",
)

TEST_FAMILIES: Tuple[str, ...] = ("paired_t", "wilcoxon", "dm")


@dataclass(frozen=True)
class ReplicateRecord:
    source_path: str
    model_name: str
    dataset: str
    model: str
    metric: str
    values: np.ndarray


def parse_run_name(run_name: str, valid_models: Sequence[str] = FILENAME_MODEL_TOKENS) -> Tuple[str, str]:
    """Return ``(dataset_experiment_type, canonical_model)`` from a run name."""
    data_name = suffix = None
    for prefix in DATASET_PREFIXES:
        if run_name == prefix:
            data_name, suffix = prefix, ""
            break
        if run_name.startswith(prefix + "_"):
            data_name, suffix = prefix, run_name[len(prefix) + 1 :]
            break
    if data_name is None:
        raise ValueError(f"Could not parse dataset prefix from run name: {run_name!r}")

    for model in sorted(valid_models, key=len, reverse=True):
        if suffix == model or suffix.startswith(model + "_"):
            return DATA_NAME_TO_EXPERIMENT.get(data_name, data_name), MODEL_ALIASES.get(model, model)

    raise ValueError(f"Could not parse model token from run name: {run_name!r}")


def split_label(dataset: str) -> str:
    if dataset in SYNTHETIC_DATASETS:
        return "synthetic"
    if dataset in REAL_DATASETS:
        return "real"
    return "other"


def load_npz_records(npz_paths: Sequence[str]) -> List[ReplicateRecord]:
    """Load schema-v1 per-replicate bootstrap files into typed records."""
    records: List[ReplicateRecord] = []
    for path in npz_paths:
        with np.load(path, allow_pickle=True) as npz:
            schema = int(npz["schema_version"])
            if schema != SCHEMA_VERSION:
                raise ValueError(f"{path}: unsupported schema_version={schema}, expected {SCHEMA_VERSION}")
            B = int(npz["B"])
            model_names = [str(x) for x in npz["model_names"].tolist()]
            metric_names = [str(x) for x in npz["metric_names"].tolist()]
            data = np.asarray(npz["data"], dtype=float)
            expected_shape = (len(model_names), len(metric_names), B)
            if data.shape != expected_shape:
                raise ValueError(f"{path}: data shape {data.shape}, expected {expected_shape}")

            for i, model_name in enumerate(model_names):
                dataset, model = parse_run_name(model_name)
                for j, metric in enumerate(metric_names):
                    records.append(
                        ReplicateRecord(
                            source_path=str(path),
                            model_name=model_name,
                            dataset=dataset,
                            model=model,
                            metric=metric,
                            values=data[i, j, :].astype(float, copy=True),
                        )
                    )
    return records


def _record_map(records: Sequence[ReplicateRecord]) -> Dict[Tuple[str, str, str], ReplicateRecord]:
    out: Dict[Tuple[str, str, str], ReplicateRecord] = {}
    duplicates: List[Tuple[str, str, str]] = []
    for record in records:
        key = (record.dataset, record.model, record.metric)
        if key in out:
            duplicates.append(key)
        out[key] = record
    if duplicates:
        sample = ", ".join(map(str, duplicates[:5]))
        raise ValueError(
            "Duplicate dataset/model/metric records in paired-test input. "
            f"Use one selected run per method before testing. Examples: {sample}"
        )
    return out


def _oriented_losses(values: np.ndarray, metric: str) -> np.ndarray:
    """Return values oriented so lower is always better."""
    if metric in HIGHER_IS_BETTER:
        return -values
    return values


def _paired_t(compared_loss: np.ndarray, baseline_loss: np.ndarray, alternative: str) -> Tuple[float, float]:
    try:
        result = stats.ttest_rel(compared_loss, baseline_loss, alternative=alternative)
        return float(result.statistic), float(result.pvalue)
    except TypeError:
        diff = compared_loss - baseline_loss
        n = len(diff)
        if n < 2:
            return float("nan"), float("nan")
        se = diff.std(ddof=1) / math.sqrt(n)
        if se == 0:
            stat = math.copysign(float("inf"), diff.mean()) if diff.mean() != 0 else 0.0
        else:
            stat = diff.mean() / se
        if alternative == "less":
            p = stats.t.cdf(stat, df=n - 1)
        elif alternative == "greater":
            p = stats.t.sf(stat, df=n - 1)
        else:
            p = 2.0 * min(stats.t.cdf(stat, df=n - 1), stats.t.sf(stat, df=n - 1))
        return float(stat), float(p)


def _wilcoxon(compared_loss: np.ndarray, baseline_loss: np.ndarray, alternative: str) -> Tuple[float, float]:
    diff = compared_loss - baseline_loss
    if np.allclose(diff, 0.0, equal_nan=False):
        return 0.0, 1.0
    try:
        result = stats.wilcoxon(
            compared_loss,
            baseline_loss,
            alternative=alternative,
            zero_method="wilcox",
            method="auto",
        )
    except TypeError:
        result = stats.wilcoxon(
            compared_loss,
            baseline_loss,
            alternative=alternative,
            zero_method="wilcox",
        )
    return float(result.statistic), float(result.pvalue)


def _diebold_mariano(
    compared_loss: np.ndarray,
    baseline_loss: np.ndarray,
    alternative: str,
    lags: int = 0,
) -> Tuple[float, float]:
    """Diebold-Mariano test on oriented paired loss differentials.

    ``lags=0`` is the h=1 case.  Larger values use Bartlett-weighted
    autocovariances as a Newey-West long-run variance estimate.
    """
    diff = compared_loss - baseline_loss
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan")

    mean_diff = float(diff.mean())
    centered = diff - mean_diff
    max_lag = min(max(0, int(lags)), n - 1)
    lr_var = float(np.dot(centered, centered) / n)
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        lr_var += 2.0 * weight * cov

    if lr_var <= 0.0 or not np.isfinite(lr_var):
        if mean_diff == 0.0:
            return 0.0, 1.0
        return math.copysign(float("inf"), mean_diff), 0.0

    stat = mean_diff / math.sqrt(lr_var / n)
    if alternative == "less":
        p = stats.t.cdf(stat, df=n - 1)
    elif alternative == "greater":
        p = stats.t.sf(stat, df=n - 1)
    else:
        p = 2.0 * min(stats.t.cdf(stat, df=n - 1), stats.t.sf(stat, df=n - 1))
    return float(stat), float(p)


def adjust_pvalues(p_values: Sequence[float], method: str, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return adjusted p-values and reject flags for a correction method."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    finite = p[finite_mask]
    if len(finite) == 0:
        return adjusted, np.zeros_like(p, dtype=bool)

    method = method.lower()
    if method in {"none", "uncorrected"}:
        adjusted[finite_mask] = finite
    elif method == "bonferroni":
        adjusted[finite_mask] = np.minimum(finite * len(finite), 1.0)
    elif method == "holm":
        order = np.argsort(finite)
        sorted_p = finite[order]
        sorted_adj = np.empty_like(sorted_p)
        running = 0.0
        m = len(sorted_p)
        for rank, pv in enumerate(sorted_p):
            running = max(running, (m - rank) * pv)
            sorted_adj[rank] = min(running, 1.0)
        unsorted = np.empty_like(sorted_adj)
        unsorted[order] = sorted_adj
        adjusted[finite_mask] = unsorted
    elif method in {"fdr_bh", "bh"}:
        order = np.argsort(finite)
        sorted_p = finite[order]
        m = len(sorted_p)
        sorted_adj = np.empty_like(sorted_p)
        running = 1.0
        for idx in range(m - 1, -1, -1):
            running = min(running, sorted_p[idx] * m / (idx + 1))
            sorted_adj[idx] = running
        unsorted = np.empty_like(sorted_adj)
        unsorted[order] = np.minimum(sorted_adj, 1.0)
        adjusted[finite_mask] = unsorted
    else:
        raise ValueError(f"Unknown p-value correction method: {method!r}")

    return adjusted, np.asarray(adjusted <= alpha, dtype=bool)


def run_paired_tests(
    records: Sequence[ReplicateRecord],
    *,
    compared_model: str = "sigtpp",
    baseline_models: Optional[Sequence[str]] = DEFAULT_BASELINES,
    metrics: Sequence[str] = ANALYSIS_METRICS,
    alternative: str = "two-sided",
    correction: str = "holm",
    alpha: float = 0.05,
    tie_atol: float = 1e-12,
    tie_rtol: float = 1e-12,
    dm_lags: int = 0,
) -> pd.DataFrame:
    """Compute paired tests for all available dataset / metric pairs."""
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError(f"alternative must be one of two-sided/less/greater, got {alternative!r}")

    by_key = _record_map(records)
    datasets = sorted({r.dataset for r in records})
    metrics_set = set(metrics)
    rows: List[Dict[str, object]] = []

    for dataset in datasets:
        for metric in metrics:
            compared = by_key.get((dataset, compared_model, metric))
            if compared is None:
                continue

            if baseline_models is None:
                baselines = sorted(
                    model for ds, model, m in by_key if ds == dataset and m == metric and model != compared_model
                )
            else:
                baselines = list(baseline_models)

            for baseline_model in baselines:
                baseline = by_key.get((dataset, baseline_model, metric))
                if baseline is None:
                    continue

                compared_values = np.asarray(compared.values, dtype=float)
                baseline_values = np.asarray(baseline.values, dtype=float)
                mask = np.isfinite(compared_values) & np.isfinite(baseline_values)
                compared_valid = compared_values[mask]
                baseline_valid = baseline_values[mask]
                n = len(compared_valid)
                if n < 2:
                    continue

                compared_loss = _oriented_losses(compared_valid, metric)
                baseline_loss = _oriented_losses(baseline_valid, metric)
                diff_loss = compared_loss - baseline_loss

                ties = np.isclose(compared_loss, baseline_loss, atol=tie_atol, rtol=tie_rtol)
                wins = compared_loss < baseline_loss
                losses = compared_loss > baseline_loss

                paired_t_stat, paired_t_p = _paired_t(compared_loss, baseline_loss, alternative)
                wilcoxon_stat, wilcoxon_p = _wilcoxon(compared_loss, baseline_loss, alternative)
                dm_stat, dm_p = _diebold_mariano(compared_loss, baseline_loss, alternative, lags=dm_lags)

                baseline_mean = float(np.mean(baseline_valid))
                compared_mean = float(np.mean(compared_valid))
                rel_imp = (
                    (baseline_mean - compared_mean) / baseline_mean * 100.0
                    if metric not in HIGHER_IS_BETTER and baseline_mean != 0
                    else float("nan")
                )
                if metric in HIGHER_IS_BETTER and baseline_mean != 0:
                    rel_imp = (compared_mean - baseline_mean) / abs(baseline_mean) * 100.0

                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_display": DATASET_DISPLAY_NAMES.get(dataset, dataset),
                        "split": split_label(dataset),
                        "metric": metric,
                        "alternative": alternative,
                        "baseline_method": baseline_model,
                        "baseline_display": MODEL_DISPLAY_NAMES.get(baseline_model, baseline_model),
                        "compared_method": compared_model,
                        "compared_display": MODEL_DISPLAY_NAMES.get(compared_model, compared_model),
                        "n": n,
                        "baseline_mean": baseline_mean,
                        "compared_mean": compared_mean,
                        "mean_oriented_loss_diff": float(np.mean(diff_loss)),
                        "relative_improvement_pct": rel_imp,
                        "replicate_wins": int(np.sum(wins & ~ties)),
                        "replicate_ties": int(np.sum(ties)),
                        "replicate_losses": int(np.sum(losses & ~ties)),
                        "paired_t_stat": paired_t_stat,
                        "paired_t_p": paired_t_p,
                        "wilcoxon_stat": wilcoxon_stat,
                        "wilcoxon_p": wilcoxon_p,
                        "dm_stat": dm_stat,
                        "dm_p": dm_p,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    missing_metrics = sorted(metrics_set - set(df["metric"]))
    if missing_metrics:
        print(f"WARNING: no paired rows emitted for metrics: {missing_metrics}")

    for family in TEST_FAMILIES:
        adj, reject = adjust_pvalues(df[f"{family}_p"].to_numpy(dtype=float), correction, alpha)
        df[f"{family}_p_adj"] = adj
        df[f"{family}_reject"] = reject

    return df


def build_outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate corrected-test outcomes into win/tie/loss counts."""
    if df.empty:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    split_order = ["synthetic", "real", "all"]
    pairs = df[["baseline_method", "baseline_display", "compared_method", "compared_display"]].drop_duplicates()

    for family in TEST_FAMILIES:
        reject_col = f"{family}_reject"
        p_adj_col = f"{family}_p_adj"
        for _, pair in pairs.iterrows():
            pair_df = df[
                (df["baseline_method"] == pair["baseline_method"]) & (df["compared_method"] == pair["compared_method"])
            ]
            metrics = list(pair_df["metric"].drop_duplicates()) + ["__all_metrics__"]
            for metric in metrics:
                metric_df = pair_df if metric == "__all_metrics__" else pair_df[pair_df["metric"] == metric]
                for split in split_order:
                    split_df = metric_df if split == "all" else metric_df[metric_df["split"] == split]
                    if split_df.empty:
                        continue

                    alt = split_df["alternative"].iloc[0] if "alternative" in split_df.columns else "two-sided"
                    if alt == "less":
                        is_win = split_df[reject_col]
                        is_loss = split_df[reject_col] & False
                    elif alt == "greater":
                        is_win = split_df[reject_col] & False
                        is_loss = split_df[reject_col]
                    else:
                        is_win = (split_df[reject_col]) & (split_df["mean_oriented_loss_diff"] < 0)
                        is_loss = (split_df[reject_col]) & (split_df["mean_oriented_loss_diff"] > 0)
                    records.append(
                        {
                            "test_family": family,
                            "baseline_method": pair["baseline_method"],
                            "baseline_display": pair["baseline_display"],
                            "compared_method": pair["compared_method"],
                            "compared_display": pair["compared_display"],
                            "metric": metric,
                            "split": split,
                            "significant_wins": int(is_win.sum()),
                            "ties_or_not_significant": int((~(is_win | is_loss)).sum()),
                            "significant_losses": int(is_loss.sum()),
                            "total": int(len(split_df)),
                            "median_adjusted_p": float(split_df[p_adj_col].median()),
                        }
                    )
    return pd.DataFrame(records)


def write_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    """Write a compact paper-ready LaTeX table."""
    if summary.empty:
        path.write_text("% No paired-test rows were available.\n", encoding="utf-8")
        return

    table = summary[(summary["metric"] == "__all_metrics__") & (summary["split"] == "all")].copy()
    table["comparison"] = table["compared_display"] + " vs " + table["baseline_display"]
    table["W/T/L"] = (
        table["significant_wins"].astype(str)
        + "/"
        + table["ties_or_not_significant"].astype(str)
        + "/"
        + table["significant_losses"].astype(str)
    )
    table["median_adjusted_p"] = table["median_adjusted_p"].map(lambda x: f"{x:.3g}")
    table = table[["test_family", "comparison", "W/T/L", "total", "median_adjusted_p"]]
    latex = table.to_latex(index=False, escape=True)
    path.write_text(latex, encoding="utf-8")


def _default_latest_npz(root_dir: str) -> List[str]:
    pattern = os.path.join(root_dir, "test", "paper_experiments", "out", "results_bootstrap*.npz")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1:] if matches else []


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    argv = [a for a in argv if a != "--"]
    parser = argparse.ArgumentParser(description="Run paired tests on per-replicate bootstrap .npz files.")
    parser.add_argument("npz_paths", nargs="*", help="Per-replicate bootstrap .npz files.")
    parser.add_argument("--compared-model", default="sigtpp", help="Model token to compare against baselines.")
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=list(DEFAULT_BASELINES),
        help="Baseline model tokens. Pass no values after the flag to compare against every other present model.",
    )
    parser.add_argument("--metrics", nargs="*", default=list(ANALYSIS_METRICS), help="Metrics to test.")
    parser.add_argument(
        "--alternative",
        choices=["two-sided", "less", "greater"],
        default="two-sided",
        help="Paired-test alternative on oriented losses. 'less' means compared model is better.",
    )
    parser.add_argument(
        "--correction",
        choices=["holm", "bonferroni", "fdr_bh", "none"],
        default="holm",
        help="Multiple-comparison correction applied separately to each test family.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dm-lags", type=int, default=0, help="HAC lags for the Diebold-Mariano variance estimate.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("test", "paper_extra_experiments", "out", "tables", "paired_bootstrap"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    root_dir = os.getcwd()
    npz_paths = list(args.npz_paths) or _default_latest_npz(root_dir)
    if not npz_paths:
        raise SystemExit("No .npz path supplied and no results_bootstrap*.npz found in test/paper_experiments/out.")

    baselines = args.baselines
    if baselines == []:
        baselines = None

    records = load_npz_records(npz_paths)
    tests = run_paired_tests(
        records,
        compared_model=args.compared_model,
        baseline_models=baselines,
        metrics=args.metrics,
        alternative=args.alternative,
        correction=args.correction,
        alpha=args.alpha,
        dm_lags=args.dm_lags,
    )
    summary = build_outcome_summary(tests)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(output_dir / "paired_tests.csv", index=False)
    summary.to_csv(output_dir / "paired_outcomes.csv", index=False)
    write_latex_summary(summary, output_dir / "paired_outcomes_latex.tex")

    print(f"Loaded {len(records)} metric vectors from {len(npz_paths)} .npz file(s).")
    print(f"Wrote {len(tests)} paired-test rows to {output_dir / 'paired_tests.csv'}.")
    print(f"Wrote {len(summary)} outcome rows to {output_dir / 'paired_outcomes.csv'}.")
    print(f"Wrote LaTeX summary to {output_dir / 'paired_outcomes_latex.tex'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
