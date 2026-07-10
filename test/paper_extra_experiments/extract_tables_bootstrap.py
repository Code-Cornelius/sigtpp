"""Script 2: rebuild tables from a bootstrap-enriched results file.

Port of ``extract_data_from_raw_results.py`` adapted for the bootstrap format
produced by :mod:`recompute_bootstrap`.  The bootstrap txt has an explicit
header row with ``<metric>_mean`` / ``<metric>_std`` column names; parsing reads
that header dynamically instead of relying on a fixed positional column list.

Per-loss tables combine mean and std in the compact notation ``11.1(3)``, where
the parenthesised digit is the std expressed in units of the last displayed
decimal place (standard concise uncertainty notation).  Best cell per column
gets ``\\bfseries``, second-best gets ``\\underline{}`` (from ulem).

Two-step pipeline:

Step 1 – Parse bootstrap txt into a structured DataFrame.
Step 2 – Reshape into per-loss CSV pivot files and a combined structured CSV.

Formulas used in pair_scores / pair_aggregates
-----------------------------------------------
All analysis metrics are lower-is-better.  "compared" = SigTPP, "baseline" = reference model.

relative_improvement_pct (per dataset)::

    100 * (baseline_mean - compared_mean) / baseline_mean

    Positive → compared outperforms baseline.

score_ratio (per dataset)::

    compared_mean / baseline_mean

    < 1 → compared is better.  Only defined when both values are strictly positive.

geo_mean_score_ratio (aggregate over datasets)::

    exp( mean( log(score_ratio) ) )

    Geometric mean of per-dataset score ratios.  Equivalent to the exponent of
    the mean log-ratio, which treats multiplicative deviations symmetrically.

geo_relative_improvement_pct (aggregate over datasets)::

    100 * (1 - geo_mean_score_ratio)

    Positive → compared outperforms baseline on average across datasets.

Usage::

    python -c "import sys; sys.path.insert(0, 'src'); \\
        exec(open('test/paper_extra_experiments/extract_tables_bootstrap.py').read())"
"""

import math
import os
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from test.paper_experiments.recompute_bootstrap import DATA_NAME_TO_EXPERIMENT
from test.paper_extra_experiments._metric_format import (
    DECIMAL_PLACES as _SHARED_DECIMAL_PLACES,
    SCALE_EXPONENTS as _SHARED_SCALE_EXPONENTS,
    format_mean_std_cell,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Longest-first so prefix matching never stops at a partial match.
# These are the *data-name* prefixes embedded in run names, not experiment types.
DATASET_PREFIXES = sorted(
    [
        "hp_three_marks",
        "ihp_three_marks",
        "hawkes_3x3",
        "hawkes",
        "taxi",
        "stackoverflow",
        "taobao",
        "yelp_mississauga",
        "earthquake",
    ],
    key=len,
    reverse=True,
)

# Keys are experiment_type names (values of DATA_NAME_TO_EXPERIMENT) so that
# dataset labels in the parsed DataFrame match DATASET_DISPLAY_NAMES.
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
    # "score": "DDPM",  # legacy filename token (renamed to "ddpm")
    "vae": "VAE",
    "wgan": "WGAN",
    "sigtpp": "SigTPP",
    # "sigwgan": "SigTPP",  # legacy filename token (renamed to "sigtpp")
    "ssm": "SSM",
}

# Result files on disk may use legacy model tokens ("score", "sigwgan") or the
# current ones ("ddpm", "sigtpp"). parse_run_name normalises legacy tokens to the
# canonical form so downstream selection/aggregation is name-agnostic; because both
# tokens already share a display name, generated tables are unchanged.
MODEL_ALIASES: Dict[str, str] = {
    "score": "ddpm",
    "sigwgan": "sigtpp",
}

# experiment_type names (values of DATA_NAME_TO_EXPERIMENT).
SYNTHETIC_DATASETS: FrozenSet[str] = frozenset(
    {
        "poisson_three_marks",
        "inh_poisson_three_marks",
        "hawkes",
        "hawkes_3x3",
    }
)

REAL_DATASETS: FrozenSet[str] = frozenset(
    {
        "earthquake",
        "stackoverflow",
        "taobao",
        "taxi",
        "yelp_mississauga",
    }
)

# Metrics where higher is better (bold the maximum instead of the minimum).
HIGHER_IS_BETTER: FrozenSet[str] = frozenset(
    {
        "top1_mark_acc",
        "top3_mark_acc",
    }
)

# Shared with the sig-degree ablation report (test/paper_experiments/sig_degree_report.py) via _metric_format.py.
TABLE_SCALE_EXPONENTS: Dict[str, int] = _SHARED_SCALE_EXPONENTS
TABLE_DECIMAL_PLACES: Dict[str, int] = _SHARED_DECIMAL_PLACES

# Losses to extract for the output tables. save_per_loss_files reads
# <loss_name>_mean (and _std) columns from the DataFrame.
LOSSES_TO_EXTRACT: List[str] = [
    "sigW_loword_notstd",
    "hist_it",
    "hist_int",
    "hist_it_flat",
    "hist_int_flat",
    "ED",
    "W1",
    "CRPS",
    "corr",
    "corr_short",
    "autocorr_it",
    "autocorr_it_short",
    "autocorr",
    "autocorr_short",
    "MAE_proper",
    "MSE_proper",
    "MAE",
    "mark_ce",
    "top1_mark_acc",
    "top3_mark_acc",
    "train_time",
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_rank_cell(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1f}"


# ---------------------------------------------------------------------------
# Step 1 – parsing
# ---------------------------------------------------------------------------


def parse_run_name(run_name: str, valid_models: Sequence[str]) -> Tuple[str, str]:
    """Return (dataset_experiment_type, model) parsed from a bootstrap run name.

    Run names embed the data-name prefix (e.g. ``hp_three_marks``).  The
    function maps that prefix to its experiment_type via DATA_NAME_TO_EXPERIMENT
    so the returned ``dataset`` key matches DATASET_DISPLAY_NAMES.

    ``valid_models`` is iterated longest-first so that prefixes like ``score``
    don't shadow longer names like ``score_v2``. Callers that invoke this in a
    hot loop should pre-sort once (see parse_results_file).
    """
    data_name = suffix = None
    for prefix in DATASET_PREFIXES:
        if run_name == prefix:
            data_name, suffix = prefix, ""
            break
        if run_name.startswith(prefix + "_"):
            data_name, suffix = prefix, run_name[len(prefix) + 1 :]
            break

    if data_name is None:
        raise ValueError(f"Could not parse dataset prefix from run name: {run_name}")

    experiment_type = DATA_NAME_TO_EXPERIMENT.get(data_name, data_name)

    for model in valid_models:
        if suffix == model or suffix.startswith(model + "_"):
            return experiment_type, MODEL_ALIASES.get(model, model)

    raise ValueError(f"Could not parse model from run name: {run_name}")


def parse_results_file(
    input_path: Union[str, Path],
    valid_models: Sequence[str],
    deduplicate_first: bool = True,
) -> pd.DataFrame:
    """Read the bootstrap-enriched txt file and return a structured DataFrame.

    The txt file has an explicit header row produced by ExperimentResults._write_txt.
    Column names are read from that header; metric values are parsed positionally.
    Empty lines are skipped silently.
    Multi-word ERROR strings are reconstructed by joining tokens beyond the
    known metric columns.
    """
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Sort once: parse_run_name needs longest-first iteration so prefix matching never
    # stops at a partial match like "score" inside "score_v2".
    sorted_models = sorted(valid_models, key=len, reverse=True)

    # First non-empty line must be the MODEL header.
    header_tokens = None
    header_line_idx = 0
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "MODEL":
            header_tokens = parts[1:]
            header_line_idx = i
            break

    if header_tokens is None:
        raise ValueError(f"No MODEL header found in {input_path!r}")

    # ERROR is appended as the last column only when rows failed.
    has_error_col = bool(header_tokens) and header_tokens[-1] == "ERROR"
    metric_columns = header_tokens[:-1] if has_error_col else header_tokens
    n_metrics = len(metric_columns)

    for line_number, line in enumerate(lines[header_line_idx + 1 :], start=header_line_idx + 2):
        parts = line.strip().split()
        if not parts:
            continue  # skip empty lines

        run_name, value_tokens = parts[0], parts[1:]

        # Metric values occupy the first n_metrics tokens.
        # Everything after is the (possibly multi-word) ERROR message.
        raw_values = value_tokens[:n_metrics]
        error_str = " ".join(value_tokens[n_metrics:]) if len(value_tokens) > n_metrics else None

        if len(raw_values) != n_metrics:
            print(
                f"WARNING: Line {line_number}: expected {n_metrics} metric values, "
                f"got {len(raw_values)} — skipping (run_name={run_name!r})"
            )
            continue

        try:
            dataset, model = parse_run_name(run_name, sorted_models)
        except ValueError as exc:
            print(f"WARNING: Line {line_number}: {exc} — skipping")
            continue

        metric_values = pd.to_numeric(pd.Series(raw_values, dtype="string"), errors="coerce").tolist()

        row = {"dataset": dataset, "model": model}
        row.update(dict(zip(metric_columns, metric_values)))
        if error_str:
            row["error"] = error_str
        rows.append(row)

    df = pd.DataFrame(rows)

    duplicates = df[df.duplicated(subset=["dataset", "model"], keep="first")]
    if not duplicates.empty:
        pairs = duplicates[["dataset", "model"]].drop_duplicates()
        print(f"WARNING: {len(duplicates)} duplicate row(s) found and dropped (keeping first):")
        for _, row in pairs.iterrows():
            print(f"  - dataset={row['dataset']!r}, model={row['model']!r}")

    if deduplicate_first:
        df = df.drop_duplicates(subset=["dataset", "model"], keep="first").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 2 – reshaping
# ---------------------------------------------------------------------------


def _prepare(
    df: pd.DataFrame,
    model_display_names: Optional[Mapping[str, str]],
    dataset_display_names: Optional[Mapping[str, str]],
    dataset_filter: Optional[str] = None,
) -> pd.DataFrame:
    work = df.copy()
    if dataset_filter is not None:
        work = work.loc[work["dataset"] == dataset_filter].copy()
    if model_display_names is not None:
        work["model"] = work["model"].map(model_display_names).fillna(work["model"])
    if dataset_display_names is not None:
        work["dataset"] = work["dataset"].map(dataset_display_names).fillna(work["dataset"])
    return work


def _mean_rank_by_dataset_group(
    pivot: pd.DataFrame,
    dataset_columns: Sequence[str],
    ascending: bool,
) -> pd.Series:
    """Average model rank over the provided dataset columns."""
    if not dataset_columns:
        return pd.Series(float("nan"), index=pivot.index, dtype="float64")
    return pivot.loc[:, dataset_columns].rank(axis=0, ascending=ascending, na_option="keep").mean(axis=1)


def _write_formatted_pivot_csv(
    mean_pivot: pd.DataFrame,
    std_pivot: Optional[pd.DataFrame],
    out_path: Path,
    loss_name: str,
    scale_exponent: Optional[int],
    decimal_places: int,
    higher_is_better: bool,
) -> None:
    """Write a pivot CSV with LaTeX formatting.

    Value cells use the concise ``mean(std)`` notation.  Rank cells use plain
    floats.  Best cell per column gets ``\\bfseries``, second-best ``\\underline{}``.
    All comparisons use the raw mean float so the string format never interferes.
    """
    export_mean = mean_pivot.reset_index()
    export_std = std_pivot.reset_index() if std_pivot is not None else None

    formatted = export_mean.copy().astype(object)

    for col in export_mean.columns:
        if col == "model":
            continue

        is_rank = str(col).startswith("rank")

        # Collect finite mean values to determine best / second-best.
        try:
            col_means = pd.to_numeric(export_mean[col], errors="coerce").dropna()
        except Exception:
            col_means = pd.Series(dtype=float)

        sorted_unique = sorted(col_means.unique(), reverse=(higher_is_better and not is_rank))
        best_val = sorted_unique[0] if sorted_unique else None
        second_val = sorted_unique[1] if len(sorted_unique) > 1 else None

        for i in export_mean.index:
            try:
                mean_f = float(export_mean.loc[i, col])
                if pd.isna(mean_f):
                    raise ValueError
            except (TypeError, ValueError):
                formatted.loc[i, col] = ""
                continue

            # Build the cell string.
            if is_rank:
                cell = _format_rank_cell(mean_f)
            else:
                std_f = float("nan")
                if export_std is not None and col in export_std.columns:
                    try:
                        std_f = float(export_std.loc[i, col])
                    except (TypeError, ValueError):
                        pass
                cell = format_mean_std_cell(mean_f, std_f, scale_exponent, decimal_places)

            # Apply LaTeX highlighting.
            if not is_rank and best_val is not None and mean_f == best_val:
                cell = r"\bfseries " + cell
            elif not is_rank and second_val is not None and mean_f == second_val:
                cell = r"\underline{" + cell + "}"

            formatted.loc[i, col] = cell

    formatted.to_csv(out_path, index=False)


def save_per_loss_files(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    losses: Sequence[str],
    valid_models: Sequence[str],
    model_display_names: Optional[Mapping[str, str]] = None,
    dataset_display_names: Optional[Mapping[str, str]] = None,
    dataset_filter: Optional[str] = None,
) -> None:
    """Save one CSV per loss.

    Format: rows = models, columns = datasets.
    Missing (model, dataset) combinations are left empty.
    Value cells: ``mean(std)`` with LaTeX highlighting.

        model,H1,EQ
        \\bfseries 9.2(3),\\uline{10.1(2)}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work = _prepare(df, model_display_names, dataset_display_names, dataset_filter)

    dupes = work[work.duplicated(subset=["dataset", "model"], keep=False)]
    if not dupes.empty:
        pairs = dupes[["dataset", "model"]].drop_duplicates()
        raise ValueError(
            "Duplicate (dataset, model) pairs after applying display names — "
            "likely two raw datasets share the same display name:\n"
            + "\n".join(f"  dataset={r['dataset']!r}, model={r['model']!r}" for _, r in pairs.iterrows())
        )

    if model_display_names is not None:
        model_order = [model_display_names.get(m, m) for m in valid_models]
    else:
        model_order = list(valid_models)

    synthetic_labels = {
        dataset_display_names.get(ds, ds) if dataset_display_names is not None else ds for ds in SYNTHETIC_DATASETS
    }
    real_labels = {
        dataset_display_names.get(ds, ds) if dataset_display_names is not None else ds for ds in REAL_DATASETS
    }

    for loss_name in losses:
        mean_col = f"{loss_name}_mean"
        std_col = f"{loss_name}_std"

        if mean_col not in work.columns:
            print(f"WARNING: column {mean_col!r} not found — skipping loss {loss_name!r}")
            continue

        all_dataset_labels = [
            lbl
            for lbl in (
                dataset_display_names.get(ds, ds) if dataset_display_names is not None else ds
                for ds in DATASET_DISPLAY_NAMES
            )
            if lbl in synthetic_labels | real_labels
        ]

        mean_pivot = work.pivot(index="model", columns="dataset", values=mean_col)
        mean_pivot = mean_pivot.reindex(index=model_order, columns=all_dataset_labels)
        mean_pivot.index.name = "model"
        mean_pivot.columns.name = None

        std_pivot: Optional[pd.DataFrame] = None
        if std_col in work.columns:
            std_pivot = work.pivot(index="model", columns="dataset", values=std_col)
            std_pivot = std_pivot.reindex(index=model_order, columns=all_dataset_labels)
            std_pivot.index.name = "model"
            std_pivot.columns.name = None

        known_labels = synthetic_labels | real_labels
        unknown_labels = [col for col in mean_pivot.columns if col not in known_labels]
        if unknown_labels:
            raise ValueError(f"Unclassified datasets for ranking in loss '{loss_name}': {unknown_labels}")

        synthetic_columns = [col for col in mean_pivot.columns if col in synthetic_labels]
        real_columns = [col for col in mean_pivot.columns if col in real_labels]

        ascending = loss_name not in HIGHER_IS_BETTER
        mean_pivot["rank_synthetic"] = _mean_rank_by_dataset_group(mean_pivot, synthetic_columns, ascending=ascending)
        mean_pivot["rank_real"] = _mean_rank_by_dataset_group(mean_pivot, real_columns, ascending=ascending)

        out_path = output_dir / f"{loss_name}.csv"
        scale_exponent = TABLE_SCALE_EXPONENTS.get(loss_name)
        decimal_places = TABLE_DECIMAL_PLACES.get(loss_name, 3)
        _write_formatted_pivot_csv(
            mean_pivot,
            std_pivot,
            out_path,
            loss_name,
            scale_exponent,
            decimal_places,
            higher_is_better=(loss_name in HIGHER_IS_BETTER),
        )


# ---------------------------------------------------------------------------
# Metrics included in SigTPP analysis tables
# ---------------------------------------------------------------------------

ANALYSIS_METRICS: List[str] = [
    "ED",
    "W1",
    "sigW_loword_notstd",
    "CRPS",
    "hist_int_flat",
    "hist_it_flat",
    "autocorr_it_short",
    "corr",
]

PAIRWISE_BASELINE_MODELS: List[str] = [
    "wgan",
    "vae",
    "ddpm",
]


def score_ratio(sigtpp_mean: float, best_baseline_mean: float) -> float:
    """Return ``sigtpp_mean / best_baseline_mean`` when both inputs are positive."""
    if pd.isna(sigtpp_mean) or pd.isna(best_baseline_mean):
        return float("nan")
    if sigtpp_mean <= 0 or best_baseline_mean <= 0:
        return float("nan")

    return sigtpp_mean / best_baseline_mean


def _geometric_mean_ratio(score_ratios: Sequence[float]) -> float:
    if not score_ratios:
        return float("nan")

    valid_logs = []
    for ratio in score_ratios:
        if pd.isna(ratio) or ratio <= 0:
            return float("nan")
        valid_logs.append(math.log(float(ratio)))

    return math.exp(sum(valid_logs) / len(valid_logs))


def geometric_relative_improvement_pct(score_ratios: Sequence[float]) -> float:
    """Return the geometric relative improvement in percent from positive score ratios."""
    geo_mean_ratio = _geometric_mean_ratio(score_ratios)
    if pd.isna(geo_mean_ratio):
        return float("nan")

    return 100.0 * (1.0 - geo_mean_ratio)


# ---------------------------------------------------------------------------
# SigTPP analysis: pair_scores, pair_aggregates, model_ranks
# ---------------------------------------------------------------------------


def build_sigtpp_analysis_tables(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    sigtpp_model: str = "sigtpp",
    analysis_metrics: Optional[Sequence[str]] = None,
    deter_model: str = "deter",
    model_display_names: Optional[Mapping[str, str]] = None,
) -> None:
    """Build sigTPP analysis tables comparing models pairwise.

    Convention: all metrics are treated as lower-is-better.
    Uncertainty values in the source tables are bootstrap standard errors.

    Datasets whose name is absent from SYNTHETIC_DATASETS / REAL_DATASETS
    are silently excluded.

    Ranking ties are broken by average rank (pandas method='average').

    Cross-metric aggregates are stored as metric="__all_metrics__" sentinel rows
    within the per-metric tables (same convention everywhere).

    Saves CSV files to output_dir:
        relative_improv.csv
            One row per (baseline_method, compared_method, metric, dataset).
            Columns: baseline_method, compared_method, metric, dataset, split,
            baseline_mean, compared_mean, relative_improvement_pct.
            relative_improvement_pct = 100 * (baseline_mean - compared_mean) / baseline_mean;
            positive => `compared_method` outperforms `baseline_method`.
            baseline_method in {wgan, vae, DDPM, deter, __best_excl_SigTPP__}.

        pair_aggregates.csv
            Aggregates of pair_scores per (baseline_method, compared_method,
            metric, split, formula). Columns: baseline_method, compared_method,
            metric, split, formula, value, n.
            formula in {mean_relative_improvement_pct, geo_mean_score_ratio, geo_relative_improvement_pct}.

        model_ranks.csv
            One row per (model, metric, split). Columns: model, metric, split,
            avg_rank, std_rank, n. Rank 1 = best (lowest mean).
            metric="__all_metrics__" pools across metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if analysis_metrics is None:
        analysis_metrics = ANALYSIS_METRICS

    def _rn(series: pd.Series) -> pd.Series:
        if model_display_names is None:
            return series
        return series.map(lambda x: model_display_names.get(x, x))

    _sigtpp_display = (model_display_names or {}).get(sigtpp_model, sigtpp_model)
    _best_excl_sentinel = f"__best_excl_{_sigtpp_display}__"

    # Only classified datasets
    classified = SYNTHETIC_DATASETS | REAL_DATASETS
    work = df[df["dataset"].isin(classified)].copy()

    datasets_all = sorted(work["dataset"].unique())
    datasets_syn = sorted(d for d in datasets_all if d in SYNTHETIC_DATASETS)
    datasets_real = sorted(d for d in datasets_all if d in REAL_DATASETS)

    def split_label(ds: str) -> str:
        return "synthetic" if ds in SYNTHETIC_DATASETS else "real"

    # ------------------------------------------------------------------
    # Build cell_data[metric][dataset] = {model: (mean, se), ...}
    # ------------------------------------------------------------------
    cell_data: Dict[str, Dict[str, Dict[str, tuple]]] = {}
    valid_metrics: List[str] = []
    for metric in analysis_metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in work.columns:
            print(f"WARNING: {mean_col!r} not found — skipping metric {metric!r}")
            continue
        cell_data[metric] = {}
        for ds in datasets_all:
            ds_df = work[work["dataset"] == ds]
            cell: Dict[str, tuple] = {}
            for _, row in ds_df.iterrows():
                m_val = row[mean_col]
                s_val = row[std_col] if std_col in work.columns else float("nan")
                if not pd.isna(m_val):
                    cell[row["model"]] = (float(m_val), float(s_val) if not pd.isna(s_val) else float("nan"))
            cell_data[metric][ds] = cell
        valid_metrics.append(metric)

    # ------------------------------------------------------------------
    # relative_improv.csv  – one row per (baseline_method, compared_method,
    # metric, dataset). Subsumes former Tables A, A_pairwise, H_A.
    # Convention: relative_improvement_pct = 100*(baseline - compared)/baseline,
    # so positive values mean `compared_method` outperforms `baseline_method`.
    # All ANALYSIS_METRICS are lower-is-better.
    # ------------------------------------------------------------------
    split_sort_order = {"synthetic": 0, "real": 1, "all": 2}
    records_pair_scores: List[Dict[str, object]] = []

    def _emit_pair(
        baseline_method: str,
        compared_method: str,
        baseline_mean: float,
        compared_mean: float,
        metric: str,
        ds: str,
    ) -> None:
        if pd.isna(baseline_mean) or baseline_mean == 0:
            rel_imp = float("nan")
        else:
            rel_imp = (baseline_mean - compared_mean) / baseline_mean * 100.0
        records_pair_scores.append(
            {
                "baseline_method": baseline_method,
                "compared_method": compared_method,
                "metric": metric,
                "dataset": DATASET_DISPLAY_NAMES.get(ds, ds),
                "split": split_label(ds),
                "baseline_mean": baseline_mean,
                "compared_mean": compared_mean,
                "relative_improvement_pct": rel_imp,
            }
        )

    for metric in valid_metrics:
        for ds in datasets_all:
            cell = cell_data[metric][ds]

            # sigwgan compared against each named baseline + the best non-sigwgan baseline
            if sigtpp_model in cell:
                sig_mean = cell[sigtpp_model][0]
                for baseline_model in PAIRWISE_BASELINE_MODELS:
                    if baseline_model in cell:
                        _emit_pair(
                            baseline_model,
                            sigtpp_model,
                            cell[baseline_model][0],
                            sig_mean,
                            metric,
                            ds,
                        )
                baselines = {m: v[0] for m, v in cell.items() if m != sigtpp_model}
                if baselines:
                    best = min(baselines, key=baselines.__getitem__)
                    _emit_pair(
                        _best_excl_sentinel,
                        sigtpp_model,
                        baselines[best],
                        sig_mean,
                        metric,
                        ds,
                    )

            # deter as baseline vs every other model present
            if deter_model in cell:
                deter_mean_val = cell[deter_model][0]
                if not pd.isna(deter_mean_val) and deter_mean_val > 0:
                    for model_name, (model_mean_val, _) in cell.items():
                        if model_name != deter_model:
                            _emit_pair(
                                deter_model,
                                model_name,
                                deter_mean_val,
                                model_mean_val,
                                metric,
                                ds,
                            )

    df_pair_scores = pd.DataFrame(records_pair_scores)
    df_pair_scores_save = (
        df_pair_scores.assign(
            _split_order=df_pair_scores["split"].map(split_sort_order).fillna(len(split_sort_order)),
        )
        .sort_values(["baseline_method", "compared_method", "metric", "_split_order", "dataset"])
        .drop(columns=["_split_order"])
        .reset_index(drop=True)
    )
    df_pair_scores_save = df_pair_scores_save.assign(
        baseline_method=_rn(df_pair_scores_save["baseline_method"]),
        compared_method=_rn(df_pair_scores_save["compared_method"]),
    )
    df_pair_scores_save.round(4).to_csv(output_dir / "relative_improv.csv", index=False)
    print(f"[relative_improv] {len(df_pair_scores_save)} rows.")

    # ------------------------------------------------------------------
    # pair_aggregates.csv  – aggregates of pair_scores per
    # (baseline_method, compared_method, metric, split, formula).
    # metric="__all_metrics__" pools across metrics.
    # Subsumes former Tables B, B_pairwise, H_B.
    # ------------------------------------------------------------------
    df_ps = df_pair_scores.copy()
    df_ps["score_ratio"] = [
        score_ratio(sigtpp_mean=c, best_baseline_mean=b) for c, b in zip(df_ps["compared_mean"], df_ps["baseline_mean"])
    ]

    records_pair_agg: List[Dict[str, object]] = []
    pairs = df_ps[["baseline_method", "compared_method"]].drop_duplicates().itertuples(index=False, name=None)
    for baseline_method, compared_method in pairs:
        pair_df = df_ps[(df_ps["baseline_method"] == baseline_method) & (df_ps["compared_method"] == compared_method)]
        for metric in valid_metrics + ["__all_metrics__"]:
            m_df = pair_df if metric == "__all_metrics__" else pair_df[pair_df["metric"] == metric]
            for split_name in ["synthetic", "real", "all"]:
                split_df = m_df if split_name == "all" else m_df[m_df["split"] == split_name]

                rel_vals = split_df["relative_improvement_pct"].dropna()
                ratios = split_df["score_ratio"].dropna().tolist()

                common = {
                    "baseline_method": baseline_method,
                    "compared_method": compared_method,
                    "metric": metric,
                    "split": split_name,
                }
                records_pair_agg.append(
                    {
                        **common,
                        "formula": "mean_relative_improvement_pct",
                        "value": rel_vals.mean() if len(rel_vals) else float("nan"),
                        "n": len(rel_vals),
                    }
                )
                records_pair_agg.append(
                    {
                        **common,
                        "formula": "geo_mean_score_ratio",
                        "value": _geometric_mean_ratio(ratios),
                        "n": len(ratios),
                    }
                )
                records_pair_agg.append(
                    {
                        **common,
                        "formula": "geo_relative_improvement_pct",
                        "value": geometric_relative_improvement_pct(ratios),
                        "n": len(ratios),
                    }
                )

    df_pair_agg = pd.DataFrame(records_pair_agg)
    df_pair_agg = df_pair_agg.assign(
        baseline_method=_rn(df_pair_agg["baseline_method"]),
        compared_method=_rn(df_pair_agg["compared_method"]),
    )
    df_pair_agg.round(4).to_csv(output_dir / "pair_aggregates.csv", index=False)
    print(f"[pair_aggregates] {len(df_pair_agg)} rows.")

    # ------------------------------------------------------------------
    # win_counts.csv  – win / within-std / underperform counts for SigTPP.
    # win        : SigTPP has the lowest mean of ALL methods on that (metric, dataset).
    # within_std : not a win, but sig_mean ≤ best_mean + best_se.
    # underperform: otherwise.
    # metric="__all_metrics__" rows pool across all metrics.
    # ------------------------------------------------------------------
    records_wc_raw = []
    for metric in valid_metrics:
        for ds in datasets_all:
            cell = cell_data[metric][ds]
            if sigtpp_model not in cell or len(cell) < 2:
                continue
            all_means = {m: v[0] for m, v in cell.items()}
            best_model = min(all_means, key=all_means.__getitem__)
            best_mean = all_means[best_model]
            best_se = cell[best_model][1]
            sig_mean = all_means[sigtpp_model]
            if sig_mean == best_mean:
                cat = "win"
            elif not pd.isna(best_se) and sig_mean <= best_mean + best_se:
                cat = "within_std"
            else:
                cat = "underperform"
            records_wc_raw.append({"metric": metric, "dataset": ds, "split": split_label(ds), "category": cat})

    df_wc_raw = pd.DataFrame(records_wc_raw)
    records_wc = []
    for metric in valid_metrics + ["__all_metrics__"]:
        m_df = df_wc_raw if metric == "__all_metrics__" else df_wc_raw[df_wc_raw["metric"] == metric]
        for split_name, split_ds in [("all", datasets_all), ("synthetic", datasets_syn), ("real", datasets_real)]:
            sub = m_df[m_df["dataset"].isin(split_ds)]
            if sub.empty:
                continue
            records_wc.append(
                {
                    "metric": metric,
                    "split": split_name,
                    "wins": int((sub["category"] == "win").sum()),
                    "within_std": int((sub["category"] == "within_std").sum()),
                    "underperform": int((sub["category"] == "underperform").sum()),
                    "total": len(sub),
                }
            )

    pd.DataFrame(records_wc).to_csv(output_dir / "win_counts.csv", index=False)
    print(f"[win_counts] {len(records_wc)} rows.")

    # ------------------------------------------------------------------
    # model_ranks.csv  – one row per (model, metric, split).
    # Rank 1 = lowest mean = best. Ties resolved by average rank.
    # metric="__all_metrics__" pools across metrics.
    # ------------------------------------------------------------------
    rank_records = []
    for metric in valid_metrics:
        for ds in datasets_all:
            cell = cell_data[metric][ds]
            if len(cell) < 2:
                continue
            means_s = pd.Series({m: v[0] for m, v in cell.items()})
            ranks_s = means_s.rank(method="average", ascending=True)
            for model, rank in ranks_s.items():
                rank_records.append(
                    {
                        "metric": metric,
                        "dataset": ds,
                        "split": split_label(ds),
                        "model": model,
                        "rank": rank,
                    }
                )

    df_ranks = pd.DataFrame(rank_records)

    records_ranks: List[Dict[str, object]] = []
    for model_name in sorted(df_ranks["model"].unique()):
        m_df_full = df_ranks[df_ranks["model"] == model_name]
        for metric in valid_metrics + ["__all_metrics__"]:
            m_df = m_df_full if metric == "__all_metrics__" else m_df_full[m_df_full["metric"] == metric]
            for split_name, split_ds in [
                ("synthetic", datasets_syn),
                ("real", datasets_real),
                ("all", datasets_all),
            ]:
                s_df = m_df[m_df["dataset"].isin(split_ds)]
                records_ranks.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "split": split_name,
                        "avg_rank": round(float(s_df["rank"].mean()), 2) if len(s_df) else float("nan"),
                        "std_rank": round(float(s_df["rank"].std()), 2) if len(s_df) > 1 else float("nan"),
                        "n": len(s_df),
                    }
                )

    df_model_ranks = pd.DataFrame(records_ranks)

    # Sort: specific metrics by split → model → metric;
    # __all_metrics__ sentinel rows last, ordered model → split.
    _split_ord = {"synthetic": 0, "real": 1, "all": 2}
    _metric_ord = {m: i for i, m in enumerate(valid_metrics)}
    _model_ord_map = {m: i for i, m in enumerate(sorted(df_model_ranks["model"].unique()))}
    _is_all = (df_model_ranks["metric"] == "__all_metrics__").astype(int)
    _split_key = df_model_ranks["split"].map(_split_ord)
    _metric_key = df_model_ranks["metric"].map(_metric_ord).fillna(len(valid_metrics)).astype(int)
    _model_key = df_model_ranks["model"].map(_model_ord_map)
    df_model_ranks = (
        df_model_ranks.assign(
            _k0=_is_all,
            _k2=_is_all * _model_key + (1 - _is_all) * _split_key,
            _k3=_is_all * _split_key + (1 - _is_all) * _model_key,
            _k4=(1 - _is_all) * _metric_key,
        )
        .sort_values(["_k0", "_k2", "_k3", "_k4"])
        .drop(columns=["_k0", "_k2", "_k3", "_k4"])
        .reset_index(drop=True)
    )

    df_model_ranks = df_model_ranks.assign(model=_rn(df_model_ranks["model"]))
    df_model_ranks.to_csv(output_dir / "model_ranks.csv", index=False)
    print(f"[model_ranks] {len(df_model_ranks)} rows.")

    print(f"\nAll analysis tables saved to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import ROOT_DIR

    _paper_exp_out = os.path.join(ROOT_DIR, "test/paper_experiments/out")
    _extra_exp_out = os.path.join(ROOT_DIR, "test/paper_extra_experiments/out")

    # Tokens that may appear in result filenames: legacy ("score", "sigwgan") and
    # current ("ddpm", "sigtpp"). parse_run_name normalises legacy → canonical.
    filename_tokens = [
        "deter",
        "gamma",
        "score",
        "ddpm",
        "vae",
        "wgan",
        "sigwgan",
        "sigtpp",
    ]
    # Canonical models used for coverage / pivoting / display (post-normalisation).
    valid_models = [
        "deter",
        "gamma",
        "ddpm",
        "vae",
        "wgan",
        "sigtpp",
    ]

    # Step 1 – parse bootstrap txt into structured DataFrame
    df = parse_results_file(
        input_path=os.path.join(_paper_exp_out, "results_bootstrap.txt"),
        valid_models=filename_tokens,
    )

    print(df.head())
    print(f"\nParsed {len(df)} rows from results_bootstrap.txt")

    # Coverage check – which models are present / missing per dataset
    all_datasets = sorted(df["dataset"].unique())
    present = set(zip(df["dataset"], df["model"]))
    coverage = pd.DataFrame(
        {m: ["OK" if (ds, m) in present else "MISS" for ds in all_datasets] for m in valid_models},
        index=all_datasets,
    )
    coverage.index.name = "dataset"
    print("\nModel coverage:")
    print(coverage.to_string())

    missing_models = [m for m in valid_models if all((ds, m) not in present for ds in all_datasets)]
    if missing_models:
        print(f"\nWARNING: these models appear in no dataset: {missing_models}")

    unmapped_datasets = [ds for ds in all_datasets if ds not in DATASET_DISPLAY_NAMES]
    if unmapped_datasets:
        print(f"\nWARNING: these datasets have no display name mapping: {unmapped_datasets}")

    unseen_dataset_keys = [k for k in DATASET_DISPLAY_NAMES if k not in all_datasets]
    if unseen_dataset_keys:
        print(f"\nWARNING: these DATASET_DISPLAY_NAMES keys are not in the data (typo?): {unseen_dataset_keys}")

    # Step 2 – reshape into pivot tables (rows=model, cols=dataset)
    save_per_loss_files(
        df=df,
        output_dir=os.path.join(_extra_exp_out, "tables/per_loss"),
        losses=LOSSES_TO_EXTRACT,
        valid_models=valid_models,
        model_display_names=MODEL_DISPLAY_NAMES,
        dataset_display_names=DATASET_DISPLAY_NAMES,
    )
    print(f"\nTables saved to {os.path.join(_extra_exp_out, 'tables/per_loss')}")

    # Step 3 – SigTPP analysis tables (pair_scores, pair_aggregates, model_ranks, table_D)
    print("\n--- Building SigTPP analysis tables ---")
    build_sigtpp_analysis_tables(
        df=df,
        output_dir=os.path.join(_extra_exp_out, "tables/sigtpp_analysis"),
        sigtpp_model="sigtpp",
        analysis_metrics=ANALYSIS_METRICS,
        model_display_names=MODEL_DISPLAY_NAMES,
    )
