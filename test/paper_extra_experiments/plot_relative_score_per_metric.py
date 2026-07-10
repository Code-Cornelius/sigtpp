"""Plot normalised score (GMRAE vs Deter) — average + all metrics on one axis, models as hue.

X-axis: Average (leftmost) then individual metrics.
Y-axis: Normalised score = geo-mean(model / deter) across datasets. Lower is better.

Reference: Armstrong & Collopy (1992), Fleming & Wallace (1986).

Usage::

    python -c "import sys; sys.path.insert(0, 'src'); exec(open('test/paper_extra_experiments/plot_relative_score_per_metric.py').read())"
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ROOT_DIR
from src.logger.init_logger import set_config_logging
from test.paper_extra_experiments._plot_style import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    apply_paper_style,
)

set_config_logging()
logger = logging.getLogger(__name__)

apply_paper_style()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(ROOT_DIR)
INPUT_CSV_CANDIDATES = (
    ROOT / "test/paper_extra_experiments/out/tables/sigtpp_analysis/pair_aggregates.csv",
    ROOT / "test/paper_extra_experiments/out/tables/sigTPP_analysis/pair_aggregates.csv",
)
INPUT_CSV = next((path for path in INPUT_CSV_CANDIDATES if path.exists()), INPUT_CSV_CANDIDATES[0])
OUTPUT_DIR = ROOT / "test/paper_extra_experiments/out/figures"

SPLITS = ["all", "synthetic", "real"]

MODEL_DISPLAY = {
    "Gamma": r"\textsc{Gamma}",
    "DDPM": r"\textsc{DDPM}",
    "VAE": r"\textsc{VAE}",
    "WGAN": r"\textsc{WGAN}",
    "sigtpp": r"\textsc{SigTPP}",
}

MODEL_NAME_ALIASES = {
    "SigTPP": "sigtpp",
    "Sig-TPP": "sigtpp",
    "sig-tpp": "sigtpp",
}

METRIC_DISPLAY = {
    "ED": r"$\mathcal{E}$",
    "W1": r"$\mathrm{W}_1$",
    "sigW_loword_notstd": r"$\mathrm{Sig\text{-}W}_1$",
    "CRPS": r"CRPS",
    "hist_int_flat": r"$L_{\lambda}$",
    "hist_it_flat": r"$L_{\log(\tau)}$",
    "autocorr_it_short": r"$\mathrm{ACD}$",
    "corr": r"$\mathrm{PCD}$",
}

AVG_LABEL = r"Average"


def _geo_mean(values: pd.Series) -> float:
    """Geometric mean of strictly-positive values (NaNs ignored). Returns NaN if empty."""
    arr = values.dropna().to_numpy()
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.log(arr).mean()))


# ---------------------------------------------------------------------------
# Load and reshape
# ---------------------------------------------------------------------------

df_raw = pd.read_csv(INPUT_CSV)
df_raw["baseline_method"] = df_raw["baseline_method"].replace(MODEL_NAME_ALIASES)
df_raw["compared_method"] = df_raw["compared_method"].replace(MODEL_NAME_ALIASES)
logger.info("Loaded %d rows from %s", len(df_raw), INPUT_CSV)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    # ---------------------------------------------------------------------------
    # Load and reshape
    # ---------------------------------------------------------------------------

    df = df_raw[
        (df_raw["baseline_method"] == "Deter")
        & (df_raw["compared_method"].isin(MODEL_DISPLAY))
        & (df_raw["formula"] == "geo_mean_score_ratio")
        & (df_raw["split"] == split)
        & (df_raw["metric"] != "__all_metrics__")
    ].copy()
    df["Model"] = df["compared_method"].map(MODEL_DISPLAY)
    df["Metric"] = df["metric"].map(METRIC_DISPLAY).fillna(df["metric"])
    score_col = "value"

    # Order models by mean score across metrics (best = lowest first)
    geo_by_model = df.groupby("compared_method")[score_col].apply(_geo_mean).sort_values()
    model_order_raw = [m for m in geo_by_model.index if m in MODEL_DISPLAY]
    model_order = [MODEL_DISPLAY[m] for m in model_order_raw]

    # Per-model "Average" rows
    avg_rows = []
    for model_raw, model_label in MODEL_DISPLAY.items():
        geo_avg = _geo_mean(df.loc[df["compared_method"] == model_raw, score_col])
        if not np.isnan(geo_avg):
            avg_rows.append({"Model": model_label, "Metric": AVG_LABEL, score_col: geo_avg})
    df_avg = pd.DataFrame(avg_rows)

    df_plot = pd.concat([df_avg, df[["Model", "Metric", score_col]]], ignore_index=True)

    # Metric order: Average first, then individual metrics in original order
    metric_order = [AVG_LABEL] + [METRIC_DISPLAY.get(m, m) for m in df["metric"].unique()]

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------

    palette = sns.color_palette("tab10", n_colors=len(model_order))

    fig, ax = plt.subplots(figsize=(14, 3.5))

    sns.barplot(
        data=df_plot,
        x="Metric",
        y=score_col,
        hue="Model",
        hue_order=model_order,
        order=metric_order,
        palette=palette,
        ax=ax,
    )

    # Vertical separator between Average and the individual metrics
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Relative Score", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(split.capitalize())
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.legend(
        title="Model",
        title_fontsize=LEGEND_FONTSIZE,
        fontsize=LEGEND_FONTSIZE,
        loc="upper right",
    )

    sns.despine()
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"relative_score_per_metric_{split}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    logger.info("Saved to %s", out_path)
    plt.close(fig)
