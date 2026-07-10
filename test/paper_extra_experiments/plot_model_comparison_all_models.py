#### Script to redo the plots but comparing the different methods.

import logging
import math
import os
import typing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import Normalize

from src.logger.init_logger import set_config_logging

# Put here to shut all logs from usual libraries but keep the logs from this project.
set_config_logging()
logger = logging.getLogger(__name__)
from src.utils.fix_seq_ends import set_seq_to_nan_from_index

from config import OUT_FILE_NAME, ROOT_DIR
from src.metrics.crosscor import autocorr
from src.utils import tpp_utils
from src.utils.utils_os import factory_fct_linked_path


from src.metrics.corrloss import CorrLoss
from src.plot.tpp_plot_components import ensure_numpy_array

sns.set()
plt.rcParams["text.usetex"] = True
plt.rcParams["font.weight"] = "bold"

PANEL_TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 17
LEGEND_FONTSIZE = 14
DIST_TITLE_FONTSIZE = PANEL_TITLE_FONTSIZE + 4
DIST_TICK_LABEL_FONTSIZE = TICK_LABEL_FONTSIZE + 4
QQ_TITLE_FONTSIZE = DIST_TITLE_FONTSIZE
QQ_TICK_LABEL_FONTSIZE = DIST_TICK_LABEL_FONTSIZE
ACF_LEGEND_FONTSIZE = LEGEND_FONTSIZE + 2
ACF_TITLE_FONTSIZE = PANEL_TITLE_FONTSIZE + 6
CORR_HEATMAP_TITLE_FONTSIZE = PANEL_TITLE_FONTSIZE + 6
CORR_CBAR_TICK_LABEL_FONTSIZE = TICK_LABEL_FONTSIZE + 6
DIST_AXIS_LABEL_FONTSIZE = AXIS_LABEL_FONTSIZE + 4
QQ_AXIS_LABEL_FONTSIZE = DIST_AXIS_LABEL_FONTSIZE
AUTOCORR_MAX_LAG = 5
CORR_MAX_LAG = 20
SHARED_SUBTITLE_FONTSIZE = DIST_AXIS_LABEL_FONTSIZE
SHARED_SUBTITLE_Y_OFFSET = 0.13
SHARED_SUBTITLE_BOTTOM_MARGIN = 0.06
TARGET_LABEL = r"\textsc{Target}"
SIGTPP_LABEL = r"\textsc{SigTPP}"
WGAN_LABEL = r"\textsc{WGAN}"
DDPM_LABEL = r"\textsc{DDPM}"
VAE_LABEL = r"\textsc{VAE}"
DETER_LABEL = r"\textsc{DETER}"
GAMMA_LABEL = r"\textsc{GAMMA}"
DELTA_ACF_LABEL = r"$\Delta$ACF"
DELTA_TARGET_LABEL = rf"$\Delta$ {TARGET_LABEL}"

MODEL_SPECS = [
    ("model_sigwgan", SIGTPP_LABEL),
    ("model_wgan", WGAN_LABEL),
    ("model_score", DDPM_LABEL),
    ("model_vae", VAE_LABEL),
    ("model_deter", DETER_LABEL),
    ("model_gamma", GAMMA_LABEL),
]
DEFAULT_MODELS_TO_SHOW = [True, True, True, True, True, True]


def select_model_specs(models_to_show: typing.Sequence[bool]) -> typing.List[typing.Tuple[str, str]]:
    if len(models_to_show) != len(MODEL_SPECS):
        raise ValueError(
            f"models_to_show must have {len(MODEL_SPECS)} booleans "
            f"for {[label for _, label in MODEL_SPECS]}, got {len(models_to_show)}."
        )
    if any(not isinstance(show_model, bool) for show_model in models_to_show):
        raise TypeError("models_to_show must contain only True/False values.")

    selected = [spec for spec, show_model in zip(MODEL_SPECS, models_to_show) if show_model]
    if not selected:
        raise ValueError("models_to_show must enable at least one model.")
    return selected


def add_shared_xlabel(fig: plt.Figure, axes: typing.Union[np.ndarray, typing.Sequence[plt.Axes]], label: str) -> None:
    """Place a single x label under a row of axes with a small bottom margin."""
    fig.tight_layout(rect=(0.0, SHARED_SUBTITLE_BOTTOM_MARGIN, 1, 1))
    axes_arr = np.atleast_1d(axes).ravel()
    bottom = min(ax.get_position().y0 for ax in axes_arr)
    y = max(0.01, bottom - SHARED_SUBTITLE_Y_OFFSET)
    fig.text(0.5, y, label, ha='center', va='top', fontsize=SHARED_SUBTITLE_FONTSIZE, fontweight='bold')


def _as_ordered_models(
    models: typing.Sequence[typing.Tuple[str, np.typing.ArrayLike]],
) -> typing.List[typing.Tuple[str, np.ndarray]]:
    return [(label, np.asarray(data)) for label, data in models]


def plot_distributions_row(
    models: typing.Sequence[typing.Tuple[str, np.typing.ArrayLike]],
    dist_target: np.typing.ArrayLike,
    log_scale: bool = None,
):
    """Plot target plus model distributions side by side in a single row."""

    def get_num_bin_int_hist(arr: np.ndarray) -> int:
        try:
            # Manually check if all inputs are integer valued
            if np.all(np.isclose(arr, arr.astype(int), atol=1e-10)):
                unique_values = np.unique(arr)
                num_categories = len(unique_values)

                # Choose a compact divisor so categorical histograms remain readable.
                if num_categories > 100:
                    dividers = [d for d in range(2, int(math.sqrt(num_categories) + 1)) if num_categories % d == 0]
                    allowed_dividers = [d for d in dividers if num_categories // d <= 100]
                    # num_categories might be empty if we can't find any dividers. Then, we resort to use a default value.
                    num_categories = max(allowed_dividers) if allowed_dividers else 100
                if num_categories < 5:
                    num_categories = 25
            else:
                num_categories = 50  # Default number of bins if not integer valued
        except Exception:
            num_categories = 50
        return num_categories

    panels = [(TARGET_LABEL, np.asarray(dist_target)), *_as_ordered_models(models)]

    fig, axes = plt.subplots(1, len(panels), sharey=True, sharex=True, figsize=(3 * len(panels), 3))
    axes = np.atleast_1d(axes).ravel()
    color_scheme = sns.color_palette('inferno', max(a.shape[1] if a.ndim > 1 else 1 for _, a in panels))

    for ax, (name, data) in zip(axes, panels):
        assert len(data.shape) == 2, f"{name} expected 2D array, got shape {data.shape}."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            lo = np.nanmin(data)
            hi = np.nanmax(data)
            if hi - lo < 1e-1:
                binrange = (lo - 0.5, hi + 0.5)
                bins = 5
            else:
                binrange = None
                bins = get_num_bin_int_hist(data)

            sns.histplot(
                data,
                bins=bins,
                palette=color_scheme,
                ax=ax,
                stat="density",
                element="bars",
                edgecolor="none",
                linewidth=0,
                binrange=binrange,
                common_norm=True,
                kde=False,
                multiple="stack",
                legend=False,
            )
            for patch in ax.patches:
                patch.set_rasterized(True)

        if log_scale:
            ax.set_yscale('log')
        ax.text(
            0.5,
            0.95,
            name,
            transform=ax.transAxes,
            fontsize=DIST_TITLE_FONTSIZE,
            ha='center',
            va='top',
            fontweight='bold',
        )
        ax.set_ylabel("PDF", fontsize=DIST_AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.tick_params(axis='x', labelbottom=True, labelsize=DIST_TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=DIST_TICK_LABEL_FONTSIZE)

    add_shared_xlabel(fig, axes, "Interarrival Time")
    return fig


def qq_plot_multi_models_against_targets(
    models: typing.Sequence[typing.Tuple[str, np.typing.ArrayLike]],
    targets_sampled: np.typing.ArrayLike,
    add_legend: bool = True,
    max_val: typing.Optional[float] = None,
):
    """
    Make QQ plots of multiple model sequences against the same targets on one row of axes.
    Matches the quantile-matching and reference-line logic used by
    `qq_plot_multi_seqs_against_targets` in src/plot/tpp_plot_components.py.
    If `max_val` is provided, truncate the displayed axis range to that upper bound.
    """
    models_np = [(name, ensure_numpy_array(seqs)) for name, seqs in models]
    targets_sampled = ensure_numpy_array(targets_sampled)

    for name, X in models_np:
        if X.shape != targets_sampled.shape:
            raise ValueError(f"Shape mismatch between {name} and targets: {X.shape} vs {targets_sampled.shape}")

    _, n_features = targets_sampled.shape
    color_scheme = sns.color_palette('inferno', n_features)

    SUBSAMPLE_FREQ = 1
    MAX_SAMPLES_PLOT = 10_000

    # Reference line range taken across models and targets (skip col 0: tau1 sampled artificially),
    # matching qq_plot_multi_seqs_against_targets.
    ref_vals = np.concatenate([*[seqs[:, 1:].flatten() for _, seqs in models_np], targets_sampled[:, 1:].flatten()])
    ref_vals = ref_vals[~np.isnan(ref_vals)]
    if ref_vals.size == 0:
        raise ValueError("Models and targets contain only NaNs; cannot draw QQ plots.")
    ref_min, ref_max = 0.0, float(ref_vals.max())

    fig, axes = plt.subplots(1, len(models_np), sharex=True, sharey=True, figsize=(3.2 * len(models_np), 3.4))
    axes = np.atleast_1d(axes).ravel()

    for panel_idx, (ax, (title, X)) in enumerate(zip(axes, models_np)):
        ax.plot([ref_min, ref_max], [ref_min, ref_max], color='red', linestyle='--', linewidth=1)

        # Skip col 0: tau1 is sampled artificially.
        for feat_idx in range(1, n_features):
            if feat_idx % SUBSAMPLE_FREQ != 0:
                continue
            x = X[:, feat_idx]
            y = targets_sampled[:, feat_idx]

            # Filter NaNs independently: QQ plots compare sorted quantiles, not paired values.
            # A joint mask would drop valid data when length distributions differ across sets.
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]

            if x.size == 0 or y.size == 0:
                continue

            # Use evenly-spaced percentile positions so quantiles are comparable even when the
            # two sides have different valid sizes (matches canonical QQ logic).
            n = min(x.size, y.size, MAX_SAMPLES_PLOT)
            x_quantiles = np.percentile(x, np.linspace(0, 100, n))
            y_quantiles = np.percentile(y, np.linspace(0, 100, n))
            ax.scatter(y_quantiles, x_quantiles, s=1.0, color=color_scheme[feat_idx], rasterized=True)

        ax.text(
            0.5,
            0.95,
            title,
            transform=ax.transAxes,
            fontsize=QQ_TITLE_FONTSIZE,
            ha='center',
            va='top',
            fontweight='bold',
        )
        if panel_idx == 0:
            ax.set_ylabel("Sampled Quantiles", fontsize=QQ_AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.tick_params(axis='both', labelsize=QQ_TICK_LABEL_FONTSIZE)
        ax.grid(True)

    if max_val is not None:
        for ax in axes:
            ax.set_xlim(0.0, float(max_val))
            ax.set_ylim(0.0, float(max_val))

    if not add_legend:
        for ax in axes:
            ax.set_ylabel("")

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    if add_legend:
        bottom = min(ax.get_position().y0 for ax in axes)
        fig.text(
            0.5,
            max(0.01, bottom - 0.13),
            "Target Quantiles",
            ha='center',
            va='top',
            fontsize=QQ_AXIS_LABEL_FONTSIZE,
            fontweight='bold',
        )
    return fig


# New method in case needed.
def qq_plot_overlay_models(
    models: typing.List[typing.Tuple[str, np.ndarray]],
    targets: np.ndarray,
    ax: plt.Axes,
    add_legend: bool = True,
):
    """
    Flatten all dimensions (cols 1:) for each model and overlay them on one axis.
    Each model gets one color from a qualitative palette; the x=y reference line
    is drawn once from the flattened targets.

    Parameters
    ----------
    models : list of (name, seqs) pairs
        Each seqs has shape (n_samples, n_features). Can contain NaNs for
        variable-length sequences.
    targets : np.ndarray, shape (n_samples, n_features)
    ax : plt.Axes
    add_legend : bool
    """
    targets = ensure_numpy_array(targets)

    # Flatten targets (skip col 0: tau1 is sampled artificially), strip NaNs.
    if targets.shape[1] < 2:
        raise ValueError(f"targets must have at least 2 columns (got {targets.shape[1]}); col 0 is always skipped.")
    tgt_flat = targets[:, 1:].flatten()
    tgt_flat = tgt_flat[~np.isnan(tgt_flat)]
    if tgt_flat.size == 0:
        raise ValueError("Targets contain only NaNs.")

    MAX_SAMPLES_PLOT = 10_000

    # Pass 1: flatten and validate each model; track global value range for the reference line.
    color_scheme = sns.color_palette('tab10', len(models))
    precomputed: typing.List[typing.Optional[np.ndarray]] = []
    ref_min, ref_max = float(tgt_flat.min()), float(tgt_flat.max())
    for name, seqs in models:
        seqs = ensure_numpy_array(seqs)
        if seqs.shape[1] < 2:
            logger.warning(f"Model '{name}' has fewer than 2 columns (got {seqs.shape[1]}); skipping.")
            precomputed.append(None)
            continue
        x_all = seqs[:, 1:].flatten()
        x_all = x_all[~np.isnan(x_all)]
        if x_all.size == 0:
            logger.warning(f"Model '{name}' has no valid (non-NaN) values; skipping.")
            precomputed.append(None)
            continue
        ref_min = min(ref_min, float(x_all.min()))
        ref_max = max(ref_max, float(x_all.max()))
        precomputed.append(x_all)

    # Draw x=y reference line spanning the full range (targets + all models).
    ax.plot([ref_min, ref_max], [ref_min, ref_max], color='red', linestyle='--', linewidth=1)

    # Pass 2: scatter each model's quantiles.
    for (name, _), x_all, color in zip(models, precomputed, color_scheme):
        if x_all is None:
            continue
        n = min(x_all.size, tgt_flat.size, MAX_SAMPLES_PLOT)
        x_quantiles = np.percentile(x_all, np.linspace(0, 100, n))
        y_quantiles = np.percentile(tgt_flat, np.linspace(0, 100, n))
        ax.scatter(y_quantiles, x_quantiles, s=2.0, color=color, label=name, rasterized=True)

    if add_legend:
        ax.set_title('QQ Plot – Model Comparison (Flattened)')
        ax.set_xlabel('Target Quantiles')
        ax.set_ylabel('Sampled Quantiles')
        ax.legend(loc='upper left', markerscale=5)
    else:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax.grid(True)


def plot_hist_lens_four(dist_sigwgan, dist_wgan, dist_score, dist_target):
    raise NotImplementedError


def plot_hist_lens_models(
    models: typing.Sequence[typing.Tuple[str, torch.Tensor]],
    dist_target: torch.Tensor,
):
    series = [(TARGET_LABEL, dist_target), *models]
    assert all(len(x.shape) == 1 for _, x in series)

    fig, ax = plt.subplots()
    palette = sns.color_palette("tab10", len(series))

    min_len = int(min(t.min().item() for _, t in series))
    max_len = int(max(t.max().item() for _, t in series))
    bins = max_len - min_len + 1
    for (name, data), color in zip(series, palette):
        if name == TARGET_LABEL:
            alpha = 1.0
        else:
            alpha = 0.35
        sns.histplot(
            data,
            bins=bins,
            ax=ax,
            color=color,
            stat='density',
            discrete=True,
            label=name,
            alpha=alpha,
            edgecolor='none',
        )

    ax.set_title("Length Distributions", fontsize=PANEL_TITLE_FONTSIZE, fontweight='bold')
    ax.set_xlabel("Length", fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel("Density (log scale)", fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE)
    sns.despine()
    return fig


# expects an existing function: autocorr(tensor, max_lag:int) -> torch.Tensor [L]
# L = max_lag + 1 typically


def _to_acf_frame(x: torch.Tensor, label: str, max_lag: int) -> pd.DataFrame:
    acf = autocorr(x, max_lag=max_lag).detach().cpu().numpy().reshape(-1)  # skip lag 0
    L = acf.shape[0]
    return pd.DataFrame(
        {
            "Lags": np.arange(1, L),
            "Autocorrelation": acf[1:],
            "Type": np.full(L - 1, label, dtype=object),
        }
    )


def plot_compare_autocorr_models(
    models: typing.Sequence[typing.Tuple[str, torch.Tensor]],
    dist_target: torch.Tensor,
    max_lag: int,
) -> plt.Figure:
    # raw data
    frames_raw = [_to_acf_frame(dist_target, TARGET_LABEL, max_lag)] + [
        _to_acf_frame(dist_model, model_label, max_lag) for model_label, dist_model in models
    ]
    df_raw = pd.concat(frames_raw, ignore_index=True)

    fig, ax_raw = plt.subplots(1, 1, figsize=(4.8, 4.8))

    # colors and line styles
    model_labels = [label for label, _ in models]
    order_raw = [TARGET_LABEL, *model_labels]
    base_colors = sns.color_palette("tab10", len(order_raw))
    palette_raw = dict(zip(order_raw, base_colors))

    # helper to compute discrepancies
    def _plot_discrepancies(df, target_label, model_labels, palette, ax):
        tgt = df[df["Type"] == target_label][["Lags", "Autocorrelation"]].rename(columns={"Autocorrelation": "tgt"})
        for m in model_labels:
            mod = df[df["Type"] == m][["Lags", "Autocorrelation"]].rename(columns={"Autocorrelation": "mod"})
            merged = pd.merge(tgt, mod, on="Lags", how="inner").dropna()
            diff = merged["mod"] - merged["tgt"]
            ax.plot(
                merged["Lags"],
                diff,
                linestyle=":",
                linewidth=2.5,
                alpha=1.0,
                color=palette[m],
                label=m,
            )

    _plot_discrepancies(
        df_raw,
        TARGET_LABEL,
        model_labels,
        palette_raw,
        ax_raw,
    )

    ax_raw.axhline(0.0, color="black", alpha=0.4, linewidth=1)

    # labels
    ax_raw.set_title(r"Interarrival times $\tau$", fontsize=ACF_TITLE_FONTSIZE, fontweight='bold')
    ax_raw.set_xlabel("Lags", fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax_raw.set_ylabel(DELTA_ACF_LABEL, fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax_raw.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    # legends with ACF/diff meaning
    handles_raw, labels_raw = ax_raw.get_legend_handles_labels()
    ax_raw.legend(
        handles_raw
        + [
            # plt.Line2D([], [], color="black", linestyle="-", label="ACF"),
            plt.Line2D([], [], color="black", linestyle=":", label=DELTA_TARGET_LABEL),
        ],
        labels_raw + [DELTA_TARGET_LABEL],
        loc="best",
        fontsize=ACF_LEGEND_FONTSIZE,
    )

    fig.tight_layout()
    return fig


def plot_corr_err(
    models: typing.Sequence[typing.Tuple[str, torch.Tensor]],
    dist_target: torch.Tensor,
    vmax=0.3,
    bound=50,
):
    """
    Plot heatmaps of |Corr(model) - Corr(target)| with one shared colorbar on the right.
    Shapes: (n_samples, n_features) or (n_samples, n_features, 1).
    Returns (fig, axes, errs_by_label).
    """

    errs_by_label = {
        label: CorrLoss(dist_target[:, 1:, :])(dist_model[:, 1:, :])[:bound, :bound] for label, dist_model in models
    }

    # shared color scale
    if vmax is None:
        vmax = torch.nanmax(torch.stack([torch.nanmax(err) for err in errs_by_label.values()])).item()
        if not np.isfinite(vmax):
            vmax = 1.0  # fallback

    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = "coolwarm"

    fig = plt.figure(figsize=(3.0 * len(errs_by_label) + 0.5, 4.8))

    gs = fig.add_gridspec(
        nrows=1,
        ncols=len(errs_by_label) + 1,
        width_ratios=[*[1] * len(errs_by_label), 0.06],
    )
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(len(errs_by_label))]
    cax = fig.add_subplot(gs[0, len(errs_by_label)])

    # inline heatmap plotting with shared colorbar on the last plot
    datas = list(errs_by_label.values())
    titles = list(errs_by_label.keys())

    for i, (ax, data, title) in enumerate(zip(axes, datas, titles)):
        d = data
        if d.numel() == 1:
            d = d.view(1, 1)
        assert d.dim() == 2, "Heatmap data must be 2D."
        dnp = d.detach().cpu().numpy()

        # skip entirely-NaN matrices to avoid seaborn errors
        if np.isnan(dnp).all():
            ax.set_axis_off()
            ax.set_title(title, fontsize=CORR_HEATMAP_TITLE_FONTSIZE, fontweight='bold')
            continue

        heatmap = sns.heatmap(
            dnp,
            ax=ax,
            cmap=cmap,
            norm=norm,
            linewidths=0.5,
            cbar=(i == len(datas) - 1),
            cbar_ax=cax if i == len(datas) - 1 else None,
            vmin=0.0,
            vmax=vmax,
            square=False,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Index", fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
        ax.set_title(title, fontsize=CORR_HEATMAP_TITLE_FONTSIZE, fontweight='bold')
        if i == len(datas) - 1 and heatmap.collections:
            heatmap.collections[0].colorbar.ax.tick_params(labelsize=CORR_CBAR_TICK_LABEL_FONTSIZE)

    fig.tight_layout(rect=(0.0, SHARED_SUBTITLE_BOTTOM_MARGIN, 1, 1))
    bottom = min(ax.get_position().y0 for ax in axes)
    fig.text(
        0.5, max(0.01, bottom - 0.03), "Index", ha='center', va='top', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold'
    )
    return fig, axes, errs_by_label


def plot_hist_intensity(
    models: typing.Sequence[typing.Tuple[str, np.typing.ArrayLike]],
    dist_target: np.typing.ArrayLike,
    num_bins: int,
    time_horizon: typing.Optional[float] = None,
) -> typing.Tuple[plt.Figure, np.ndarray]:
    """
    Plot target plus model stacked-count histograms (as intensity after renormalisation) in one row.
    Uses a single shared bin width across all axes. No try/except path.

    Inputs are 2D arrays shaped (num_sequences, num_events_per_sequence).
    """

    assert num_bins > 0, f"Expected a positive number of bins, got {num_bins}."

    dists = [(TARGET_LABEL, np.asarray(dist_target)), *_as_ordered_models(models)]

    # Validate shapes and collect the shared horizon
    for name, arr in dists:
        assert arr.ndim == 2, f"{name} expected 2D array, got shape {arr.shape}."

    finite_values = [arr[np.isfinite(arr)] for _, arr in dists]
    finite_values = [values for values in finite_values if values.size > 0]
    if not finite_values:
        raise ValueError("Input distributions contain only NaNs or non-finite values.")

    observed_max = max(float(values.max()) for values in finite_values)
    horizon = observed_max
    if time_horizon is not None:
        horizon = max(horizon, float(time_horizon))
    if horizon <= 0.0:
        horizon = max(observed_max, 1e-12)

    bin_edges = np.linspace(0.0, horizon, num_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    # Pre-compute target histogram counts for overlay on model subplots
    target_arr = np.asarray(dist_target)
    target_flat = target_arr.flatten()
    target_flat = target_flat[np.isfinite(target_flat)]
    target_counts, _ = np.histogram(target_flat, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    N_target = target_arr.shape[0]

    # Sequence count per distribution. With sharey=True we cannot safely relabel y-ticks per axis
    # (last writer wins), so require a common N and rescale once.
    row_counts = {name: arr.shape[0] for name, arr in dists}
    if len(set(row_counts.values())) != 1:
        raise ValueError(
            f"plot_hist_intensity expects the same number of sequences across distributions; got {row_counts}."
        )
    N_common = next(iter(row_counts.values()))

    # Figure and axes
    fig, axes = plt.subplots(1, len(dists), sharey=True, sharex=True, figsize=(3 * len(dists), 3))
    axes = np.atleast_1d(axes).ravel()
    color_counts = {name: (arr.shape[1] if arr.ndim > 1 else 1) for name, arr in dists}
    max_colors = max(color_counts.values())
    # Sample from a fixed 256-step palette so indices are stable regardless of max_colors
    _inferno256 = sns.color_palette('inferno', 256)
    color_purple = _inferno256[60]
    color_orange = _inferno256[170]
    base_palette_purple = [color_purple] * max_colors
    base_palette_orange = [color_orange] * max_colors
    # Plot each histogram with shared bins
    for ax, (name, data) in zip(axes, dists):
        if name == TARGET_LABEL:
            palette = base_palette_orange[: color_counts[name]]
        else:
            palette = base_palette_purple[: color_counts[name]]
        sns.histplot(
            data,
            bins=bin_edges,  # fixed shared bins
            stat='count',  # raw counts; we rescale ticks to intensity
            multiple='stack',
            common_norm=False,  # keep raw counts per stack
            legend=False,
            edgecolor='none',
            linewidth=0,
            palette=palette,
            ax=ax,
        )
        for patch in ax.patches:
            patch.set_rasterized(True)

        if name != TARGET_LABEL:
            # N_common == N_target == arr.shape[0] for every panel (enforced above).
            ax.plot(
                bin_centers,
                target_counts * (N_common / N_target),
                color=base_palette_orange[0],
                linestyle=(0, (1, 3)),
                linewidth=3.0,
                # label='Target',
            )
            # ax.legend(fontsize=8, loc='best')

        ax.text(
            0.5,
            0.95,
            name,
            transform=ax.transAxes,
            fontsize=DIST_TITLE_FONTSIZE,
            ha='center',
            va='top',
            fontweight='bold',
        )
        ax.set_xlim(0.0, horizon)
        ax.tick_params(axis='both', labelsize=DIST_TICK_LABEL_FONTSIZE)

    # Renormalise y-axis tick labels once to intensity = counts / (N * bin_width).
    # Safe because sharey=True ties all four axes and we've asserted a common N.
    yticks = axes[0].get_yticks()
    axes[0].set_yticks(yticks)
    with np.errstate(invalid='ignore', divide='ignore'):
        axes[0].set_yticklabels(np.round(yticks / (N_common * bin_width), 1))

    axes[0].set_ylabel("Intensity", fontsize=DIST_AXIS_LABEL_FONTSIZE, fontweight='bold')
    add_shared_xlabel(fig, axes, "Time")
    return fig, axes


def run_model_comparison(
    experiment_type: str = "hawkes3",
    models_to_show: typing.Optional[typing.Sequence[bool]] = None,
) -> None:
    # ── User config — edit here when running from the IDE ──────────────────────
    # Available keys: "taobao", "taxi", "hawkes3", "ihp3", "eq", "so", "yelp_mississauga"
    # Aliases also accepted: "earthquake"→"eq", "hawkes_3x3"→"hawkes3",
    #                        "inh_poisson_three_marks"→"ihp3",
    #                        "stackoverflow"→"so"
    # Requires samples_gen.pth under out/<dataset>/models/<model_name>/ and a shared
    # test_targets.pth (or legacy samples_tgt.pth) under out/<dataset>/
    # Order: SigTPP, WGAN, DDPM, VAE, DETER, GAMMA.
    # Original 3-model comparison: [True, True, True, False, False, False]
    if models_to_show is None:
        models_to_show = DEFAULT_MODELS_TO_SHOW
    # ──────────────────────────────────────────────────────────────────────────

    # Best models per type read from results_raw.txt (norm_score = Borda rank, lower is better).
    # Keys use the short dataset aliases requested for the comparison plot.
    CONFIGS = {
        "taobao": dict(
            dataset_dir="taobao",
            model_sigwgan="taobao_sigwgan_TX13_traiT_hid_16_conc1_lr_g0,001_mark0,1_sig_6_use_T_anchfree_detaF",  # norm_score 108
            model_wgan="taobao_wgan_TX13_conc1_hid_32_lips0,01_lr_d0,001_lr_g5e-05",  # norm_score 65
            model_score="taobao_score_TX13_batc1024_conc1_hid_16_lr0,001_num_10",  # norm_score 49
            model_vae="taobao_vae_TX13_conc1_free0,1_hid_32_kl_a50_late8_lr0,01_reco10",  # norm_score 92
            model_deter="taobao_deter_TX13_hid_16_lr_g0,01",  # norm_score 20
            model_gamma="taobao_gamma_TX13_lear0,0001",  # norm_score 12
            max_lag=10,
            qq_max_val=1.6,
            vmax=0.20,
            bound=12,
            out_name="taobao",
            time_horizon=13.0,
        ),
        "taxi": dict(
            dataset_dir="taxi",
            model_sigwgan="taxi_sigwgan_TX23_traiF_hid_16_conc1_lr_g0,0005_mark10_sig_8_use_T_anchfree_detaF",  # norm_score 402
            model_wgan="taxi_wgan_TX23_conc1_hid_32_lips0,01_lr_d0,01_lr_g5e-05",  # norm_score 70
            model_score="taxi_score_TX23_batc1024_conc1_hid_32_lr0,001_num_50",  # norm_score 30
            model_vae="taxi_vae_TX23_conc1_free0,1_hid_32_kl_a50_late16_lr0,01_reco10",  # norm_score 78
            model_deter="taxi_deter_TX23_hid_16_lr_g0,01",  # norm_score 17
            model_gamma="taxi_gamma_TX23_lear1",  # norm_score 15
            max_lag=20,
            qq_max_val=5.0,
            vmax=0.30,
            bound=23,
            out_name="taxi",
            time_horizon=23.0,
        ),
        "hawkes3": dict(
            dataset_dir="hawkes_3x3",
            model_sigwgan="hawkes_3x3_sigwgan_TX15_traiT_hid_16_conc1_lr_g0,0005_mark0,1_sig_6_use_T_anchresi_detaT",  # norm_score 89
            model_wgan="hawkes_3x3_wgan_TX15_conc1_hid_32_lips0,01_lr_d0,01_lr_g0,0001",  # norm_score 47
            model_score="hawkes_3x3_score_TX15_batc1024_conc1_hid_32_lr0,001_num_100",  # norm_score 35
            model_vae="hawkes_3x3_vae_TX15_conc1_free0,1_hid_32_kl_a50_late16_lr0,01_reco10",  # norm_score 32
            model_deter="hawkes_3x3_deter_TX15_hid_32_lr_g0,01",  # norm_score 20
            model_gamma="hawkes_3x3_gamma_TX15_lear0,0001",  # norm_score 12
            max_lag=12,
            qq_max_val=6.0,
            vmax=0.10,
            bound=12,
            out_name="hawkes3x3",
            time_horizon=15.0,
        ),
        "ihp3": dict(
            dataset_dir="inh_poisson_three_marks",
            # Best-result names come from results_raw.txt; the corresponding artifact folders are not present locally.
            model_sigwgan="ihp_three_marks_sigwgan_TX10_traiF_hid_16_conc1_lr_g0,001_mark0,1_sig_8_use_T_anchfree_detaF",  # norm_score 53
            model_wgan="ihp_three_marks_wgan_TX10_conc1_hid_32_lips0,0001_lr_d0,01_lr_g0,0001",  # norm_score 35
            model_score="ihp_three_marks_score_TX10_batc4096_conc1_hid_32_lr0,001_num_50",  # norm_score 50
            model_vae="ihp_three_marks_vae_TX10_conc1_free0,1_hid_16_kl_a50_late8_lr0,01_reco10",  # norm_score 55
            model_deter="ihp_three_marks_deter_TX10_hid_32_lr_g0,01",  # norm_score 18
            model_gamma="ihp_three_marks_gamma_TX10_lear0,0001",  # norm_score 15
            max_lag=10,
            qq_max_val=50.0,
            vmax=0.10,
            bound=10,
            out_name="ihp3",
            time_horizon=10.0,
        ),
        "eq": dict(
            dataset_dir="earthquake",
            model_sigwgan="earthquake_sigwgan_TX78_traiF_hid_16_conc1_lr_g0,0005_mark0,1_sig_8_use_T_anchfree_detaF",  # norm_score 83
            model_wgan="earthquake_wgan_TX78_conc1_hid_16_lips0,001_lr_d0,0001_lr_g1e-05",  # norm_score 110
            model_score="earthquake_score_TX78_batc1024_conc1_hid_16_lr0,001_num_10",  # norm_score 41
            model_vae="earthquake_vae_TX78_conc1_free0,1_hid_32_kl_a50_late16_lr0,0001_reco1",  # norm_score 55
            model_deter="earthquake_deter_TX78_hid_16_lr_g0,01",  # norm_score 18
            model_gamma="earthquake_gamma_TX78_lear0,01",  # norm_score 15
            max_lag=20,
            qq_max_val=15.0,
            vmax=0.20,
            bound=20,
            out_name="eq",
            time_horizon=78.0,
        ),
        "so": dict(
            dataset_dir="stackoverflow",
            model_sigwgan="stackoverflow_sigwgan_TX64_traiF_hid_32_conc1_lr_g0,0005_mark0,1_sig_8_use_T_anchfree_detaT",  # norm_score 45
            model_wgan="stackoverflow_wgan_TX64_conc1_hid_32_lips0,001_lr_d0,001_lr_g5e-05",  # norm_score 64
            model_score="stackoverflow_score_TX64_batc1024_conc1_hid_16_lr0,001_num_50",  # norm_score 39
            model_vae="stackoverflow_vae_TX64_conc1_free0,1_hid_32_kl_a50_late8_lr0,001_reco10",  # norm_score 38
            model_deter="stackoverflow_deter_TX64_hid_32_lr_g0,01",  # norm_score 17
            model_gamma="stackoverflow_gamma_TX64_lear0,01",  # norm_score 14
            max_lag=32,
            qq_max_val=18.0,
            vmax=0.15,
            bound=15,
            out_name="so",
            time_horizon=64.0,
        ),
        "yelp_mississauga": dict(
            dataset_dir="yelp_mississauga",
            model_sigwgan="yelp_mississauga_sigwgan_TX24_traiF_hid_16_conc1_lr_g0,001_mark0,1_sig_8_use_T_anchresi_detaT",  # norm_score 52
            model_wgan="yelp_mississauga_wgan_TX24_conc1_hid_32_lips0,0001_lr_d0,01_lr_g5e-05",  # norm_score 105
            model_score="yelp_mississauga_score_TX24_batc1024_conc1_hid_32_lr0,001_num_10",  # norm_score 41
            model_vae="yelp_mississauga_vae_TX24_conc1_free0,1_hid_32_kl_a50_late8_lr0,01_reco10",  # norm_score 52
            model_deter="yelp_mississauga_deter_TX24_hid_8_lr_g0,01",  # norm_score 11
            model_gamma="yelp_mississauga_gamma_TX24_lear0,0001",  # norm_score 14
            max_lag=12,
            qq_max_val=10.0,
            vmax=0.25,
            bound=24,
            out_name="yelp_mississauga",
            time_horizon=24.0,
        ),
    }

    experiment_aliases = {
        "earthquake": "eq",
        "hawkes_3x3": "hawkes3",
        "ihp_three_marks": "ihp3",
        "inh_poisson_three_marks": "ihp3",
        "stackoverflow": "so",
    }
    experiment_key = experiment_aliases.get(experiment_type.lower(), experiment_type.lower())
    if experiment_key not in CONFIGS:
        raise KeyError(f"Unknown experiment_type={experiment_type!r}. Available: {sorted(CONFIGS)}")

    cfg = CONFIGS[experiment_key]
    dataset_dir = cfg.get("dataset_dir", experiment_key)
    dataset_root = os.path.join(ROOT_DIR, "test/paper_experiments/out", dataset_dir)
    models_dir = os.path.join(dataset_root, "models")
    results_dir = os.path.join(dataset_root, "results")

    def resolve_sample_path(model_key: str, sample_file: str) -> str:
        model_name = cfg[model_key]
        if not os.path.isdir(models_dir):
            has_results = os.path.isdir(results_dir)
            raise FileNotFoundError(
                f"No models directory found for dataset_dir={dataset_dir!r} at "
                f"{os.path.join('test/paper_experiments/out', dataset_dir, 'models')}. "
                + (
                    "The best model ids were verified from the local results tables, but the sampled artifacts "
                    "for this dataset are not present in this checkout."
                    if has_results
                    else "Neither sampled artifacts nor results tables are present for this dataset in this checkout."
                )
            )

        sample_path = os.path.join(models_dir, model_name, sample_file)
        if not os.path.exists(sample_path):
            if sample_file == "samples_tgt.pth":
                # Test targets are now saved once per experiment (not duplicated
                # into every model's directory); fall back to the shared file.
                shared_path = os.path.join(dataset_root, "test_targets.pth")
                if os.path.exists(shared_path):
                    return shared_path
            family = model_key.removeprefix("model_")
            family_candidates = []
            if os.path.isdir(models_dir):
                family_candidates = sorted(
                    name for name in os.listdir(models_dir) if f"_{family}_" in name or name.startswith(family)
                )
            preview = ", ".join(family_candidates[:5]) if family_candidates else "none"
            raise FileNotFoundError(
                f"Missing {sample_file!r} for {model_key}={model_name!r} in "
                f"{os.path.join('test/paper_experiments/out', dataset_dir, 'models')}. "
                "The dict points to the best run listed in results_raw.txt, but that artifact is not present here. "
                f"Available local candidates for family {family!r}: {preview}."
            )
        return sample_path

    selected_model_specs = select_model_specs(models_to_show)
    model_dists = []
    model_lens = []
    for model_key, model_label in selected_model_specs:
        dist_model, lens_model = tpp_utils.load_samples(resolve_sample_path(model_key, "samples_gen.pth"))[:2]
        dist_model = set_seq_to_nan_from_index(dist_model, lens_model - 1)
        model_dists.append((model_label, dist_model))
        model_lens.append((model_label, lens_model))

    target_source_key = selected_model_specs[0][0]
    dist_target, lens_target = tpp_utils.load_samples(resolve_sample_path(target_source_key, "samples_tgt.pth"))[:2]
    dist_target = set_seq_to_nan_from_index(dist_target, lens_target - 1)

    n = min(*[dist_model.shape[0] for _, dist_model in model_dists], dist_target.shape[0])
    model_dists = [(label, dist_model[:n]) for label, dist_model in model_dists]
    model_lens = [(label, lens_model[:n]) for label, lens_model in model_lens]
    dist_target = dist_target[:n]
    lens_target = lens_target[:n]

    fig1 = plot_distributions_row(
        [(label, dist_model.squeeze(-1).detach().cpu().numpy()) for label, dist_model in model_dists],
        dist_target.squeeze(-1).detach().cpu().numpy(),
        log_scale=True,
    )
    fig2 = qq_plot_multi_models_against_targets(
        [(label, dist_model.squeeze(-1)) for label, dist_model in model_dists],
        dist_target.squeeze(-1),
        max_val=cfg.get("qq_max_val"),
    )
    effective_max_lag = max(
        0,
        min(
            AUTOCORR_MAX_LAG,
            *[dist_model[:, 1:, :].shape[1] // 2 - 1 for _, dist_model in model_dists],
            dist_target[:, 1:, :].shape[1] // 2 - 1,
        ),
    )
    # ACF: strip tau_1 to match the TPPMetrics convention used during training
    # (src/plot/tpp_plots.py passes sampled_fake_sample[:, 1:, :] to plot_compare_autocorr).
    fig_acf = plot_compare_autocorr_models(
        [(label, dist_model[:, 1:, :]) for label, dist_model in model_dists],
        dist_target[:, 1:, :],
        max_lag=effective_max_lag,
    )
    fig_corr, axes, corr_errs = plot_corr_err(
        model_dists,
        dist_target,
        vmax=cfg["vmax"],
        bound=CORR_MAX_LAG,
    )
    # Lengths: +1 to recover total time points including the t0 anchor,
    # matching training (src/plot/tpp_plots.py: sampled_lens + 1).
    fig_lengths = plot_hist_lens_models([(label, lens_model + 1) for label, lens_model in model_lens], lens_target + 1)
    fig_intensity, ax = plot_hist_intensity(
        [(label, dist_model[:, :, 0].cumsum(1).detach().cpu().numpy()) for label, dist_model in model_dists],
        dist_target[:, :, 0].cumsum(1).detach().cpu().numpy(),
        num_bins=25,
        time_horizon=cfg.get("time_horizon"),
    )

    plots_dir = os.path.join(ROOT_DIR, "test/paper_extra_experiments/out/figures")
    os.makedirs(plots_dir, exist_ok=True)
    name = cfg["out_name"]
    model_set_name = "all" if len(selected_model_specs) == len(MODEL_SPECS) else f"{len(selected_model_specs)}model"
    fig1.savefig(os.path.join(plots_dir, f"{name}_{model_set_name}_interarrival_dist.pdf"), dpi=300)
    fig2.savefig(os.path.join(plots_dir, f"{name}_{model_set_name}_qq.pdf"), dpi=300)
    fig_acf.savefig(os.path.join(plots_dir, f"{name}_{model_set_name}_acf_delta.pdf"))
    fig_corr.savefig(os.path.join(plots_dir, f"{name}_{model_set_name}_corr_error.pdf"))
    fig_lengths.savefig(os.path.join(plots_dir, f"{name}_{model_set_name}_length_dist.pdf"))
    fig_intensity.savefig(os.path.join(plots_dir, f"{name}_{model_set_name}_intensity.pdf"), dpi=300)
    plt.show()


if __name__ == "__main__":
    run_model_comparison()
