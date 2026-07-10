"""Dataset diagnostic plot functions for temporal point processes.

All public functions return a matplotlib Figure. Internal helpers operate on
existing Axes. No file I/O here — see dataset_diagnostics.py for export.
"""

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from src.diagnostics._tpp_features import (
    DEFAULT_ACF_DISPLAY_MAX_LAG,
    DEFAULT_MAX_CORRELATION_HEATMAP_DIM,
    DEFAULT_MAX_LAG,
    DEFAULT_MAX_PATHS,
    DEFAULT_MIN_SAMPLES_FOR_CORR,
    SPLITS,
    SPLIT_LINESTYLES,
    _dataset_report_slug,
    _event_times_naned,
    _interarrivals_drop_first,
    _interarrivals_naned,
    _overlay_alpha,
    _prepare_corr_features,
    _resolve_indices,
    _shared_max_paths,
    _split_attrs,
    _split_color,
    _split_label,
)
from src.diagnostics.dataset_summary import _mark_frequency_stats
from src.metrics.corrloss import corr_torch
from src.metrics.crosscor import autocorr

logger = logging.getLogger(__name__)

_DIAG_AXIS_LABEL_FONTSIZE = 18
_DIAG_TICK_LABEL_FONTSIZE = 17
_DIAG_LEGEND_FONTSIZE = 14
_DIAG_INTENSITY_PDF_AXIS_LABEL_FONTSIZE = _DIAG_AXIS_LABEL_FONTSIZE + 4
_DIAG_FLAT_FIGSIZE = (6.4, 4.1)
_DIAG_STACKED_FLAT_FIGSIZE = (6.4, 4.6)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _set_window_title(fig: Figure, title: str) -> None:
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def _truncate_corr_matrix(data_np: np.ndarray, max_lag: int) -> np.ndarray:
    """Return the displayed correlation block capped to the requested lag window."""
    if max_lag <= 0:
        return data_np[:0, :0]
    limit = min(max_lag, data_np.shape[0], data_np.shape[1])
    return data_np[:limit, :limit]


def _correlation_heatmap_lag_limit(max_lag: int) -> int:
    if max_lag <= 0:
        return max_lag
    return min(max_lag, DEFAULT_MAX_CORRELATION_HEATMAP_DIM)


def _shared_effective_lag(series_by_split: Dict[str, torch.Tensor], max_lag: int) -> int:
    return min(
        max_lag,
        DEFAULT_ACF_DISPLAY_MAX_LAG,
        min(max(series.shape[1] - 1, 0) for series in series_by_split.values()),
    )


def _merged_its_tensor(dm, *, drop_first: bool) -> torch.Tensor:
    """Inter-arrivals from all splits, padded to a common max L."""
    its_list = []
    for split in SPLITS:
        cum = getattr(dm, f"{split}_in")
        lens = getattr(dm, f"{split}_in_len")
        its = _interarrivals_drop_first(cum, lens) if drop_first else _interarrivals_naned(cum, lens)
        its_list.append(its)

    max_L = max(t.shape[1] for t in its_list)
    padded = []
    for t in its_list:
        if t.shape[1] < max_L:
            pad = torch.full((t.shape[0], max_L - t.shape[1], t.shape[2]), float("nan"))
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


def _merged_event_times_flat(dm) -> np.ndarray:
    """Flat array of all valid event times across train + val + test."""
    parts = []
    align_to_anchor = bool(getattr(dm, "align_diagnostic_event_times_to_anchor", False))
    for split in SPLITS:
        cum = getattr(dm, f"{split}_in")
        lens = getattr(dm, f"{split}_in_len")
        et = _event_times_naned(cum, lens, align_to_anchor=align_to_anchor)[:, :, 0].detach().cpu().numpy().reshape(-1)
        parts.append(et[~np.isnan(et)])
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Internal axes-level helpers
# ---------------------------------------------------------------------------


def _plot_single_split_intensity(
    ax: plt.Axes,
    event_times: np.ndarray,
    title: str = "",
    time_horizon: float = 0.0,
    ylabel: str = "Empirical Intensity",
    num_bins: int = 25,
) -> None:
    finite = event_times[np.isfinite(event_times)]
    if finite.size == 0:
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.set_title("")
        return

    horizon = max(float(time_horizon), float(finite.max()))
    if horizon <= 0.0:
        horizon = max(float(finite.max()), 1e-12)

    ax.hist(
        finite,
        bins=num_bins,
        range=(0.0, horizon),
        density=True,
        color=sns.color_palette("inferno")[3],
        alpha=0.85,
        edgecolor="none",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.0, horizon)
    return


def _plot_single_split_its(
    ax: plt.Axes,
    its: np.ndarray,
    title: str = "",
    *,
    add_legend: bool = True,
    xlabel: str = "Value",
    ylabel: str = "PDF",
) -> None:
    from src.plot.tpp_plot_components import plot_one_distribution_against_another

    finite = its[np.isfinite(its)]
    if finite.size == 0:
        ax.set_title("")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    plot_one_distribution_against_another(
        its,
        None,
        ax,
        add_legend=add_legend,
        log_scale=True,
        title=title,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plot_correlation_heatmap_on_ax(ax: plt.Axes, data_np: np.ndarray, title: str, show_cbar: bool) -> None:
    annot = data_np.shape[0] <= 15
    sns.heatmap(
        data_np,
        ax=ax,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        annot=annot,
        fmt=".2f",
        linewidths=0.0,
        cbar=show_cbar,
    )
    ax.set_xlabel("Index", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Index", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def _plot_acf_bar(
    ax: plt.Axes,
    acf: np.ndarray,
    title: str,
    color: str = "steelblue",
    ylabel: str = "ACF",
) -> None:
    lags = np.arange(len(acf))
    ax.bar(lags, acf, color=color, alpha=0.85, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plot_acf_overlay(
    ax: plt.Axes,
    series_by_split: Dict[str, torch.Tensor],
    title: str,
    max_lag: int,
) -> None:
    effective_lag = _shared_effective_lag(series_by_split, max_lag)
    if effective_lag <= 0:
        ax.set_title(f"{title} [insufficient length]")
        return

    for split in SPLITS:
        acf = autocorr(series_by_split[split], effective_lag).detach().cpu().numpy().reshape(-1)
        ax.plot(
            np.arange(len(acf)),
            acf,
            label=_split_label(split),
            color=_split_color(split),
            linestyle=SPLIT_LINESTYLES[split],
            linewidth=1.6,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(title)
    ax.legend()


def _plot_sequence_marks_on_ax(
    ax: plt.Axes,
    fig: Figure,
    dm,
    split: str,
    indices: Sequence[int],
    mode: str,
    title: str,
    add_colorbar: bool = True,
):
    split_data = _split_attrs(dm, split)
    cum = split_data["cum"]
    lens = split_data["lens"]
    marks_tensor = split_data["marks"]
    assert marks_tensor is not None, f"{type(dm).__name__} has no marks for split '{split}'."
    if marks_tensor.ndim == 3:
        marks_tensor = marks_tensor.squeeze(-1)
    num_marks = int(dm.num_marks)

    scatter = None
    if mode == "counting":
        for index in indices:
            length = int(lens[index].item())
            times = cum[index, :length, 0].detach().cpu().numpy()
            times = times - times[0]
            marks = marks_tensor[index, :length].detach().cpu().numpy().astype(int)

            event_times = times[1:]
            event_marks = marks[1:]
            for mark_id in range(num_marks):
                counts = np.concatenate(([0], np.cumsum(event_marks == mark_id)))
                x = np.concatenate(([0.0], event_times))
                x = np.append(x, float(dm.time_max))
                y = np.append(counts, counts[-1])
                ax.step(x, y, where="post", label=f"Seq {index} - mark {mark_id}")
        ax.set_ylabel("Event count")
        ax.legend(loc="best", fontsize=7)
    else:
        all_marks: List[int] = []
        for row_id, index in enumerate(indices):
            length = int(lens[index].item())
            times = cum[index, :length, 0].detach().cpu().numpy()
            times = times - times[0]
            marks = marks_tensor[index, :length].detach().cpu().numpy().astype(int)
            event_times = times[1:]
            event_marks = marks[1:]
            y = np.full_like(event_times, row_id, dtype=float)
            scatter = ax.scatter(
                event_times,
                y,
                c=event_marks,
                cmap="tab10",
                vmin=0,
                vmax=max(num_marks - 1, 0),
                s=14,
            )
            all_marks.extend(event_marks.tolist())
        ax.set_ylabel("Sequence index")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([str(i) for i in indices])
        if add_colorbar and all_marks and scatter is not None:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Mark")

    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True)
    return scatter


def _plot_intensity_all_splits_on_ax(ax: plt.Axes, dm) -> None:
    """Draw the pooled empirical intensity histogram onto an existing axes."""
    event_times = _merged_event_times_flat(dm)
    _plot_single_split_intensity(
        ax,
        event_times,
        time_horizon=float(dm.time_max),
        title="",
        ylabel=r"$\lambda$",
    )
    for patch in ax.patches:
        patch.set_rasterized(True)
    ax.set_xlabel("Time", fontsize=_DIAG_INTENSITY_PDF_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Intensity", fontsize=_DIAG_INTENSITY_PDF_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(labelsize=_DIAG_TICK_LABEL_FONTSIZE)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    sns.despine(ax=ax)


def _plot_its_all_splits_on_ax(ax: plt.Axes, dm) -> None:
    """Draw the pooled inter-arrival PDF onto an existing axes."""
    its = _merged_its_tensor(dm, drop_first=False)[:, :, 0].detach().cpu().numpy()
    _plot_single_split_its(
        ax,
        its,
        title="",
        add_legend=False,
        xlabel="",
        ylabel="PDF Interarrival Times",
    )
    for patch in ax.patches:
        patch.set_rasterized(True)
    ax.set_xlabel("Interarrival Time", fontsize=_DIAG_INTENSITY_PDF_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("PDF", fontsize=_DIAG_INTENSITY_PDF_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(labelsize=_DIAG_TICK_LABEL_FONTSIZE)
    sns.despine(ax=ax)


# ---------------------------------------------------------------------------
# Public plot functions — single split
# ---------------------------------------------------------------------------


def plot_sample_paths(
    dm, indices: Optional[Iterable[int]] = None, split: str = "train", max_paths: int = DEFAULT_MAX_PATHS
) -> Figure:
    from src.plot.tpp_plot_components import plot_temporal_point_process

    cum = getattr(dm, f"{split}_in")
    lens = getattr(dm, f"{split}_in_len")
    indices = _resolve_indices(indices, cum.shape[0], max_paths)
    seqs_for_plot = _event_times_naned(
        cum,
        lens,
        align_to_anchor=bool(getattr(dm, "align_diagnostic_event_times_to_anchor", False)),
    )

    fig, ax = plt.subplots()
    plot_temporal_point_process(
        target_seqs=seqs_for_plot[indices].detach().cpu().numpy(),
        target_lens=(lens[indices] - 1).detach().cpu().numpy(),
        time_max=float(dm.time_max),
        ax=ax,
        max_paths=len(indices),
    )
    return fig


def plot_inter_arrival_distribution(dm, split: str = "train") -> Figure:
    """Single-split inter-arrival histogram."""
    cum = getattr(dm, f"{split}_in")
    lens = getattr(dm, f"{split}_in_len")
    its_np = _interarrivals_naned(cum, lens)[:, :, 0].detach().cpu().numpy().reshape(-1)
    its_np = its_np[~np.isnan(its_np)]

    fig, ax = plt.subplots()
    sns.histplot(its_np, bins=50, ax=ax, stat="density", edgecolor="none", color=sns.color_palette("flare")[3])
    ax.set_xlabel("Inter-arrival time")
    ax.set_ylabel("Density")
    ax.set_title(f"{type(dm).__name__}: inter-arrival distribution ({split})")
    return fig


def plot_correlation_heatmap(
    dm,
    split: str = "train",
    min_samples: int = DEFAULT_MIN_SAMPLES_FOR_CORR,
    max_lag: int = DEFAULT_MAX_LAG,
) -> Figure:
    """Pearson correlation across inter-arrival positions (drop tau_1, NaN-masked)."""
    cum = getattr(dm, f"{split}_in")
    lens = getattr(dm, f"{split}_in_len")
    its = _interarrivals_drop_first(cum, lens)
    heatmap_max_lag = _correlation_heatmap_lag_limit(max_lag)
    corr_features = _prepare_corr_features(its, min_samples, max_lag=heatmap_max_lag)

    fig, ax = plt.subplots()
    if corr_features.shape[1] < 2:
        ax.set_title(f"{type(dm).__name__}: correlation heatmap ({split}) [insufficient usable columns]")
        return fig

    corr = corr_torch(corr_features)
    data_np = corr.detach().cpu().numpy()
    if np.isnan(data_np).all():
        ax.set_title(f"{type(dm).__name__}: correlation heatmap ({split}) [all NaN]")
        return fig
    data_np = _truncate_corr_matrix(data_np, heatmap_max_lag)
    if data_np.shape[0] < 2:
        ax.set_title(f"{type(dm).__name__}: correlation heatmap ({split}) [insufficient lag window]")
        return fig

    annot = data_np.shape[0] <= 15
    sns.heatmap(
        data_np,
        ax=ax,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        annot=annot,
        fmt=".2f",
        linewidths=0.0,
        cbar=True,
    )
    ax.set_xlabel("Index", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Index", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{type(dm).__name__}: corr(tau_i, tau_j), i,j <= {data_np.shape[0]} ({split})")
    return fig


def plot_autocorrelation_inter_arrivals(dm, split: str = "train", max_lag: int = DEFAULT_MAX_LAG) -> Figure:
    cum = getattr(dm, f"{split}_in")
    lens = getattr(dm, f"{split}_in_len")
    its = _interarrivals_drop_first(cum, lens)
    effective_lag = min(max_lag, DEFAULT_ACF_DISPLAY_MAX_LAG, max(its.shape[1] - 1, 0))

    fig, ax = plt.subplots()
    if effective_lag <= 0:
        ax.set_title(f"{type(dm).__name__}: ACF inter-arrivals ({split}) [insufficient length]")
        return fig

    acf = autocorr(its, effective_lag).detach().cpu().numpy().reshape(-1)
    _plot_acf_bar(
        ax,
        acf,
        f"{type(dm).__name__}: ACF inter-arrivals ({split})",
    )
    return fig


def plot_autocorrelation_cumulative(dm, split: str = "train", max_lag: int = DEFAULT_MAX_LAG) -> Figure:
    cum = getattr(dm, f"{split}_in")
    lens = getattr(dm, f"{split}_in_len")
    its = _interarrivals_drop_first(cum, lens)
    cumrel = its.cumsum(dim=1)
    cumrel = cumrel.masked_fill(torch.isnan(its), float("nan"))
    effective_lag = min(max_lag, DEFAULT_ACF_DISPLAY_MAX_LAG, max(cumrel.shape[1] - 1, 0))

    fig, ax = plt.subplots()
    if effective_lag <= 0:
        ax.set_title(f"{type(dm).__name__}: ACF cumulative ({split}) [insufficient length]")
        return fig

    acf = autocorr(cumrel, effective_lag).detach().cpu().numpy().reshape(-1)
    _plot_acf_bar(
        ax,
        acf,
        f"{type(dm).__name__}: ACF cumulative time ({split})",
        color="darkorange",
    )
    return fig


def plot_mark_frequencies(dm, split: str = "train") -> Figure:
    """Bar chart of mark frequencies. Only meaningful when num_marks > 1."""
    if int(dm.num_marks) <= 1:
        raise ValueError(f"{type(dm).__name__} has num_marks=1; mark plot not meaningful.")

    marks = getattr(dm, f"{split}_marks")
    lens = getattr(dm, f"{split}_in_len")
    stats = _mark_frequency_stats(marks, lens, dm.num_marks)

    fig, ax = plt.subplots()
    ks = np.arange(stats["num_marks"])
    ax.bar(ks, stats["frequencies"], color=sns.color_palette("tab10")[: stats["num_marks"]])
    ax.set_xticks(ks)
    ax.set_xlabel("Mark")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{type(dm).__name__}: mark frequencies ({split}, total events={stats['total_events']})")
    return fig


def plot_sequence_marks(dm, indices: Iterable[int], mode: str = "counting", split: str = "train") -> Figure:
    """Per-mark counting paths or mark raster for a few sequences."""
    if int(dm.num_marks) <= 1:
        raise ValueError(f"{type(dm).__name__} has num_marks=1; mark plot not meaningful.")
    if mode not in {"counting", "raster"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'counting' or 'raster'.")

    indices = list(indices)
    split_data = _split_attrs(dm, split)
    cum = split_data["cum"]
    assert all(0 <= i < len(cum) for i in indices), f"indices out of range for {split}_in"

    fig, ax = plt.subplots()
    title = (
        f"{type(dm).__name__}: per-mark counting paths ({split})"
        if mode == "counting"
        else f"{type(dm).__name__}: mark raster ({split})"
    )
    _plot_sequence_marks_on_ax(ax, fig, dm, split, indices, mode, title)
    return fig


# ---------------------------------------------------------------------------
# Public plot functions — comparative (train / val / test side-by-side)
# ---------------------------------------------------------------------------


def plot_sample_paths_comparison(dm, max_paths: int = DEFAULT_MAX_PATHS) -> Figure:
    from src.plot.tpp_plot_components import plot_temporal_point_process

    n_paths = _shared_max_paths(dm, max_paths)
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    for ax, split in zip(axes, SPLITS):
        split_data = _split_attrs(dm, split)
        cum = split_data["cum"]
        lens = split_data["lens"]
        indices = _resolve_indices(None, cum.shape[0], n_paths)
        seqs_for_plot = _event_times_naned(
            cum,
            lens,
            align_to_anchor=bool(getattr(dm, "align_diagnostic_event_times_to_anchor", False)),
        )
        if not indices:
            ax.set_title(f"{_split_label(split)} [no sequences]")
            continue
        plot_temporal_point_process(
            target_seqs=seqs_for_plot[indices].detach().cpu().numpy(),
            target_lens=(lens[indices] - 1).detach().cpu().numpy(),
            time_max=float(dm.time_max),
            ax=ax,
            max_paths=len(indices),
        )
        ax.set_title("")
    _set_window_title(fig, f"{type(dm).__name__}: sample paths")
    fig.tight_layout()
    return fig


def plot_sequence_length_histogram(dm) -> Figure:
    fig, ax = plt.subplots()

    lengths_by_split = {split: getattr(dm, f"{split}_in_len").detach().cpu().numpy() for split in SPLITS}
    non_empty = {s: arr for s, arr in lengths_by_split.items() if arr.size > 0}
    if not non_empty:
        ax.set_title(f"{type(dm).__name__}: sequence length distribution [no data]")
        return fig
    min_len = min(int(arr.min()) for arr in non_empty.values())
    max_len = max(int(arr.max()) for arr in non_empty.values())

    for idx, split in enumerate(SPLITS):
        arr = lengths_by_split[split]
        if arr.size == 0:
            continue
        sns.histplot(
            arr,
            bins=max_len - min_len + 1,
            binrange=(min_len - 0.5, max_len + 0.5),
            discrete=True,
            label=_split_label(split),
            ax=ax,
            stat="density",
            edgecolor="none",
            color=_split_color(split),
            alpha=_overlay_alpha(idx, len(SPLITS)),
        )
        ax.axvline(
            arr.mean(),
            color=_split_color(split),
            linestyle=SPLIT_LINESTYLES[split],
            linewidth=1.1,
            alpha=0.9,
        )

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Density")
    ax.set_title(f"{type(dm).__name__}: sequence length distribution")
    ax.legend(fontsize=7)
    _set_window_title(fig, f"{type(dm).__name__}: sequence length distribution")
    return fig


def plot_inter_arrival_train_vs_val(dm) -> Figure:
    fig, ax = plt.subplots()

    for idx, split in enumerate(SPLITS):
        split_data = _split_attrs(dm, split)
        its_np = _interarrivals_naned(split_data["cum"], split_data["lens"])[:, :, 0].detach().cpu().numpy().reshape(-1)
        its_np = its_np[~np.isnan(its_np)]
        sns.histplot(
            its_np,
            bins=50,
            ax=ax,
            label=_split_label(split),
            stat="density",
            edgecolor="none",
            color=_split_color(split),
            alpha=_overlay_alpha(idx, len(SPLITS)),
        )
    ax.set_xlabel("Inter-arrival time")
    ax.set_ylabel("Density")
    ax.set_title(f"{type(dm).__name__}: inter-arrival distribution")
    ax.legend()
    _set_window_title(fig, f"{type(dm).__name__}: inter-arrival distribution")
    return fig


def plot_intensity_and_its(dm) -> Figure:
    fig, axes = plt.subplots(2, 3, sharey="row")
    align_to_anchor = bool(getattr(dm, "align_diagnostic_event_times_to_anchor", False))
    for col, split in enumerate(SPLITS):
        split_data = _split_attrs(dm, split)
        event_times = (
            _event_times_naned(
                split_data["cum"],
                split_data["lens"],
                align_to_anchor=align_to_anchor,
            )[:, :, 0]
            .detach()
            .cpu()
            .numpy()
        )
        _plot_single_split_intensity(
            axes[0, col],
            event_times,
            f"{_split_label(split)} intensity",
            time_horizon=float(dm.time_max),
        )

        its = _interarrivals_naned(split_data["cum"], split_data["lens"])[:, :, 0].detach().cpu().numpy()
        _plot_single_split_its(axes[1, col], its, f"{_split_label(split)} I.T.S.")
    _set_window_title(fig, f"{type(dm).__name__}: intensity and ITS")
    fig.tight_layout()
    return fig


def plot_correlation_heatmaps(
    dm,
    min_samples: int = DEFAULT_MIN_SAMPLES_FOR_CORR,
    max_lag: int = DEFAULT_MAX_LAG,
) -> Figure:
    fig, axes = plt.subplots(1, 3)
    heatmap_max_lag = _correlation_heatmap_lag_limit(max_lag)

    for idx, split in enumerate(SPLITS):
        split_data = _split_attrs(dm, split)
        its = _interarrivals_drop_first(split_data["cum"], split_data["lens"])
        corr_features = _prepare_corr_features(its, min_samples, max_lag=heatmap_max_lag)
        ax = axes[idx]
        if corr_features.shape[1] < 2:
            ax.set_title(f"{_split_label(split)} [insufficient usable columns]")
            continue

        corr = corr_torch(corr_features)
        data_np = corr.detach().cpu().numpy()
        if np.isnan(data_np).all():
            ax.set_title(f"{_split_label(split)} [all NaN]")
            continue
        data_np = _truncate_corr_matrix(data_np, heatmap_max_lag)
        if data_np.shape[0] < 2:
            ax.set_title(f"{_split_label(split)} [insufficient lag window]")
            continue

        _plot_correlation_heatmap_on_ax(
            ax,
            data_np,
            f"{_split_label(split)} (first {data_np.shape[0]} taus)",
            show_cbar=idx == len(SPLITS) - 1,
        )
        if idx > 0:
            ax.set_ylabel("")

    _set_window_title(fig, f"{type(dm).__name__}: correlation heatmaps")
    fig.tight_layout()
    return fig


def plot_autocorrelation_inter_arrivals_comparison(dm, max_lag: int = DEFAULT_MAX_LAG) -> Figure:
    series_by_split = {
        split: _interarrivals_drop_first(getattr(dm, f"{split}_in"), getattr(dm, f"{split}_in_len")) for split in SPLITS
    }
    fig, ax = plt.subplots()
    _plot_acf_overlay(ax, series_by_split, f"{type(dm).__name__}: ACF inter-arrivals", max_lag=max_lag)
    _set_window_title(fig, f"{type(dm).__name__}: ACF inter-arrivals")
    return fig


def plot_autocorrelation_cumulative_comparison(dm, max_lag: int = DEFAULT_MAX_LAG) -> Figure:
    series_by_split = {}
    for split in SPLITS:
        its = _interarrivals_drop_first(getattr(dm, f"{split}_in"), getattr(dm, f"{split}_in_len"))
        cumrel = its.cumsum(dim=1)
        series_by_split[split] = cumrel.masked_fill(torch.isnan(its), float("nan"))

    fig, ax = plt.subplots()
    _plot_acf_overlay(ax, series_by_split, f"{type(dm).__name__}: ACF cumulative time", max_lag=max_lag)
    _set_window_title(fig, f"{type(dm).__name__}: ACF cumulative time")
    return fig


def plot_mark_frequencies_comparison(dm) -> Figure:
    if int(dm.num_marks) <= 1:
        raise ValueError(f"{type(dm).__name__} has num_marks=1; mark plot not meaningful.")

    stats_by_split = {
        split: _mark_frequency_stats(getattr(dm, f"{split}_marks"), getattr(dm, f"{split}_in_len"), dm.num_marks)
        for split in SPLITS
    }

    fig, ax = plt.subplots()
    ks = np.arange(int(dm.num_marks))
    width = 0.24
    offsets = np.array([-width, 0.0, width])
    for offset, split in zip(offsets, SPLITS):
        ax.bar(
            ks + offset,
            stats_by_split[split]["frequencies"],
            width=width,
            color=_split_color(split),
            label=_split_label(split),
        )

    ax.set_xticks(ks)
    ax.set_xlabel("Mark")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{type(dm).__name__}: mark frequencies")
    ax.legend()
    _set_window_title(fig, f"{type(dm).__name__}: mark frequencies")
    return fig


def plot_sequence_marks_comparison(dm, mode: str = "raster", max_paths: int = 5) -> Figure:
    if int(dm.num_marks) <= 1:
        raise ValueError(f"{type(dm).__name__} has num_marks=1; mark plot not meaningful.")
    if mode not in {"counting", "raster"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'counting' or 'raster'.")

    n_paths = _shared_max_paths(dm, max_paths)
    fig, axes = plt.subplots(3, 1, sharex=True)
    shared_scatter = None
    for ax, split in zip(axes, SPLITS):
        split_len = getattr(dm, f"{split}_in").shape[0]
        indices = _resolve_indices(None, split_len, n_paths)
        if not indices:
            ax.set_title(f"{_split_label(split)} [no sequences]")
            continue
        scatter = _plot_sequence_marks_on_ax(
            ax,
            fig,
            dm,
            split,
            indices,
            mode,
            _split_label(split),
            add_colorbar=False,
        )
        if scatter is not None:
            shared_scatter = scatter
    if mode == "raster" and shared_scatter is not None:
        fig.subplots_adjust(right=0.88)
        cbar = fig.colorbar(shared_scatter, ax=np.atleast_1d(axes), location="right", pad=0.02)
        cbar.set_label("Mark")
    else:
        fig.tight_layout()
    _set_window_title(fig, f"{type(dm).__name__}: mark {mode}")
    return fig


# ---------------------------------------------------------------------------
# Public plot functions — router
# ---------------------------------------------------------------------------


def plot_sequences(dm, indices, kind: str = "sample_paths", **kwargs) -> Figure:
    """Unified router: kind in {sample_paths, trading_times, marks, marks_counting, marks_raster}."""
    if kind in {"sample_paths", "trading_times"}:
        return plot_sample_paths(dm, indices, split=kwargs.get("split", "train"))
    if kind in {"marks", "marks_counting"}:
        return plot_sequence_marks(dm, indices, mode=kwargs.get("mode", "counting"), split=kwargs.get("split", "train"))
    if kind == "marks_raster":
        return plot_sequence_marks(dm, indices, mode="raster", split=kwargs.get("split", "train"))
    raise ValueError(
        f"Unsupported kind '{kind}'. Expected one of: "
        f"'sample_paths', 'trading_times', 'marks', 'marks_counting', 'marks_raster'."
    )


# ---------------------------------------------------------------------------
# Public plot functions — impact visualization
# ---------------------------------------------------------------------------


def plot_exp_scaling_impact(dm, concentration_factor: float = 1.0, shift_param: float = 0.0) -> Figure:
    """Visualize the effect of ExpScaler on inter-arrival distributions (train + val)."""
    from src.plot.tpp_plot_components import convert_seqs_to_masked_df
    from src.data_transformations.expscaler import ExpScaler, ScalingStrategy

    scaler = ExpScaler(
        dm.train_in.diff(dim=1), dm.train_in_len - 1, concentration_factor, shift_param, ScalingStrategy.NAIVE
    )
    transformed_train = scaler(dm.train_in.diff(dim=1))
    transformed_val = scaler(dm.val_in.diff(dim=1))

    fig, axes = plt.subplots(1, 2)

    x = dm.train_in.diff(dim=1)[:, :, 0]
    train_df = convert_seqs_to_masked_df(x.detach().cpu().numpy(), dm.train_in_len.detach().cpu().numpy() - 1)
    x1 = transformed_train[:, :, 0]
    train_t_df = convert_seqs_to_masked_df(x1.detach().cpu().numpy(), dm.train_in_len.detach().cpu().numpy() - 1)
    x2 = dm.val_in.diff(dim=1)[:, :, 0]
    val_df = convert_seqs_to_masked_df(x2.detach().cpu().numpy(), dm.val_in_len.detach().cpu().numpy() - 1)
    x3 = transformed_val[:, :, 0]
    val_t_df = convert_seqs_to_masked_df(x3.detach().cpu().numpy(), dm.val_in_len.detach().cpu().numpy() - 1)

    sns.histplot(
        train_df,
        x="value",
        bins=50,
        ax=axes[0],
        label="Train original",
        stat="density",
        edgecolor="none",
        color=sns.color_palette("flare")[5],
    )
    sns.histplot(
        val_df,
        x="value",
        bins=50,
        ax=axes[0],
        label="Val original",
        stat="density",
        edgecolor="none",
        color=sns.color_palette("flare")[0],
        alpha=0.5,
    )
    axes[0].set_title("Original")
    axes[0].set_xlabel("Inter-arrival")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    sns.histplot(
        train_t_df,
        x="value",
        bins=50,
        ax=axes[1],
        label="Train transformed",
        stat="density",
        edgecolor="none",
        color=sns.color_palette("flare")[5],
    )
    sns.histplot(
        val_t_df,
        x="value",
        bins=50,
        ax=axes[1],
        label="Val transformed",
        stat="density",
        edgecolor="none",
        color=sns.color_palette("flare")[0],
        alpha=0.5,
    )
    axes[1].set_title("Transformed")
    axes[1].set_xlabel("Inter-arrival")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    return fig


# ---------------------------------------------------------------------------
# Public plot functions — all splits pooled
# ---------------------------------------------------------------------------


def plot_sample_paths_all_splits(dm, max_paths: int = DEFAULT_MAX_PATHS) -> Figure:
    """Sample paths from all splits pooled onto a single axes."""
    from src.plot.tpp_plot_components import plot_temporal_point_process

    n_per_split = max(1, max_paths // len(SPLITS))
    align_to_anchor = bool(getattr(dm, "align_diagnostic_event_times_to_anchor", False))

    all_seqs, all_lens = [], []
    for split in SPLITS:
        cum = getattr(dm, f"{split}_in")
        lens = getattr(dm, f"{split}_in_len")
        n = min(n_per_split, cum.shape[0])
        seqs = _event_times_naned(cum[:n], lens[:n], align_to_anchor=align_to_anchor)
        all_seqs.append(seqs)
        all_lens.append(lens[:n])

    max_L = max(s.shape[1] for s in all_seqs)
    padded = []
    for seqs in all_seqs:
        if seqs.shape[1] < max_L:
            pad = torch.full((seqs.shape[0], max_L - seqs.shape[1], seqs.shape[2]), float("nan"))
            seqs = torch.cat([seqs, pad], dim=1)
        padded.append(seqs)

    merged_seqs = torch.cat(padded, dim=0).detach().cpu().numpy()
    merged_lens = (torch.cat(all_lens, dim=0) - 1).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=_DIAG_FLAT_FIGSIZE)
    plot_temporal_point_process(
        target_seqs=merged_seqs,
        target_lens=merged_lens,
        time_max=float(dm.time_max),
        ax=ax,
        max_paths=len(merged_seqs),
    )
    ax.set_xlabel("Time", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Samples", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(labelsize=_DIAG_TICK_LABEL_FONTSIZE)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_intensity_all_splits(dm) -> Figure:
    """Empirical intensity histogram using all splits pooled."""
    fig, ax = plt.subplots(figsize=_DIAG_FLAT_FIGSIZE)
    _plot_intensity_all_splits_on_ax(ax, dm)
    _set_window_title(fig, f"{_dataset_report_slug(dm)}: intensity")
    fig.tight_layout()
    return fig


def plot_its_all_splits(dm) -> Figure:
    """Inter-arrival time PDF using all splits pooled."""
    fig, ax = plt.subplots()
    _plot_its_all_splits_on_ax(ax, dm)
    _set_window_title(fig, f"{_dataset_report_slug(dm)}: ITS")
    fig.tight_layout()
    return fig


def plot_intensity_pdf_all_splits(dm) -> Figure:
    """Combined 2x1 figure with pooled intensity on top and PDF below."""
    fig, axes = plt.subplots(2, 1, figsize=_DIAG_STACKED_FLAT_FIGSIZE)
    _plot_intensity_all_splits_on_ax(axes[0], dm)
    _plot_its_all_splits_on_ax(axes[1], dm)
    _set_window_title(fig, f"{_dataset_report_slug(dm)}: intensity + pdf")
    fig.tight_layout()
    return fig


def plot_acf_all_splits(dm, max_lag: int = DEFAULT_MAX_LAG) -> Figure:
    """ACF of inter-arrivals using all splits pooled."""
    its = _merged_its_tensor(dm, drop_first=True)
    effective_lag = min(max_lag, DEFAULT_ACF_DISPLAY_MAX_LAG, max(its.shape[1] - 1, 0))
    fig, ax = plt.subplots(figsize=_DIAG_FLAT_FIGSIZE)
    if effective_lag > 0:
        acf = autocorr(its, effective_lag).detach().cpu().numpy().reshape(-1)
        lags = np.arange(len(acf))
        ax.plot(lags, acf, linewidth=2.5, color=sns.color_palette("magma")[3])
        ax.axhline(0.0, color="black", alpha=0.4, linewidth=1)
    ax.set_xlabel("Lag", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Autocorrelation", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(labelsize=_DIAG_TICK_LABEL_FONTSIZE)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    sns.despine(ax=ax)
    _set_window_title(fig, f"{_dataset_report_slug(dm)}: ACF")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap_all_splits(
    dm,
    min_samples: int = DEFAULT_MIN_SAMPLES_FOR_CORR,
    max_lag: int = DEFAULT_MAX_LAG,
) -> Figure:
    """Pearson correlation heatmap using all splits pooled."""
    its = _merged_its_tensor(dm, drop_first=True)
    heatmap_max_lag = _correlation_heatmap_lag_limit(max_lag)
    corr_features = _prepare_corr_features(its, min_samples, max_lag=heatmap_max_lag)

    fig, ax = plt.subplots(figsize=_DIAG_FLAT_FIGSIZE)
    if corr_features.shape[1] < 2:
        ax.set_title("")
        return fig

    corr = corr_torch(corr_features)
    data_np = corr.detach().cpu().numpy()
    if np.isnan(data_np).all():
        ax.set_title("")
        return fig

    data_np = _truncate_corr_matrix(data_np, heatmap_max_lag)
    if data_np.shape[0] < 2:
        ax.set_title("")
        return fig

    hm = sns.heatmap(
        data_np,
        ax=ax,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        annot=data_np.shape[0] <= 15,
        fmt=".2f",
        linewidths=0.0,
        cbar=True,
    )
    ax.set_xlabel("Index", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Index", fontsize=_DIAG_AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if hm.collections:
        hm.collections[0].colorbar.ax.tick_params(labelsize=_DIAG_TICK_LABEL_FONTSIZE)
    _set_window_title(fig, f"{_dataset_report_slug(dm)}: correlation")
    fig.tight_layout()
    return fig
