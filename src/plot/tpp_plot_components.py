import logging.config
import math
import typing
import warnings

import numpy as np
import numpy.typing
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from src.utils.fix_seq_ends import get_masked_array_on_lengths

sns.set_theme()

logger = logging.getLogger(__name__)


def get_histogram_outline(
    data: np.ndarray,
    bins: int,
    density: bool = True,
    hist_range: typing.Optional[typing.Tuple[float, float]] = None,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    # Linear interpolation (mid-point) of the histogram to get an outline of the histogram.
    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        return np.array([]), np.array([])
    pdf, bin_edges = np.histogram(finite_data, bins=bins, density=density, range=hist_range)

    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_midpoints, pdf


def ensure_numpy_array(data):
    """Ensure the input is a NumPy array. Convert tensors to arrays if needed."""
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def _format_stat_float(value: float) -> str:
    """Format scalar stats without fixed decimal rounding."""
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return np.format_float_scientific(value, unique=True, trim="-")
    return np.format_float_positional(value, unique=True, trim="-")


def qq_plot_multi_seqs_against_targets(
    seqs_sampled: np.typing.ArrayLike,
    targets_sampled: np.typing.ArrayLike,
    ax: plt.Axes,
    add_legend: bool = True,
    flatten: bool = False,
):
    """
    For each of the components (columns), make a QQ plot against the targets.
    All QQ plots are drawn on the same axes with different colors.
    The red x=y reference line is drawn only once.

    If flatten=True, all dimensions (cols 1:) are pooled into a single QQ plot
    instead of one colored line per feature.
    """
    # Ensure inputs are NumPy arrays. Otherwise, needs to be converted bc np.isnan does not work on tensors.
    seqs_sampled = ensure_numpy_array(seqs_sampled)
    targets_sampled = ensure_numpy_array(targets_sampled)

    if seqs_sampled.shape != targets_sampled.shape:
        raise ValueError("Shape mismatch between seqs_sampled and targets_sampled.")

    n_samples, n_features = seqs_sampled.shape

    # Draw x=y reference line using the same channel filter as the per-feature path.
    SUBSAMPLE_FREQ = 2
    all_vals = np.concatenate(
        [seqs_sampled[:, 1::SUBSAMPLE_FREQ].flatten(), targets_sampled[:, 1::SUBSAMPLE_FREQ].flatten()]
    )
    all_vals = all_vals[~np.isnan(all_vals)]
    if all_vals.size == 0:
        logger.warning("qq_plot: no valid values to draw reference line; x=y line will be omitted.")
    else:
        min_val, max_val = float(all_vals.min()), float(all_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)

    MAX_SAMPLES_PLOT = 10000

    if flatten:
        # Pool the same time steps as the per-time-step path so both views are comparable.
        # Skip col 0 (tau1 is sampled artificially); apply SUBSAMPLE_FREQ to select the
        # same time indices that the per-feature loop would plot.
        cols = [i for i in range(1, n_features) if i % SUBSAMPLE_FREQ == 0]
        x_all = seqs_sampled[:, cols].flatten()
        y_all = targets_sampled[:, cols].flatten()
        # Filter NaNs independently: QQ plots compare sorted quantiles, not paired values.
        x_all = x_all[~np.isnan(x_all)]
        y_all = y_all[~np.isnan(y_all)]

        if x_all.size == 0 or y_all.size == 0:
            logger.warning("qq_plot flatten=True: no valid (non-NaN) values after stripping; plot will be empty.")
        else:
            n = min(x_all.size, y_all.size, MAX_SAMPLES_PLOT)
            x_quantiles = np.percentile(x_all, np.linspace(0, 100, n))
            y_quantiles = np.percentile(y_all, np.linspace(0, 100, n))
            ax.scatter(y_quantiles, x_quantiles, s=1.0, color='steelblue', rasterized=True)
    else:
        color_scheme = sns.color_palette('inferno', n_features)
        # Per feature/time step we plot the QQ plot.
        for i in range(1, n_features):  # skip i=0: tau1 is sampled artificially
            # Plot a SUBSAMPLE_FREQth of the qq plots.
            if i % SUBSAMPLE_FREQ == 0:
                x = seqs_sampled[:, i]
                y = targets_sampled[:, i]

                # Filter NaNs independently: QQ plots compare sorted quantiles, not paired values.
                # A joint mask would drop valid data when length distributions differ across sets.
                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]

                if x.shape[0] == 0 or y.shape[0] == 0:
                    continue  # skip empty sequences

                # Use evenly-spaced percentile indices so both sides cover quantiles 0–1
                n = min(len(x), len(y), MAX_SAMPLES_PLOT)
                x_quantiles = np.percentile(x, np.linspace(0, 100, n))
                y_quantiles = np.percentile(y, np.linspace(0, 100, n))
                ax.scatter(y_quantiles, x_quantiles, s=1.0, color=color_scheme[i])

    if add_legend:
        title = 'QQ Plot (Flattened)' if flatten else 'QQ Plots for Each Time Stamp'
        ax.set_title(title)
        ax.set_xlabel('Target Quantiles')
        ax.set_ylabel('Sampled Quantiles')
    else:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax.grid(True)
    ax.get_figure().tight_layout()
    return


def plot_one_distribution_against_another(
    seqs_sampled: np.typing.ArrayLike,
    targets_sampled: np.typing.ArrayLike,
    ax: plt.Axes,
    add_legend: bool = True,
    log_scale: bool = None,
    title: str = "",
):
    # Ensure inputs are NumPy arrays. Otherwise, needs to be converted bc np.isnan does not work on tensors.
    # To mask, replace values with NaNs outside.

    seqs_sampled = ensure_numpy_array(seqs_sampled)
    if targets_sampled is not None:
        targets_sampled = ensure_numpy_array(targets_sampled)
        assert len(targets_sampled.shape) == 2, f"Expected 2D tensor, but got {list(targets_sampled.shape)}."

    assert len(seqs_sampled.shape) == 2, f"Expected 2D tensor, but got {list(seqs_sampled.shape)}."

    if add_legend:
        xlabel = 'Value'
        ylabel = 'PDF'
    else:
        title = ""
        xlabel = ""
        ylabel = ""

    # Define the color scheme
    color_scheme = sns.color_palette('inferno', seqs_sampled.shape[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warnings, e.g. for NaNs in histograms.
        # Casting the nans makes them throw a warning.
        # Plot histogram for seqs_sampled

        # If the data range is small relative to its magnitude, all values land in one bin.
        # Use a scale-relative threshold and padding so this works for both unit-scale and tiny-scale data.
        lo = np.nanmin(seqs_sampled)
        hi = np.nanmax(seqs_sampled)
        scale = max(abs(lo), abs(hi), 1e-10)
        if hi - lo < 1e-1 * scale:
            padding = max(0.5 * scale, 1e-6)  # floor avoids zero-width binrange when all values are 0
            binrange = (lo - padding, hi + padding)
            bins = 5
        else:
            binrange = None
            bins = get_num_bin_int_hist(seqs_sampled)
        sns.histplot(
            seqs_sampled,
            bins=bins,
            palette=color_scheme,
            ax=ax,
            stat="density",
            element="bars",
            edgecolor="none",
            binrange=binrange,
            common_norm=True,
            kde=False,
            multiple="stack",
            legend=False,
        )
        if targets_sampled is not None:
            # Get histogram outline for targets_sampled and plot
            hist_outline: typing.Tuple[np.ndarray, np.ndarray] = get_histogram_outline(
                targets_sampled, get_num_bin_int_hist(targets_sampled), True
            )
            ax.plot(hist_outline[0], hist_outline[1], color='red', linestyle=(0, (1, 3)), label='Shadow target dist.')
            ax.legend(loc='upper right', fontsize='small')
    if log_scale:
        ax.set_yscale('log')
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.get_figure().tight_layout()
    return


def get_num_bin_int_hist(data_to_plot):
    try:
        # Manually check if all inputs are integer valued
        if np.all(np.isclose(data_to_plot, data_to_plot.astype(int), atol=1e-10)):
            unique_values = np.unique(data_to_plot)
            num_categories = len(unique_values)

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


def convert_seqs_to_masked_df(seqs: np.typing.ArrayLike, seqs_lens: np.typing.ArrayLike) -> pd.DataFrame:
    assert len(seqs.shape) == 2, f"Expected a 2D tensor, but got {list(seqs.shape)}."
    assert len(seqs_lens.shape) == 1, f"Expected a 1D tensor, but got {list(seqs_lens.shape)}."

    # First get the times of the events.
    # Since the sequences are of different lengths, we need to mask the values that are not defined.
    # There are two edge cases:
    # 1. No masking required. Then, the function below returns a mask = False - Case all masked still returns an array.
    seqs_masked = get_masked_array_on_lengths(seqs, seqs_lens)
    if type(seqs_masked.mask) is bool or type(seqs_masked.mask) is np.bool_:
        indices = np.tile(np.arange(seqs_masked.shape[1]), seqs_masked.shape[0])
    else:
        # 2. When all of the sequences are masked fully, we end up with no value at all.
        indices = np.argwhere(~seqs_masked.mask)
        if indices.size == 0:
            logger.warning("No valid sequences with positive length. No empirical intensity plot will be drawn.")
            return pd.DataFrame()
        indices = indices[:, 1]

    # Create a DataFrame with the masked sequences
    seqs_as_df = pd.DataFrame(
        {
            'value': seqs_masked.compressed(),
            'index': indices,
        }
    )
    return seqs_as_df


def plot_hist_intensity(
    seqs_cum_times: np.typing.ArrayLike,
    num_bins: int,
    two_axes: plt.Axes,
    seqs_targets_cum_times: np.typing.ArrayLike = None,
    time_horizon: typing.Optional[float] = None,
):
    """
    Rescales the intensity only if true intensity is provided. Otherwise returns counts.
    """
    seqs_cum_times = ensure_numpy_array(seqs_cum_times)
    if seqs_targets_cum_times is not None:
        seqs_targets_cum_times = ensure_numpy_array(seqs_targets_cum_times)

    assert len(seqs_cum_times.shape) == 2, f"Expected a 2D tensor, but got {list(seqs_cum_times.shape)}."
    assert num_bins > 0, f"Expected a positive number of bins, but got {num_bins}."
    assert len(two_axes) == 2, f"Expected two axes, but got {len(two_axes)}."
    ax_target = two_axes[0]
    ax_samples = two_axes[1]

    finite_values = [seqs_cum_times[np.isfinite(seqs_cum_times)]]
    if seqs_targets_cum_times is not None:
        finite_values.append(seqs_targets_cum_times[np.isfinite(seqs_targets_cum_times)])
    finite_values = [values for values in finite_values if values.size > 0]
    if not finite_values:
        raise ValueError("Expected at least one finite cumulative event time for the intensity histogram.")

    observed_max = max(float(values.max()) for values in finite_values)
    horizon = observed_max
    if time_horizon is not None:
        horizon = max(horizon, float(time_horizon))
    if horizon <= 0.0:
        horizon = max(observed_max, 1e-12)

    bin_edges = np.linspace(0.0, horizon, num_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    if seqs_targets_cum_times is not None:
        hist_outline: typing.Tuple[np.ndarray, np.ndarray] = get_histogram_outline(
            seqs_targets_cum_times, num_bins, False, hist_range=(0.0, horizon)
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        intensity_plot = sns.histplot(
            seqs_cum_times,
            palette=sns.color_palette('inferno', seqs_cum_times.shape[1]),
            stat='count',
            multiple='stack',
            ax=ax_samples,
            bins=bin_edges,
            ######################
            # Common_norm needs to be false because of the nans. We rescale the y-axis manually.
            common_norm=False,
            legend=False,
            edgecolor='none',
        )
    intensity_plot.set_xlabel("Time")
    intensity_plot.set_ylabel("Frequency")
    ax_samples.set_xlim(0.0, horizon)

    if seqs_targets_cum_times is not None:
        ax_samples.plot(
            hist_outline[0],
            # the normalisation is a safety measure
            hist_outline[1] * (seqs_cum_times.shape[0] / max(seqs_targets_cum_times.shape[0], 1)),
            color='red',
            linestyle=(0, (1, 3)),
            label='Target Intensity',
        )
        ax_samples.legend(fontsize='small')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            intensity_plot = sns.histplot(
                seqs_targets_cum_times,
                palette=sns.color_palette('inferno', seqs_targets_cum_times.shape[1]),
                stat='count',
                multiple='stack',
                ax=ax_target,
                bins=bin_edges,
                # Common_norm needs to be false because of the nans. We rescale the y-axis manually.
                common_norm=False,
                legend=False,
                edgecolor='none',
            )
        ax_target.set_xlim(0.0, horizon)

        if bin_width > 0:
            ax_samples.set_yticks(ax_samples.get_yticks())
            # Rescale to get the intensity. Difficult to do it differently because of the stacked histograms.
            # Divide by the total count and the bin width.
            ax_samples.set_yticklabels(np.round(ax_samples.get_yticks() / (seqs_cum_times.shape[0] * bin_width), 2))

            ax_target.set_yticks(ax_target.get_yticks())
            # Rescale to get the intensity. Difficult to do it differently because of the stacked histograms.
            # Divide by the total count and the bin width.
            ax_target.set_yticklabels(
                np.round(ax_target.get_yticks() / (seqs_targets_cum_times.shape[0] * bin_width), 2)
            )
        else:
            logger.warning("bin_width is zero (degenerate data), skipping intensity y-axis rescaling.")

    ax_samples.set_title("Model's Sampled Intensity")
    if seqs_targets_cum_times is not None:
        ax_target.set_title("Target Intensity")
    intensity_plot.set_xlabel("Time")
    intensity_plot.set_ylabel("Frequency")
    ax_target.get_figure().tight_layout()
    return two_axes


def plot_hist_lens(
    path_lens: torch.Tensor,
    ax,
    target_path_lens: typing.Optional[torch.Tensor] = None,
    path_lens_hist_outline=None,
    sample_min_intertime: typing.Optional[float] = None,
    target_min_intertime: typing.Optional[float] = None,
):
    # Both can't be given together. One or the other.
    # add types as well etc. why in validation plotter and not generic?

    assert len(path_lens.shape) == 1, f"Expected a 1D tensor, but got {list(path_lens.shape)}."

    if target_path_lens is not None:
        sns.histplot(
            target_path_lens,
            bins=target_path_lens.max() - target_path_lens.min() + 1,
            ax=ax,
            color=sns.color_palette('flare')[0],
            linestyle=(0, (1, 3)),
            stat='density',
            edgecolor='none',
            label='Target',
            discrete=True,
        )

    elif path_lens_hist_outline is not None:
        ax.plot(
            path_lens_hist_outline[0], path_lens_hist_outline[1], color='red', linestyle=(0, (1, 3)), label='Targets'
        )

    hist = sns.histplot(
        path_lens,
        bins=path_lens.max() - path_lens.min() + 1,
        discrete=True,
        label='Sample',
        ax=ax,
        stat='density',
        edgecolor='none',
        color=sns.color_palette('flare')[5],
    )

    hist.set_title("Length Sequences")
    hist.set_xlabel("Length")
    hist.set_ylabel("Probability")
    hist.set_yscale('log')
    if sample_min_intertime is not None:
        ax.plot([], [], linestyle="none", label=f"Sample min intertime: {_format_stat_float(sample_min_intertime)}")
    if target_min_intertime is not None:
        ax.plot([], [], linestyle="none", label=f"Target min intertime: {_format_stat_float(target_min_intertime)}")
    hist.legend()
    return hist


def plot_jitter_effect(
    inputs_before: torch.Tensor,
    inputs_after: torch.Tensor,
    inputs_len_before: torch.Tensor,
    inputs_len_after: torch.Tensor,
    time_max: float,
    n_sequences: int = 10,
) -> plt.Figure:
    """
    Visualize the effect of jittering on TPP sequences.

    Layout: 2x2 grid
    - (0,0) TPP step-functions BEFORE | (0,1) TPP step-functions AFTER
    - (1,0) Inter-arrival distribution BEFORE | (1,1) Inter-arrival distribution AFTER

    Args:
        inputs_before: (N, L+1, 1) cumulative times before jitter.
        inputs_after: (N, L+1, 1) cumulative times after jitter.
        inputs_len_before: (N,) sequence lengths before jitter.
        inputs_len_after: (N,) sequence lengths after jitter.
        time_max: maximum time horizon for step-function plots.
        n_sequences: number of sequences to plot in step-function panels.

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top row: step-function plots
    n = min(n_sequences, inputs_before.shape[0])
    plot_temporal_point_process(
        inputs_before[:n],
        inputs_len_before[:n],
        time_max,
        ax=axes[0, 0],
        max_paths=n,
    )
    axes[0, 0].set_title("Step Functions: Before Jitter")

    plot_temporal_point_process(
        inputs_after[:n],
        inputs_len_after[:n],
        time_max,
        ax=axes[0, 1],
        max_paths=n,
    )
    axes[0, 1].set_title("Step Functions: After Jitter")

    # Bottom row: inter-arrival distributions
    before_np = ensure_numpy_array(inputs_before)
    after_np = ensure_numpy_array(inputs_after)
    len_before_np = ensure_numpy_array(inputs_len_before)
    len_after_np = ensure_numpy_array(inputs_len_after)

    # Collect valid inter-arrivals
    inter_before = []
    for i in range(before_np.shape[0]):
        sl = int(len_before_np[i])
        if sl > 1:
            cum = before_np[i, :sl, 0]
            inter_before.extend((cum[1:] - cum[:-1]).tolist())
    inter_after = []
    for i in range(after_np.shape[0]):
        sl = int(len_after_np[i])
        if sl > 1:
            cum = after_np[i, :sl, 0]
            inter_after.extend((cum[1:] - cum[:-1]).tolist())

    inter_before = np.array(inter_before)
    inter_after = np.array(inter_after)

    # Filter out NaN/inf for histograms
    inter_before = inter_before[np.isfinite(inter_before)]
    inter_after = inter_after[np.isfinite(inter_after)]

    for arr, ax, title in [
        (inter_before, axes[1, 0], "Inter-arrival Distribution: Before"),
        (inter_after, axes[1, 1], "Inter-arrival Distribution: After"),
    ]:
        if arr.size > 0:
            # Log-scale histogram to see near-zero values
            log_arr = np.log10(np.clip(arr, 1e-10, None))
            ax.hist(log_arr, bins=80, edgecolor='none', alpha=0.8, color='steelblue')
            ax.set_xlabel("log10(inter-arrival time)")
            ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.4)

    n_zeros_before = int((inter_before < 1e-6).sum()) if inter_before.size > 0 else 0
    n_zeros_after = int((inter_after < 1e-6).sum()) if inter_after.size > 0 else 0
    fig.suptitle(
        f"Jitter Effect: zeros before: {n_zeros_before}, after: {n_zeros_after}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_temporal_point_process(
    target_seqs: np.typing.ArrayLike,
    target_lens: np.typing.ArrayLike,
    time_max: float,
    comparison_seqs_list: typing.Optional[typing.List[np.typing.ArrayLike]] = None,
    comparison_lens_list: typing.Optional[typing.List[np.typing.ArrayLike]] = None,
    comparison_names: typing.Optional[typing.List[str]] = None,
    ax: typing.Optional[plt.Axes] = None,
    max_paths: int = 100,
) -> plt.Axes:
    """
    Plots samples temporal point process sequences as step functions (counts vs cumulative time).

    Parameters:
    -----------
    target_seqs: np.ndarray
        Target sequences of shape (N, L, D) where N is number of sequences, L is length, D is dimension.
        Should contain cumulative times (not interarrival times).
    target_lens: np.ndarray
        Actual lengths of target sequences, shape (N,).
    time_max: float
        Maximum time horizon. If provided, trajectories extend horizontally to time_max.
    comparison_seqs_list: List[np.ndarray], optional
        List of comparison datasets, each of shape (N, L, D). If provided, target is plotted in green
        and comparisons are overlaid with different colors.
    comparison_lens_list: List[np.ndarray], optional
        List of actual lengths for comparison sequences.
    ax: plt.Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.
    max_paths: int, default=100
        Maximum number of sequences to plot.

    Returns:
    --------
    plt.Axes
        The axes object with the plot.
    """
    # Ensure inputs are numpy arrays
    target_seqs = ensure_numpy_array(target_seqs)
    target_lens = ensure_numpy_array(target_lens)

    if ax is None:
        fig, ax = plt.subplots()

    # Determine number of sequences to plot
    n_seqs = min(target_seqs.shape[0], max_paths)

    # Define color schemes
    has_comparisons = comparison_seqs_list is not None and len(comparison_seqs_list) > 0

    if has_comparisons:
        # Target in green
        target_color = 'green'
        target_alpha = 0.9
        # Colors for comparison datasets - use bright palette for stronger colors
        comparison_colors = sns.color_palette('bright', len(comparison_seqs_list))
        # Progressive alphas for comparisons: 0.6 down to 0.3
        n_comps = len(comparison_seqs_list)
        if n_comps == 1:
            comparison_alphas = [0.7]
        else:
            comparison_alphas = [round(0.7 - i * (0.7 - 0.3) / (n_comps - 1), 2) for i in range(n_comps)]
    else:
        # Multiple colors per sequence if no comparisons - use bright palette
        bright_palette = sns.color_palette('bright')  # 9 colors
        target_colors = bright_palette * (n_seqs // len(bright_palette) + 1)
        target_colors = target_colors[:n_seqs]
        target_alpha = 0.8

    # Plot target sequences
    for idx in range(n_seqs):
        cumulative_times = target_seqs[idx, :, 0]
        length = int(target_lens[idx])

        # Remove NaNs and extract valid cumulative times
        valid_mask = ~np.isnan(cumulative_times)
        if not valid_mask.any() or length == 0:
            continue

        valid_cum_times = cumulative_times[valid_mask][:length]

        # Prepend (time=0, count=0) so the step function includes the vertical
        # jump from 0 to 1 at the first event time, rather than starting abruptly at count=1.
        steps = np.arange(0, len(valid_cum_times) + 1)
        valid_cum_times = np.concatenate([[0], valid_cum_times])

        # Extend to time_max with a horizontal line at the final count
        valid_cum_times = np.concatenate([valid_cum_times, [time_max]])
        steps = np.concatenate([steps, [steps[-1]]])

        if has_comparisons:
            ax.step(valid_cum_times, steps, where="post", alpha=target_alpha, color=target_color, linewidth=1)
        else:
            # Cycle through colors if we have more sequences than colors
            color_idx = idx % len(target_colors)
            ax.step(
                valid_cum_times, steps, where="post", alpha=target_alpha, color=target_colors[color_idx], linewidth=1
            )

    # Plot comparison datasets if provided
    if has_comparisons:
        for comp_idx, (comp_seqs, comp_lens) in enumerate(zip(comparison_seqs_list, comparison_lens_list)):
            comp_seqs = ensure_numpy_array(comp_seqs)
            comp_lens = ensure_numpy_array(comp_lens)

            n_comp_seqs = min(comp_seqs.shape[0], max_paths)
            color = comparison_colors[comp_idx]

            for idx in range(n_comp_seqs):
                cumulative_times = comp_seqs[idx, :, 0]
                length = int(comp_lens[idx])

                valid_mask = ~np.isnan(cumulative_times)
                if not valid_mask.any() or length == 0:
                    continue

                valid_cum_times = cumulative_times[valid_mask][:length]

                # Prepend (time=0, count=0): same as target sequences above
                steps = np.arange(0, len(valid_cum_times) + 1)
                valid_cum_times = np.concatenate([[0], valid_cum_times])

                # Extend to time_max with a horizontal line at the final count
                valid_cum_times = np.concatenate([valid_cum_times, [time_max]])
                steps = np.concatenate([steps, [steps[-1]]])

                ax.step(
                    valid_cum_times, steps, where="post", alpha=comparison_alphas[comp_idx], color=color, linewidth=1
                )

    # Finalize the plot
    ax.set_xlabel("Cumulative Time")
    ax.set_ylabel("Event Count")
    ax.set_title("")
    ax.grid(True, alpha=0.4)

    # Create custom legend if using comparisons
    if has_comparisons:
        from matplotlib.lines import Line2D

        legend_elements = [Line2D([0], [0], color=target_color, lw=2, label='Target', alpha=target_alpha)]
        for comp_idx, color in enumerate(comparison_colors):
            label = (
                comparison_names[comp_idx]
                if comparison_names and comp_idx < len(comparison_names)
                else f'Comparison {comp_idx}'
            )
            legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label, alpha=comparison_alphas[comp_idx]))
        ax.legend(handles=legend_elements, loc='upper left')

    ax.get_figure().tight_layout()
    return ax
