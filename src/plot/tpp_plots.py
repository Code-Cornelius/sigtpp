import logging.config

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

from src.plot.metric_plots import heatmap, plot_compare_autocorr
from src.plot.tpp_plot_components import (
    plot_one_distribution_against_another,
    plot_hist_intensity,
    qq_plot_multi_seqs_against_targets,
    plot_hist_lens,
    plot_temporal_point_process,
)


def diagnostic_plots_tpp(
    sampled_fake_sample: torch.Tensor,
    real_samples: torch.Tensor,
    sampled_lens: torch.Tensor,
    real_lens: torch.Tensor,
    fig_histograms: Figure,
    ax_intensity: plt.Axes,
    fig_acf: Figure,
    fig_cov_err: Figure,
    corr_loss: torch.Tensor = None,
    ax_temporal_plot: plt.Axes = None,
    time_max: float = None,
):
    """Generate diagnostic plots for temporal point process (TPP) data.

    Creates visualizations comparing generated and real samples, including histogram plots,
    QQ plots, intensity plots, autocorrelation functions, and correlation error heatmaps.

    Args:
        sampled_fake_sample: Generated samples of shape (N, L, D) containing inter-arrival times.
            Variable-length sequences should be padded with NaN values.
        real_samples: Real/target samples of shape (N, L, D) containing inter-arrival times.
            Variable-length sequences should be padded with NaN values.
        sampled_lens: Sequence lengths for generated samples, shape (N,).
        real_lens: Sequence lengths for real samples, shape (N,).
        fig_histograms: Matplotlib figure containing 4 subplots for various histograms
            and QQ plots of inter-arrival times.
        ax_intensity: Matplotlib axes for intensity/cumulative inter-arrival time histogram.
        fig_acf: Matplotlib figure for autocorrelation function plots.
        fig_cov_err: Matplotlib figure for correlation error heatmap.
        corr_loss: Pre-computed correlation loss tensor for heatmap. Defaults to None.
        ax_temporal_plot: Matplotlib axes for temporal point process plot. Defaults to None.
        time_max: Maximum time horizon for temporal plot. Defaults to None.

    Note:
        - Input tensors should contain inter-arrival times, NOT cumulative times.
        - Currently assumes D=1 (single dimension) and slices to first dimension for plotting.
        - Variable-length sequences are handled via NaN padding. Samples should be preprocessed
          with set_seq_to_nan_from_index() before passing to this function.
        - Returns early if all sampled values are NaN or batch is empty.
    """
    assert (
        len(sampled_fake_sample.shape) == len(real_samples.shape) == 3
    ), f"Expected 3D tensors, but got {list(sampled_fake_sample.shape)} and {list(real_samples.shape)}."
    assert (
        len(sampled_lens.shape) == len(real_lens.shape) == 1
    ), f"Expected a 1D tensor, but got {list(sampled_lens.shape)} and {list(real_lens.shape)}."

    assert torch.isfinite(
        sampled_fake_sample[~torch.isnan(sampled_fake_sample)]
    ).all(), f"All data points in {sampled_fake_sample} must be finite"
    assert torch.isfinite(
        real_samples[~torch.isnan(real_samples)]
    ).all(), f"All data points in {real_samples} must be finite"

    assert (
        sampled_fake_sample.shape[1] == real_samples.shape[1]
    ), "The sequences should have the same length but got {} and {}.".format(
        sampled_fake_sample.shape[1], real_samples.shape[1]
    )

    # Check if samples have been properly "naned" for variable-length sequence handling.
    # Only warn if some sequences are shorter than the tensor (i.e. variable-length sequences exist
    # that should carry NaN padding). When all sequences span the full tensor no padding is needed.
    if (sampled_lens < sampled_fake_sample.shape[1]).any() and not torch.isnan(sampled_fake_sample).any():
        logger.warning(
            "Generated samples do not contain any NaN values. "
            "For variable-length sequences, samples should be padded with NaN using set_seq_to_nan_from_index(). "
            "This may indicate improper preprocessing."
        )

    if (real_lens < real_samples.shape[1]).any() and not torch.isnan(real_samples).any():
        logger.warning(
            "Real samples do not contain any NaN values. "
            "For variable-length sequences, samples should be padded with NaN using set_seq_to_nan_from_index(). "
            "This may indicate improper preprocessing."
        )

    # The plotting does not work if all values are nans.
    if torch.all(torch.isnan(sampled_fake_sample)) or sampled_fake_sample.shape[0] == 0:
        return

    # Correlation/ACF metrics follow the TPPMetrics convention and exclude tau_1.
    sampled_fake_sample_metric = sampled_fake_sample[:, 1:, :]
    real_samples_metric = real_samples[:, 1:, :]

    # Align the plot lag with the metric convention, but derive it from the plotted tensors instead of requiring a metric object to be passed in.
    acf_max_lag = min(
        sampled_fake_sample_metric.shape[1] // 2 - 1,
        real_samples_metric.shape[1] // 2 - 1,
        50,
    )

    # --- Length histogram ---
    try:
        # We add one: lens counts ITs including τ₁; adding 1 recovers total time points including the t0 anchor.
        plot_hist_lens(
            sampled_lens.detach().cpu().numpy() + 1,
            fig_histograms.axes[0],
            real_lens.detach().cpu().numpy() + 1,
        )
    except Exception as e:
        logger.error(f"Length histogram plot failed: {e}")

    # --- IT distribution ---
    try:
        plot_one_distribution_against_another(
            sampled_fake_sample[:, :, 0].detach().cpu().numpy(),
            real_samples[:, :, 0].detach().cpu().numpy(),
            fig_histograms.axes[3],
            True,
            True,
            title="Model's Intertimes per Time Step",
        )
        plot_one_distribution_against_another(
            real_samples[:, :, 0].detach().cpu().numpy(),
            None,
            fig_histograms.axes[2],
            True,
            True,
            title='Targets Intertimes per Time Step',
        )
    except Exception as e:
        logger.error(f"IT distribution plot failed: {e}")

    # --- QQ plot ---
    try:
        qq_plot_multi_seqs_against_targets(
            sampled_fake_sample[:, :, 0].detach().cpu().numpy(),
            real_samples[:, :, 0].detach().cpu().numpy(),
            fig_histograms.axes[1],
            True,
        )
    except Exception as e:
        logger.error(f"QQ plot failed: {e}")

    # --- Intensity histogram ---
    try:
        plot_hist_intensity(
            sampled_fake_sample[:, :, 0].cumsum(1).detach().cpu().numpy(),
            25,
            ax_intensity,
            real_samples[:, :, 0].cumsum(1).detach().cpu().numpy(),
            time_horizon=time_max,
        )
    except Exception as e:
        logger.error(f"Intensity histogram plot failed: {e}")

    # --- Correlation error heatmap ---
    try:
        if corr_loss is not None:
            heatmap(corr_loss, fig_cov_err.axes[0], "Correlation error", vmax=0.3)
    except Exception as e:
        logger.error(f"Correlation heatmap plot failed: {e}")

    # --- ACF ---
    try:
        plot_compare_autocorr(
            real_samples_metric,
            sampled_fake_sample_metric,
            ax=fig_acf.axes[0],
            max_lag=acf_max_lag,
            names_seqs_on_plot=('I.T. Targets', 'I.T. Generated'),
        )
        plot_compare_autocorr(
            torch.cumsum(real_samples_metric, dim=1),
            torch.cumsum(sampled_fake_sample_metric, dim=1),
            ax=fig_acf.axes[1],
            max_lag=acf_max_lag,
            names_seqs_on_plot=('Cum. Time Targets', 'Cum. Time Generated'),
        )
        fig_acf.tight_layout()
    except Exception as e:
        logger.error(f"ACF plot failed: {e}")

    # --- Temporal point process ---
    if ax_temporal_plot is not None and time_max is not None:
        try:
            # The stored sequences contain inter-arrival times starting from t1:
            # - t0 is the origin (always 0) and is excluded.
            # - cumsum therefore gives cumulative times relative to t0 = 0.
            sampled_cumulative = sampled_fake_sample.cumsum(dim=1)
            real_cumulative = real_samples.cumsum(dim=1)

            plot_temporal_point_process(
                target_seqs=real_cumulative.detach().cpu().numpy(),
                target_lens=real_lens.detach().cpu().numpy(),
                time_max=time_max,
                comparison_seqs_list=[sampled_cumulative.detach().cpu().numpy()],
                comparison_lens_list=[sampled_lens.detach().cpu().numpy()],
                comparison_names=["Predictive Model"],
                ax=ax_temporal_plot,
                max_paths=100,
            )
            ax_temporal_plot.set_title("Samples of the Distributions $N_t$")
        except Exception as e:
            logger.error(f"Temporal point process plot failed: {e}")

    return
