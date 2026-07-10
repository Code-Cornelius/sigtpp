import logging
import typing

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from src.metrics.crosscor import autocorr

logger = logging.getLogger(__name__)

SHOW_ANNOT_ON_HEATMAP = True


def heatmap(data: torch.Tensor, ax: plt.Axes, title_plot: str = "", vmax=None):
    """
    Generate a heatmap from a 2D torch tensor.

    Args:
        data (torch.Tensor): A 2D tensor containing the data to be visualized.
        ax (plt.Axes): Matplotlib Axes object to plot the heatmap on.
        title_plot (str): Optional title suffix.
        vmax (float): Maximum value for the color scale. Default is automatic scaling.
    """
    # Ensure data is a 2D tensor
    if data.numel() == 1:
        data = data.view(1, 1)

    assert data.dim() == 2, "The data must be a 2D array."

    # Convert to NumPy for compatibility with seaborn
    data_np = data.cpu().numpy()

    # Check if all values are NaN, skip plotting if true
    if np.isnan(data_np).all():
        logger.warning("Heatmap data is all NaN, skipping plot.")
        return

    sns.heatmap(
        data_np,
        annot=SHOW_ANNOT_ON_HEATMAP and data_np.shape[0] < 15,
        fmt=".5f",
        ax=ax,
        linewidths=0.5,
        cmap="coolwarm",
        vmin=0.0,
        vmax=vmax,
    )
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax.set_title(f"Heatmap" + (" of " + title_plot if title_plot else ""))
    return


def plot_compare_autocorr(
    seqs_1,
    seqs_2,
    ax: plt.Axes,
    max_lag: int,
    errorbar: typing.Optional[str] = None,
    names_seqs_on_plot=('Targets', 'Generated'),
):
    """
    Compute and plot the autocorrelation for two sequences as a lineplot (ACF vs lags).
    References:
        for errorbar, check https://seaborn.pydata.org/tutorial/error_bars.html.
    """
    acf_1 = autocorr(seqs_1, max_lag=max_lag).detach().cpu().numpy().reshape(-1)
    acf_2 = autocorr(seqs_2, max_lag=max_lag).detach().cpu().numpy().reshape(-1)

    acf_1_dict = {
        'Lags': np.arange(len(acf_1)),
        'Autocorrelation': acf_1,
        'Type': np.full(len(acf_1), names_seqs_on_plot[0]),
    }
    acf_2_dict = {
        'Lags': np.arange(len(acf_2)),
        'Autocorrelation': acf_2,
        'Type': np.full(len(acf_2), names_seqs_on_plot[1]),
    }
    data = pd.concat([pd.DataFrame(acf_1_dict), pd.DataFrame(acf_2_dict)])

    sns.lineplot(
        x='Lags',
        y='Autocorrelation',
        hue='Type',
        style='Type',
        style_order=[names_seqs_on_plot[1], names_seqs_on_plot[0]],
        # Remove NAs which can happen when the ACF is not computed properly. Then, it is not shown on the plot.
        data=data.dropna(),
        errorbar=errorbar,
        err_style='band',
        ax=ax,
    )
    lines = ax.get_lines()
    if len(lines) >= 2:
        lines[0].set_linewidth(2.0)
        lines[1].set_linewidth(1.0)


    # Adjusting plot aesthetics
    ax.set_xlabel('Lags')
    return ax
