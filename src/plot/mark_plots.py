"""
Mark diagnostic plots for temporal point process models.

This module provides standalone plot functions for mark diagnostics.
All functions accept pre-computed logit/mark tensors and matplotlib
figure/axes objects, following the same convention as tpp_plots.py.

Two diagnostic figures are produced for any mark payload:
  - Marginal class fit     : grouped bar chart of true vs predicted class frequencies
  - Conditional structure  : two-panel heatmap P(current | previous), empirical vs predicted
"""

import logging
import typing

import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from src.nn.architectures.mark_prediction_utils import MARK_IGNORE_INDEX

logger = logging.getLogger(__name__)

sns.set_theme()


# ---------------------------------------------------------------------------
# Data-preparation helpers
# ---------------------------------------------------------------------------


def mask_mark_sequences(
    previous_marks: torch.Tensor,
    current_targets: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Mask padding positions in (N, L) mark tensors.

    Positions at or beyond each sequence's valid length are replaced with
    ``ignore_index`` so they are excluded from downstream metric and plot
    computations.

    Args:
        previous_marks:  (N, L) int64 - mark at position t-1.
        current_targets: (N, L) int64 - mark at position t (prediction target).
        valid_lengths:   (N,)   long  - number of valid steps per sequence.

    Returns:
        Masked copies of (previous_marks, current_targets).
    """
    # Build a (1, L) position index and broadcast against (N, 1) lengths.
    pos = torch.arange(current_targets.shape[1], device=valid_lengths.device).unsqueeze(0)
    valid_mask = pos < valid_lengths.unsqueeze(1)  # (N, L) bool
    previous_marks = previous_marks.clone()
    current_targets = current_targets.clone()
    previous_marks[~valid_mask] = MARK_IGNORE_INDEX
    current_targets[~valid_mask] = MARK_IGNORE_INDEX
    return previous_marks, current_targets


# ---------------------------------------------------------------------------
# Main diagnostic plotting entry point
# ---------------------------------------------------------------------------


def diagnostic_plots_marks(
    mark_logits: torch.Tensor,
    previous_marks: torch.Tensor,
    current_targets: torch.Tensor,
    fig_marginal: plt.Figure,
    ax_marginal: plt.Axes,
    fig_conditional: plt.Figure,
    axes_conditional: typing.Sequence[plt.Axes],
) -> None:
    """Compute statistics and draw the canonical mark diagnostic plots.

    The caller is responsible for creating and clearing the figures/axes
    before calling this function, and for saving them afterwards.

    The function is a no-op when the valid mask contains no positions,
    which avoids crashes for degenerate validation batches.

    Args:
        mark_logits:      (N, L, K) float  - raw class logits.
        previous_marks:   (N, L)    int64  - mark at t-1, padded with -1.
        current_targets:  (N, L)    int64  - mark at t,   padded with -1.
        fig_marginal:     Figure for the marginal class-fit bar chart.
        ax_marginal:      Axes inside fig_marginal.
        fig_conditional:  Figure for the conditional-structure two-panel heatmap.
        axes_conditional: Sequence of two axes [ax_empirical, ax_predicted].

    Plots produced:
        - Marginal   : grouped bar chart of empirical vs predicted class frequencies.
        - Conditional: P(current | previous), empirical (left) and predicted (right).
    """
    # Guard: nothing to draw if every position is masked out.
    valid_mask = current_targets != MARK_IGNORE_INDEX  # (N, L) bool
    if not valid_mask.any():
        return

    # K = number of mark classes.
    num_marks = mark_logits.shape[-1]

    # Convert logits to probabilities; select only valid (unmasked) positions.
    probs = torch.softmax(mark_logits, dim=-1)  # (N, L, K)
    valid_probs = probs[valid_mask]  # (M, K)
    valid_targets = current_targets[valid_mask]  # (M,)
    valid_previous_marks = previous_marks[valid_mask]  # (M,)

    # ------------------------------------------------------------------
    # 1. Marginal class distribution
    #    Empirical  : fraction of true labels belonging to each class
    #    Predicted  : mean predicted probability mass per class
    # ------------------------------------------------------------------
    empirical_marginal = torch.bincount(valid_targets, minlength=num_marks).float()
    empirical_marginal = empirical_marginal / empirical_marginal.sum().clamp(min=1.0)
    predicted_marginal = valid_probs.mean(dim=0)  # (K,) - mean over M valid positions

    bright = sns.color_palette('bright')
    true_color = bright[2]
    pred_color = bright[0]

    mark_indices = torch.arange(num_marks).cpu().numpy()
    width = 0.38
    ax_marginal.bar(
        mark_indices - width / 2,
        empirical_marginal.detach().cpu().numpy(),
        width=width,
        color=true_color,
        label='True',
    )
    ax_marginal.bar(
        mark_indices + width / 2,
        predicted_marginal.detach().cpu().numpy(),
        width=width,
        color=pred_color,
        label='Predicted',
    )
    # Uniform baseline: the single most informative reference for reviewers.
    ax_marginal.axhline(
        1.0 / num_marks,
        color='red',
        linestyle=(0, (1, 3)),
        linewidth=1.0,
        label='Uniform (1/K)',
    )
    ax_marginal.set_title('Mark Marginal Class Fit')
    ax_marginal.set_xlabel('Mark class')
    ax_marginal.set_ylabel('Probability')
    ax_marginal.set_xticks(mark_indices)
    ax_marginal.legend()
    fig_marginal.tight_layout()

    # ------------------------------------------------------------------
    # 2. Conditional structure: P(current | previous)
    #    Left panel  : empirical frequencies from the validation set
    #    Right panel : mean predicted probabilities from the model
    #    Both panels share the same colour scale for a fair visual comparison.
    # ------------------------------------------------------------------
    empirical_conditional_counts = torch.zeros(num_marks, num_marks, device=mark_logits.device)
    empirical_conditional_counts.index_put_(
        (valid_previous_marks, valid_targets),
        torch.ones_like(valid_targets, dtype=torch.float),
        accumulate=True,
    )
    # How often each previous-mark value appears (denominator for row normalisation).
    previous_mark_counts = torch.bincount(valid_previous_marks, minlength=num_marks).float().unsqueeze(1)
    # Row r = P(current | previous = r).
    empirical_conditional = (empirical_conditional_counts / previous_mark_counts.clamp(min=1.0)).detach().cpu().numpy()

    # Sum model probabilities grouped by previous mark, then row-normalise.
    predicted_conditional_sums = torch.zeros(num_marks, num_marks, device=mark_logits.device)
    predicted_conditional_sums.index_add_(0, valid_previous_marks, valid_probs)
    predicted_conditional = (predicted_conditional_sums / previous_mark_counts.clamp(min=1.0)).detach().cpu().numpy()

    # Shared colour scale: take the max across both matrices (floor at 1e-8 to avoid zero vmax).
    vmax_conditional = float(max(np.nanmax(empirical_conditional), np.nanmax(predicted_conditional), 1e-8))

    annotate_cond = num_marks < 15
    sns.heatmap(
        empirical_conditional,
        annot=annotate_cond,
        fmt='.2f',
        cmap='coolwarm',
        vmin=0.0,
        vmax=vmax_conditional,
        linewidths=0.3,
        ax=axes_conditional[0],
        cbar=False,
    )
    axes_conditional[0].set_title('Empirical P(current | previous)')
    axes_conditional[0].set_xlabel('Current mark')
    axes_conditional[0].set_ylabel('Previous mark')

    sns.heatmap(
        predicted_conditional,
        annot=annotate_cond,
        fmt='.2f',
        cmap='coolwarm',
        vmin=0.0,
        vmax=vmax_conditional,
        linewidths=0.3,
        ax=axes_conditional[1],
        cbar=False,
    )
    axes_conditional[1].set_title('Predicted P(current | previous)')
    axes_conditional[1].set_xlabel('Current mark')
    # Remove duplicate y-axis tick labels from the right panel (sharey=True already links them).
    axes_conditional[1].tick_params(labelleft=False)

    # Adaptive annotation color for both conditional panels.
    if annotate_cond:
        threshold = 0.5 * vmax_conditional
        for ax in axes_conditional:
            for text in ax.texts:
                val = float(text.get_text())
                text.set_color('white' if val > threshold else 'black')

    # Single shared colorbar anchored to the right of the figure.
    # Do NOT call tight_layout() after this: fig.colorbar(ax=list) already
    # repositions the panels to make room, and a subsequent tight_layout()
    # recalculates layout without accounting for the colorbar reservation,
    # pushing the colorbar back on top of the panels.
    sm = mpl.cm.ScalarMappable(
        cmap='coolwarm',
        norm=mpl.colors.Normalize(vmin=0.0, vmax=vmax_conditional),
    )
    sm.set_array([])
    fig_conditional.colorbar(sm, ax=axes_conditional, fraction=0.046, pad=0.04)
