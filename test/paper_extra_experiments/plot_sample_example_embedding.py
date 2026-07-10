"""Plot a single Poisson trajectory together with its linear embedding Phi(eta).

Output: ``embedding_data_sample.pdf`` under ``test/paper_extra_experiments/out/figures``.

Shows three overlaid series on one axis:
  - the counting path t -> eta_t (step plot)
  - the continuous embedding Phi(eta)_t (dashed line)
  - the grid points (t_k, tau_k) that define Phi (markers)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from config import ROOT_DIR
from test.paper_extra_experiments._plot_style import (
    GRID_ALPHA,
    GRID_LINEWIDTH,
    apply_paper_style,
)

apply_paper_style(use_stix=True)

_FIG_DIR = os.path.join(ROOT_DIR, "test/paper_extra_experiments/out/figures")
os.makedirs(_FIG_DIR, exist_ok=True)

AXIS_LABEL_FONTSIZE = 22
LEGEND_FONTSIZE = 16
TITLE_FONTSIZE = 20
TICK_FONTSIZE = 14

LAMBDA_RATE = 1.0
T = 7.0


def main() -> None:
    # Use the legacy global-state API so the exact figure remains reproducible
    # (default_rng draws different samples for the same seed).
    np.random.seed(42)
    interarrival_times = np.random.exponential(1.0 / LAMBDA_RATE, size=25)
    arrival_times = np.concatenate(([0.0], np.cumsum(interarrival_times)))
    arrival_times = arrival_times[arrival_times <= T]

    t_vals = np.linspace(0, T, 8000)
    eta_t = np.searchsorted(arrival_times[1:], t_vals, side="right")

    palette = sns.color_palette("dark")
    tau = np.diff(arrival_times)
    tau_with_0 = np.concatenate(([0.0], tau))
    tau_final = T - arrival_times[-1]
    t_grid = np.concatenate((arrival_times, [T]))
    tau_grid = np.concatenate((tau_with_0, [tau_final]))
    phi = np.interp(t_vals, t_grid, tau_grid)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.step(t_vals, eta_t, where="post", color=palette[0], linewidth=2.2, label=r"Counting path: $t \mapsto \eta_t$")
    ax.plot(t_vals, phi, color=palette[2], linewidth=2.2, linestyle="--", label=r"Embedding $\Phi(\eta)_t$")
    ax.scatter(t_grid, tau_grid, marker="o", s=40, color=palette[1], zorder=5, label=r"Grid points $(t_k,\,\tau_k)$")

    ax.set_xticks(np.arange(0, int(T) + 1, 1))
    ax.set_xlabel(r"Time $t$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(r"Counting path $\eta_t$ and embedding $\Phi$", fontsize=TITLE_FONTSIZE)
    ax.legend(loc="upper left", fontsize=LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

    fig.tight_layout()
    fig.savefig(os.path.join(_FIG_DIR, "embedding_data_sample.pdf"))

    plt.show()


if __name__ == "__main__":
    main()
