"""Plot expected counting path and its linear embedding for a 1D Poisson process.

Draws two figures (saved as PDFs under ``test/paper_extra_experiments/out/figures``):

- ``poisson_sig_embedding.pdf``    : N step paths + their mean E[eta_t]
- ``embedding_model_sample.pdf``   : the linear embedding Phi(eta)_t (continuous)
                                     + its mean

Each figure shows N_SAMPLES Monte-Carlo trajectories in light colour and the
empirical mean in a darker shade.
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

apply_paper_style(use_stix=True, boldmath=True)

_FIG_DIR = os.path.join(ROOT_DIR, "test/paper_extra_experiments/out/figures")
os.makedirs(_FIG_DIR, exist_ok=True)

LAMBDA_RATE = 1.0
T = 7.0
N_SAMPLES = 20
RNG = np.random.default_rng(42)
T_VALS = np.linspace(0, T, 4000)
PALETTE = sns.color_palette("dark")

AXIS_LABEL_FONTSIZE = 28
LEGEND_FONTSIZE = 26
TICK_FONTSIZE = 22


def simulate_poisson(rng: np.random.Generator, lam: float, tmax: float) -> np.ndarray:
    """Simulate a homogeneous Poisson process on [0, tmax]; returns arrival times incl. anchor 0.

    Uses a one-draw-per-iteration loop (rather than a bulk size= draw) so that the
    number of RNG samples consumed per call matches the original script. Switching
    to a bulk draw would change the generator state between samples and produce a
    visually different (but statistically equivalent) figure for the same seed.
    """
    times = [0.0]
    while True:
        t_next = times[-1] + rng.exponential(1.0 / lam)
        if t_next > tmax:
            break
        times.append(t_next)
    return np.array(times)


def counting_process(arrival_times: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(arrival_times[1:], t_grid, side="right").astype(float)


def linear_embedding(arrival_times: np.ndarray, tmax: float, t_eval: np.ndarray) -> np.ndarray:
    tau = np.diff(arrival_times)
    tau_with_0 = np.concatenate(([0.0], tau))
    tau_final = tmax - arrival_times[-1]
    t_grid = np.concatenate((arrival_times, [tmax]))
    tau_grid = np.concatenate((tau_with_0, [tau_final]))
    return np.interp(t_eval, t_grid, tau_grid)


def place_axis_labels(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, rotation=0)
    ax.xaxis.set_label_coords(0.98, 0.03, transform=ax.transAxes)
    ax.yaxis.set_label_coords(0.03, 0.97, transform=ax.transAxes)
    ax.xaxis.label.set_horizontalalignment("right")
    ax.xaxis.label.set_verticalalignment("bottom")
    ax.yaxis.label.set_horizontalalignment("left")
    ax.yaxis.label.set_verticalalignment("top")


def main() -> None:
    samples = [simulate_poisson(RNG, LAMBDA_RATE, T) for _ in range(N_SAMPLES)]
    eta_curves = np.stack([counting_process(s, T_VALS) for s in samples])
    embedding_curves = np.stack([linear_embedding(s, T, T_VALS) for s in samples])
    mean_eta = eta_curves.mean(axis=0)
    mean_embedding = embedding_curves.mean(axis=0)

    fig, ax = plt.subplots()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(N_SAMPLES):
        ax.step(T_VALS, eta_curves[i], where="post", color=PALETTE[0], alpha=0.35, linewidth=1.4)
    ax.step(T_VALS, mean_eta, where="post", color=PALETTE[0], linewidth=3.5, label=r"$\mathbb{E}[\eta_t]$")
    place_axis_labels(ax, r"Time $t$", r"$\eta_t$")
    ax.legend(loc="upper center", fontsize=LEGEND_FONTSIZE)
    ax.set_xticks(np.arange(0, T + 1, 1))
    ax.tick_params(labelsize=TICK_FONTSIZE, labelbottom=False, labelleft=False)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    fig.tight_layout()
    fig.savefig(os.path.join(_FIG_DIR, "poisson_sig_embedding.pdf"))

    fig2, ax2 = plt.subplots()
    for i in range(N_SAMPLES):
        ax2.plot(T_VALS, embedding_curves[i], color=PALETTE[2], alpha=0.45, linewidth=1.4)
    ax2.plot(T_VALS, mean_embedding, color=PALETTE[2], linewidth=3.5, label=r"$\mathbb{E}[\Phi(\eta)_t]$")
    place_axis_labels(ax2, r"Time $t$", r"$\Phi(\eta)_t$")
    ax2.legend(loc="upper center", fontsize=LEGEND_FONTSIZE)
    ax2.set_xticks(np.arange(0, T + 1, 1))
    ax2.tick_params(labelsize=TICK_FONTSIZE, labelbottom=False, labelleft=False)
    ax2.set_axisbelow(True)
    ax2.grid(True, which="major", alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
    fig2.tight_layout()
    fig2.savefig(os.path.join(_FIG_DIR, "embedding_model_sample.pdf"))

    plt.show()


if __name__ == "__main__":
    main()
