"""Shared matplotlib + seaborn style for paper_extra_experiments plots.

Call ``apply_paper_style()`` once at module level to set rcParams and font sizes
consistently across plot scripts. Constants are re-exported for direct use.
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 17
TITLE_FONTSIZE = 20
GRID_ALPHA = 0.4
GRID_LINEWIDTH = 0.99

# Comparison-plot fonts (used by plot_model_comparison).
PANEL_TITLE_FONTSIZE = 18
COMP_AXIS_LABEL_FONTSIZE = 18
COMP_TICK_LABEL_FONTSIZE = 17
COMP_LEGEND_FONTSIZE = 14


def apply_paper_style(*, use_stix: bool = False, boldmath: bool = False) -> None:
    """Apply the shared style. Idempotent.

    use_stix:   set STIX fonts for the math/text family (used by embedding plots).
    boldmath:   add ``\\boldmath`` to the LaTeX preamble.
    """
    sns.set()
    if use_stix:
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["text.usetex"] = True
    preamble = r"\usepackage{amsfonts}\usepackage{amsmath}"
    if boldmath:
        preamble += r"\boldmath"
    plt.rcParams["text.latex.preamble"] = preamble
    plt.rcParams["font.weight"] = "bold"
