"""Shared per-metric formatting tables for paper_extra_experiments reports.

SCALE_EXPONENTS:  multiplicative scale applied before display, expressed as a
                  power of 10. ``-2`` means values are divided by ``10**-2`` =
                  multiplied by 100.
DECIMAL_PLACES:   number of digits after the decimal point in the displayed
                  cell. Falls back to 2 if absent.
"""

from typing import Dict, Optional

import pandas as pd

SCALE_EXPONENTS: Dict[str, int] = {
    "sigW_loword_notstd": -5,
    "hist_it": -2,
    "hist_int": -2,
    "hist_it_flat": -2,
    "hist_int_flat": -2,
    "ED": -3,
    "W1": -2,
    "corr": -2,
    "corr_short": -2,
    "autocorr_it": -2,
    "autocorr_it_short": -2,
    "autocorr": -2,
    "autocorr_short": -2,
    "CRPS": -1,
    "MAE_proper": -1,
    "MSE_proper": -1,
    "MAE": -1,
    "top1_mark_acc": -2,
    "top3_mark_acc": -2,
}

DECIMAL_PLACES: Dict[str, int] = {
    "hist_it": 2,
    "hist_int": 2,
    "hist_it_flat": 2,
    "hist_int_flat": 2,
    "W1": 2,
    "ED": 2,
    "sigW_loword_notstd": 2,
    "corr": 2,
    "corr_short": 2,
    "autocorr_it": 2,
    "autocorr_it_short": 2,
    "autocorr": 2,
    "autocorr_short": 2,
    "CRPS": 2,
    "MAE_proper": 2,
    "MSE_proper": 2,
    "MAE": 2,
    "top1_mark_acc": 1,
    "top3_mark_acc": 1,
}


def scale_value(value: float, exponent: Optional[int]) -> float:
    if exponent is None:
        return value
    return value / (10**exponent)


def format_mean_std_cell(
    mean: float,
    std: float,
    scale_exponent: Optional[int] = None,
    decimal_places: int = 3,
) -> str:
    """Format as ``11.1(3)`` or ``6.05(1.46)`` — mean with std in parentheses.

    Both mean and std are scaled (via :func:`scale_value`) and formatted to the
    same number of decimal places. If ``std`` is NaN or absent the cell shows the
    mean alone; if ``mean`` is NaN the cell is empty.

    Shared by test/paper_experiments/sig_degree_report.py and
    extract_tables_bootstrap.py so the two reports render bootstrap uncertainty
    byte-for-byte identically.
    """
    if pd.isna(mean):
        return ""
    mean_s = scale_value(float(mean), scale_exponent)
    mean_str = f"{mean_s:.{decimal_places}f}"
    if pd.isna(std):
        return mean_str
    std_s = scale_value(float(std), scale_exponent)
    std_str = f"{std_s:.{decimal_places}f}"
    return f"{mean_str}({std_str})"
