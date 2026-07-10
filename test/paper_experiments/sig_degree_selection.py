"""Validation-file parsing and per-sig-degree winner selection.

Pure-text, torch-free helpers used by the in-pipeline sig-degree ablation
(``sig_degree_ablation.py``, driven by ``TrainingManager``) and its unit tests.
Selection reads only the validation tuning file and never touches the test split
(data-leakage guard): each degree's winner is the row with the lowest
``val_norm_score``.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Column in the validation tuning file used to pick each degree's winner. Same
# criterion as the main pipeline's winner evaluation path.
_SELECTION_COLUMN = "val_norm_score"

# Degree tokens from the run-name grammar (get_dir_name_from_params abbreviates
# keys to 4 chars): absolute mode `sig_degree` -> `_sig_<d>`, relative mode
# `relative_sig_degree` -> `_rela<offset>` (offsets may be negative). Both are
# anchored on token boundaries so unrelated `_sig*` substrings never match.
_ABS_SIG_PATTERN = re.compile(r"_sig_(\d+)(?=_|$)")
_REL_SIG_PATTERN = re.compile(r"_rela(-?\d+)(?=_|$)")


def _extract_sig_token(run_name: str) -> Optional[Tuple[str, int]]:
    """Return ``("absolute" | "relative", degree)`` or ``None`` if no token."""
    abs_match = _ABS_SIG_PATTERN.search(run_name)
    if abs_match:
        return "absolute", int(abs_match.group(1))
    rel_match = _REL_SIG_PATTERN.search(run_name)
    if rel_match:
        return "relative", int(rel_match.group(1))
    return None


def _extract_sig_degree(run_name: str) -> Optional[int]:
    token = _extract_sig_token(run_name)
    return None if token is None else token[1]


def parse_results(path: Path) -> pd.DataFrame:
    """Read a whitespace-separated results file into a DataFrame.

    Handles both the validation tuning schema and bootstrap-enriched schemas.
    A trailing ``ERROR`` column, present only when some rows failed, is parsed as
    a single possibly multi-word message so metric columns stay aligned and
    failed rows are kept with NaN metrics rather than dropped.
    """
    with Path(path).open(encoding="utf-8") as handle:
        lines = handle.readlines()

    if not lines:
        raise ValueError(f"Empty results file: {path}")

    header = lines[0].split()
    if not header or header[0].upper() != "MODEL":
        got = header[0] if header else "<empty line>"
        raise ValueError(f"Expected first token of header to be MODEL, got {got!r}")

    col_names = ["run_name"] + header[1:]
    has_error_col = header[-1] == "ERROR"
    n_value_cols = len(col_names) - 1 - (1 if has_error_col else 0)

    rows: List[List[str]] = []
    for lineno, line in enumerate(lines[1:], start=2):
        parts = line.strip().split()
        if not parts:
            continue
        run_name, value_tokens = parts[0], parts[1:]

        if has_error_col:
            if len(value_tokens) < n_value_cols:
                logger.warning(
                    "line %d: expected >= %d value tokens, got %d - skipping",
                    lineno,
                    n_value_cols,
                    len(value_tokens),
                )
                continue
            values = value_tokens[:n_value_cols]
            error_str = " ".join(value_tokens[n_value_cols:])
            rows.append([run_name] + values + [error_str])
        else:
            if len(value_tokens) != n_value_cols:
                logger.warning(
                    "line %d: expected %d value tokens, got %d - skipping",
                    lineno,
                    n_value_cols,
                    len(value_tokens),
                )
                continue
            rows.append([run_name] + value_tokens)

    df = pd.DataFrame(rows, columns=col_names)
    for col in col_names[1:]:
        if col == "ERROR":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _selection_candidates(val_results_path: Path) -> pd.DataFrame:
    """Return rankable validation rows with an integer ``sig_degree`` column."""
    df = parse_results(Path(val_results_path))
    if _SELECTION_COLUMN not in df.columns:
        raise ValueError(
            f"Validation tuning file is missing the {_SELECTION_COLUMN!r} column; got columns {list(df.columns)}"
        )

    df = df.copy()
    tokens = df["run_name"].map(_extract_sig_token)
    df["sig_degree"] = tokens.map(lambda token: None if token is None else token[1])
    modes = {token[0] for token in tokens if token is not None}
    if len(modes) > 1:
        logger.warning(
            "mixed absolute and relative sig-degree tokens in %s - degrees from " "different modes are not comparable",
            val_results_path,
        )

    missing = int(df["sig_degree"].isna().sum())
    if missing:
        logger.warning("%d row(s) had no sig_degree in run name - excluded", missing)
        df = df.dropna(subset=["sig_degree"])
    df["sig_degree"] = df["sig_degree"].astype(int)

    if "ERROR" in df.columns:
        failed = df["ERROR"].fillna("").str.strip() != ""
        if failed.any():
            logger.warning("%d failed row(s) excluded from winner selection", int(failed.sum()))
            df = df[~failed]

    finite_score = df[_SELECTION_COLUMN].map(lambda value: pd.notna(value) and np.isfinite(float(value)))
    invalid_scores = int((~finite_score).sum())
    if invalid_scores:
        logger.warning(
            "%d row(s) had non-finite %s - excluded from winner selection",
            invalid_scores,
            _SELECTION_COLUMN,
        )
        df = df[finite_score]

    return df


def select_winner_rows_by_sig_degree(val_results_path: Path) -> Dict[int, pd.Series]:
    """Return ``{sig_degree: row}``, winner = min validation ``val_norm_score``."""
    df = _selection_candidates(Path(val_results_path))
    winners: Dict[int, pd.Series] = {}
    for degree, group in df.groupby("sig_degree", sort=True):
        best_idx = group[_SELECTION_COLUMN].idxmin()
        winners[int(degree)] = group.loc[best_idx]
    return winners


def winner_names_from_rows(winner_rows: Dict[int, pd.Series]) -> Dict[int, str]:
    """Return ``{sig_degree: winner_run_name}`` from selected winner rows."""
    return {degree: str(row["run_name"]) for degree, row in winner_rows.items()}


def select_winners_by_sig_degree(val_results_path: Path) -> Dict[int, str]:
    """Return ``{sig_degree: winner_run_name}``, winner = min ``val_norm_score``.

    Selection reads only the validation tuning file. Rows whose run name has no
    degree token (absolute ``_sig_<d>`` or relative ``_rela<offset>``), rows
    marked failed, and rows with non-finite selection scores are excluded with
    warnings.
    """
    return winner_names_from_rows(select_winner_rows_by_sig_degree(Path(val_results_path)))


def find_latest_val_tuning_file(results_dir: Path, version: str) -> Path:
    """Find the newest timestamped ``<version>_val_tuning_<ts>.txt`` file.

    Manual/ad-hoc use only (e.g. an interactive re-run when the exact file path
    isn't known). Do not call this from an automated pipeline step that means
    "the file I just wrote": if another run shares ``results_dir`` and
    ``version`` and writes a newer file first, this silently returns the wrong
    run's file. ``TrainingManager`` avoids this by threading the exact path
    returned from ``ExperimentResults.save()`` straight into the ablation
    instead of calling this function.
    """
    candidates = sorted(Path(results_dir).glob(f"{version}_val_tuning_*.txt"), key=lambda path: path.name)
    if not candidates:
        raise FileNotFoundError(f"No validation tuning file matching {version}_val_tuning_*.txt found in {results_dir}")
    return candidates[-1]
