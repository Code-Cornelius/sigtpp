"""Typed container for experiment results with txt output.

Bootstrap-aware schema
----------------------
Bootstrap metrics are stored under two keys: ``<name>_mean`` and
``<name>_std``. Metrics that are computed once, outside the bootstrap loop, are
stored as plain ``<name>`` columns. Models that fail to evaluate keep the row
but with all metric fields set to NaN; the optional ``error`` key explains why.
Ranking uses bootstrap means only.
"""

import copy
import logging
import os
import typing
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)
from src.utils.result_helpers import apply_competition_ranking, write_result_rows_fixed_width


class ExperimentResults:
    """Holds results from a training run and provides save/normalize operations."""

    # Display order for metric base names. The saver expands each name into
    # ``<name>_mean`` and ``<name>_std`` columns.
    DISPLAY_METRICS = [
        "sigW_loword_notstd",
        "hist_it",
        "hist_int",
        "hist_it_flat",
        "hist_int_flat",
        "ED",
        "W1",
        "CRPS",
        "corr",
        "corr_short",
        "autocorr_it",
        "autocorr_it_short",
        "autocorr",
        "autocorr_short",
        "MAE_proper",
        "MSE_proper",
        "MAE",
        "mark_ce",
        "top1_mark_acc",
        "top3_mark_acc",
        "train_time",
    ]

    # Extra metrics: saved to file but never ranked or included in norm_score.
    # Controlled by TPPMetricsConfig.save_extra_metrics; absent keys write as nan.
    EXTRA_METRICS: typing.ClassVar[typing.List[str]] = []

    # Metrics computed once outside the bootstrap loop. They are saved as plain
    # scalar columns, not as ``*_mean`` / ``*_std`` pairs.
    NON_BOOTSTRAP_METRICS: typing.ClassVar[typing.FrozenSet[str]] = frozenset(
        {"train_time", "mark_ce", "top1_mark_acc", "top3_mark_acc"}
    )

    # Metrics that describe the run, not the evaluated split. They keep their
    # plain name in split-prefixed (e.g. ``val_``) tuning tables.
    SPLIT_INDEPENDENT_METRICS: typing.ClassVar[typing.FrozenSet[str]] = frozenset({"train_time"})

    # All ranking metrics assume lower is better. Ranking reads ``<name>_mean``.
    RANKING_METRICS = [
        "sigW_loword_notstd",
        "hist_it",
        "hist_int",
        "ED",
        "W1",
        "autocorr_it_short",
        "corr",
        "CRPS",
    ]

    def __init__(self, rows: List[Dict[str, Any]], version: str):
        self.rows = rows
        self.version = version
        # Set by TrainingManager after the validation-selected winner's single
        # test pass; None when no winner was evaluated (e.g. all configs failed,
        # diagnostics were skipped, or evaluate_winner_on_test was disabled).
        self.final_test_row: Optional[Dict[str, Any]] = None
        # Set by TrainingManager.run() at the end of the grid loop: model_name ->
        # the exact cfg used to train it. Multi-seed finalization uses this to
        # resolve which cfg to reload when evaluating the cross-seed winner.
        self.config_by_model_name: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _mean_key(metric: str) -> str:
        return f"{metric}_mean"

    @staticmethod
    def _std_key(metric: str) -> str:
        return f"{metric}_std"

    def normalize_and_rank(self, prefix: str = "") -> "ExperimentResults":
        """Return a NEW ExperimentResults with rank_ columns and a norm score.
        Pure: self is never modified. Safe to call multiple times.

        Ranking skips metrics with no valid values. Ties receive the same min-rank.

        ``prefix`` selects the metric namespace: the default ranks unprefixed
        ``<metric>_mean`` keys into ``rank_<metric>`` / ``norm_score`` (final
        test reporting), while ``prefix="val_"`` ranks ``val_<metric>_mean``
        into ``rank_val_<metric>`` / ``val_norm_score`` (hyperparameter tuning).
        """
        new_rows = copy.deepcopy(self.rows)

        # --- Ranking ---
        try:
            self._apply_ranking(new_rows, prefix=prefix)
        except Exception:
            logger.exception("normalize_and_rank: ranking failed.")
            raise

        return ExperimentResults(new_rows, self.version)

    @classmethod
    def _apply_ranking(cls, rows: List[Dict[str, Any]], prefix: str = "") -> None:
        """Borda-count ranking over RANKING_METRICS (lower is better).

        Reads ``<prefix><metric>_mean`` from each row.
        - Metrics with no valid (non-NaN) values are skipped.
        - Ties receive the same rank (min/competition ranking).
        - NaN entries get rank=NaN (excluded from per-metric ranking) but are
          penalised with rank=n+1 in the norm score.
        """
        apply_competition_ranking(
            rows,
            cls.RANKING_METRICS,
            lambda metric: f"{prefix}{cls._mean_key(metric)}",
            score_key=f"{prefix}norm_score",
            rank_key_for_metric=lambda metric: f"rank_{prefix}{metric}",
            logger=logger,
        )
        return

    def save(self, results_dir: str, prefix: str = "") -> Optional[str]:
        """Write ranked results as timestamped txt to results_dir.

        With ``prefix="val_"`` this writes the hyperparameter tuning table
        (val-prefixed columns, ranked by ``val_norm_score``) to a
        ``<version>_val_tuning_<ts>.txt`` file, keeping it visually and
        programmatically distinct from final test reports.

        Returns the exact path written, or ``None`` when there were no rows to
        save. Callers that need to re-read this exact file later (e.g. the
        sig-degree ablation) must use this returned path rather than
        re-discovering "the latest" file in ``results_dir``: two runs sharing
        the same ``version`` and ``results_dir`` (e.g. a plain run and a
        concurrent multiseed sub-run) can otherwise race, and a glob-latest
        lookup can silently pick up the wrong run's file.
        """
        if not self.rows:
            logger.error("No results to save.")
            return None

        ranked = self.normalize_and_rank(prefix=prefix)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(results_dir, exist_ok=True)

        stem = f"{self.version}_{prefix}tuning" if prefix else self.version
        txt_path = os.path.join(results_dir, f"{stem}_{timestamp}.txt")
        ranked._write_txt(txt_path, prefix=prefix)
        logger.info(f"Results TXT saved to {txt_path}")
        return txt_path

    @classmethod
    def build_column_names(cls, prefix: str = "") -> List[str]:
        """Build the ordered list of metric column names for the enriched output.

        Bootstrap metrics expand into ``<name>_mean`` followed by
        ``<name>_std``. Non-bootstrap metrics are written as plain scalar
        columns. Extra metrics stay with the main metric block; rank columns
        and the norm score are appended last.

        With a split ``prefix`` (e.g. ``"val_"``) every split-dependent metric
        column is prefixed, ``_flat`` histogram columns are dropped (they are
        test-only artifacts), and split-independent columns such as
        ``train_time`` keep their plain name.
        """

        def name_for(metric: str) -> str:
            return metric if metric in cls.SPLIT_INDEPENDENT_METRICS else f"{prefix}{metric}"

        column_names: List[str] = []
        for m in cls.DISPLAY_METRICS + cls.EXTRA_METRICS:
            if prefix and m.endswith("_flat"):
                continue
            if m in cls.NON_BOOTSTRAP_METRICS:
                column_names.append(name_for(m))
            else:
                column_names.append(cls._mean_key(name_for(m)))
                column_names.append(cls._std_key(name_for(m)))
        for m in cls.RANKING_METRICS:
            column_names.append(f"rank_{prefix}{m}")
        column_names.append(f"{prefix}norm_score")
        return column_names

    def _write_txt(self, path: str, prefix: str = "") -> None:
        """Write formatted fixed-width text table (human-readable)."""
        metric_names = self.build_column_names(prefix=prefix)
        write_result_rows_fixed_width(
            self.rows,
            ["MODEL"] + metric_names,
            path,
            metric_names,
            include_seed=False,
            align_error_header_left=True,
        )
