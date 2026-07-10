"""High-level runner for single-seed and multi-seed experiment configs."""

import logging
from typing import Any, Dict, List, Tuple

from config import OUT_FILE_NAME
from src.utils.utils_dict import verbose_get
from test.paper_experiments.experiment_results import ExperimentResults
from test.paper_experiments.multiseed_helpers import (
    aggregate_across_seeds,
    aggregate_test_rows_across_seeds,
    build_seed_config,
    seeds_from_config,
    strip_seed_suffix,
    write_multiseed_outputs,
    write_multiseed_test_by_seed_txt,
    write_multiseed_test_summary_txt,
)
from test.paper_experiments.training_helpers import get_experiment_entry
from test.paper_experiments.trainingmanager import TrainingManager

logger = logging.getLogger(__name__)

# Local runs use this fixed sub-name so they don't clobber server results. Being a
# fixed custom_file_name, it makes get_model_name drop the per-config tokens, so a
# whole grid collapses onto one name. Fine for a single config; on a grid the
# checkpoints then collide (bogus state_dict mismatches) and the `_sig_<d>` token
# the ablation needs is lost. Use server_training: true for grids/ablation.
_LOCAL_RESULTS_FILE_NAME = "test_debug_warning"


def run_experiment_config(cfg: Dict[str, Any]) -> None:
    """Run one loaded experiment config.

    Single-seed configs preserve the original ``TrainingManager`` path.  Configs
    with multiple ``seeds`` are expanded into isolated per-seed runs and then
    summarized with explicit ``*_seed_*`` aggregate columns.
    """
    seeds = seeds_from_config(cfg)
    if len(seeds) == 1:
        _run_training(cfg)
        return

    seed_results = []
    for seed in seeds:
        logger.info("=== MULTI-SEED RUN seed=%d/%s ===", seed, seeds)
        seed_cfg = build_seed_config(cfg, seed)
        result = _run_training(seed_cfg)
        seed_results.append((seed, result))

    path_link = TrainingManager.get_pathlinker(cfg)
    # Each sub-run's own val_tuning/sig_degree-ablation raw files are skipped (see
    # the is_multiseed_subrun guard in TrainingManager.run()); these aggregates,
    # built from seed_results' in-memory rows, are the only per-seed val output
    # written to disk. Lands in its own folder, not the results_on_val_txt/
    # results_on_test_txt a single-seed run uses.
    multiseed_folder = path_link([OUT_FILE_NAME, cfg["experiment_type"], "results_on_multiseed", ""])
    write_multiseed_outputs(seed_results, cfg["version"], multiseed_folder)

    _finalize_multiseed_test(cfg, seed_results, multiseed_folder)
    return


def _finalize_multiseed_test(
    cfg: Dict[str, Any],
    seed_results: List[Tuple[int, ExperimentResults]],
    multiseed_folder: str,
) -> None:
    """Evaluate the cross-seed-selected config's test performance for every seed.

    ``aggregate_across_seeds`` ranks configs by validation performance
    averaged across seeds; its top row is the single config a paper-level
    result should report. Each seed's own local validation winner
    (``TrainingManager.run()``'s per-seed grid search) may be a *different*
    config, which is exactly the ambiguity a plain per-seed test collection
    can't resolve. This instead evaluates the SAME globally selected config on
    the held-out test split for every seed, then prunes every other
    checkpoint for that seed now that the winner is known (per-seed pruning
    was deferred in ``TrainingManager.run()`` for this reason).

    Honors ``evaluate_winner_on_test: false`` the same way the single-seed
    path in ``TrainingManager.run()`` does. Also refuses to pick a winner
    whose ``val_norm_score`` is NaN (e.g. every seed ran with
    ``skip_diagnostics: true``, or every config failed): ranking is
    degenerate in that case, so ``aggregate_across_seeds``'s row order is
    arbitrary insertion order, not a real ranking.
    """
    if not cfg.get("evaluate_winner_on_test", True):
        logger.info("Multi-seed test finalization skipped: evaluate_winner_on_test is False.")
        return

    summary_rows = aggregate_across_seeds(seed_results)
    if not summary_rows:
        logger.warning("Multi-seed test finalization skipped: no rankable config in the validation summary.")
        return
    winner_row = summary_rows[0]
    winner_score = winner_row["metrics"].get("val_norm_score")
    if winner_score is None or winner_score != winner_score:  # NaN-safe: NaN != NaN
        logger.warning(
            "Multi-seed test finalization skipped: top config %r has no valid validation ranking "
            "(val_norm_score=%r); nothing to reliably select as the cross-seed winner.",
            winner_row["model_name"],
            winner_score,
        )
        return
    winner_base_name = winner_row["model_name"]

    refine_b = cfg.get("refine_best_n_bootstraps")
    refine_b = int(refine_b) if refine_b is not None else None
    test_rows = []
    for seed, result in seed_results:
        config_by_model_name = getattr(result, "config_by_model_name", {})
        seed_model_name = next(
            (name for name in config_by_model_name if strip_seed_suffix(name) == winner_base_name), None
        )
        if seed_model_name is None:
            logger.warning(
                "Multi-seed test finalization: seed %d has no surviving checkpoint for winner %r; "
                "skipping this seed's test row.",
                seed,
                winner_base_name,
            )
            continue
        seed_cfg = config_by_model_name[seed_model_name]
        manager = _build_manager(seed_cfg)
        try:
            test_row = manager.evaluate_named_model_on_test(
                seed_model_name,
                seed_cfg,
                multiseed_folder,
                multiseed_folder,
                refine_b,
                write_report=False,
            )
        except Exception:
            logger.exception("Multi-seed test finalization: evaluating winner failed for seed %d.", seed)
            continue
        test_rows.append((seed, test_row))
        manager.prune_all_except(result, {seed_model_name})

    if not test_rows:
        logger.warning("Multi-seed test finalization: no seed produced a test row; no test summary written.")
        return

    write_multiseed_test_by_seed_txt(test_rows, cfg["version"], multiseed_folder)
    write_multiseed_test_summary_txt(
        winner_base_name,
        aggregate_test_rows_across_seeds(test_rows),
        cfg["version"],
        multiseed_folder,
    )
    return


def _build_manager(config: Dict[str, Any]) -> TrainingManager:
    experiment_entry = get_experiment_entry(
        verbose_get(config, "experiment_type", logger, None),
        logger_override=logger,
    )
    return TrainingManager(
        **experiment_entry,
        config=config,
        custom_file_name_results=_LOCAL_RESULTS_FILE_NAME if not config["server_training"] else None,
    )


def _run_training(config: Dict[str, Any]):
    return _build_manager(config).run()
