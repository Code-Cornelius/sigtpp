"""
Unified training manager for different experiment types.
"""

import logging
import os
import signal
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


from config import ROOT_DIR, OUT_FILE_NAME
from src.data_types.exceptions import SkipConfig
from src.data_types.sigw_loss_data_props import SkipSigDegreeConfig
from src.data_types.tppmetrics import DatasetSplitType
from src.nn.architectures.architecture_types import Architectures
from src.utils.training_sig_err_history_logger import TrainingSigErrHistoryLogger
from src.utils.utils_file import delete_dir_tree_safe, remove_files_from_dir
from src.utils.utils_os import factory_fct_linked_path, savefig
from test.paper_experiments.experiment_results import ExperimentResults
from test.paper_experiments.recompute_bootstrap import (
    BootstrapEvalSettings,
    _write_per_replicate_npz,
    recompute_one_row,
)
from test.paper_experiments.sig_degree_ablation import run_sig_degree_ablation_from_val_file
from test.paper_experiments.training_helpers import get_model_chkpt_path, log_config_message
from src.utils.parameters_product import parameters_product
from src.utils.progress_bar_without_val_batch_update import ProgressbarWithoutValBatchUpdate
from src.utils.utils_dict import verbose_get


class NaNDetectorCallback(Callback):
    """Checks model weights for NaN at each validation epoch. Zero per-step overhead."""

    def on_validation_epoch_end(self, trainer, pl_module):
        nan_modules: Dict[str, list] = {}
        for name, param in pl_module.named_parameters():
            if torch.isnan(param).any():
                module_name, _, param_name = name.rpartition(".")
                key = module_name or name
                nan_modules.setdefault(key, []).append(param_name or name)
        if nan_modules:
            modules_str = ", ".join(nan_modules.keys())
            logger.critical(
                "NaN weights at epoch %d in: %s",
                trainer.current_epoch,
                modules_str,
            )
            for module_name, param_names in nan_modules.items():
                logger.debug(
                    "NaN parameters in '%s': %s",
                    module_name,
                    ", ".join(param_names),
                )
            trainer.should_stop = True
            setattr(trainer, "_nan_weights_detected", True)
            logger.warning(
                "Stopping training early due to NaN weights; diagnostics will use the best saved checkpoint if one exists."
            )
        return


class TrainingManager(object):
    """Unified training manager for different experiment types."""

    _PERIOD_PLOT_VAL_LOCAL = {
        Architectures.DETER: 1,
        Architectures.GAMMA: 100,
        Architectures.DDPM: 500,
        Architectures.SIGTPP: 500,
        Architectures.WGAN: 500,
        Architectures.VAE: 500,
    }

    _PERIOD_PLOT_VAL_SERVER = {
        Architectures.DETER: 100_000,
        Architectures.GAMMA: 100_000,
        Architectures.DDPM: 100_000,
        Architectures.SIGTPP: 100_000,
        Architectures.WGAN: 100_000,
        Architectures.VAE: 100_000,
    }

    # Bootstrap-aware schema: every metric is reported as ``<name>_mean``.
    # Derived from ExperimentResults.DISPLAY_METRICS; excludes train_time
    # (reported separately) and _flat histogram variants (less informative for a quick recap).
    _RECAP_METRICS = [
        m if m in ExperimentResults.NON_BOOTSTRAP_METRICS else f"{m}_mean"
        for m in ExperimentResults.DISPLAY_METRICS
        if m != "train_time" and not m.endswith("_flat")
    ]

    @staticmethod
    def get_pathlinker(cfg: Dict[str, Any]) -> Callable:
        return factory_fct_linked_path(ROOT_DIR, cfg["output_dir"])

    @staticmethod
    def _eval_device(gpu_id: Any) -> torch.device:
        """Device for direct (non-Trainer) diagnostics, matching ``Trainer(gpus=gpu_id)``.

        Configs pass ``gpu_id`` as a list of GPU indices (e.g. ``[0]``); an empty
        list (or ``0`` / ``None``) means CPU. Diagnostics use the first listed GPU
        so they land on the same device as the trainer-driven test pass. An
        unrecognised spec raises rather than silently running a long diagnostic
        pass on CPU.
        """
        if isinstance(gpu_id, (list, tuple)):
            return torch.device(f"cuda:{int(gpu_id[0])}") if gpu_id else torch.device("cpu")
        if gpu_id is None or (isinstance(gpu_id, int) and gpu_id <= 0):
            return torch.device("cpu")
        if isinstance(gpu_id, int):
            return torch.device("cuda:0")
        raise ValueError(f"Unsupported gpu_id {gpu_id!r}: expected a list of GPU indices (e.g. [0]) or an int count.")

    def _load_best_and_validate(
        cfg: Dict[str, Any],
        latest_checkpoint: Optional[str],
        datamodel_name: str,
        data: Any,
        model_factory: Callable[..., Any],
        plot_val: Any,
        logger_custom: Any,
        datamodel_path: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Evaluate the best checkpoint on the FULL validation split.

        Returns ``val_``-prefixed diagnostic metrics for cross-config
        hyperparameter ranking. The test split is never touched here: it is
        reserved for the single winner evaluation in ``_evaluate_winner_on_test``.

        Calls ``model.evaluate_split_no_grad(split=VAL)`` directly instead of
        ``trainer.test``: validation no longer rides the Lightning loop, so the
        split is an explicit argument and ``on_test_end`` (artifact saving)
        cannot fire. ``eval`` / ``no_grad`` / input-device placement live in
        ``TPPArchitecture.evaluate_split_no_grad``; this manager only supplies the
        target device (``_eval_device``) and moves the model onto it.
        """
        try:
            model = model_factory(cfg, data, plot_val, datamodel_path, logger_custom, latest_checkpoint)
            logger.info("Running validation diagnostics on model loaded from: %s", latest_checkpoint)
            # Distribution metrics need the whole split in one batch (see
            # evaluate_split); val_in/val_in_len/val_marks already hold it.
            device = TrainingManager._eval_device(cfg.get("gpu_id"))
            model.to(device)
            raw_metrics = model.evaluate_split_no_grad(
                data.val_in,
                data.val_in_len,
                data.val_marks,
                split=DatasetSplitType.VAL,
            )
            metrics = {f"val_{k}": v for k, v in raw_metrics.items()}
        except KeyError as e:
            logger.error(
                f"Failed to perform validation diagnostics: missing required config parameter {e}. Check your YAML config."
            )
            metrics = {"error": f"Missing config parameter: {e}"}
        except Exception as e:
            logger.error(f"Failed to perform validation diagnostics: {e}")
            metrics = {"error": str(e)}
        return datamodel_name, metrics

    def __init__(
        self,
        data_factory: Callable[[Dict[str, Any]], Any],
        model_factory: Callable[[Dict[str, Any], Any, int], Any],
        model_namer: Callable[[float, Dict[str, Any], str], str],
        loss_metrics_fn: Callable[[Any, int], List[str]],
        config: Dict[str, Any],
        custom_file_name_results: Optional[str] = None,
    ):
        self.data_factory = data_factory
        self.model_factory = model_factory
        self.model_namer = model_namer
        self.loss_metrics_fn = loss_metrics_fn
        self.cfg = config
        self.custom_file_name = custom_file_name_results
        self._current_trainer = None
        self._stop_requested = False

    def run(self) -> ExperimentResults:
        """Train all parameter combinations in sequence and save results.

        SIGINT/SIGTERM are caught via a flag so the current config completes before the loop
        exits, avoiding interference with PyTorch Lightning internals.
        """
        run_seed = self._resolve_single_seed(self.cfg)
        param_grid = parameters_product(self.cfg["parameter_sets"])
        logger.info("Number of configurations: %d", len(param_grid))

        # A fixed custom_file_name collapses the whole grid onto one name, so the
        # checkpoints collide and the `_sig_<d>` ablation token is lost. Flag it now
        # rather than let it surface as confusing failures later on.
        if self.custom_file_name is not None and len(param_grid) > 1:
            logger.warning(
                "custom_file_name=%r with a %d-config grid: all configs share one name, so "
                "checkpoints collide (expect bogus state_dict mismatches) and the sig-degree "
                "ablation finds no winners. Unset it (server_training: true) for grid runs.",
                self.custom_file_name,
                len(param_grid),
            )

        results = ExperimentResults([], version=self.cfg["version"])
        path_link = self.get_pathlinker(self.cfg)
        if self.cfg.get("_multiseed_seed_tag") is not None:
            # Per-seed sub-run of a multiseed sweep (see build_seed_config in
            # multiseed_helpers.py). This seed's own val_tuning/sig_degree-ablation/
            # winner-on-test writes are all skipped below (is_multiseed_subrun):
            # they're superseded by the cross-seed aggregate files
            # run_experiment_config writes into this same folder from in-memory
            # rows, and by the cross-seed finalization step's own winner-on-test
            # pass. These folder variables stay assigned for the log message below
            # and for symmetry with the single-seed branch.
            multiseed_folder = path_link([OUT_FILE_NAME, self.cfg["experiment_type"], "results_on_multiseed", ""])
            results_val_txt_folder = multiseed_folder
            results_test_txt_folder = multiseed_folder
            results_test_npz_folder = multiseed_folder
            results_ablation_folder = multiseed_folder
        else:
            results_val_txt_folder = path_link([OUT_FILE_NAME, self.cfg["experiment_type"], "results_on_val_txt", ""])
            results_test_txt_folder = path_link(
                [OUT_FILE_NAME, self.cfg["experiment_type"], "results_on_test_txt", ""]
            )
            results_test_npz_folder = path_link(
                [OUT_FILE_NAME, self.cfg["experiment_type"], "results_on_test_npz", ""]
            )
            results_ablation_folder = path_link(
                [OUT_FILE_NAME, self.cfg["experiment_type"], "results_on_ablation", ""]
            )
        train_times = []
        n_failed = 0
        n_skipped = 0
        # Per-row cfg snapshot so the post-loop refine hook can rebuild data/model
        # for the winner. Each ``param`` combo produces a unique ``model_name`` via
        # the deterministic namer; later combos overwrite earlier ones only on a
        # genuine collision, which would already break ``ModelCheckpoint`` paths.
        config_by_model_name: Dict[str, Dict[str, Any]] = {}

        self._stop_requested = False

        def termination_handler(signum: int, _frame: Any) -> None:
            logger.critical("Signal %d received, requesting graceful stop", signum)
            self._stop_requested = True
            if self._current_trainer is not None:
                self._current_trainer.should_stop = True

        prev_sigint = signal.signal(signal.SIGINT, termination_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, termination_handler)

        try:
            for i, param in enumerate(param_grid):
                if self._stop_requested:
                    logger.warning("Training interrupted, saving partial results.")
                    break

                seed_everything(run_seed, workers=True)
                cfg = self.cfg.copy()
                cfg["seed"] = run_seed
                cfg.pop("seeds", None)
                cfg["parameter_sets"] = param
                log_config_message(i + 1, len(param_grid), 80, train_times)
                logger.info(f"Parameters: {param}")
                config_start = time.perf_counter()
                try:
                    result = self._train_single_config(cfg)
                except (KeyboardInterrupt, SystemExit):
                    logger.warning("Training interrupted, saving partial results.")
                    if results.rows:
                        # Rows carry val_-prefixed diagnostics; save the partial
                        # tuning table with the same schema as the full run.
                        results.save(results_val_txt_folder, prefix="val_")
                        logger.info("Partial results saved in %s", results_val_txt_folder)
                    self._print_recap(
                        results,
                        train_times,
                        n_failed,
                        n_total_expected=len(param_grid),
                        n_skipped=n_skipped,
                        interrupted=True,
                    )
                    return results
                except SkipConfig as e:
                    logger.info("Skipping config %d/%d — invalid combination: %s", i + 1, len(param_grid), e)
                    n_skipped += 1
                    try:
                        data = self.data_factory(cfg)
                        model_name_skip = self.model_namer(data.time_max, cfg, self.custom_file_name)
                    except Exception as name_err:
                        logger.warning(
                            "Could not reconstruct model_name for skipped config %d: %s",
                            i + 1,
                            name_err,
                        )
                        experiment = cfg.get("experiment_type", "unknown")
                        version = cfg.get("version", f"config_{i + 1}")
                        model_name_skip = f"{experiment}/{version}"
                    results.rows.append(
                        {
                            "model_name": model_name_skip,
                            "metrics": {"error": f"skipped: {e}"},
                        }
                    )
                    continue
                except Exception as e:
                    logger.error("Config %s failed: %s", param, e)
                    n_failed += 1
                    train_times.append(np.round(time.perf_counter() - config_start, 2))
                    try:
                        data = self.data_factory(cfg)
                        model_name_err = self.model_namer(data.time_max, cfg, self.custom_file_name)
                    except Exception as name_err:
                        logger.warning(
                            "Could not reconstruct model_name for failed config %d: %s",
                            i + 1,
                            name_err,
                        )
                        experiment = cfg.get("experiment_type", "unknown")
                        version = cfg.get("version", f"config_{i + 1}")
                        model_name_err = f"{experiment}/{version}"
                    result = {
                        "model_name": model_name_err,
                        "metrics": {"error": str(e)},
                    }
                    results.rows.append(result)
                    # Do not throw here otherwise multi hyperparameter training can't finish.
                    # raise e
                else:
                    if result.get("train_time") is not None:
                        train_times.append(result["train_time"])
                    if "error" in result.get("metrics", {}):
                        n_failed += 1
                    results.rows.append(result)
                    config_by_model_name[result["model_name"]] = cfg
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

        # Multi-seed sub-runs defer the winner-on-test pass and disk pruning to
        # the cross-seed finalization step in run_experiment_config: the
        # per-seed local winner may not be the config that wins across all
        # seeds, and its checkpoint must stay on disk until that is known.
        # Evaluating/pruning here would (a) write a per-seed final_test report
        # that becomes redundant once the fixed-config test summary exists, and
        # (b) risk deleting the eventual cross-seed winner's checkpoint before
        # it can be evaluated on test for this seed.
        is_multiseed_subrun = self.cfg.get("_multiseed_seed_tag") is not None

        # All-config validation tuning table: val-prefixed columns, ranked by val_norm_score.
        # Capture the exact path written: the sig-degree ablation below must read
        # back this run's own file, not rediscover "the latest" one in a shared
        # directory (a concurrent run sharing this version/folder could win a
        # glob-latest race -- see ExperimentResults.save). Multi-seed sub-runs skip
        # this write: it is this seed's own local grid-search table, fully
        # superseded by the cross-seed ``multiseed_per_seed``/``multiseed_summary``
        # files ``run_experiment_config`` writes afterwards from the in-memory rows
        # of every seed (``write_multiseed_by_seed_txt`` reads ``result.rows``
        # directly, not this file, so nothing downstream needs it on disk).
        if is_multiseed_subrun:
            val_tuning_path = None
        else:
            val_tuning_path = results.save(results_val_txt_folder, prefix="val_")
        results.config_by_model_name = config_by_model_name

        # Single test pass for the validation-selected winner. This is the only
        # place the test split is evaluated; it always runs unless explicitly
        # disabled via ``evaluate_winner_on_test: false``.
        refine_b = self.cfg.get("refine_best_n_bootstraps")
        if not is_multiseed_subrun and not self._stop_requested and self.cfg.get("evaluate_winner_on_test", True):
            if self.cfg.get("skip_diagnostics", False):
                logger.info("Winner test evaluation skipped because diagnostics were disabled for this run.")
            else:
                try:
                    results.final_test_row = self._evaluate_winner_on_test(
                        results,
                        config_by_model_name,
                        results_test_txt_folder,
                        results_test_npz_folder,
                        int(refine_b) if refine_b is not None else None,
                    )
                except Exception:
                    # The winner evaluation is layered on an already-saved tuning
                    # table; don't propagate failures up the run() boundary.
                    logger.exception("Winner test evaluation failed; validation tuning table already saved.")

        # Per-sig-degree ablation: opt-in via ``sig_degree_ablation: true``. Runs
        # here, BEFORE pruning, so every degree's winner checkpoint is still on
        # disk. Additive to the single-winner pass above; never selects on test.
        # Multi-seed sub-runs skip this too: it needs val_tuning_path (this seed's
        # own local grid-search table on disk), which multi-seed sub-runs no longer
        # write -- each seed's own local per-degree winner would anyway be exactly
        # the kind of per-seed-independent selection the cross-seed finalization
        # step in run_experiment_config was built to avoid for the single-winner
        # test pass.
        if is_multiseed_subrun and self.cfg.get("sig_degree_ablation", False):
            logger.info(
                "Sig-degree ablation skipped for multi-seed sub-run: per-seed ablation files are "
                "no longer written (see is_multiseed_subrun in TrainingManager.run())."
            )
        elif not self._stop_requested and self.cfg.get("sig_degree_ablation", False):
            if self.cfg.get("skip_diagnostics", False):
                logger.info("Sig-degree ablation skipped because diagnostics were disabled for this run.")
            elif val_tuning_path is None:
                logger.warning("Sig-degree ablation skipped: no validation tuning file was written (no results).")
            else:
                try:
                    self._evaluate_sig_degree_ablation_on_test(
                        val_tuning_path,
                        results_ablation_folder,
                        int(refine_b) if refine_b is not None else None,
                    )
                except Exception:
                    logger.exception("Sig-degree ablation failed; validation tuning table already saved.")

        # Disk cleanup: keep only the top-K model dirs by val_norm_score
        # (default K=10). Set keep_top_k_models <= 0 to keep all; skipped on
        # interruption. Multi-seed sub-runs skip this too (see
        # is_multiseed_subrun above): pruning happens once, after the finalization
        # step in run_experiment_config, to keep only the cross-seed winner.
        if not is_multiseed_subrun:
            self._prune_non_top_models(results)

        logger.info("All trainings completed, results saved in %s", results_val_txt_folder)
        self._print_recap(
            results,
            train_times,
            n_failed,
            n_total_expected=len(param_grid),
            n_skipped=n_skipped,
            interrupted=self._stop_requested,
        )
        return results

    @staticmethod
    def _resolve_single_seed(cfg: Dict[str, Any]) -> int:
        """Return the scalar seed for one TrainingManager run."""
        seeds = cfg.get("seeds")
        assert isinstance(seeds, list) and seeds, f"Config must define 'seeds', e.g. seeds: [42]. Got {seeds!r}."
        assert len(seeds) == 1, (
            "TrainingManager handles one seed only; use run_experiment_config for multi-seed configs. "
            f"Got {seeds!r}."
        )
        return int(seeds[0])

    def _evaluate_winner_on_test(
        self,
        results: ExperimentResults,
        config_by_model_name: Dict[str, Dict[str, Any]],
        results_test_txt_folder: str,
        results_test_npz_folder: str,
        refine_b: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the single test pass for the validation-selected winner.

        Configs are ranked by ``val_norm_score``; only the top-1 model's
        checkpoint is reloaded and evaluated on the test split, exactly once.
        The final report keeps unprefixed metric names (``ED_mean``,
        ``norm_score``) and lives in ``results_test_txt_folder`` as
        ``<version>_final_test[_B<b>]_<model_name>_<ts>.txt`` (plus a matching ``.npz`` with
        per-replicate vectors in ``results_test_npz_folder`` when bootstrapped).

        ``refine_b`` optionally raises the bootstrap replicate count for this
        pass only (grid search uses a small B for ranking speed).

        Returns the winner's test row, or ``None`` when no config is rankable.
        """
        valid_rows = []
        for row in results.rows:
            metrics = row.get("metrics", {})
            if "error" in metrics:
                continue
            has_rankable_metrics = False
            for metric in ExperimentResults.RANKING_METRICS:
                value = metrics.get(f"val_{metric}_mean")
                if value is None:
                    continue
                try:
                    if not np.isnan(float(value)):
                        has_rankable_metrics = True
                        break
                except (TypeError, ValueError):
                    continue
            if has_rankable_metrics:
                valid_rows.append(row)
        if not valid_rows:
            logger.warning("Winner test evaluation: no successful runs with validation diagnostics; skipping.")
            return None

        ranked = ExperimentResults(valid_rows, results.version).normalize_and_rank(prefix="val_")
        winner_name = ranked.rows[0]["model_name"]
        winner_cfg = config_by_model_name.get(winner_name)
        if winner_cfg is None:
            logger.warning("Winner test evaluation: cannot resolve cfg for winner %r; skipping.", winner_name)
            return None

        return self.evaluate_named_model_on_test(
            winner_name, winner_cfg, results_test_txt_folder, results_test_npz_folder, refine_b
        )

    def evaluate_named_model_on_test(
        self,
        model_name: str,
        cfg: Dict[str, Any],
        results_test_txt_folder: str,
        results_test_npz_folder: str,
        refine_b: Optional[int] = None,
        write_report: bool = True,
    ) -> Dict[str, Any]:
        """Reload ``model_name``'s checkpoint (trained under ``cfg``) and test it once.

        Shared by ``_evaluate_winner_on_test`` (single-seed: ranks then calls
        this on the top-1 config) and the multi-seed finalization step in
        ``run_experiment_config`` (calls this once per seed, forcing
        ``model_name`` to the cross-seed winner rather than each seed's local
        top-1).

        The per-replicate ``.npz`` (when bootstrapped) is always written to
        ``results_test_npz_folder``, regardless of ``write_report`` -- it is
        the only copy of that seed's bootstrap replicates, and the caller
        typically deletes the checkpoint right after via ``prune_all_except``.
        The filename includes ``model_name`` (not just a timestamp): multi-seed
        finalization calls this once per seed, back-to-back, with an identical
        ``final_version`` (``self.cfg['version']`` is not seed-specific) into
        the same folder, so a timestamp alone could collide between seeds that
        finish evaluating within the same second.
        When ``write_report`` is true (the default, used by
        ``_evaluate_winner_on_test``), also writes the timestamped
        ``<version>_final_test[_B<b>]_<model_name>_<ts>.txt`` report. Multi-seed
        finalization passes ``write_report=False``: it consolidates every
        seed's row into ``multiseed_test_by_seed``/``multiseed_test_summary``
        instead, so a per-seed txt report here would just duplicate that.

        Returns the ranked test row (``rank_<metric>``/``norm_score`` columns
        populated from this pass).
        """
        cfg = cfg.copy()
        original_b = int(cfg.get("n_bootstraps", 1))
        if refine_b is not None:
            cfg["n_bootstraps"] = int(refine_b)
            logger.info(
                "Evaluating %r on test (bootstrap B: %d -> %d).",
                model_name,
                original_b,
                refine_b,
            )
        else:
            logger.info("Evaluating %r on test (B=%d).", model_name, original_b)

        data = self.data_factory(cfg)
        path_link = self.get_pathlinker(cfg)
        datamodel_path = path_link([OUT_FILE_NAME, cfg["experiment_type"], "models", model_name, ""])
        checkpoint = get_model_chkpt_path(datamodel_path)
        period_plot_val = 10**9  # suppress validation plotting

        trainer = Trainer(
            default_root_dir=path_link([OUT_FILE_NAME]),
            gpus=cfg["gpu_id"],
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )
        seed_everything(int(cfg.get("seed", 42)), workers=True)
        model = self.model_factory(cfg, data, period_plot_val, datamodel_path, None, checkpoint)
        trainer.test(model, datamodule=data, verbose=False)
        metrics = model.metrics_test or {}
        per_replicate = getattr(model, "_bootstrap_per_replicate", None)

        b_suffix = f"_B{refine_b}" if refine_b is not None else ""
        final_version = f"{self.cfg['version']}_final_test{b_suffix}"

        # Rank the single test row so the final report carries unprefixed
        # rank_<metric> / norm_score columns (the schema downstream readers
        # expect), populated from the test pass rather than left as NaN.
        ranked_final = ExperimentResults(
            [{"model_name": model_name, "metrics": dict(metrics)}],
            version=final_version,
        ).normalize_and_rank()
        winner_row = ranked_final.rows[0]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # model_name disambiguates the filename: multi-seed finalization calls this
        # once per seed, back-to-back, for the SAME final_version (self.cfg['version']
        # is not seed-specific) into the SAME results_on_multiseed folder -- without
        # this, two evaluations completing within the same wall-clock second would
        # overwrite each other's bootstrap replicates.
        file_stem = f"{final_version}_{model_name}_{timestamp}"

        if per_replicate is not None:
            # Lazy import to avoid a circular dependency at module load:
            # recompute_bootstrap imports trainingmanager for side-effect settings registration.
            from test.paper_experiments.recompute_bootstrap import _write_per_replicate_npz

            os.makedirs(results_test_npz_folder, exist_ok=True)
            npz_path = os.path.join(results_test_npz_folder, f"{file_stem}.npz")
            _write_per_replicate_npz(
                [{"model_name": model_name, "per_replicate": per_replicate}],
                npz_path,
                int(cfg["n_bootstraps"]),
            )

        if not write_report:
            return winner_row

        os.makedirs(results_test_txt_folder, exist_ok=True)
        txt_path = os.path.join(results_test_txt_folder, f"{file_stem}.txt")
        ranked_final._write_txt(txt_path)
        logger.info("Final test report saved to %s", txt_path)

        return winner_row

    def _evaluate_sig_degree_ablation_on_test(
        self,
        val_tuning_path: str,
        results_ablation_folder: str,
        refine_b: Optional[int],
    ) -> None:
        """Per-sig-degree ablation: evaluate each degree's val-winner on test.

        Reads ``val_tuning_path`` -- the exact file this run's ``results.save()``
        call just wrote, not a re-discovered "latest" file -- selects each
        signature-degree group's winner by lowest ``val_norm_score``, and reruns
        the bootstrap test step for each via ``recompute_one_row``. Ablation
        reports and bootstrap vectors are written to their own
        ``results_on_ablation`` folder, kept
        separate from the single-winner ``results_on_test_*`` outputs.
        """
        # Mirror _eval_device's semantics: configs pass gpu_id as a list of
        # device indices; [] / None / int <= 0 mean CPU; a bare positive int is
        # a device COUNT (first device), not an index.
        cfg_gpu = self.cfg["gpu_id"]
        gpu_id: Optional[int]
        if isinstance(cfg_gpu, (list, tuple)):
            gpu_id = int(cfg_gpu[0]) if cfg_gpu else None
        elif cfg_gpu is None or (isinstance(cfg_gpu, int) and cfg_gpu <= 0):
            gpu_id = None
        elif isinstance(cfg_gpu, int):
            gpu_id = 0
        else:
            raise ValueError(f"Unsupported gpu_id {cfg_gpu!r}: expected a list of GPU indices or an int count.")
        # Same fallback as the winner pass (cfg.get("n_bootstraps", 1)) so both
        # test passes agree when the key is absent.
        n_bootstraps = int(refine_b) if refine_b is not None else int(self.cfg.get("n_bootstraps", 1))
        trainer_seed = self._resolve_single_seed(self.cfg)

        outputs = run_sig_degree_ablation_from_val_file(
            val_tuning_path,
            self.cfg["version"],
            n_bootstraps=n_bootstraps,
            trainer_seed=trainer_seed,
            gpu_id=gpu_id,
            recompute_one_row_fn=recompute_one_row,
            write_npz_fn=_write_per_replicate_npz,
            settings_cls=BootstrapEvalSettings,
            results_dir=results_ablation_folder,
        )
        logger.info("Sig-degree ablation report saved to %s", outputs.report_path)
        keep_k = self.cfg.get("keep_top_k_models")
        if keep_k is None or int(keep_k) > 0:
            logger.warning(
                "Model pruning runs after this ablation: per-degree winner checkpoints outside the "
                "top-K will be deleted. Re-running the ablation later requires keep_top_k_models <= 0."
            )
        return

    def _prune_non_top_models(self, results: ExperimentResults) -> None:
        """Delete model dirs outside the top-K by ``val_norm_score``.

        Controlled via the ``keep_top_k_models`` config key. Defaults to ``10``
        when unset. Set to ``<= 0`` to keep every dir (e.g. for
        ``recompute_bootstrap.py``, which needs all checkpoints). The run also
        keeps every dir when interrupted. Deletes whole ``models/<model_name>/``
        dirs, including failed/skipped configs (non-rankable, never in the
        top-K). The winner is rank-1 and is always kept. Never deletes outside
        the experiment's ``models/`` root.
        """
        if self._stop_requested:
            return
        keep_k = self.cfg.get("keep_top_k_models")
        try:
            keep_k = int(keep_k) if keep_k is not None else 10
        except (TypeError, ValueError):
            keep_k = 10
        if keep_k <= 0:
            return

        valid_rows = []
        for row in results.rows:
            metrics = row.get("metrics", {})
            if "error" in metrics:
                continue
            for metric in ExperimentResults.RANKING_METRICS:
                value = metrics.get(f"val_{metric}_mean")
                try:
                    if value is not None and not np.isnan(float(value)):
                        valid_rows.append(row)
                        break
                except (TypeError, ValueError):
                    continue
        if not valid_rows:
            logger.info("Model pruning skipped: no rankable configs (need val diagnostics).")
            return

        ranked = ExperimentResults(valid_rows, results.version).normalize_and_rank(prefix="val_")
        keep_names = {row["model_name"] for row in ranked.rows[:keep_k]}
        if results.final_test_row is not None:
            keep_names.add(results.final_test_row["model_name"])

        self.prune_all_except(results, keep_names)

    def prune_all_except(self, results: ExperimentResults, keep_names: Set[str]) -> None:
        """Delete every trained model dir for this experiment except ``keep_names``.

        Shared by ``_prune_non_top_models`` (keeps the top-K by validation rank
        plus the local winner) and the multi-seed finalization step in
        ``run_experiment_config`` (keeps only the cross-seed winner's
        checkpoint for one seed, once the winner is known across all seeds).
        ``results.rows`` supplies the full set of model names trained in this
        run/sub-run; never deletes outside the experiment's ``models/`` root.
        """
        if self._stop_requested:
            return
        all_names = {row["model_name"] for row in results.rows if row.get("model_name")}
        delete_names = all_names - keep_names
        if not delete_names:
            logger.info(
                "Model pruning: nothing to delete (%d config(s), keeping %d).", len(all_names), len(keep_names)
            )
            return

        path_link = self.get_pathlinker(self.cfg)
        experiment_type = self.cfg["experiment_type"]
        models_root = path_link([OUT_FILE_NAME, experiment_type, "models", ""])

        freed = 0
        n_deleted = 0
        for name in sorted(delete_names):
            model_dir = path_link([OUT_FILE_NAME, experiment_type, "models", name, ""])
            if not os.path.isdir(model_dir):
                continue
            freed_bytes = delete_dir_tree_safe(model_dir, must_be_under=models_root)
            if not os.path.isdir(model_dir):
                freed += freed_bytes
                n_deleted += 1

        logger.info(
            "Model pruning: kept %d, deleted %d model dir(s), freed %.1f MB.",
            len(all_names & keep_names),
            n_deleted,
            freed / 1e6,
        )
        return

    def _train_single_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Train, evaluate, and persist one config.

        On training failure diagnostics are skipped; the error is recorded in metrics so the
        config is counted as failed rather than silently appearing as a success.
        """
        do_only_diagnostic = verbose_get(cfg, "diagnostic_only", logger, False)
        data = self.data_factory(cfg)
        model_name = self.model_namer(data.time_max, cfg, self.custom_file_name)
        path_link = self.get_pathlinker(cfg)
        datamodel_path = path_link([OUT_FILE_NAME, cfg["experiment_type"], "models", model_name, ""])
        if not do_only_diagnostic:
            remove_files_from_dir(datamodel_path)

        trainer, custom_logger = self._build_trainer(cfg, datamodel_path, path_link, data.num_marks)
        period_plot_val = self._compute_period_plot_val(cfg)
        if cfg.get("skip_diagnostics", False):
            period_plot_val = 999_999  # suppress all validation plotting

        # Covers the case we do only diagnostic or if fails.
        train_time = 0.0
        interrupted = False
        metrics_result = None
        logger_with_fig_error_losses = None
        current_out_path = None

        if not do_only_diagnostic:
            if cfg["verbose"]:
                logger.info("Creating the model.")
            model = self.model_factory(cfg, data, period_plot_val, datamodel_path, custom_logger)
            if cfg["verbose"]:
                logger.info("Model created.")

            logger_with_fig_error_losses = custom_logger
            current_out_path = datamodel_path

            try:
                train_time, interrupted = self._fit(cfg, trainer, model, data)
                if getattr(trainer, "_nan_weights_detected", False):
                    logger.warning(
                        "Training stopped after NaN detection; evaluating the best checkpoint saved before instability."
                    )
            except (KeyboardInterrupt, SystemExit):
                # Preserve interruption signal, skip diagnostics
                interrupted = True
                self._terminating_operations(logger_with_fig_error_losses, current_out_path)
                metrics_result = {"error": "Training interrupted"}
            except Exception as e:
                logger.error("Training failed, skipping diagnostics: %s", e)
                metrics_result = {"error": str(e)}
                # raise e

        if interrupted:
            # Re-raise interruption to be caught in run()
            self._flush_figures(logger_with_fig_error_losses, current_out_path)
            raise KeyboardInterrupt

        # If training failed, skip diagnostics but still persist the error metrics
        if metrics_result is not None:
            self._flush_figures(logger_with_fig_error_losses, current_out_path)
            return {"model_name": model_name, "metrics": metrics_result, "train_time": train_time}

        # skip_diagnostics=True: training verified, bypass test phase and plotting entirely. More often used for tests.
        if cfg.get("skip_diagnostics", False):
            self._flush_figures(logger_with_fig_error_losses, current_out_path)
            return {"model_name": model_name, "metrics": {}, "train_time": train_time}

        # Training succeeded: run validation diagnostics for hyperparameter ranking.
        # The test split is reserved for the post-grid winner evaluation.
        try:
            latest_checkpoint = self._select_checkpoint(datamodel_path, trainer)
            model_name, metrics_result = self._run_val_diagnostics(
                cfg,
                latest_checkpoint,
                model_name,
                data,
                period_plot_val,
                custom_logger,
                datamodel_path,
            )
        except Exception as e:
            # Diagnostic failure is also counted as a config failure
            logger.error("Diagnostics failed: %s", e)
            metrics_result = {"error": str(e)}
            # raise e

        metrics_result["train_time"] = train_time
        self._flush_figures(logger_with_fig_error_losses, current_out_path)
        return {"model_name": model_name, "metrics": metrics_result, "train_time": train_time}

    def _build_trainer(
        self,
        cfg: Dict[str, Any],
        datamodel_path: str,
        path_link: Callable,
        num_marks: int = 1,
    ) -> Tuple[Trainer, TrainingSigErrHistoryLogger]:
        period_log = cfg['period_log']
        if 'patience' in cfg:
            patience = cfg['patience']
            early_stop = EarlyStopping(
                monitor="val_epdf",
                min_delta=1e-4,
                verbose=False,
                patience=patience // period_log,
                mode='min',
                # In score version, we put epdf == nan to avoid computing it at every step, which is incompatible with True below.
                check_finite=False if Architectures(cfg['version']) is Architectures.DDPM else True,
            )
        else:
            early_stop = None

        chkpt = ModelCheckpoint(
            monitor="val_epdf",
            mode="min",
            verbose=False,
            save_top_k=1,
            dirpath=datamodel_path,
            filename="model-{epoch:04d}-{val_epdf:.4f}",
        )

        custom_logger = TrainingSigErrHistoryLogger(
            metrics=self.loss_metrics_fn(Architectures(cfg['version']), num_marks),
            plot_loss_history=True,
            period_logging_pt_lightning=period_log,
            period_in_logs_plotting=cfg['period_plotting_in_logs'],
            output_dir=datamodel_path,
        )

        callbacks = [
            chkpt,
            NaNDetectorCallback(),
            ProgressbarWithoutValBatchUpdate(refresh_rate=10 if cfg["verbose"] else 0),
        ]
        if early_stop is not None:
            callbacks.append(early_stop)

        trainer = Trainer(
            default_root_dir=path_link([OUT_FILE_NAME]),
            gpus=cfg['gpu_id'],
            max_epochs=cfg['epochs'],
            logger=[custom_logger],
            check_val_every_n_epoch=period_log,
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )
        self._current_trainer = trainer
        return trainer, custom_logger

    def _fit(
        self,
        cfg: Dict[str, Any],
        trainer: Trainer,
        model: Any,
        data: Any,
    ) -> Tuple[float, bool]:
        """
        Caller must handle training failures and decide whether to skip diagnostics.
        """
        start = time.perf_counter()
        interrupted = False
        try:
            trainer.fit(model, datamodule=data)
        except (KeyboardInterrupt, SystemExit):
            interrupted = True
            raise
        except Exception as e:
            logger.error("Training failed: %s", e)
            raise
        finally:
            self._current_trainer = None
            train_time = np.round(time.perf_counter() - start, 2)
            if cfg["verbose"]:
                logger.info(
                    "Total time training: %s seconds. On average, it took: %s seconds per epoch.",
                    train_time,
                    np.round(train_time / (trainer.current_epoch + 1), 4),
                )
        return train_time, interrupted

    def _select_checkpoint(
        self,
        datamodel_path: str,
        trainer: Trainer,
    ) -> str:
        """Find the best checkpoint path, or raise on failure."""
        try:
            return get_model_chkpt_path(datamodel_path)
        except FileNotFoundError as e:
            epochs_done = trainer.current_epoch
            raise FileNotFoundError(
                f"{e}. Model trained for {epochs_done} epoch(s). A checkpoint is only saved after a validation step; check that 'epochs' and 'period_log' allow at least one validation to complete."
            ) from e

    def _run_val_diagnostics(
        self,
        cfg: Dict[str, Any],
        latest_checkpoint: str,
        model_name: str,
        data: Any,
        period_plot_val: int,
        custom_logger: Any,
        datamodel_path: str,
    ) -> Tuple[str, Dict[str, Any]]:
        # No trainer: validation diagnostics run via model.evaluate_split directly.
        return TrainingManager._load_best_and_validate(
            cfg,
            latest_checkpoint,
            model_name,
            data,
            self.model_factory,
            period_plot_val,
            custom_logger,
            datamodel_path,
        )

    def _flush_figures(
        self,
        logger_with_fig_error_losses: Optional[Any] = None,
        current_out_path: Optional[str] = None,
    ) -> None:
        """Save the loss-history SVG and close all matplotlib figures."""
        if logger_with_fig_error_losses is not None:
            self._terminating_operations(logger_with_fig_error_losses, current_out_path)
        plt.close("all")

    def _compute_period_plot_val(self, cfg: Dict[str, Any]) -> int:
        """Return the validation plot period for this config's architecture and environment."""
        version = Architectures(cfg['version'])
        server = verbose_get(cfg, 'server_training', logger, False)
        if server:
            return self._PERIOD_PLOT_VAL_SERVER[version]
        return self._PERIOD_PLOT_VAL_LOCAL[version]

    def _print_recap(
        self,
        results: ExperimentResults,
        train_times: list,
        n_failed: int,
        n_total_expected: int,
        n_skipped: int = 0,
        interrupted: bool = False,
    ) -> None:
        n_ran = len(results.rows)
        n_total = n_total_expected if n_total_expected > 0 else n_ran
        sep = "=" * 60
        lines = [sep, "  TRAINING COMPLETE"]

        if interrupted:
            lines.append(f"  *** Stopped early by user interruption ({n_ran}/{n_total} configs run) ***")

        # Grid search summary
        if n_total > 1:
            n_ok = n_ran - n_failed
            fail_str = f" ({n_failed} failed)" if n_failed else ""
            skip_str = f", {n_skipped} skipped" if n_skipped else ""
            lines.append(f"  Grid search: {n_ok}/{n_total} configs succeeded{fail_str}{skip_str}")

        # Training duration
        if train_times:
            avg_time = float(np.mean(train_times))
            lines.append(f"  Avg fit time: {avg_time:.1f}s/config")

        # Best model performance (selection is validation-based; rows carry val_ metrics)
        success_rows = [r for r in results.rows if "error" not in r["metrics"]]
        if success_rows:
            if len(success_rows) > 1:
                ranked = ExperimentResults(success_rows, results.version).normalize_and_rank(prefix="val_")
                best = ranked.rows[0]
                score = best["metrics"].get("val_norm_score", np.nan)
                if np.isnan(score):
                    lines.append("  Best validation-selected model: could not rank (insufficient metrics)")
                else:
                    lines.append(f"  Best validation-selected model: {best['model_name']}")
            else:
                best = success_rows[0]
            shown = []
            for k in self._RECAP_METRICS:
                # Rows hold val_ metrics; fall back to plain keys for legacy rows.
                v = best["metrics"].get(f"val_{k}", best["metrics"].get(k))
                if v is not None:
                    try:
                        fv = float(v)
                        if not np.isnan(fv):
                            shown.append(f"{k}: {fv:.4f}")
                    except (TypeError, ValueError):
                        pass
            if shown:
                lines.append("  Performance (validation): " + " | ".join(shown))
            if results.final_test_row is not None:
                lines.append(f"  Final test report written for: {results.final_test_row['model_name']}")

        # Error messages for failed configs
        if n_failed > 0:
            lines.append("")
            lines.append(f"  ERRORS ({n_failed} configs failed):")
            for i, row in enumerate(results.rows, 1):
                if "error" in row["metrics"]:
                    error_msg = row["metrics"]["error"]
                    lines.append(f"  - Config {i}: {error_msg}")

        lines.append(sep)
        for line in lines:
            logger.info(line)

    def _terminating_operations(
        self,
        logger_with_fig_error_losses: Optional[Any] = None,
        current_out_path: Optional[str] = None,
    ) -> None:
        """Flush the loss-history figure to disk if logger and output path are provided."""
        if logger_with_fig_error_losses is not None and current_out_path is not None:
            savefig(logger_with_fig_error_losses.fig, current_out_path + "loss_history.svg")
        elif logger_with_fig_error_losses is None or current_out_path is None:
            logger.warning("No logger or output path set for terminating operations.")


###################################
# required to register experiments
###################################
import test.paper_experiments.settings.poisson as _  # noqa: F401
import test.paper_experiments.settings.inh_poisson as _  # noqa: F401
import test.paper_experiments.settings.taxi as _  # noqa: F401
import test.paper_experiments.settings.hawkes as _  # noqa: F401
import test.paper_experiments.settings.stackoverflow as _  # noqa: F401
import test.paper_experiments.settings.taobao as _  # noqa: F401
import test.paper_experiments.settings.earthquake as _  # noqa: F401
import test.paper_experiments.settings.hawkes_3x3 as _  # noqa: F401
import test.paper_experiments.settings.yelp_mississauga as _  # noqa: F401
