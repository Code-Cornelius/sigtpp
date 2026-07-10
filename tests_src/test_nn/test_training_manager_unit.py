"""
Unit tests for TrainingManager extracted methods.

Each method is tested in isolation using mocks so no real training,
checkpoints, or file I/O occurs.  See docs/tasks/training_manager/add_training_manager_tests.md
for the full test plan.
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import logging
import os
import signal
import unittest

from unittest.mock import ANY, MagicMock, patch, call

import torch
import torch.nn as nn

from config import OUT_FILE_NAME
from test.paper_experiments.trainingmanager import TrainingManager, NaNDetectorCallback
from test.paper_experiments.experiment_results import ExperimentResults
from src.nn.architectures.architecture_types import Architectures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_cfg(version="sigtpp", **overrides):
    cfg = {
        "version": version,
        "gpu_id": [],
        "epochs": 3,
        "period_log": 1,
        "period_plotting_in_logs": 10,
        "verbose": False,
        "output_dir": "/tmp/test_tm",
        "experiment_type": "poisson_three_marks",
        "seeds": [0],
        "parameter_sets": {},
    }
    cfg.update(overrides)
    return cfg


def _make_manager(cfg=None):
    if cfg is None:
        cfg = _minimal_cfg()
    return TrainingManager(
        data_factory=lambda c: MagicMock(time_max=1.0),
        model_factory=MagicMock(),
        model_namer=lambda tm, c, f: "test_model",
        loss_metrics_fn=lambda arch, num_marks: ["val_epdf"],
        config=cfg,
    )


def _mock_trainer(current_epoch=2):
    return MagicMock(current_epoch=current_epoch)


# ---------------------------------------------------------------------------
# _compute_period_plot_val
# ---------------------------------------------------------------------------


class TestComputePeriodPlotVal(unittest.TestCase):

    def test_local_mode_uses_local_table(self):
        cfg = _minimal_cfg(version="sigtpp")
        manager = _make_manager(cfg)
        result = manager._compute_period_plot_val(cfg)
        self.assertEqual(result, TrainingManager._PERIOD_PLOT_VAL_LOCAL[Architectures.SIGTPP])

    def test_server_mode_uses_server_table(self):
        cfg = _minimal_cfg(version="sigtpp", server_training=True)
        manager = _make_manager(cfg)
        result = manager._compute_period_plot_val(cfg)
        self.assertEqual(result, TrainingManager._PERIOD_PLOT_VAL_SERVER[Architectures.SIGTPP])


# ---------------------------------------------------------------------------
# _build_trainer
# ---------------------------------------------------------------------------


class TestBuildTrainer(unittest.TestCase):
    """Patch Trainer and TrainingSigErrHistoryLogger to avoid real PL setup."""

    def _call_build_trainer(self, cfg):
        manager = _make_manager(cfg)
        path_link = lambda parts: "/tmp/" + "/".join(str(p) for p in parts)
        return manager._build_trainer(cfg, "/tmp/model/", path_link)

    @patch("test.paper_experiments.trainingmanager.Trainer")
    @patch("test.paper_experiments.trainingmanager.TrainingSigErrHistoryLogger")
    def test_patience_present_includes_early_stopping(self, MockLogger, MockTrainer):
        cfg = _minimal_cfg(version="sigtpp", patience=100)
        self._call_build_trainer(cfg)

        callbacks = MockTrainer.call_args[1]["callbacks"]
        cb_names = [type(cb).__name__ for cb in callbacks]
        self.assertIn("EarlyStopping", cb_names)

    @patch("test.paper_experiments.trainingmanager.Trainer")
    @patch("test.paper_experiments.trainingmanager.TrainingSigErrHistoryLogger")
    def test_patience_absent_excludes_early_stopping(self, MockLogger, MockTrainer):
        cfg = _minimal_cfg(version="sigtpp")  # no 'patience' key
        self._call_build_trainer(cfg)

        callbacks = MockTrainer.call_args[1]["callbacks"]
        cb_names = [type(cb).__name__ for cb in callbacks]
        self.assertNotIn("EarlyStopping", cb_names)

    @patch("test.paper_experiments.trainingmanager.Trainer")
    @patch("test.paper_experiments.trainingmanager.TrainingSigErrHistoryLogger")
    def test_score_architecture_check_finite_false(self, MockLogger, MockTrainer):
        from pytorch_lightning.callbacks import EarlyStopping

        cfg = _minimal_cfg(version="ddpm", patience=100)
        self._call_build_trainer(cfg)

        callbacks = MockTrainer.call_args[1]["callbacks"]
        early_stop = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        self.assertFalse(early_stop.check_finite)

    @patch("test.paper_experiments.trainingmanager.Trainer")
    @patch("test.paper_experiments.trainingmanager.TrainingSigErrHistoryLogger")
    def test_non_score_architecture_check_finite_true(self, MockLogger, MockTrainer):
        from pytorch_lightning.callbacks import EarlyStopping

        cfg = _minimal_cfg(version="sigtpp", patience=100)
        self._call_build_trainer(cfg)

        callbacks = MockTrainer.call_args[1]["callbacks"]
        early_stop = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        self.assertTrue(early_stop.check_finite)

    @patch("test.paper_experiments.trainingmanager.Trainer")
    @patch("test.paper_experiments.trainingmanager.TrainingSigErrHistoryLogger")
    def test_patience_is_scaled_by_period_log(self, MockLogger, MockTrainer):
        from pytorch_lightning.callbacks import EarlyStopping

        cfg = _minimal_cfg(version="sigtpp", patience=100, period_log=5)
        self._call_build_trainer(cfg)

        callbacks = MockTrainer.call_args[1]["callbacks"]
        early_stop = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        self.assertEqual(early_stop.patience, 20)

    @patch("test.paper_experiments.trainingmanager.Trainer")
    @patch("test.paper_experiments.trainingmanager.TrainingSigErrHistoryLogger")
    def test_verbose_false_sets_progress_bar_refresh_rate_zero(self, MockLogger, MockTrainer):
        cfg = _minimal_cfg(verbose=False)
        self._call_build_trainer(cfg)

        callbacks = MockTrainer.call_args[1]["callbacks"]
        progress_bar = next(cb for cb in callbacks if cb.__class__.__name__ == "ProgressbarWithoutValBatchUpdate")
        self.assertEqual(progress_bar.refresh_rate, 0)

    @patch("test.paper_experiments.trainingmanager.Trainer")
    @patch("test.paper_experiments.trainingmanager.TrainingSigErrHistoryLogger")
    def test_verbose_true_sets_progress_bar_refresh_rate_ten(self, MockLogger, MockTrainer):
        cfg = _minimal_cfg(verbose=True)
        self._call_build_trainer(cfg)

        callbacks = MockTrainer.call_args[1]["callbacks"]
        progress_bar = next(cb for cb in callbacks if cb.__class__.__name__ == "ProgressbarWithoutValBatchUpdate")
        self.assertEqual(progress_bar.refresh_rate, 10)

    @patch("test.paper_experiments.trainingmanager.Trainer")
    @patch("test.paper_experiments.trainingmanager.TrainingSigErrHistoryLogger")
    def test_build_trainer_passes_expected_logger_and_trainer_args(self, MockLogger, MockTrainer):
        cfg = _minimal_cfg(version="sigtpp", epochs=7, gpu_id=[0], period_log=3, period_plotting_in_logs=12)
        manager = _make_manager(cfg)
        path_link = lambda parts: "/tmp/" + "/".join(str(p) for p in parts)

        trainer, custom_logger = manager._build_trainer(cfg, "/tmp/model/", path_link)

        MockLogger.assert_called_once_with(
            metrics=["val_epdf"],
            plot_loss_history=True,
            period_logging_pt_lightning=3,
            period_in_logs_plotting=12,
            output_dir="/tmp/model/",
        )
        self.assertIs(custom_logger, MockLogger.return_value)
        self.assertIs(trainer, MockTrainer.return_value)
        trainer_kwargs = MockTrainer.call_args.kwargs
        chkpt = next(cb for cb in trainer_kwargs["callbacks"] if cb.__class__.__name__ == "ModelCheckpoint")
        self.assertEqual(os.path.abspath(chkpt.dirpath), os.path.abspath("/tmp/model/"))
        self.assertEqual(trainer_kwargs["default_root_dir"], path_link([OUT_FILE_NAME]))
        self.assertEqual(trainer_kwargs["gpus"], [0])
        self.assertEqual(trainer_kwargs["max_epochs"], 7)
        self.assertEqual(trainer_kwargs["check_val_every_n_epoch"], 3)
        self.assertEqual(trainer_kwargs["logger"], [MockLogger.return_value])
        self.assertEqual(trainer_kwargs["num_sanity_val_steps"], 0)


# ---------------------------------------------------------------------------
# _fit
# ---------------------------------------------------------------------------


class TestFit(unittest.TestCase):

    def test_normal_completion_returns_not_interrupted(self):
        cfg = _minimal_cfg()
        manager = _make_manager(cfg)
        trainer = _mock_trainer()

        train_time, interrupted = manager._fit(cfg, trainer, MagicMock(), MagicMock())

        self.assertFalse(interrupted)
        self.assertGreaterEqual(train_time, 0.0)

    def test_keyboard_interrupt_reraises_to_caller(self):
        cfg = _minimal_cfg()
        manager = _make_manager(cfg)
        trainer = _mock_trainer(current_epoch=0)
        trainer.fit.side_effect = KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            manager._fit(cfg, trainer, MagicMock(), MagicMock())

    def test_generic_exception_reraises_to_caller(self):
        cfg = _minimal_cfg()
        manager = _make_manager(cfg)
        trainer = _mock_trainer(current_epoch=0)
        trainer.fit.side_effect = RuntimeError("exploded")

        with self.assertLogs("test.paper_experiments.trainingmanager", level="ERROR") as cm:
            with self.assertRaises(RuntimeError) as ctx:
                manager._fit(cfg, trainer, MagicMock(), MagicMock())

        self.assertEqual(str(ctx.exception), "exploded")
        self.assertTrue(any("exploded" in line for line in cm.output))


# ---------------------------------------------------------------------------
# _select_checkpoint
# ---------------------------------------------------------------------------


class TestSelectCheckpoint(unittest.TestCase):

    @patch("test.paper_experiments.trainingmanager.get_model_chkpt_path")
    def test_checkpoint_found_returns_path(self, mock_get_chkpt):
        expected = "/tmp/model/model-epoch=0001-val_epdf=0.1234.ckpt"
        mock_get_chkpt.return_value = expected
        manager = _make_manager()

        result = manager._select_checkpoint("/tmp/dummy_path/", _mock_trainer())

        self.assertEqual(result, expected)

    @patch("test.paper_experiments.trainingmanager.get_model_chkpt_path")
    def test_not_found_raises_file_not_found_with_epoch_count(self, mock_get_chkpt):
        mock_get_chkpt.side_effect = FileNotFoundError("not found")
        manager = _make_manager()
        trainer = _mock_trainer(current_epoch=7)

        with self.assertRaises(FileNotFoundError) as ctx:
            manager._select_checkpoint("/tmp/nonexistent_path_12345/", trainer)
        self.assertIn("7", str(ctx.exception))

    @patch("test.paper_experiments.trainingmanager.get_model_chkpt_path")
    def test_generic_exception_after_failed_fit_raises_file_not_found(self, mock_get_chkpt):
        """Companion to _fit generic-exception test: downstream still raises FileNotFoundError."""
        mock_get_chkpt.side_effect = FileNotFoundError("not found")
        manager = _make_manager()
        trainer = _mock_trainer(current_epoch=0)

        with self.assertRaises(FileNotFoundError):
            manager._select_checkpoint("/tmp/nonexistent_path_12345/", trainer)


# ---------------------------------------------------------------------------
# _flush_figures
# ---------------------------------------------------------------------------


class TestFlushFigures(unittest.TestCase):

    def test_no_logger_terminating_operations_not_called(self):
        manager = _make_manager()
        with patch.object(manager, "_terminating_operations") as mock_terminate:
            manager._flush_figures()
            mock_terminate.assert_not_called()

    @patch("test.paper_experiments.trainingmanager.savefig")
    def test_with_logger_terminating_operations_called(self, mock_savefig):
        manager = _make_manager()
        mock_log = MagicMock()
        mock_log.fig = MagicMock()

        manager._flush_figures(mock_log, "/tmp/model/")
        mock_savefig.assert_called()


# ---------------------------------------------------------------------------
# _load_best_and_validate / val diagnostics
# ---------------------------------------------------------------------------


class TestLoadBestAndValidate(unittest.TestCase):

    def test_happy_path_evaluates_full_val_split_and_prefixes_metrics(self):
        from src.data_types.tppmetrics import DatasetSplitType

        data = MagicMock()
        custom_logger = MagicMock()
        model = MagicMock()
        model.evaluate_split_no_grad.return_value = {"ED_mean": 0.5, "W1_mean": 0.7}
        model_factory = MagicMock(return_value=model)

        model_name, metrics = TrainingManager._load_best_and_validate(
            _minimal_cfg(),
            "best.ckpt",
            "named_model",
            data,
            model_factory,
            123,
            custom_logger,
            "/tmp/model/",
        )

        self.assertEqual(model_name, "named_model")
        # All diagnostic metrics come back val_-prefixed for the tuning table.
        self.assertEqual(metrics, {"val_ED_mean": 0.5, "val_W1_mean": 0.7})
        # Model is moved to the diagnostic device (gpu_id=[] -> cpu), then
        # evaluate_split_no_grad runs eval()/no_grad()/device-move and evaluate_split(split=VAL).
        model.to.assert_called_once_with(torch.device("cpu"))
        model.evaluate_split_no_grad.assert_called_once_with(
            data.val_in, data.val_in_len, data.val_marks, split=DatasetSplitType.VAL
        )

    def test_exception_returns_error_metrics_without_evaluation(self):
        model_factory = MagicMock(side_effect=RuntimeError("load failed"))

        _, metrics = TrainingManager._load_best_and_validate(
            _minimal_cfg(),
            "best.ckpt",
            "named_model",
            MagicMock(),
            model_factory,
            123,
            MagicMock(),
            "/tmp/model/",
        )

        self.assertIn("error", metrics)
        self.assertIn("load failed", metrics["error"])


# ---------------------------------------------------------------------------
# _eval_device (gpu_id -> torch.device for direct, non-Trainer diagnostics)
# ---------------------------------------------------------------------------


class TestEvalDevice(unittest.TestCase):
    """Locks the gpu_id -> device mapping the direct VAL pass relies on.

    Constructing ``torch.device("cuda:N")`` does not allocate, so the GPU
    branch is testable on CPU-only CI (the integration tests only cover ``[]``).
    """

    def test_empty_gpu_list_is_cpu(self):
        self.assertEqual(TrainingManager._eval_device([]), torch.device("cpu"))

    def test_gpu_list_uses_first_index(self):
        self.assertEqual(TrainingManager._eval_device([3]), torch.device("cuda:3"))
        self.assertEqual(TrainingManager._eval_device([0, 1]), torch.device("cuda:0"))

    def test_none_and_zero_int_are_cpu(self):
        self.assertEqual(TrainingManager._eval_device(None), torch.device("cpu"))
        self.assertEqual(TrainingManager._eval_device(0), torch.device("cpu"))

    def test_unsupported_spec_raises_instead_of_silent_cpu(self):
        with self.assertRaises(ValueError):
            TrainingManager._eval_device("0,1")


# ---------------------------------------------------------------------------
# winner test evaluation
# ---------------------------------------------------------------------------


class TestEvaluateWinnerOnTest(unittest.TestCase):
    """The single test pass must target the validation-ranked winner only."""

    def _run_winner_eval(self, rows, config_by_model_name, tmpdir, refine_b=None):
        manager = _make_manager(_minimal_cfg(output_dir=tmpdir))
        manager.data_factory = MagicMock(return_value=MagicMock(time_max=1.0))
        model = MagicMock(metrics_test={"ED_mean": 0.42}, _bootstrap_per_replicate=None)
        manager.model_factory = MagicMock(return_value=model)
        results = ExperimentResults(rows, version="sigtpp")

        with patch("test.paper_experiments.trainingmanager.Trainer") as MockTrainer, patch(
            "test.paper_experiments.trainingmanager.get_model_chkpt_path", return_value="best.ckpt"
        ), patch("test.paper_experiments.trainingmanager.seed_everything"):
            row = manager._evaluate_winner_on_test(results, config_by_model_name, tmpdir, tmpdir, refine_b)
        return row, MockTrainer.return_value, manager

    def test_winner_is_selected_by_val_score_not_test_score(self):
        import tempfile, shutil

        tmpdir = tempfile.mkdtemp(prefix="tm_winner_test_")
        # Config A: best on validation, worst on (stale) test metrics.
        rows = [
            {"model_name": "A", "metrics": {"val_ED_mean": 0.1, "val_W1_mean": 0.1}},
            {"model_name": "B", "metrics": {"val_ED_mean": 0.9, "val_W1_mean": 0.9}},
        ]
        cfg_map = {"A": _minimal_cfg(output_dir=tmpdir), "B": _minimal_cfg(output_dir=tmpdir)}
        try:
            row, trainer, manager = self._run_winner_eval(rows, cfg_map, tmpdir)

            self.assertEqual(row["model_name"], "A")
            trainer.test.assert_called_once()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_final_test_row_carries_unprefixed_norm_score(self):
        """The final report keeps unprefixed names; ranking the single test row must
        populate ``norm_score`` (not ``val_norm_score``) so downstream readers see it."""
        import math
        import tempfile, shutil

        tmpdir = tempfile.mkdtemp(prefix="tm_winner_test_")
        rows = [
            {"model_name": "A", "metrics": {"val_ED_mean": 0.1, "val_W1_mean": 0.1}},
            {"model_name": "B", "metrics": {"val_ED_mean": 0.9, "val_W1_mean": 0.9}},
        ]
        cfg_map = {"A": _minimal_cfg(output_dir=tmpdir), "B": _minimal_cfg(output_dir=tmpdir)}
        try:
            row, _trainer, _manager = self._run_winner_eval(rows, cfg_map, tmpdir)

            self.assertIn("norm_score", row["metrics"])
            self.assertNotIn("val_norm_score", row["metrics"])
            self.assertFalse(math.isnan(float(row["metrics"]["norm_score"])))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_no_val_rankable_rows_skips_test_pass(self):
        import tempfile, shutil

        tmpdir = tempfile.mkdtemp(prefix="tm_winner_test_")
        rows = [{"model_name": "A", "metrics": {}}]
        try:
            row, trainer, manager = self._run_winner_eval(rows, {"A": _minimal_cfg(output_dir=tmpdir)}, tmpdir)

            self.assertIsNone(row)
            trainer.test.assert_not_called()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEvaluateNamedModelOnTest(unittest.TestCase):
    """``write_report=False`` (multi-seed finalization) must still persist the
    per-replicate bootstrap npz: it's the only copy of that seed's bootstrap
    replicates, and the caller (training_runner._finalize_multiseed_test)
    deletes the checkpoint right after via prune_all_except."""

    def _run(self, tmpdir, write_report, refine_b=None, model_name="m1"):
        manager = _make_manager(_minimal_cfg(output_dir=tmpdir, n_bootstraps=1))
        manager.data_factory = MagicMock(return_value=MagicMock(time_max=1.0))
        model = MagicMock(metrics_test={"ED_mean": 0.42}, _bootstrap_per_replicate={"ED": [0.42]})
        manager.model_factory = MagicMock(return_value=model)

        with patch("test.paper_experiments.trainingmanager.Trainer") as MockTrainer, patch(
            "test.paper_experiments.trainingmanager.get_model_chkpt_path", return_value="best.ckpt"
        ), patch("test.paper_experiments.trainingmanager.seed_everything"):
            row = manager.evaluate_named_model_on_test(
                model_name, _minimal_cfg(output_dir=tmpdir, n_bootstraps=1), tmpdir, tmpdir, refine_b, write_report
            )
        return row

    def test_write_report_false_still_writes_npz_but_not_txt(self):
        import glob
        import tempfile, shutil

        tmpdir = tempfile.mkdtemp(prefix="tm_named_eval_test_")
        try:
            row = self._run(tmpdir, write_report=False)

            self.assertEqual(row["model_name"], "m1")
            npz_files = glob.glob(os.path.join(tmpdir, "*.npz"))
            txt_files = glob.glob(os.path.join(tmpdir, "*.txt"))
            self.assertEqual(len(npz_files), 1, "bootstrap replicates must survive write_report=False")
            self.assertEqual(len(txt_files), 0, "the per-seed txt report is retired for write_report=False")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_report_true_writes_both_npz_and_txt(self):
        import glob
        import tempfile, shutil

        tmpdir = tempfile.mkdtemp(prefix="tm_named_eval_test_")
        try:
            self._run(tmpdir, write_report=True)

            npz_files = glob.glob(os.path.join(tmpdir, "*.npz"))
            txt_files = glob.glob(os.path.join(tmpdir, "*.txt"))
            self.assertEqual(len(npz_files), 1)
            self.assertEqual(len(txt_files), 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_same_second_different_seeds_do_not_collide_on_disk(self):
        """Multi-seed finalization calls evaluate_named_model_on_test back-to-back
        for two different seeds under the SAME final_version (self.cfg['version']
        is not seed-specific); the model_name in the filename must keep their
        bootstrap replicates from overwriting each other even within one second."""
        import glob
        import tempfile, shutil

        tmpdir = tempfile.mkdtemp(prefix="tm_named_eval_test_")
        try:
            with patch("test.paper_experiments.trainingmanager.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "2026-01-01_00-00-00"
                self._run(tmpdir, write_report=False, model_name="cfg_a_seed1")
                self._run(tmpdir, write_report=False, model_name="cfg_a_seed2")

            npz_files = glob.glob(os.path.join(tmpdir, "*.npz"))
            self.assertEqual(len(npz_files), 2, f"expected one npz per seed, got {npz_files}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# _train_single_config
# ---------------------------------------------------------------------------


class TestTrainSingleConfig(unittest.TestCase):

    def test_diagnostic_only_skips_training_setup_but_runs_diagnostics(self):
        cfg = _minimal_cfg(diagnostic_only=True)
        manager = _make_manager(cfg)
        trainer = MagicMock()
        custom_logger = MagicMock()
        path_link = lambda parts: "/tmp/" + "/".join(str(p) for p in parts)
        datamodel_path = path_link([OUT_FILE_NAME, "poisson_three_marks", "models", "test_model", ""])

        with patch.object(manager, "get_pathlinker", return_value=path_link), patch(
            "test.paper_experiments.trainingmanager.remove_files_from_dir"
        ) as mock_remove, patch.object(
            manager, "_build_trainer", return_value=(trainer, custom_logger)
        ) as mock_build, patch.object(
            manager, "_compute_period_plot_val", return_value=123
        ), patch.object(
            manager, "_fit"
        ) as mock_fit, patch.object(
            manager, "_select_checkpoint", return_value="best.ckpt"
        ) as mock_select, patch.object(
            manager, "_run_val_diagnostics", return_value=("test_model", {"val_ED_mean": 0.1})
        ) as mock_diag, patch.object(
            manager, "_flush_figures"
        ) as mock_flush:
            result = manager._train_single_config(cfg)

        expected_metrics = {"val_ED_mean": 0.1, "train_time": 0.0}
        self.assertEqual(result, {"model_name": "test_model", "metrics": expected_metrics, "train_time": 0.0})
        mock_remove.assert_not_called()
        manager.model_factory.assert_not_called()
        mock_fit.assert_not_called()
        mock_build.assert_called_once()
        mock_select.assert_called_once_with(ANY, trainer)
        # _run_val_diagnostics no longer takes a trainer (validation runs via
        # model.evaluate_split directly, not trainer.test).
        mock_diag.assert_called_once_with(
            cfg,
            "best.ckpt",
            "test_model",
            ANY,
            123,
            custom_logger,
            datamodel_path,
        )
        mock_flush.assert_called_once_with(None, None)

    def test_interrupted_training_flushes_figures_before_reraising(self):
        cfg = _minimal_cfg()
        manager = _make_manager(cfg)
        trainer = MagicMock()
        custom_logger = MagicMock()
        model = MagicMock()
        manager.model_factory.return_value = model
        path_link = lambda parts: "/tmp/" + "/".join(str(p) for p in parts)
        datamodel_path = path_link([OUT_FILE_NAME, "poisson_three_marks", "models", "test_model", ""])

        with patch.object(manager, "get_pathlinker", return_value=path_link), patch(
            "test.paper_experiments.trainingmanager.remove_files_from_dir"
        ), patch.object(manager, "_build_trainer", return_value=(trainer, custom_logger)), patch.object(
            manager, "_compute_period_plot_val", return_value=123
        ), patch.object(
            manager, "_fit", side_effect=KeyboardInterrupt()
        ), patch.object(
            manager, "_terminating_operations"
        ), patch.object(
            manager, "_flush_figures"
        ) as mock_flush:
            with self.assertRaises(KeyboardInterrupt):
                manager._train_single_config(cfg)

        mock_flush.assert_called_once_with(custom_logger, datamodel_path)


# ---------------------------------------------------------------------------
# _print_recap
# ---------------------------------------------------------------------------


class TestPrintRecap(unittest.TestCase):

    def _recap(self, rows, train_times, n_failed):
        manager = _make_manager()
        results = ExperimentResults(rows, version="sigtpp")
        with self.assertLogs("test.paper_experiments.trainingmanager", level="INFO") as cm:
            manager._print_recap(results, train_times, n_failed, n_total_expected=len(rows))
        return "\n".join(cm.output)

    def test_single_config_no_grid_search_line(self):
        rows = [{"model_name": "m1", "metrics": {"ED": 0.5}}]
        log = self._recap(rows, [10.0], 0)
        self.assertNotIn("Grid search", log)

    def test_multi_config_with_failure_shows_summary(self):
        rows = [
            {"model_name": "m1", "metrics": {"ED": 0.5}},
            {"model_name": "m2", "metrics": {"error": "bad config"}},
        ]
        log = self._recap(rows, [10.0, 5.0], 1)
        self.assertIn("1/2 configs succeeded", log)
        self.assertIn("1 failed", log)

    def test_no_train_times_no_avg_fit_line(self):
        rows = [{"model_name": "m1", "metrics": {"ED": 0.5}}]
        log = self._recap(rows, [], 0)
        self.assertNotIn("Avg fit time", log)

    def test_with_train_times_shows_average(self):
        rows = [{"model_name": "m1", "metrics": {"ED": 0.5}}]
        log = self._recap(rows, [10.0, 20.0], 0)
        self.assertIn("15.0", log)

    def test_all_failed_grid_search_present_no_performance_line(self):
        rows = [
            {"model_name": "m1", "metrics": {"error": "fail1"}},
            {"model_name": "m2", "metrics": {"error": "fail2"}},
        ]
        log = self._recap(rows, [], 2)
        self.assertIn("Grid search", log)
        self.assertNotIn("Performance", log)

    def test_failed_row_error_message_appears_in_log(self):
        rows = [{"model_name": "m1", "metrics": {"error": "UNIQUE_ERROR_XYZ"}}]
        log = self._recap(rows, [], 1)
        self.assertIn("UNIQUE_ERROR_XYZ", log)

    def test_multi_success_rows_show_best_model_when_rankable(self):
        # Rows now carry validation diagnostics; ranking is val-based.
        rows = [
            {"model_name": "m1", "metrics": {"val_ED_mean": 0.5, "val_W1_mean": 0.7}},
            {"model_name": "m2", "metrics": {"val_ED_mean": 0.1, "val_W1_mean": 0.2}},
        ]
        log = self._recap(rows, [1.0, 2.0], 0)
        self.assertIn("Best validation-selected model: m2", log)

    def test_recap_prints_normalized_mark_metrics(self):
        rows = [
            {
                "model_name": "m1",
                "metrics": {
                    "val_ED_mean": 0.5,
                    "val_mark_ce": torch.tensor(0.25),
                    "val_top1_mark_acc": 0.8,
                    "val_top3_mark_acc": 1.0,
                },
            }
        ]
        log = self._recap(rows, [1.0], 0)
        self.assertIn("mark_ce: 0.2500", log)
        self.assertIn("top1_mark_acc: 0.8000", log)
        self.assertIn("top3_mark_acc: 1.0000", log)

    def test_multi_success_rows_show_could_not_rank_when_metrics_missing(self):
        rows = [
            {"model_name": "m1", "metrics": {}},
            {"model_name": "m2", "metrics": {}},
        ]
        log = self._recap(rows, [], 0)
        self.assertIn("Best validation-selected model: could not rank", log)


# ---------------------------------------------------------------------------
# NaNDetectorCallback
# ---------------------------------------------------------------------------


class _NaNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.fill_(float("nan"))

    def forward(self, x):
        return x


class TestNaNDetectorCallback(unittest.TestCase):

    def test_nan_weights_logs_critical_with_module_name(self):
        model = _NaNModel()
        cb = NaNDetectorCallback()
        trainer = MagicMock(current_epoch=0)

        with self.assertLogs("test.paper_experiments.trainingmanager", level="CRITICAL") as cm:
            cb.on_validation_epoch_end(trainer, model)

        log_output = "\n".join(cm.output)
        self.assertIn("linear", log_output)

    @patch("test.paper_experiments.trainingmanager.logger")
    def test_clean_model_no_critical_logged(self, mock_logger):
        model = nn.Linear(2, 2)
        cb = NaNDetectorCallback()
        trainer = MagicMock(current_epoch=0)

        cb.on_validation_epoch_end(trainer, model)

        mock_logger.critical.assert_not_called()


# ---------------------------------------------------------------------------
# signal handling / termination
# ---------------------------------------------------------------------------


class TestSignalHandling(unittest.TestCase):

    @patch("test.paper_experiments.trainingmanager.signal.signal")
    def test_run_registers_sigint_and_sigterm_at_start_and_restores_after(self, mock_signal):
        # Handlers are registered once in run(), not per-config.
        manager = _make_manager(_minimal_cfg(parameter_sets={"lr": [1e-4]}))
        mock_signal.return_value = signal.SIG_DFL  # simulate previous handler

        with patch.object(
            manager, "_train_single_config", return_value={"model_name": "m", "metrics": {}}
        ), patch.object(manager, "_print_recap"), patch(
            "test.paper_experiments.trainingmanager.ExperimentResults.save"
        ):
            manager.run()

        calls = mock_signal.call_args_list
        sigint_calls = [c for c in calls if c.args[0] == signal.SIGINT]
        sigterm_calls = [c for c in calls if c.args[0] == signal.SIGTERM]
        # Two calls each: one to install, one to restore
        self.assertEqual(len(sigint_calls), 2)
        self.assertEqual(len(sigterm_calls), 2)

    @patch("test.paper_experiments.trainingmanager.signal.signal")
    def test_stop_requested_flag_breaks_config_loop(self, mock_signal):
        # Signal sets a local flag; loop checks it between configs.
        # We capture the installed handler and call it to simulate a signal arriving
        # between the first and second config.
        captured_handler = [None]

        def capture(sig, handler):
            if sig == signal.SIGINT:
                captured_handler[0] = handler
            return signal.SIG_DFL

        mock_signal.side_effect = capture

        manager = _make_manager(_minimal_cfg(parameter_sets={"lr": [1e-4, 1e-3]}))
        call_count = 0

        def side_effect(_cfg):
            nonlocal call_count
            call_count += 1
            # Fire the signal handler after the first config, simulating Ctrl-C between configs
            if captured_handler[0] is not None:
                captured_handler[0](signal.SIGINT, None)
            return {"model_name": "m", "metrics": {}}

        with patch.object(manager, "_train_single_config", side_effect=side_effect), patch.object(
            manager, "_print_recap"
        ), patch("test.paper_experiments.trainingmanager.ExperimentResults.save"):
            manager.run()

        self.assertEqual(call_count, 1)  # second config never runs

    @patch("test.paper_experiments.trainingmanager.savefig")
    def test_terminating_operations_saves_loss_history_svg(self, mock_savefig):
        # Logger and path are now passed as arguments, not read from instance state.
        manager = _make_manager()
        mock_log = MagicMock(fig="figure")

        manager._terminating_operations(mock_log, "/tmp/model/")

        mock_savefig.assert_called_once_with("figure", "/tmp/model/loss_history.svg")

    def test_terminating_operations_logs_warning_when_state_missing(self):
        manager = _make_manager()

        with self.assertLogs("test.paper_experiments.trainingmanager", level="WARNING") as cm:
            manager._terminating_operations()

        self.assertIn("No logger or output path set", "\n".join(cm.output))


# ---------------------------------------------------------------------------
# run() failure counting and model-name recovery
# ---------------------------------------------------------------------------


class TestRunFailureCountAndNameRecovery(unittest.TestCase):
    """Tests for C1 (n_failed incremented on error metrics) and C4 (safe name recovery)."""

    def _make_simple_manager(self, n_configs=2):
        cfg = _minimal_cfg(seeds=[42])
        cfg["parameter_sets"] = {"lr": [1e-4] * n_configs}
        return TrainingManager(
            data_factory=lambda c: MagicMock(time_max=1.0),
            model_factory=MagicMock(),
            model_namer=lambda tm, c, f: "test_model",
            loss_metrics_fn=lambda arch, num_marks: ["val_epdf"],
            config=cfg,
        )

    def test_error_metrics_returned_by_train_single_config_counted_as_failure(self):
        """C1: _train_single_config returning {error: ...} must increment n_failed."""
        manager = self._make_simple_manager(n_configs=2)
        call_count = [0]

        def side_effect(cfg):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"model_name": "m1", "metrics": {}, "train_time": 1.0}
            return {"model_name": "m2", "metrics": {"error": "training failed"}, "train_time": 0.5}

        with patch.object(manager, "_train_single_config", side_effect=side_effect), patch.object(
            ExperimentResults, "save"
        ), self.assertLogs("test.paper_experiments.trainingmanager", level="INFO") as cm:
            manager.run()

        log = "\n".join(cm.output)
        self.assertIn("1/2", log, "Expected '1/2 configs succeeded' in recap")
        self.assertIn("1 failed", log, "Expected '1 failed' in recap")

    def test_error_path_when_data_factory_also_fails_records_error_with_fallback_name(self):
        """C4: if data_factory raises in the error path, the training error is recorded with a fallback name."""
        cfg = _minimal_cfg(seeds=[42])
        cfg["parameter_sets"] = {"lr": [1e-4]}

        manager = TrainingManager(
            data_factory=MagicMock(side_effect=RuntimeError("data load failed")),
            model_factory=MagicMock(),
            model_namer=lambda tm, c, f: "test_model",
            loss_metrics_fn=lambda arch, num_marks: ["val_epdf"],
            config=cfg,
        )

        with patch.object(
            manager, "_train_single_config", side_effect=ValueError("original training error")
        ), patch.object(ExperimentResults, "save"):
            results = manager.run()

        self.assertEqual(len(results.rows), 1)
        self.assertIn("error", results.rows[0]["metrics"])
        self.assertIn("original training error", results.rows[0]["metrics"]["error"])


# ---------------------------------------------------------------------------
# run() error handling
# ---------------------------------------------------------------------------


class TestRunErrorHandling(unittest.TestCase):

    def _make_run_manager(self, n_lr_values=1, **cfg_overrides):
        """Manager wired to the poisson_three_marks experiment with a minimal param grid."""
        from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY
        from src.utils.utils_dict import verbose_get
        import tempfile, os

        cfg = _minimal_cfg(version="sigtpp", seeds=[42], **cfg_overrides)
        cfg["experiment_type"] = "poisson_three_marks"
        cfg["parameter_sets"] = {"lr_gen": [1e-4] * n_lr_values}
        cfg["output_dir"] = tempfile.mkdtemp(prefix="tm_run_test_")

        manager = TrainingManager(
            **verbose_get(EXPERIMENT_REGISTRY, "poisson_three_marks", MagicMock(), None),
            config=cfg,
        )
        return manager, cfg["output_dir"]

    def test_config_exception_records_error_in_results(self):
        # run() catches generic exceptions from _train_single_config and records them.
        manager, tmp = self._make_run_manager()
        try:
            with patch.object(manager, "_train_single_config", side_effect=ValueError("bad param")):
                with patch.object(ExperimentResults, "save"):
                    results = manager.run()

            self.assertEqual(len(results.rows), 1)
            self.assertIn("error", results.rows[0]["metrics"])
            self.assertIn("bad param", results.rows[0]["metrics"]["error"])
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_keyboard_interrupt_with_existing_rows_saves_partial(self):
        manager, tmp = self._make_run_manager(n_lr_values=2)
        call_count = 0

        def side_effect(_cfg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"model_name": "m1", "metrics": {"ED": 0.5}, "train_time": 5.0}
            raise KeyboardInterrupt()

        try:
            with patch.object(manager, "_train_single_config", side_effect=side_effect):
                with patch.object(ExperimentResults, "save") as mock_save:
                    results = manager.run()

            self.assertEqual(len(results.rows), 1)
            mock_save.assert_called()
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_keyboard_interrupt_with_no_rows_returns_empty_without_saving(self):
        manager, tmp = self._make_run_manager()
        try:
            with patch.object(manager, "_train_single_config", side_effect=KeyboardInterrupt()):
                with patch.object(ExperimentResults, "save") as mock_save:
                    results = manager.run()

            self.assertEqual(len(results.rows), 0)
            mock_save.assert_not_called()
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_config_exception_calls_data_factory_for_model_name_records_error(self):
        # run() calls data_factory in error path to reconstruct model_name, then records error.
        manager, tmp = self._make_run_manager()
        data_factory_mock = MagicMock(return_value=MagicMock(time_max=1.0))
        manager.data_factory = data_factory_mock
        try:
            with patch.object(manager, "_train_single_config", side_effect=ValueError("bad param")):
                with patch.object(ExperimentResults, "save"):
                    results = manager.run()

            data_factory_mock.assert_called_once()
            self.assertEqual(len(results.rows), 1)
            self.assertIn("error", results.rows[0]["metrics"])
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_evaluates_val_winner_on_test_exactly_once(self):
        """Two configs -> two val diagnostic rows -> a single winner test pass."""
        manager, tmp = self._make_run_manager(n_lr_values=2)
        call_count = 0

        def side_effect(_cfg):
            nonlocal call_count
            call_count += 1
            return {
                "model_name": f"m{call_count}",
                "metrics": {"val_ED_mean": 0.1 * call_count, "val_W1_mean": 0.1 * call_count},
                "train_time": 1.0,
            }

        try:
            with patch.object(manager, "_train_single_config", side_effect=side_effect), patch.object(
                ExperimentResults, "save"
            ), patch.object(
                manager, "_evaluate_winner_on_test", return_value={"model_name": "m1", "metrics": {}}
            ) as mock_winner:
                results = manager.run()

            mock_winner.assert_called_once()
            # The winner test row is exposed on the returned results.
            self.assertEqual(results.final_test_row, {"model_name": "m1", "metrics": {}})
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_saves_val_tuning_table(self):
        """The all-config save must be the val-prefixed tuning table."""
        manager, tmp = self._make_run_manager(n_lr_values=1)
        try:
            with patch.object(
                manager,
                "_train_single_config",
                return_value={"model_name": "m1", "metrics": {"val_ED_mean": 0.1}, "train_time": 1.0},
            ), patch.object(ExperimentResults, "save") as mock_save, patch.object(
                manager, "_evaluate_winner_on_test", return_value=None
            ):
                manager.run()

            mock_save.assert_called_once()
            self.assertEqual(mock_save.call_args.kwargs.get("prefix"), "val_")
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_multiseed_subrun_skips_winner_test_eval_and_pruning(self):
        """Multi-seed sub-runs defer the winner-on-test pass and disk pruning
        to the cross-seed finalization step in run_experiment_config (see
        training_runner._finalize_multiseed_test): the per-seed local winner
        may not be the config that wins across all seeds. They also skip
        writing their own local val_tuning table and sig_degree ablation
        outputs entirely (even when sig_degree_ablation=True): both are
        superseded by the cross-seed aggregate files run_experiment_config
        writes from in-memory rows."""
        manager, tmp = self._make_run_manager(n_lr_values=2, _multiseed_seed_tag=7, sig_degree_ablation=True)
        call_count = 0

        def side_effect(_cfg):
            nonlocal call_count
            call_count += 1
            return {
                "model_name": f"m{call_count}",
                "metrics": {"val_ED_mean": 0.1 * call_count, "val_W1_mean": 0.1 * call_count},
                "train_time": 1.0,
            }

        try:
            with patch.object(manager, "_train_single_config", side_effect=side_effect), patch.object(
                ExperimentResults, "save"
            ) as mock_save, patch.object(manager, "_evaluate_winner_on_test") as mock_winner, patch.object(
                manager, "_evaluate_sig_degree_ablation_on_test"
            ) as mock_ablation, patch.object(
                manager, "_prune_non_top_models"
            ) as mock_prune:
                results = manager.run()

            mock_save.assert_not_called()
            mock_winner.assert_not_called()
            mock_ablation.assert_not_called()
            mock_prune.assert_not_called()
            self.assertIsNone(results.final_test_row)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_multiseed_subrun_exposes_config_by_model_name(self):
        """run_experiment_config's finalization step resolves each seed's exact
        cfg for the cross-seed winner via results.config_by_model_name."""
        manager, tmp = self._make_run_manager(n_lr_values=2, _multiseed_seed_tag=7)
        call_count = 0

        def side_effect(cfg):
            nonlocal call_count
            call_count += 1
            return {
                "model_name": f"m{call_count}",
                "metrics": {"val_ED_mean": 0.1 * call_count},
                "train_time": 1.0,
            }

        try:
            with patch.object(manager, "_train_single_config", side_effect=side_effect), patch.object(
                ExperimentResults, "save"
            ):
                results = manager.run()

            self.assertEqual(set(results.config_by_model_name), {"m1", "m2"})
            self.assertEqual(results.config_by_model_name["m1"]["_multiseed_seed_tag"], 7)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_single_seed_run_still_prunes_and_evaluates_winner(self):
        """Regression guard: only multiseed sub-runs skip this; plain runs unchanged."""
        manager, tmp = self._make_run_manager(n_lr_values=1)
        try:
            with patch.object(
                manager,
                "_train_single_config",
                return_value={"model_name": "m1", "metrics": {"val_ED_mean": 0.1}, "train_time": 1.0},
            ), patch.object(ExperimentResults, "save"), patch.object(
                manager, "_evaluate_winner_on_test", return_value={"model_name": "m1", "metrics": {}}
            ) as mock_winner, patch.object(
                manager, "_prune_non_top_models"
            ) as mock_prune:
                manager.run()

            mock_winner.assert_called_once()
            mock_prune.assert_called_once()
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_system_exit_from_select_checkpoint_saves_partial_results(self):
        """SystemExit(0) raised by _select_checkpoint is caught by run() same as KeyboardInterrupt."""
        manager, tmp = self._make_run_manager(n_lr_values=2)
        call_count = 0

        def side_effect(_cfg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"model_name": "m1", "metrics": {"val_ED_mean": 0.5}, "train_time": 5.0}
            raise SystemExit(0)

        try:
            with patch.object(manager, "_train_single_config", side_effect=side_effect):
                with patch.object(ExperimentResults, "save") as mock_save:
                    results = manager.run()

            self.assertEqual(len(results.rows), 1)
            mock_save.assert_called_once()
            # The partial save must use the val_ tuning schema (rows carry val_ metrics).
            self.assertEqual(mock_save.call_args.kwargs.get("prefix"), "val_")
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
