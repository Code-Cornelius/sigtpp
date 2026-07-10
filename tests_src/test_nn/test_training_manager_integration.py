"""
Integration tests for TrainingManager.

These tests run real training (minimal epochs, CPU only) to verify that
the end-to-end pipeline behaviour is preserved after the modularisation
refactor.

All tests write to a temporary directory and clean up after themselves.
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import logging
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

# Reset logging before unit tests run
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

from unittest.mock import patch

logger = logging.getLogger(__name__)

from src.utils.utils_dict import verbose_get
from test.paper_experiments.training_helpers import load_experiment_config
from test.paper_experiments.trainingmanager import TrainingManager
from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY


_SHARED_TRAINING_RUN = None


# ---------------------------------------------------------------------------
# Shared minimal config for the Poisson/SigWGAN experiment
# ---------------------------------------------------------------------------


def _poisson_cfg(tmp_dir: str) -> dict:
    cfg = load_experiment_config("poisson_three_marks/sigtpp_test.yaml")
    cfg["seeds"] = [42]
    cfg["epochs"] = 1
    cfg["period_log"] = 1
    cfg["patience"] = 9999
    cfg["diagnostic_only"] = False
    cfg["gpu_id"] = []
    cfg["verbose"] = False
    cfg["output_dir"] = tmp_dir
    cfg["smoke_data_size"] = 512
    cfg["parameter_sets"] = {
        "lr_gen": [1.0e-4],
        "sig_degree": [2],
        "concentration_factor": [1.0],
        "hid_size_rep": [8],
        "use_teacher_forcing": [False],
        "terminal_anchor": ["free_endpoint"],
        "detach_cum_channel": [False],
        "mark_loss_weight": [1.0],
    }
    return cfg


def _make_manager(cfg: dict) -> TrainingManager:
    return TrainingManager(
        **verbose_get(EXPERIMENT_REGISTRY, cfg["experiment_type"], logger, None),
        config=cfg,
        custom_file_name_results="integration_test",
    )


def _ensure_shared_training_run() -> dict:
    global _SHARED_TRAINING_RUN

    if _SHARED_TRAINING_RUN is None:
        tmp = tempfile.mkdtemp(prefix="tm_shared_train_")
        cfg = _poisson_cfg(tmp)
        manager = _make_manager(cfg)
        try:
            train_results = manager.run()
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        _SHARED_TRAINING_RUN = {
            "tmp": tmp,
            "train_results": train_results,
        }

    return _SHARED_TRAINING_RUN


def tearDownModule():
    global _SHARED_TRAINING_RUN

    if _SHARED_TRAINING_RUN is not None:
        shutil.rmtree(_SHARED_TRAINING_RUN["tmp"], ignore_errors=True)
        _SHARED_TRAINING_RUN = None


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestDiagnosticOnlyMode(unittest.TestCase):
    """diagnostic_only=True must skip training, run test phase, return train_time=0."""

    def test_diagnostic_only_train_time_zero_and_non_empty_metrics(self):
        shared = _ensure_shared_training_run()
        first_results = shared["train_results"]
        self.assertEqual(len(first_results.rows), 1)
        self.assertNotIn("error", first_results.rows[0]["metrics"])

        cfg2 = _poisson_cfg(shared["tmp"])
        cfg2["diagnostic_only"] = True
        manager2 = _make_manager(cfg2)
        diag_results = manager2.run()

        self.assertEqual(len(diag_results.rows), 1)
        row = diag_results.rows[0]
        self.assertNotIn("error", row["metrics"], f"Diagnostic-only run produced error: {row['metrics'].get('error')}")
        self.assertEqual(row.get("train_time"), 0.0, "diagnostic_only should set train_time=0.0")
        self.assertGreater(len(row["metrics"]), 0, "Diagnostic run returned empty metrics: test phase may not have run")


class TestGridSearch(unittest.TestCase):
    """Single training run with minimal config must complete with no errors.

    Grid search control flow (iterate configs, collect results) is tested by
    TestGridSearchWithFailingConfig using mocks. This test just verifies that
    a complete training pipeline (train + test + metrics) works end-to-end.
    """

    def test_single_config_succeeds(self):
        tmp = tempfile.mkdtemp(prefix="tm_grid_test_")
        try:
            cfg = _poisson_cfg(tmp)
            cfg["parameter_sets"] = {
                "lr_gen": [1.0e-4],
                "sig_degree": [4],
                "concentration_factor": [1.0],
                "hid_size_rep": [8],
                "use_teacher_forcing": [False],
                "terminal_anchor": ["free_endpoint"],
                "detach_cum_channel": [False],
                "mark_loss_weight": [1.0],
            }

            manager = _make_manager(cfg)
            results = manager.run()

            self.assertEqual(len(results.rows), 1, "Expected exactly 1 result row")
            row = results.rows[0]
            self.assertNotIn("error", row["metrics"], f"Training run produced error: {row['metrics'].get('error')}")

        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestGridSearchWithFailingConfig(unittest.TestCase):
    """When one config fails, run() records an error row and continues.

    Both calls to _train_single_config are mocked so no real training occurs :
    this behaviour is already exercised by TestGridSearch and TestDiagnosticOnlyMode.
    """

    def test_one_failing_config_produces_error_row_run_completes(self):
        tmp = tempfile.mkdtemp(prefix="tm_fail_test_")
        try:
            cfg = _poisson_cfg(tmp)
            # Two configs: lr_gen=[1e-4, 5e-4], all other params single-element.
            # Cartesian product = 2 configs (not 2^n).
            cfg["parameter_sets"] = {
                "lr_gen": [1.0e-4, 5.0e-4],
                "sig_degree": [4],
                "concentration_factor": [1.0],
                "hid_size_rep": [8],
                "use_teacher_forcing": [False],
                "terminal_anchor": ["free_endpoint"],
                "detach_cum_channel": [False],
            }

            manager = _make_manager(cfg)

            call_count = 0

            def patched(c):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {"model_name": "m1", "metrics": {"ED": 0.5}, "train_time": 1.0}
                raise RuntimeError("injected failure for test")

            with patch.object(manager, "_train_single_config", side_effect=patched):
                results = manager.run()

            self.assertEqual(len(results.rows), 2, "Expected 2 rows (1 ok + 1 error)")
            errors = [r for r in results.rows if "error" in r["metrics"]]
            clean = [r for r in results.rows if "error" not in r["metrics"]]
            self.assertEqual(len(errors), 1, "Expected exactly 1 error row")
            self.assertEqual(len(clean), 1, "Expected exactly 1 clean row")

        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestGridCustomFileNameWarning(unittest.TestCase):
    """A fixed custom_file_name across a multi-config grid collapses every config
    onto one model name (colliding checkpoints, lost sig-degree token). run() must
    warn about this up front; a single-config grid must stay quiet."""

    def _collect_collision_warnings(self, cfg):
        manager = _make_manager(cfg)  # custom_file_name_results="integration_test"
        tm_logger = logging.getLogger("test.paper_experiments.trainingmanager")
        messages = []

        class _Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        handler = _Capture(level=logging.WARNING)
        prev_level = tm_logger.level
        tm_logger.setLevel(logging.WARNING)
        tm_logger.addHandler(handler)
        try:
            with patch.object(
                manager,
                "_train_single_config",
                side_effect=lambda c: {"model_name": "m", "metrics": {"ED": 0.5}, "train_time": 1.0},
            ), patch.object(manager, "_evaluate_winner_on_test"):
                manager.run()
        finally:
            tm_logger.removeHandler(handler)
            tm_logger.setLevel(prev_level)
        return [m for m in messages if "share one name" in m]

    def test_multi_config_grid_with_custom_file_name_warns(self):
        tmp = tempfile.mkdtemp(prefix="tm_cfn_warn_")
        try:
            cfg = _poisson_cfg(tmp)
            cfg["parameter_sets"] = {**cfg["parameter_sets"], "sig_degree": [2, 3]}  # 2-config grid
            warnings = self._collect_collision_warnings(cfg)
            self.assertEqual(len(warnings), 1, warnings)
            self.assertIn("integration_test", warnings[0])
            self.assertIn("2-config grid", warnings[0])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_single_config_grid_does_not_warn(self):
        tmp = tempfile.mkdtemp(prefix="tm_cfn_nowarn_")
        try:
            cfg = _poisson_cfg(tmp)  # single-config grid
            self.assertEqual(self._collect_collision_warnings(cfg), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSigDegreeAblationGlue(unittest.TestCase):
    """The ablation glue must reuse the run's seed and GPU semantics, not defaults."""

    @staticmethod
    def _resolved_kwargs(cfg, refine_b=None):
        manager = TrainingManager(
            data_factory=lambda cfg_: None,
            model_factory=lambda *args, **kwargs: None,
            model_namer=lambda *args, **kwargs: "m",
            loss_metrics_fn=lambda *args, **kwargs: [],
            config=cfg,
        )
        with patch(
            "test.paper_experiments.trainingmanager.run_sig_degree_ablation_from_val_file"
        ) as mock_run:
            mock_run.return_value = SimpleNamespace(report_path="unused")
            manager._evaluate_sig_degree_ablation_on_test(
                "unused_val_tuning_path.txt",
                "unused_ablation_folder",
                refine_b,
            )
        return mock_run

    def test_ablation_uses_run_seed_gpu_index_bootstrap_count_and_results_dir(self):
        mock_run = self._resolved_kwargs({"seeds": [7], "gpu_id": [3], "n_bootstraps": 5, "version": "sigtpp"})
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["trainer_seed"], 7)
        self.assertEqual(kwargs["gpu_id"], 3)
        self.assertEqual(kwargs["n_bootstraps"], 5)
        self.assertEqual(kwargs["results_dir"], "unused_ablation_folder")
        # The manager must forward the exact val-tuning path it was given
        # straight through, never re-derive it (e.g. via a directory glob).
        self.assertEqual(mock_run.call_args.args[0], "unused_val_tuning_path.txt")

    def test_ablation_int_gpu_count_means_first_device(self):
        mock_run = self._resolved_kwargs({"seeds": [42], "gpu_id": 1, "n_bootstraps": 5, "version": "sigtpp"})
        self.assertEqual(mock_run.call_args.kwargs["gpu_id"], 0)

    def test_ablation_cpu_gpu_spec_resolves_to_none(self):
        mock_run = self._resolved_kwargs({"seeds": [42], "gpu_id": [], "n_bootstraps": 5, "version": "sigtpp"})
        self.assertIsNone(mock_run.call_args.kwargs["gpu_id"])


class TestSigDegreeAblationOutputFolders(unittest.TestCase):
    """run() must route ablation output to its own results_on_ablation folder,
    not the results_on_test_* folders the single-winner pass writes to."""

    def test_ablation_uses_dedicated_ablation_folders_not_test_folders(self):
        tmp = tempfile.mkdtemp(prefix="tm_ablation_folder_test_")
        try:
            cfg = _poisson_cfg(tmp)
            cfg["sig_degree_ablation"] = True
            manager = _make_manager(cfg)

            def fake_train(_cfg):
                return {"model_name": "m1", "metrics": {"ED": 0.5}, "train_time": 1.0}

            with patch.object(manager, "_train_single_config", side_effect=fake_train), patch.object(
                manager, "_evaluate_sig_degree_ablation_on_test"
            ) as mock_ablation:
                manager.run()

            mock_ablation.assert_called_once()
            val_tuning_path, ablation_arg, _refine_b = mock_ablation.call_args.args
            self.assertIn("results_on_ablation", ablation_arg)
            self.assertNotIn("results_on_test_txt", ablation_arg)
            self.assertNotIn("results_on_test_npz", ablation_arg)
            # Regression: the manager must pass the exact file its own
            # ExperimentResults.save() call wrote, not a folder to re-glob.
            self.assertTrue(os.path.isfile(val_tuning_path), val_tuning_path)
            self.assertIn("val_tuning", os.path.basename(val_tuning_path))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestMultiseedSubrunOutputFolder(unittest.TestCase):
    """A multiseed sub-run (cfg carries _multiseed_seed_tag) must not write any of
    its own per-seed raw output files (val_tuning txt, sig_degree ablation txt/npz,
    winner-on-test report): those are fully superseded by the cross-seed aggregate
    files ``run_experiment_config`` writes afterwards from in-memory rows, and by
    the cross-seed finalization step's own winner-on-test pass."""

    def test_multiseed_subrun_skips_all_per_seed_raw_output_writes(self):
        tmp = tempfile.mkdtemp(prefix="tm_multiseed_folder_test_")
        try:
            cfg = _poisson_cfg(tmp)
            cfg["sig_degree_ablation"] = True
            cfg["_multiseed_seed_tag"] = 42
            manager = _make_manager(cfg)

            def fake_train(_cfg):
                return {"model_name": "m1", "metrics": {"ED": 0.5}, "train_time": 1.0}

            with patch.object(manager, "_train_single_config", side_effect=fake_train), patch.object(
                manager, "_evaluate_winner_on_test"
            ) as mock_winner, patch.object(
                manager, "_evaluate_sig_degree_ablation_on_test"
            ) as mock_ablation, patch(
                "test.paper_experiments.trainingmanager.ExperimentResults.save"
            ) as mock_save:
                results = manager.run()

            # Multi-seed sub-runs defer the winner-on-test pass to the
            # cross-seed finalization step in run_experiment_config (the
            # per-seed local winner may not be the config that wins across all
            # seeds) -- see TrainingManager.run()'s is_multiseed_subrun guard.
            mock_winner.assert_not_called()

            # This seed's own local val_tuning table and sig_degree ablation
            # (even though sig_degree_ablation=True) are no longer written: the
            # former is redundant with multiseed_per_seed/multiseed_summary
            # (built from result.rows in memory, not this file), and the latter
            # depends on that file existing on disk.
            mock_save.assert_not_called()
            mock_ablation.assert_not_called()
            self.assertIsNone(results.final_test_row)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
