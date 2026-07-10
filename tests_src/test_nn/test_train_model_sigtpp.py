"""
Integration test: TrainingManager end-to-end with the SigTPP model on Poisson data.

Trains for a minimal number of epochs to verify the full pipeline (data loading,
model construction, training loop, checkpoint saving, test phase) runs without error.
This is intentionally fast â€“ correctness of metrics is covered by unit tests.
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import logging
import shutil
import tempfile
import unittest

logger = logging.getLogger(__name__)

from src.utils.utils_dict import verbose_get
from test.paper_experiments.training_helpers import load_experiment_config
from test.paper_experiments.trainingmanager import TrainingManager
from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY


class TestTrainModelSigtpp(unittest.TestCase):
    """TrainingManager smoke-test: SigTPP training must complete without error."""

    def test_sigtpp_train_model_runs_without_error(self):

        cfg = load_experiment_config("poisson_three_marks/sigtpp_test.yaml")

        tmp_dir = tempfile.mkdtemp(prefix="train_sigtpp_test_")
        try:
            # -------------------------------------------------------------------
            # Override config for speed and isolation
            # -------------------------------------------------------------------
            cfg["seeds"] = [42]
            cfg["epochs"] = 3  # minimal training so a checkpoint is saved
            cfg["period_log"] = 1  # validate every epoch so checkpoint is written
            cfg["patience"] = 9999  # disable early stopping
            cfg["diagnostic_only"] = False
            cfg["gpu_id"] = []  # CPU only
            cfg["verbose"] = False
            cfg["output_dir"] = tmp_dir  # isolate outputs from real experiment dirs

            # Single, minimal parameter set (avoids the large grid in sigtpp_test.yaml)
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

            manager = TrainingManager(
                **verbose_get(
                    EXPERIMENT_REGISTRY,
                    verbose_get(cfg, "experiment_type", logger, None),
                    logger,
                    None,
                ),
                config=cfg,
                custom_file_name_results="sigtpp_integration_test",
            )
            results = manager.run()

            # -------------------------------------------------------------------
            # Assertions: run must produce at least one result row without error
            # -------------------------------------------------------------------
            self.assertIsNotNone(results, "TrainingManager.run() returned None")
            self.assertEqual(len(results.rows), 1, "Expected exactly 1 result row")
            row = results.rows[0]
            metrics = row.get("metrics", {})
            self.assertNotIn(
                "error",
                metrics,
                f"SigTPP run produced an error: {metrics.get('error')}",
            )
            self.assertTrue(
                len(metrics) > 0,
                "SigTPP run returned an empty metrics dict: test phase may not have completed",
            )

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    unittest.main()
