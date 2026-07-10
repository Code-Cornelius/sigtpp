"""End-to-end seed reproducibility lock.

Two TrainingManager runs with identical config and the same seed must produce
identical test-phase metrics. This catches the regression class where a generator
silently uses unseeded global RNG (e.g. the ``np.random.rand`` leak previously
fixed in ``src/generators/ihp.py``): in that scenario the bootstrap helper would
still produce paired indices, but the underlying model / data path would drift
between runs and the metric diff would surface here.
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import logging
import math
import shutil
import tempfile
import unittest

logger = logging.getLogger(__name__)

from src.utils.utils_dict import verbose_get
from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY
from test.paper_experiments.training_helpers import load_experiment_config
from test.paper_experiments.trainingmanager import TrainingManager


def _poisson_seed_cfg(tmp_dir: str) -> dict:
    cfg = load_experiment_config("poisson_three_marks/sigtpp_test.yaml")
    cfg["epochs"] = 1
    cfg["period_log"] = 1
    cfg["patience"] = 9999
    cfg["diagnostic_only"] = False
    cfg["gpu_id"] = []  # CPU-only: deterministic across runs
    cfg["verbose"] = False
    cfg["output_dir"] = tmp_dir
    cfg["seeds"] = [42]
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


def _run_one(tmp_dir: str) -> dict:
    cfg = _poisson_seed_cfg(tmp_dir)
    manager = TrainingManager(
        **verbose_get(EXPERIMENT_REGISTRY, cfg["experiment_type"], logger, None),
        config=cfg,
        custom_file_name_results="seed_repro_test",
    )
    results = manager.run()
    assert len(results.rows) == 1, f"Expected 1 row, got {len(results.rows)}"
    assert "error" not in results.rows[0]["metrics"], results.rows[0]["metrics"].get("error")
    return results.rows[0]["metrics"]


class TestSeedReproducibility(unittest.TestCase):
    """Locks the seed -> identical-metrics contract end-to-end."""

    def test_two_runs_same_seed_produce_identical_test_metrics(self):
        tmp_a = tempfile.mkdtemp(prefix="seed_repro_a_")
        tmp_b = tempfile.mkdtemp(prefix="seed_repro_b_")
        try:
            metrics_a = _run_one(tmp_a)
            metrics_b = _run_one(tmp_b)
        finally:
            shutil.rmtree(tmp_a, ignore_errors=True)
            shutil.rmtree(tmp_b, ignore_errors=True)

        # Compare only numeric, non-NaN keys present in both runs. ``train_time`` and
        # other wall-clock fields are excluded since they reflect machine load.
        skip_keys = {"train_time_mean", "train_time", "error"}
        common_keys = sorted(set(metrics_a.keys()) & set(metrics_b.keys()) - skip_keys)
        self.assertTrue(common_keys, "No comparable metric keys between the two runs.")

        mismatches: list = []
        compared = 0
        for key in common_keys:
            va, vb = metrics_a[key], metrics_b[key]
            try:
                fa, fb = float(va), float(vb)
            except (TypeError, ValueError):
                continue
            if math.isnan(fa) and math.isnan(fb):
                continue
            compared += 1
            # CPU-only training with seed_everything(workers=True) is bit-exact in
            # principle; allow a small tolerance for accumulated float64 round-off
            # in metric reductions across NumPy versions.
            if not math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-8):
                mismatches.append((key, fa, fb))

        self.assertGreater(compared, 0, "No numeric metric was actually compared between runs.")
        self.assertEqual(
            mismatches,
            [],
            f"Seed reproducibility broken: {len(mismatches)} metric(s) diverged across runs:\n"
            + "\n".join(f"  {k}: {a!r} vs {b!r}" for k, a, b in mismatches),
        )


if __name__ == "__main__":
    unittest.main()
