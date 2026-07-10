import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no Tk/display required
import pytest
pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built
from matplotlib import pyplot as plt

from config import OUT_FILE_NAME
from src.utils.utils_dict import verbose_get
from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY
from test.paper_experiments.training_helpers import load_experiment_config
from test.paper_experiments.training_runner import run_experiment_config
from test.paper_experiments.trainingmanager import TrainingManager


logger = logging.getLogger(__name__)


_CONFIG_ROOT = Path("test/paper_experiments/configs")
_EXPECTED_SMOKE_CASES: Set[Tuple[str, str]] = {
    ("earthquake", "deter"),
    ("earthquake", "gamma"),
    ("earthquake", "ddpm"),
    ("earthquake", "sigtpp"),
    ("earthquake", "vae"),
    ("earthquake", "wgan"),
    ("hawkes", "deter"),
    ("hawkes", "gamma"),
    ("hawkes", "ddpm"),
    ("hawkes", "sigtpp"),
    ("hawkes", "vae"),
    ("hawkes", "wgan"),
    ("hawkes_3x3", "deter"),
    ("hawkes_3x3", "gamma"),
    ("hawkes_3x3", "ddpm"),
    ("hawkes_3x3", "sigtpp"),
    ("hawkes_3x3", "vae"),
    ("hawkes_3x3", "wgan"),
    ("poisson_three_marks", "deter"),
    ("poisson_three_marks", "gamma"),
    ("poisson_three_marks", "ddpm"),
    ("poisson_three_marks", "sigtpp"),
    ("poisson_three_marks", "vae"),
    ("poisson_three_marks", "wgan"),
    ("stackoverflow", "deter"),
    # ("stackoverflow", "gamma"),
    # ("stackoverflow", "ddpm"),
    # ("stackoverflow", "sigtpp"),
    # ("stackoverflow", "vae"),
    # ("stackoverflow", "wgan"),
    ("taobao", "deter"),
    # ("taobao", "gamma"),
    # ("taobao", "ddpm"),
    # ("taobao", "sigtpp"),
    # ("taobao", "vae"),
    # ("taobao", "wgan"),
    ("taxi", "deter"),
    # ("taxi", "gamma"),
    # ("taxi", "ddpm"),
    # ("taxi", "sigtpp"),
    # ("taxi", "vae"),
    # ("taxi", "wgan"),
    ("yelp_mississauga", "deter"),
    ("yelp_mississauga", "gamma"),
    ("yelp_mississauga", "ddpm"),
    ("yelp_mississauga", "sigtpp"),
    ("yelp_mississauga", "vae"),
    ("yelp_mississauga", "wgan"),
    ("inh_poisson_three_marks", "deter"),
    ("inh_poisson_three_marks", "gamma"),
    ("inh_poisson_three_marks", "ddpm"),
    ("inh_poisson_three_marks", "sigtpp"),
    ("inh_poisson_three_marks", "vae"),
    ("inh_poisson_three_marks", "wgan"),
}
# Datasets excluded entirely from the smoke matrix (none at present).
_EXCLUDED_SMOKE_DATASETS: set = set()
# (dataset, version) pairs that have configs on disk but are deliberately excluded
# from smoke testing (e.g. data rarely available in CI / slow real-world datasets).
_EXCLUDED_SMOKE_CASES: Set[Tuple[str, str]] = {
    ("stackoverflow", "gamma"),
    ("stackoverflow", "ddpm"),
    ("stackoverflow", "sigtpp"),
    ("stackoverflow", "vae"),
    ("stackoverflow", "wgan"),
    ("taobao", "gamma"),
    ("taobao", "ddpm"),
    ("taobao", "sigtpp"),
    ("taobao", "vae"),
    ("taobao", "wgan"),
    ("taxi", "gamma"),
    ("taxi", "ddpm"),
    ("taxi", "sigtpp"),
    ("taxi", "vae"),
    ("taxi", "wgan"),
    # sigtpp is skipped under the smoke-forced tiny sig params: this dataset's long
    # sequences (avg ~296) make the standardised-path signature decay so the largest
    # usable degree caps at 2, below the minimum of 3 (relative_sig_degree=0). This is a
    # property of the data, not a bug, so the case is excluded rather than expected.
}
_SYNTHETIC_EXPERIMENTS = {
    "poisson_three_marks",
    "inh_poisson_three_marks",
    "hawkes",
    "hawkes_3x3",
}
_SMOKE_DATA_SIZE = 64


def _parse_env_csv(name: str) -> Set[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


_SMOKE_DATASET_FILTER = _parse_env_csv("SMOKE_DATASETS")
_SMOKE_MODEL_FILTER = _parse_env_csv("SMOKE_MODELS")
_SMOKE_CASE_FILTER = _parse_env_csv("SMOKE_CASES")


def _smoke_filter_is_active() -> bool:
    return bool(_SMOKE_DATASET_FILTER or _SMOKE_MODEL_FILTER or _SMOKE_CASE_FILTER)


def _matches_smoke_filters(dataset: str, version: str) -> bool:
    case = _format_case((dataset, version))
    if _SMOKE_DATASET_FILTER and dataset not in _SMOKE_DATASET_FILTER:
        return False
    if _SMOKE_MODEL_FILTER and version not in _SMOKE_MODEL_FILTER:
        return False
    if _SMOKE_CASE_FILTER and case not in _SMOKE_CASE_FILTER:
        return False
    return True


def _format_case(case: Tuple[str, str]) -> str:
    dataset, version = case
    return f"{dataset}/{version}"


def _discover_cases() -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Return smoke cases and coverage errors for the expected dataset/model matrix.

    Discovery is the cross-product of EXPERIMENT_REGISTRY x configs/models/*_test.yaml,
    minus excluded datasets and excluded (dataset, version) pairs.
    """
    pairs: Dict[Tuple[str, str], str] = {}
    invalid_configs: List[str] = []
    for dataset in sorted(EXPERIMENT_REGISTRY):
        if dataset in _EXCLUDED_SMOKE_DATASETS:
            continue
        for cfg_path in sorted(_CONFIG_ROOT.glob("*_test.yaml")):
            rel_path = f"{dataset}/{cfg_path.name}"
            cfg = load_experiment_config(rel_path)
            version = str(cfg.get("version", "")).strip().lower()
            if not version:
                invalid_configs.append(f"{rel_path}: missing version")
                continue
            if (dataset, version) in _EXCLUDED_SMOKE_CASES:
                continue
            if not _matches_smoke_filters(dataset, version):
                continue
            pairs[(dataset, version)] = rel_path

    discovered_keys = set(pairs)
    problems = []
    if not _smoke_filter_is_active():
        missing_cases = sorted(_EXPECTED_SMOKE_CASES - discovered_keys)
        unexpected_cases = sorted(discovered_keys - _EXPECTED_SMOKE_CASES)
        if missing_cases:
            problems.append("Missing expected smoke cases: " + ", ".join(_format_case(case) for case in missing_cases))
        if unexpected_cases:
            problems.append(
                "Unexpected smoke cases discovered: " + ", ".join(_format_case(case) for case in unexpected_cases)
            )
    if invalid_configs:
        problems.append("Invalid smoke configs: " + "; ".join(sorted(invalid_configs)))

    return [(ds, version, rel) for (ds, version), rel in sorted(pairs.items())], problems


_SMOKE_LR = 1e-5  # safe lr: prevents blow-up in 1 step
_SMOKE_SMALL_PARAMS = {
    # model size → minimum meaningful values
    "hid_size_rep": 4,
    "hid_size_rnn": 4,
    "hid_size": 4,
    "sig_degree": 2,
    "relative_sig_degree": 0,
    "latent_dim": 4,
    # lr → always override to the safe value
    "lr": _SMOKE_LR,
    "lr_gen": _SMOKE_LR,
    "lr_disc": _SMOKE_LR,
}


def _collapse_parameter_sets_to_singleton(cfg: dict) -> None:
    params = cfg.get("parameter_sets", {})
    singleton = {}
    for k, v in params.items():
        if isinstance(v, list):
            if not v:
                continue
            singleton[k] = [v[0]]
        else:
            singleton[k] = [v]
    # Force a single train/val batch to make "one iteration" deterministic.
    singleton["batch_size"] = [1_000_000_000]
    # Override size and lr params so smoke is fast and numerically stable.
    for k, forced in _SMOKE_SMALL_PARAMS.items():
        if k in singleton:
            singleton[k] = [forced]
    cfg["parameter_sets"] = singleton


def _is_data_unavailable_error(error: str) -> bool:
    err = (error or "").lower()
    download_patterns = (
        "couldn't reach",
        "connectionerror",
        "max retries exceeded",
        "name or service not known",
        "temporary failure in name resolution",
    )
    missing_data_patterns = (
        "easytpp/",
        "dataset not found",
        "file not found",
        "no such file or directory",
    )
    return any(p in err for p in download_patterns) or any(p in err for p in missing_data_patterns)


@pytest.mark.smoke
class TestSmokeDatasetModelMatrix(unittest.TestCase):
    """One smoke test case per (dataset, model version) pair."""

    maxDiff = None

    def test_expected_smoke_matrix_is_current(self) -> None:
        self.assertFalse(_DISCOVERY_ERRORS, "\n".join(_DISCOVERY_ERRORS))

    def _run_case(self, dataset: str, version: str, config_rel_path: str) -> None:
        tmp = tempfile.mkdtemp(prefix=f"smoke_{dataset}_{version}_")
        try:
            cfg = load_experiment_config(config_rel_path)
            cfg["epochs"] = 1
            cfg["period_log"] = 1
            cfg["period_plotting_in_logs"] = 999_999  # no plotting during validation
            cfg["patience"] = 1
            cfg["diagnostic_only"] = False
            cfg["gpu_id"] = []
            cfg["verbose"] = False
            cfg["output_dir"] = tmp
            cfg["skip_diagnostics"] = True  # skip test phase + sample generation + plots
            # The matrix runs the single-seed manager.run() path and only checks that one
            # training step runs error-free; seed *count* is irrelevant to that. Normalize any
            # multi-seed config down to its first seed, so any *_test.yaml is free to declare
            # N seeds without breaking the matrix. Real multi-seed expansion is covered by
            # TestSmokeSeedAndBootstrap.
            seeds = cfg.get("seeds") or [42]
            cfg["seeds"] = [int(seeds[0])]
            _collapse_parameter_sets_to_singleton(cfg)
            if dataset in _SYNTHETIC_EXPERIMENTS:
                cfg["smoke_data_size"] = _SMOKE_DATA_SIZE

            manager = TrainingManager(
                **verbose_get(EXPERIMENT_REGISTRY, dataset, logger, None),
                config=cfg,
                custom_file_name_results=f"smoke_{dataset}_{version}",
            )
            results = manager.run()

            self.assertEqual(
                len(results.rows),
                1,
                f"{dataset}/{version}: expected exactly one result row, got {len(results.rows)}",
            )
            row = results.rows[0]
            error = row.get("metrics", {}).get("error")
            if error is not None and _is_data_unavailable_error(error):
                self.skipTest(f"{dataset}/{version}: data unavailable in this environment ({error})")
            self.assertNotIn(
                "error",
                row.get("metrics", {}),
                f"{dataset}/{version} failed with error: {error}",
            )
        finally:
            plt.close("all")
            shutil.rmtree(tmp, ignore_errors=True)


def _attach_smoke_case_tests() -> None:
    for dataset, version, rel_path in _DISCOVERED_CASES:
        test_name = f"test_smoke__{dataset}__{version}"

        def _test(self, d=dataset, v=version, p=rel_path):
            return self._run_case(d, v, p)

        _test.__name__ = test_name
        setattr(TestSmokeDatasetModelMatrix, test_name, _test)


_DISCOVERED_CASES, _DISCOVERY_ERRORS = _discover_cases()
_attach_smoke_case_tests()


@pytest.mark.smoke
class TestSmokeSeedAndBootstrap(unittest.TestCase):
    """Orthogonal smoke checks for the bootstrap loop and the multi-seed branch.

    ``TestSmokeDatasetModelMatrix`` runs each (dataset, model) pair with
    ``skip_diagnostics=True`` and a single seed, so it covers neither the
    bootstrap path (``compute_test_metrics_bootstrapped``) nor the multi-seed
    expansion in ``run_experiment_config``. These tests exercise the bootstrap
    path once and the multi-seed expansion for both the cheapest baseline
    (``deter``) and the headline GAN (``sigtpp``) on the cheapest synthetic.
    """

    _SMOKE_DATASET = "poisson_three_marks"
    _SMOKE_VERSION = "deter"
    _SMOKE_CONFIG = f"{_SMOKE_DATASET}/deter_test.yaml"

    def _base_cfg(self, tmp: str, config_rel: Optional[str] = None) -> dict:
        cfg = load_experiment_config(config_rel or self._SMOKE_CONFIG)
        cfg["epochs"] = 1
        cfg["period_log"] = 1
        cfg["period_plotting_in_logs"] = 999_999
        cfg["patience"] = 1
        cfg["diagnostic_only"] = False
        cfg["gpu_id"] = []
        cfg["verbose"] = False
        cfg["output_dir"] = tmp
        cfg["smoke_data_size"] = _SMOKE_DATA_SIZE
        _collapse_parameter_sets_to_singleton(cfg)
        return cfg

    def test_smoke_bootstrap_writes_mean_and_std_columns(self) -> None:
        tmp = tempfile.mkdtemp(prefix="smoke_bootstrap_")
        try:
            cfg = self._base_cfg(tmp)
            cfg["skip_diagnostics"] = False  # test phase is what runs the bootstrap loop
            cfg["n_bootstraps"] = 3  # below LOCAL_BOOTSTRAP_CAP, so no override applies

            manager = TrainingManager(
                **verbose_get(EXPERIMENT_REGISTRY, self._SMOKE_DATASET, logger, None),
                config=cfg,
                custom_file_name_results=f"smoke_bootstrap_{self._SMOKE_VERSION}",
            )
            results = manager.run()

            self.assertEqual(len(results.rows), 1)
            metrics = results.rows[0].get("metrics", {})
            error = metrics.get("error")
            if error is not None and _is_data_unavailable_error(error):
                self.skipTest(f"bootstrap smoke: data unavailable ({error})")
            self.assertNotIn("error", metrics, f"bootstrap smoke failed: {error}")
            mean_keys = [k for k in metrics if k.endswith("_mean")]
            std_keys = [k for k in metrics if k.endswith("_std")]
            self.assertGreater(len(mean_keys), 0, "expected bootstrap *_mean keys in metrics")
            self.assertGreater(len(std_keys), 0, "expected bootstrap *_std keys in metrics")
        finally:
            plt.close("all")
            shutil.rmtree(tmp, ignore_errors=True)

    def _assert_multiseed_outputs(self, version: str, config_rel: str) -> None:
        """Run the multi-seed expansion for one model and assert its summary outputs.

        Routes through ``run_experiment_config`` (not the single-seed
        ``manager.run()``) so the per-seed loop and ``write_multiseed_outputs``
        aggregation in :mod:`training_runner` are actually exercised.
        """
        tmp = tempfile.mkdtemp(prefix=f"smoke_multiseed_{version}_")
        try:
            cfg = self._base_cfg(tmp, config_rel)
            cfg["skip_diagnostics"] = True  # this branch is about the seed loop, not the test phase
            cfg["seeds"] = [42, 43]

            run_experiment_config(cfg)

            results_dir = os.path.join(tmp, OUT_FILE_NAME, self._SMOKE_DATASET, "results_on_multiseed")
            self.assertTrue(os.path.isdir(results_dir), f"{version}: missing results dir: {results_dir}")
            entries = os.listdir(results_dir)
            by_seed = [f for f in entries if "multiseed_per_seed" in f]
            summary = [f for f in entries if "multiseed_summary" in f]
            self.assertEqual(len(by_seed), 1, f"{version}: expected one by-seed file, got {by_seed}")
            self.assertEqual(len(summary), 1, f"{version}: expected one seed-summary file, got {summary}")
            with open(os.path.join(results_dir, summary[0])) as f:
                header = f.readline()
            self.assertIn("hist_it_seed_mean", header)
            self.assertIn("hist_it_seed_std", header)
            self.assertIn("hist_it_seed_n_valid", header)
        finally:
            plt.close("all")
            shutil.rmtree(tmp, ignore_errors=True)

    def test_smoke_multiseed_writes_seed_summary(self) -> None:
        """Multi-seed expansion on the cheapest baseline (deter)."""
        self._assert_multiseed_outputs("deter", self._SMOKE_CONFIG)

    def test_smoke_multiseed_sigtpp_writes_seed_summary(self) -> None:
        """Multi-seed expansion on the headline GAN (sigtpp)."""
        self._assert_multiseed_outputs("sigtpp", f"{self._SMOKE_DATASET}/sigtpp_test.yaml")
