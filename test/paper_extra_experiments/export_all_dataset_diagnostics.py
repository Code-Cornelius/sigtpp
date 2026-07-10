"""
Export diagnostic reports (PDF) for every dataset used in training.

Parameters are sourced from:
  - Synthetic: settings/<dataset>.py data_factory + configs/<dataset>/experiment.yaml
  - Real: data modules have no configurable params at construction time

Synthetic caches are regenerated on demand under the poisson_<num_marks>_marks_* prefix.

Run:
    python -c "import sys; sys.path.insert(0, 'src'); exec(open('test/paper_extra_experiments/export_all_dataset_diagnostics.py').read())"
"""

import logging
import os

import numpy as np

from config import ROOT_DIR
from src.logger.init_logger import set_config_logging

set_config_logging()
logger = logging.getLogger(__name__)

from src.diagnostics.dataset_diagnostics import export_dataset_report_merged

_DIAG_BASE = os.path.join(ROOT_DIR, "test/paper_extra_experiments/out/dataset_diagnostics")

# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------
from test.paper_experiments.data.synthetic.poisson.poisson_dataset import PoissonDataModule
from test.paper_experiments.data.synthetic.hawkes.hawkes_dataset import HawkesDataModule
from test.paper_experiments.data.synthetic.hawkes.hawkes_3x3_dataset import Hawkes3x3DataModule

# ---------------------------------------------------------------------------
# EasyTPP
# ---------------------------------------------------------------------------
from test.paper_experiments.data.real.easytpp.taxi_dataset import TaxiDataModule
from test.paper_experiments.data.real.easytpp.stackoverflow_dataset import StackOverflowDataModule
from test.paper_experiments.data.real.easytpp.taobao_dataset import TaobaoDataModule
from test.paper_experiments.data.real.easytpp.earthquake_dataset import EarthquakeDataModule

# ---------------------------------------------------------------------------
# EditTPP
# ---------------------------------------------------------------------------
from test.paper_experiments.data.real.editpp.yelp_mississauga_dataset import YelpMississaugaDataModule

# ---------------------------------------------------------------------------
# Dataset registry
# All params match the data_factory in settings/<dataset>.py + experiment.yaml.
# ---------------------------------------------------------------------------

# Each entry: (display_name, factory, output_slug)
# output_slug overrides the directory name under dataset_diagnostics/.
# Required for datasets sharing a class (e.g. all four PoissonDataModule variants)
# because _dataset_report_slug falls back to type(dm).__name__ when DATASET_NAME is absent.
DATASETS = [
    # --- Synthetic HP (poisson_three_marks) ---
    # settings/poisson.py: data_size=2_000, seed=42, use_IHP_or_HP=False
    # configs/poisson_three_marks/experiment.yaml: num_marks=3, mark_probs=[0.7, 0.05, 0.25]
    (
        "HP 3-marks",
        lambda: PoissonDataModule(
            data_size=2_000,
            seed=42,
            use_IHP_or_HP=False,
            num_marks=3,
            mark_probs=np.array([0.7, 0.05, 0.25]),
        ),
        "hp_three_marks",
    ),
    # --- Synthetic IHP (inh_poisson_three_marks) ---
    # settings/inh_poisson.py: data_size=5_000, seed=42, use_IHP_or_HP=True
    # configs/inh_poisson_three_marks/experiment.yaml: num_marks=3, mark_probs=[0.7, 0.05, 0.25]
    (
        "IHP 3-marks",
        lambda: PoissonDataModule(
            data_size=5_000,
            seed=42,
            use_IHP_or_HP=True,
            num_marks=3,
            mark_probs=np.array([0.7, 0.05, 0.25]),
        ),
        "ihp_three_marks",
    ),
    # --- Synthetic Hawkes (hawkes) ---
    # settings/hawkes.py: data_size=10_000, seed=42, mu=0.3, alpha=0.4, beta=1.0
    (
        "Hawkes",
        lambda: HawkesDataModule(data_size=10_000, seed=42, mu=0.3, alpha=0.4, beta=1.0),
        None,
    ),
    # --- Synthetic Hawkes 3x3 (hawkes_3x3) ---
    # settings/hawkes_3x3.py: data_size=2_000, seed=42
    # configs/hawkes_3x3/experiment.yaml: time_max=15.0, baseline, adjacency, decays
    (
        "Hawkes 3x3",
        lambda: Hawkes3x3DataModule(
            data_size=2_000,
            seed=42,
            time_max=15.0,
            baseline=np.array([0.5, 0.5, 0.5]),
            adjacency=np.array([[0.5, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]]),
            decays=np.ones((3, 3)),
        ),
        None,
    ),
    # --- EasyTPP ---
    ("Taxi", TaxiDataModule, None),
    ("StackOverflow", StackOverflowDataModule, None),
    ("Taobao", TaobaoDataModule, None),
    ("Earthquake", EarthquakeDataModule, None),
    # --- EditTPP ---
    ("YelpMississauga", YelpMississaugaDataModule, None),
]

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

n_total = len(DATASETS)
failures = []
for idx, (name, factory, slug) in enumerate(DATASETS, start=1):
    logger.critical("=" * 60)
    logger.critical("[%d/%d] Loading: %s", idx, n_total, name)
    logger.critical("=" * 60)
    try:
        dm = factory()
        if slug is not None:
            dm.DATASET_NAME = slug
        _slug = getattr(dm, "DATASET_NAME", type(dm).__name__.lower())
        logger.critical("[%d/%d] Exporting: %s  →  %s", idx, n_total, name, _slug)
        export_dataset_report_merged(
            dm,
            output_dir=os.path.join(_DIAG_BASE, _slug),
            fig_format="pdf",
            preview=False,
        )
        logger.critical("[%d/%d] Done: %s", idx, n_total, name)
    except Exception as exc:
        logger.exception("FAILED: %s — %s", name, exc)
        failures.append(name)

logger.critical("=" * 60)
if failures:
    logger.warning("FAILED datasets (%d): %s", len(failures), failures)
else:
    logger.critical("All %d datasets exported successfully.", n_total)
logger.critical("Reports saved under: %s", _DIAG_BASE)
