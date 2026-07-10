"""
Print summary statistics for all datasets used in the paper experiments.

For each dataset, one line is logged:
    DATASET, K=N, ev_tr=X, ev_val=X, ev_te=X, seq_tr=X, seq_val=X, seq_te=X, len_min=X, len_mean=X.X, len_max=X

Events = real events only (anchor excluded).  Sequence length = events per sequence.

Run:
    python -c "import sys; sys.path.insert(0, 'src'); exec(open('test/paper_extra_experiments/report_dataset_statistics.py').read())"
"""

import logging
from typing import Callable, Tuple

import numpy as np
import torch

from src.logger.init_logger import set_config_logging

set_config_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

# Single source of truth for column widths used in both header and rows.
_NAME_W = 35
_K_W = 2
_EV_W = 7
_SEQ_W = 5
_LEN_MIN_W = 4
_LEN_MEAN_W = 7
_LEN_MAX_W = 5

_ROW_FMT = (
    f"{{name:<{_NAME_W}s}}  K={{K:<{_K_W}d}}  "
    f"ev  tr={{tr_ev:>{_EV_W}d}}  val={{va_ev:>{_EV_W}d}}  te={{te_ev:>{_EV_W}d}}  "
    f"seq  tr={{tr_seqs:>{_SEQ_W}d}}  val={{va_seqs:>{_SEQ_W}d}}  te={{te_seqs:>{_SEQ_W}d}}  "
    f"len  min={{l_min:>{_LEN_MIN_W}d}}  mean={{l_mean:>{_LEN_MEAN_W}.1f}}  max={{l_max:>{_LEN_MAX_W}d}}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_stats(lens: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
    """Return (n_seqs, n_events, event_lengths) for one split.

    lens: (N,) long — sequence lengths *including* the anchor at position 0.
    """
    # exclude anchor; clamp at 0 so anchor-only sequences (lens=0) don't show as -1 events
    event_lens = (lens - 1).clamp_min(0).float()
    n_seqs = int(lens.shape[0])
    n_events = int(event_lens.sum().item())
    return n_seqs, n_events, event_lens


def print_stats(name: str, dm) -> None:
    tr_seqs, tr_ev, tr_lens = _split_stats(dm.train_in_len)
    va_seqs, va_ev, va_lens = _split_stats(dm.val_in_len)
    te_seqs, te_ev, te_lens = _split_stats(dm.test_in_len)

    all_lens = torch.cat([tr_lens, va_lens, te_lens])
    l_min = int(all_lens.min().item())
    l_mean = float(all_lens.mean().item())
    l_max = int(all_lens.max().item())

    logger.info(
        _ROW_FMT.format(
            name=name,
            K=int(dm.num_marks),
            tr_ev=tr_ev,
            va_ev=va_ev,
            te_ev=te_ev,
            tr_seqs=tr_seqs,
            va_seqs=va_seqs,
            te_seqs=te_seqs,
            l_min=l_min,
            l_mean=l_mean,
            l_max=l_max,
        )
    )


def try_load(name: str, factory: Callable) -> None:
    try:
        print_stats(name, factory())
    except Exception:
        logger.exception("%-35s ERROR", name)


# ---------------------------------------------------------------------------
# EasyTPP datasets
# ---------------------------------------------------------------------------

from test.paper_experiments.data.real.easytpp.taxi_dataset import TaxiDataModule
from test.paper_experiments.data.real.easytpp.stackoverflow_dataset import StackOverflowDataModule
from test.paper_experiments.data.real.easytpp.taobao_dataset import TaobaoDataModule
from test.paper_experiments.data.real.easytpp.earthquake_dataset import EarthquakeDataModule

# EditTPP datasets
from test.paper_experiments.data.real.editpp.yelp_mississauga_dataset import YelpMississaugaDataModule
# Synthetic datasets
from test.paper_experiments.data.synthetic.poisson.poisson_dataset import PoissonDataModule
from test.paper_experiments.data.synthetic.hawkes.hawkes_dataset import HawkesDataModule
from test.paper_experiments.data.synthetic.hawkes.hawkes_3x3_dataset import Hawkes3x3DataModule


logger.info(
    "%-35s  %-4s  %-40s  %-30s  %s",
    "DATASET",
    "K",
    "EVENTS (tr / val / te)",
    "SEQ (tr / val / te)",
    "SEQ LEN (min / mean / max)",
)
logger.info("-" * 160)

# ---- Retained real datasets ----
try_load("taxi", TaxiDataModule)
try_load("stackoverflow", StackOverflowDataModule)
try_load("taobao", TaobaoDataModule)
try_load("earthquake", EarthquakeDataModule)
try_load("yelp_mississauga", YelpMississaugaDataModule)
# ---- Synthetic: Poisson HP ----
try_load("poisson_hp_1mark", lambda: PoissonDataModule(data_size=2_000, seed=42, use_IHP_or_HP=False, num_marks=1))
try_load("poisson_hp_3marks", lambda: PoissonDataModule(data_size=2_000, seed=42, use_IHP_or_HP=False, num_marks=3))

# ---- Synthetic: Poisson IHP ----
try_load("poisson_ihp_1mark", lambda: PoissonDataModule(data_size=5_000, seed=42, use_IHP_or_HP=True, num_marks=1))
try_load("poisson_ihp_3marks", lambda: PoissonDataModule(data_size=5_000, seed=42, use_IHP_or_HP=True, num_marks=3))

# ---- Synthetic: Hawkes ----
try_load("hawkes", lambda: HawkesDataModule(data_size=10_000, seed=42, mu=0.3, alpha=0.4, beta=1.0))

# ---- Synthetic: Hawkes 3x3 ----
try_load(
    "hawkes_3x3",
    lambda: Hawkes3x3DataModule(
        data_size=2_000,
        seed=42,
        time_max=15.0,
        baseline=np.array([0.5, 0.5, 0.5]),
        adjacency=np.array([[0.5, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]]),
        decays=np.ones((3, 3)),
    ),
)

