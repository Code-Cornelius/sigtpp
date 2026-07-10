"""
Signature degree diagnostic.

For each dataset, instantiates a minimal ArchitectureOneToOne so the
training preprocessing runs exactly as during training, then reads
train_sig_loss_seqs directly and reports per-degree signature statistics.

Key columns
-----------
l2_mean     : L2-norm of E[Sig_k(X)] across N sequences.
std_median  : median per-term std at degree k — near-zero means the terms
              don't vary across sequences.
n_dead_1e8  : terms with std < 1e-8  (StandardScaler clamp threshold;
              note: below float32 eps ~1.2e-7).
n_dead_1e5  : terms with std < 1e-5  (practical noise floor for float32).
frac_dead   : n_dead_1e8 / n_terms.  Degrees with frac_dead ≥ 0.5 are marked.

Usage
-----
    python -c "import sys; sys.path.insert(0, 'src'); exec(open('test/paper_extra_experiments/report_sig_degree_diagnostics.py').read())"
"""

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from config import ROOT_DIR
from src.logger.init_logger import set_config_logging

set_config_logging()
logger = logging.getLogger(__name__)

try:
    import signatory
except ImportError:
    print("signatory is not installed — cannot compute signatures.")
    sys.exit(1)

from src.data_types.sigw_loss_data_props import SigWLossDataProps
from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.nn.architectures.architecture_one_to_one import ArchitectureOneToOne

# ── configuration ─────────────────────────────────────────────────────────────
MAX_DEGREE = 8
CLAMP_THR = 1e-8  # mirrors StandardScaler (below float32 eps ~1.2e-7)
SAFE_THR = 1e-5  # practical noise floor for float32 + finite N
DEAD_FRAC = 0.5  # majority-dead threshold
OUTPUT_DIR = Path(os.path.join(ROOT_DIR, "test/paper_extra_experiments/out/diagnostics"))
OUTPUT_FILE = OUTPUT_DIR / "sig_degree_analysis.txt"


# ── architecture instantiation ────────────────────────────────────────────────


def _build_model(data, anchor_mode: TerminalAnchorMode) -> ArchitectureOneToOne:
    """
    Minimal ArchitectureOneToOne that runs preprocessing but does no training.
    sig_degree=MAX_DEGREE so the full sig_loss_seqs is computed up to that level.
    RESIDUAL mode requires detach_cum_channel=True (enforced by the architecture).
    """
    return ArchitectureOneToOne(
        data_train=data.train_in,
        data_train_lens=data.train_in_len,
        data_val=data.val_in,
        data_val_lens=data.val_in_len,
        train_marks=data.train_marks,
        val_marks=data.val_marks,
        period_plot_val=1,
        loss_properties=SigWLossDataProps(
            sig_degree=MAX_DEGREE,
            scale_high_degrees=False,
            standardise_sig=True,
        ),
        learning_rate=1e-3,
        concentration_factor=1.0,
        hid_size_rep=2,
        use_teacher_forcing=False,
        t_max=data.time_max,
        num_marks=data.num_marks,
        total_epochs=1,
        enable_plot=False,
        terminal_anchor_mode=anchor_mode,
        detach_cum_channel=(anchor_mode == TerminalAnchorMode.RESIDUAL),
    )


# ── signature statistics ──────────────────────────────────────────────────────


def _degree_stats(paths: torch.Tensor, max_degree: int) -> List[Dict]:
    D = paths.shape[2]
    sigs = signatory.signature(paths.float(), depth=max_degree)  # (N, sig_len)
    mean_sig = sigs.mean(0)
    std_sig = sigs.std(0)

    rows, ptr = [], 0
    for deg in range(1, max_degree + 1):
        n = D**deg
        end = ptr + n
        s = std_sig[ptr:end]
        rows.append(
            dict(
                deg=deg,
                n_terms=n,
                l2_mean=mean_sig[ptr:end].norm().item(),
                std_median=s.median().item(),
                n_dead_1e8=int((s < CLAMP_THR).sum().item()),
                n_dead_1e5=int((s < SAFE_THR).sum().item()),
            )
        )
        ptr = end
    return rows


# ── formatting ────────────────────────────────────────────────────────────────


def _format_block(name: str, anchor: str, N: int, L: int, D: int, rows: List[Dict]) -> str:
    hdr = (
        f"\n{'='*72}\n"
        f"Dataset: {name}  [{anchor}]   N={N}  L={L}  D={D}\n"
        f"{'='*72}\n"
        f"  {'deg':>4}  {'n_terms':>8}  {'l2_mean':>9}  {'std_med':>9}"
        f"  {'n_dead(1e-8)':>12}  {'n_dead(1e-5)':>12}  {'frac':>6}\n"
        f"  {'-'*72}"
    )
    lines = [hdr]
    for r in rows:
        frac = r['n_dead_1e8'] / r['n_terms']
        marker = "  ←" if frac >= DEAD_FRAC else ""
        lines.append(
            f"  {r['deg']:>4}  {r['n_terms']:>8}  {r['l2_mean']:>9.4f}"
            f"  {r['std_median']:>9.2e}"
            f"  {r['n_dead_1e8']:>12}  {r['n_dead_1e5']:>12}  {frac:>6.3f}{marker}"
        )
    return "\n".join(lines)


# ── dataset loaders ───────────────────────────────────────────────────────────


def _from_module(module_path: str, class_name: str) -> Callable:
    """Lazy loader for a DataModule defined in another module."""

    def _load():
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)()

    return _load


def _hawkes(data_size=2000, seed=42, **kw):
    def _load():
        from test.paper_experiments.data.synthetic.hawkes.hawkes_dataset import HawkesDataModule

        return HawkesDataModule(data_size=data_size, seed=seed, **kw)

    return _load


def _poisson(use_ihp: bool, num_marks: int = 1, data_size: int = 2000, seed: int = 42):
    def _load():
        import numpy as np
        from test.paper_experiments.data.synthetic.poisson.poisson_dataset import PoissonDataModule

        return PoissonDataModule(
            data_size=data_size,
            seed=seed,
            use_IHP_or_HP=use_ihp,
            num_marks=num_marks,
        )

    return _load


def _hawkes3x3(data_size=2000, seed=42):
    def _load():
        from test.paper_experiments.data.synthetic.hawkes.hawkes_3x3_dataset import Hawkes3x3DataModule

        return Hawkes3x3DataModule(
            data_size=data_size,
            seed=seed,
            time_max=15.0,
            baseline=[0.5, 0.5, 0.5],
            adjacency=[[0.5, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1]],
            decays=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        )

    return _load



DATASETS = [
    # Synthetic
    ("hp_three_marks", _poisson(use_ihp=False, num_marks=3, data_size=2000, seed=42)),
    ("ihp_three_marks", _poisson(use_ihp=True, num_marks=3, data_size=5000, seed=42)),
    ("hawkes", _hawkes(data_size=2000, seed=42)),
    ("hawkes_3x3", _hawkes3x3(data_size=2000, seed=42)),
    # EasyTPP
    ("taxi", _from_module("test.paper_experiments.data.real.easytpp.taxi_dataset", "TaxiDataModule")),
    (
        "stackoverflow",
        _from_module("test.paper_experiments.data.real.easytpp.stackoverflow_dataset", "StackOverflowDataModule"),
    ),
    ("taobao", _from_module("test.paper_experiments.data.real.easytpp.taobao_dataset", "TaobaoDataModule")),
    ("earthquake", _from_module("test.paper_experiments.data.real.easytpp.earthquake_dataset", "EarthquakeDataModule")),
    # EditTPP
    (
        "yelp_mississauga",
        _from_module("test.paper_experiments.data.real.editpp.yelp_mississauga_dataset", "YelpMississaugaDataModule"),
    ),
]


# ── main ──────────────────────────────────────────────────────────────────────
ANCHOR_MODES = [
    ("free_endpoint", TerminalAnchorMode.FREE_ENDPOINT),
    # ("residual", TerminalAnchorMode.RESIDUAL),
]


def _dead_str(v: Optional[int]) -> str:
    return str(v) if v is not None else f"none≤{MAX_DEGREE}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    anchor_labels = [label for label, _ in ANCHOR_MODES]

    # summary_rows: dataset name -> {anchor_label: first_dead_degree | None}
    summary_rows: List[tuple] = []
    all_blocks: List[str] = []

    for name, loader in DATASETS:
        try:
            data = loader()
        except Exception as e:
            logger.warning("Skipping %s — load failed: %s", name, e)
            continue

        N = data.train_in.shape[0]
        L = data.train_in.shape[1] - 1

        first_dead_per_mode: Dict[str, Optional[int]] = {label: None for label in anchor_labels}

        for anchor_label, anchor_mode in ANCHOR_MODES:
            logger.info(
                "Building model for %s [%s]  (N=%d, L=%d, time_max=%.1f)",
                name,
                anchor_label,
                N,
                L,
                data.time_max,
            )

            try:
                model = _build_model(data, anchor_mode)
                paths = model.metrics_train.sig_loss_seqs.cpu()
            except Exception as e:
                logger.warning("Skipping %s [%s] — model init failed: %s", name, anchor_label, e)
                continue

            D = paths.shape[2]

            try:
                rows = _degree_stats(paths, MAX_DEGREE)
            except Exception as e:
                logger.warning("Skipping %s [%s] — signature failed: %s", name, anchor_label, e)
                del model
                continue

            first_dead = next((r['deg'] for r in rows if r['n_dead_1e8'] == r['n_terms']), None)
            first_dead_per_mode[anchor_label] = first_dead

            block = _format_block(name, anchor_label, N, L, D, rows)
            logger.info("%s", block)
            all_blocks.append(block)

            del model  # free GPU memory between datasets

        summary_rows.append((name, first_dead_per_mode))

    # Summary table: one column per active anchor mode.
    col_widths = {label: max(14, len(label)) for label in anchor_labels}
    header = f"  {'dataset':<22}  " + "  ".join(f"{label:>{col_widths[label]}}" for label in anchor_labels)
    summary_lines = [
        "",
        "=" * 72,
        "SUMMARY",
        "=" * 72,
        header,
        "  " + "-" * (22 + 2 + sum(col_widths.values()) + 2 * (len(anchor_labels) - 1 if anchor_labels else 0)),
    ]
    for name, deads in summary_rows:
        cells = "  ".join(f"{_dead_str(deads[label]):>{col_widths[label]}}" for label in anchor_labels)
        summary_lines.append(f"  {name:<22}  {cells}")
    summary = "\n".join(summary_lines)
    logger.info("%s", summary)
    all_blocks.append(summary)

    OUTPUT_FILE.write_text("\n".join(all_blocks), encoding="utf-8")
    logger.info("Results saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
