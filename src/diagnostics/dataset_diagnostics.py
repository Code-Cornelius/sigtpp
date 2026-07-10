"""Dataset diagnostic report orchestration for temporal point processes.

Exports figures to disk. All plot logic lives in dataset_plots.py; all tensor
helpers in _tpp_features.py.

Public API surface is intentionally flat: callers import from this module
so that test monkeypatching of plot functions works correctly.
"""

import contextlib
import logging
from pathlib import Path
from typing import Dict, Optional

from matplotlib import pyplot as plt
from matplotlib import rc_context

from config import ROOT_DIR
from src.utils.utils_os import factory_fct_linked_path

# Tensor helpers live in _tpp_features; this module and external callers reach in
# via the module alias (e.g. _features.<name>) instead of a hand-maintained re-export list.
from src.diagnostics import _tpp_features as _features

# Plot functions are accessed through this alias so tests can monkeypatch them
# at `dd._plots.<name>` and the rebind takes effect on the call inside
# export_dataset_report / export_dataset_report_merged.
from src.diagnostics import dataset_plots as _plots

logger = logging.getLogger(__name__)

_DATASET_DIAGNOSTICS_LINKER = factory_fct_linked_path(ROOT_DIR, "test/paper_experiments/out/dataset_diagnostics")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _dataset_report_panel_filename(dm, panel_name: str, fig_format: str) -> str:
    return f"{_features._dataset_report_slug(dm)}_{panel_name}.{fig_format}"


def resolve_dataset_report_dir(dm, output_dir=None, *, preview: bool = False) -> Path:
    """Resolve the report directory for a datamodule.

    When ``output_dir`` is omitted, reports land under
    ``test/paper_experiments/out/dataset_diagnostics/<dataset>/`` and preview
    exports append ``_preview`` to that dataset slug.
    """
    if output_dir is not None:
        return Path(output_dir)

    slug = _features._dataset_report_slug(dm)
    if preview:
        slug = f"{slug}_preview"
    return Path(_DATASET_DIAGNOSTICS_LINKER([slug]))


# ---------------------------------------------------------------------------
# Internal render/save helpers
# ---------------------------------------------------------------------------


def _render_panels(
    panels,
    *,
    title_prefix: Optional[str] = None,
    rc_params: Optional[Dict] = None,
):
    rendered = []
    failures = []
    ctx = rc_context(rc_params) if rc_params is not None else contextlib.nullcontext()
    with ctx:
        for name, factory in panels:
            open_before = set(plt.get_fignums())
            try:
                fig = factory()
                if title_prefix is not None:
                    _plots._set_window_title(fig, f"{title_prefix}: {name.replace('_', ' ')}")
                rendered.append((name, fig))
            except Exception as exc:  # keep rendering other panels, then fail at the end
                logger.exception("Failed to render panel '%s': %s", name, exc)
                failures.append(name)
                for fignum in set(plt.get_fignums()) - open_before:
                    plt.close(fignum)
    return rendered, failures


def _save_panels(rendered, out: Path, dm, fig_format: str, *, dpi: Optional[int] = None):
    saved = []
    failures = []
    savefig_kwargs = {"bbox_inches": "tight"}
    if dpi is not None:
        savefig_kwargs["dpi"] = dpi
    for name, fig in rendered:
        path = out / _dataset_report_panel_filename(dm, name, fig_format)
        try:
            fig.savefig(path, **savefig_kwargs)
            logger.info("Saved %s", path)
            saved.append(name)
        except Exception as exc:
            logger.exception("Failed to save panel '%s' to %s: %s", name, path, exc)
            failures.append(name)
        finally:
            plt.close(fig)
    return saved, failures


# ---------------------------------------------------------------------------
# Export orchestrators
# ---------------------------------------------------------------------------


def export_dataset_report(
    dm,
    output_dir=None,
    max_paths: int = _features.DEFAULT_MAX_PATHS,
    max_lag: int = _features.DEFAULT_MAX_LAG,
    log_dataset_samples: bool = True,
    log_sample_windows: int = _features.DEFAULT_LOG_SAMPLE_WINDOWS,
    log_sample_stride: int = _features.DEFAULT_LOG_SAMPLE_STRIDE,
    fig_format: str = "svg",
    preview: bool = False,
) -> None:
    """Generate the paper-ready per-dataset diagnostic report.

    When `preview=True`, renders all figures, shows them interactively, then
    saves the five comparative figures used in the paper. When `output_dir`
    is not provided, it is inferred from `dm` under
    `test/paper_experiments/out/dataset_diagnostics/`.
    """
    if log_dataset_samples:
        _features._log_dataset_sample_windows(dm, num_windows=log_sample_windows, stride=log_sample_stride)

    panels = [
        ("sample_paths", lambda: _plots.plot_sample_paths_comparison(dm, max_paths=max_paths)),
        ("intensity_and_its", lambda: _plots.plot_intensity_and_its(dm)),
        ("correlation_heatmap", lambda: _plots.plot_correlation_heatmaps(dm, max_lag=max_lag)),
        ("acf_inter_arrivals", lambda: _plots.plot_autocorrelation_inter_arrivals_comparison(dm, max_lag=max_lag)),
        ("acf_cumulative", lambda: _plots.plot_autocorrelation_cumulative_comparison(dm, max_lag=max_lag)),
    ]

    rendered, panel_failures = _render_panels(panels, title_prefix=type(dm).__name__)

    if preview and rendered:
        plt.show()

    out = resolve_dataset_report_dir(dm, output_dir, preview=preview)
    out.mkdir(parents=True, exist_ok=True)

    saved, save_failures = _save_panels(rendered, out, dm, fig_format)
    panel_failures.extend(save_failures)

    if panel_failures:
        raise RuntimeError(
            "Failed to export dataset diagnostics panels: " f"failed={panel_failures}, saved={saved}, output_dir={out}"
        )


def export_dataset_report_merged(
    dm,
    output_dir=None,
    max_paths: int = _features.DEFAULT_MAX_PATHS,
    max_lag: int = _features.DEFAULT_MAX_LAG,
    fig_format: str = "pdf",
    preview: bool = False,
) -> None:
    """Export one merged-split figure per diagnostic panel (4 total).

    All splits (train, val, test) are pooled before plotting so each figure
    shows the full dataset distribution rather than per-split columns.

    Panels saved:
      sample_paths, intensity_pdf, acf, correlation
    """
    panels = [
        ("sample_paths", lambda: _plots.plot_sample_paths_all_splits(dm, max_paths=max_paths)),
        ("intensity_pdf", lambda: _plots.plot_intensity_pdf_all_splits(dm)),
        ("acf", lambda: _plots.plot_acf_all_splits(dm, max_lag=max_lag)),
        ("correlation", lambda: _plots.plot_correlation_heatmap_all_splits(dm, max_lag=max_lag)),
    ]

    rendered, panel_failures = _render_panels(
        panels,
        rc_params={"text.usetex": True, "font.weight": "bold"},
    )

    if preview and rendered:
        plt.show()

    out = resolve_dataset_report_dir(dm, output_dir, preview=preview)
    out.mkdir(parents=True, exist_ok=True)

    saved, save_failures = _save_panels(rendered, out, dm, fig_format, dpi=300)
    panel_failures.extend(save_failures)

    if panel_failures:
        raise RuntimeError(
            "Failed to export merged dataset diagnostics panels: "
            f"failed={panel_failures}, saved={saved}, output_dir={out}"
        )
