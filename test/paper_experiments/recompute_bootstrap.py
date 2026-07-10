"""Script 1: bootstrap-aware recompute from a raw-results text file.

Reads ``results_raw.txt`` line-by-line, treating it purely as the list of model
identifiers to recompute. For each row:

1. parse the run-name into ``(data_name, version, model_dir)``;
2. resolve the matching ``experiment_type`` from the data-name prefix;
3. reconstruct the cfg via :func:`load_experiment_config` +
   :func:`parse_model_dir_to_cfg`;
4. build the data module + model via the registered factories;
5. load the best checkpoint from disk;
6. run the bootstrap-aware test step;
7. write one enriched row per input model. Models that fail to load keep the
   row but with all metric fields set to NaN (the failure reason is stored in
   ``error``).

Output schema: every metric is reported as ``<name>_mean`` / ``<name>_std``.
See ``docs/metrics_and_eval/04-30-13_FEAT_bootstrap_recompute_plan.md``.

Usage::

    python -c "import sys; sys.path.insert(0, 'src'); \\
        exec(open('test/paper_experiments/recompute_bootstrap.py').read())"

Or with arguments::

    python -c "..." -- --n-bootstraps 200 --output results_bootstrap.txt
    python -c "..." -- --datasets hawkes taxi
    python -c "..." -- --fast        # local sanity check (small samples)
"""

import argparse
import logging
import os
import sys
import time
import typing
from dataclasses import dataclass, replace
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
import numpy as np
from src.logger.init_logger import set_config_logging
from src.data_types.bootstrap_eval import aggregate_bootstrap_metrics, build_per_replicate_matrix

# Only reconfigure logging when this file is run as a script. Importing helpers
# from this module (e.g. ``_write_per_replicate_npz`` in unit tests) must NOT
# call dictConfig, because that disables every logger created earlier in the
# pytest session and breaks ``assertLogs`` in unrelated tests.
if __name__ == "__main__":
    set_config_logging()
logger = logging.getLogger(__name__)


from test.paper_experiments.training_helpers import (
    get_model_chkpt_path,
    load_experiment_config,
    parse_model_dir_to_cfg,
)
from test.paper_experiments.dataset_names import ALL_DATASET_NAMES, DATA_NAME_TO_EXPERIMENT


from config import ROOT_DIR, OUT_FILE_NAME
from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY
from test.paper_experiments.experiment_results import ExperimentResults


# ============================================================================ #
# User config - edit here when running from the IDE
# ============================================================================ #

# Input raw-results file listing model run names to recompute.
_INPUT = os.path.join(ROOT_DIR, "test", "paper_experiments", "out", "results_raw.txt")

# Output file base path; timestamp + gpu id are appended automatically.
_OUTPUT = os.path.join(ROOT_DIR, "test", "paper_experiments", "out", "results_bootstrap.txt")

# True  -> fast mode: B=5.
# False -> full mode: B=100.
_FAST = True

# Subset of datasets to recompute. Set to None to process every dataset.
_DATASETS = [
    "hp_three_marks",
    "ihp_three_marks",
    "hawkes",
    "hawkes_3x3",
    "earthquake",
    "stackoverflow",
    "taobao",
    "taxi",
    "yelp_mississauga",
]

# GPU device index passed to PyTorch Lightning Trainer.
_GPU_ID = 0

# True -> resolve and log model path only; skip checkpoint load + trainer.test.
_SKIP_RECOMPUTE = False

# ============================================================================ #


# --------------------------------------------------------------------------- #
# Data-name -> experiment_type resolution
# --------------------------------------------------------------------------- #

_DATA_NAMES_LONGEST_FIRST = sorted(DATA_NAME_TO_EXPERIMENT.keys(), key=len, reverse=True)


def split_run_name(run_name: str) -> typing.Tuple[str, str, str]:
    """Return ``(data_name, version, model_dir)`` from a raw-results run name.

    ``model_dir`` matches what :func:`get_dir_name_from_params` produced for the run, i.e.
    ``<data_name>_<version>_TX<time>_<param_tokens...>``. ``parse_model_dir_to_cfg`` consumes it
    directly.
    """
    for prefix in _DATA_NAMES_LONGEST_FIRST:
        if run_name == prefix or run_name.startswith(prefix + "_"):
            tail = run_name[len(prefix) + 1 :] if len(run_name) > len(prefix) else ""
            tokens = tail.split("_")
            if not tokens:
                raise ValueError(f"Run name {run_name!r} has no version token after data prefix.")
            version = tokens[0]
            return prefix, version, run_name
    raise ValueError(f"Unknown data-name prefix in run name: {run_name!r}")


# --------------------------------------------------------------------------- #
# Fast / local-mode knobs (also used as defaults from the CLI)
# --------------------------------------------------------------------------- #


@dataclass
class BootstrapEvalSettings:
    """Configurable knobs for the bootstrap-aware test step."""

    n_bootstraps: int = 100
    trainer_seed: int = 42

    @classmethod
    def fast(cls) -> "BootstrapEvalSettings":
        return cls(n_bootstraps=25, trainer_seed=42)


# --------------------------------------------------------------------------- #
# Single-row recompute
# --------------------------------------------------------------------------- #


def _ensure_experiment_registry_initialized() -> None:
    """Import experiment settings only when a recompute actually needs them.

    Importing ``trainingmanager`` eagerly pulls in optional metric dependencies
    such as ``signatory``. Utility consumers of this module, including the
    per-replicate NPZ writer, must remain importable without those dependencies.
    """
    import test.paper_experiments.trainingmanager  # noqa: F401


def _resolve_cfg(
    model_dir: str,
    experiment_type: str,
    version: str,
    data_name: str,
) -> typing.Dict[str, typing.Any]:
    """Reconstruct the full cfg dict for a checkpoint directory.

    Uses :func:`load_experiment_config` to honour the actual layout (shared root-level model YAMLs
    + per-experiment ``experiment.yaml``), and reuses the parser in :func:`parse_model_dir_to_cfg`
    via the ``ref_cfg_override`` hook to recover ``parameter_sets`` from the directory name.

    The reverse parser splits dir names on underscores and expects ``tokens[0]`` to be a
    single-token data prefix. Multi-token prefixes such as ``hp_three_marks`` violate that
    assumption, so we synthesise a single-token name (``x_<version>_<tail>``) before handing
    the dir name to the parser.
    """
    cfg = load_experiment_config(f"{experiment_type}/{version}.yaml")
    configs_root = str(Path(ROOT_DIR) / "test" / "paper_experiments" / "configs")

    expected_prefix = f"{data_name}_{version}_"
    if not model_dir.startswith(expected_prefix):
        raise ValueError(
            f"model_dir {model_dir!r} does not start with expected '<data>_<version>_' " f"prefix {expected_prefix!r}."
        )
    synthetic_dir_name = f"x_{version}_{model_dir[len(expected_prefix) :]}"
    return parse_model_dir_to_cfg(
        synthetic_dir_name,
        experiment_type,
        configs_root,
        ref_cfg_override=cfg,
    )


def recompute_one_row(
    run_name: str,
    settings: BootstrapEvalSettings,
    gpu_id: typing.Optional[int] = 0,
    skip_recompute: bool = False,
) -> typing.Dict[str, typing.Any]:
    """Recompute bootstrap-aware metrics for a single run.

    ``gpu_id`` is a CUDA device index; ``None`` means CPU.

    Returns a row dict shaped like the entries in :class:`ExperimentResults.rows`: ``model_name``
    plus a ``metrics`` dict with ``*_mean`` / ``*_std`` keys, or an ``error`` key on failure.
    """
    _ensure_experiment_registry_initialized()
    data_name, version, model_dir = split_run_name(run_name)
    experiment_type = DATA_NAME_TO_EXPERIMENT.get(data_name)
    if experiment_type is None:
        return {
            "model_name": run_name,
            "metrics": {"error": f"unknown data prefix '{data_name}'"},
        }

    factories = EXPERIMENT_REGISTRY.get(experiment_type)
    if factories is None:
        return {
            "model_name": run_name,
            "metrics": {"error": f"experiment '{experiment_type}' not registered"},
        }

    try:
        cfg = _resolve_cfg(model_dir, experiment_type, version, data_name)
        cfg["version"] = version
        cfg["skip_diagnostics"] = False
        cfg["n_bootstraps"] = settings.n_bootstraps
    except Exception as e:
        logger.error("Cfg resolution failed for %s: %s", run_name, e)
        return {"model_name": run_name, "metrics": {"error": f"cfg resolve failed: {e}"}}

    try:
        data = factories["data_factory"](cfg)
        datamodel_path = os.path.join(
            ROOT_DIR,
            cfg.get("output_dir", "test/paper_experiments"),
            OUT_FILE_NAME,
            experiment_type,
            "models",
            model_dir,
            "",
        )
        logger.info(
            "Resolved model for recompute: run=%s, experiment=%s, version=%s, model_dir=%s, model_path=%s, model_path_exists=%s",
            run_name,
            experiment_type,
            version,
            model_dir,
            datamodel_path,
            os.path.exists(datamodel_path),
        )
        if skip_recompute:
            logger.critical("Skipping recompute for %s because --skip-recompute is enabled.", run_name)
            return {"model_name": model_dir, "metrics": {"error": "recompute skipped"}}
        checkpoint = get_model_chkpt_path(datamodel_path)
        model = factories["model_factory"](
            cfg,
            data,
            10**9,  # period_plot_val: disable validation plotting
            datamodel_path,
            None,  # logger_custom not needed for test-only path
            checkpoint,
        )
    except Exception as e:
        logger.error("Model load failed for %s: %s", run_name, e)
        return {"model_name": model_dir, "metrics": {"error": f"model load failed: {e}"}}

    try:
        trainer = Trainer(
            default_root_dir=os.path.join(ROOT_DIR, OUT_FILE_NAME),
            gpus=[gpu_id] if gpu_id is not None else None,
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )
        seed_everything(settings.trainer_seed, workers=True)
        trainer.test(model, datamodule=data, verbose=False)
        metrics_dict = model.metrics_test or {}
        per_replicate = model._bootstrap_per_replicate
    except Exception as e:
        logger.exception("trainer.test failed for %s", run_name)
        return {"model_name": model_dir, "metrics": {"error": f"trainer.test failed: {e}"}}
    finally:
        import matplotlib.pyplot as plt
        import torch

        plt.close("all")
        torch.cuda.empty_cache()

    if not metrics_dict:
        return {"model_name": model_dir, "metrics": {"error": "test_step completed but metrics_test is empty"}}

    return {"model_name": model_dir, "metrics": dict(metrics_dict), "per_replicate": per_replicate}


def _write_per_replicate_npz(
    rows: typing.List[typing.Dict[str, typing.Any]],
    path: str,
    B: int,
) -> None:
    """Write per-replicate metric vectors to a compressed ``.npz`` file.

    Layout: ``model_names (M,)``, ``metric_names (K,)``, ``data (M, K, B)``.
    Failed models (``per_replicate=None``) get all-NaN slices. Schema version
    is stamped so downstream notebooks can refuse incompatible files.
    """
    all_metrics: typing.List[str] = sorted({m for row in rows for m in (row.get("per_replicate") or {}).keys()})
    if not all_metrics:
        all_metrics = list(ExperimentResults.DISPLAY_METRICS)

    model_names = np.array([row["model_name"] for row in rows], dtype=object)
    metric_names = np.array(all_metrics, dtype=object)
    M = len(rows)
    K = len(all_metrics)
    data = np.full((M, K, B), float("nan"))

    for i, row in enumerate(rows):
        per_rep = row.get("per_replicate") or {}
        for j, m in enumerate(all_metrics):
            if m in per_rep:
                arr = per_rep[m]
                if len(arr) != B:
                    raise ValueError(
                        f"per-replicate vector for model={row['model_name']!r} metric={m!r} "
                        f"has length {len(arr)}, expected {B}. Pairing across models is broken; "
                        "refusing to silently truncate or pad."
                    )
                data[i, j, :] = arr

    np.savez_compressed(
        path,
        model_names=model_names,
        metric_names=metric_names,
        data=data,
        schema_version=np.array(1),
        B=np.array(B),
    )
    logger.info(
        "Per-replicate .npz written to %s (%d models, %d metrics, B=%d)",
        path,
        M,
        K,
        B,
    )


def _read_run_names(input_path: str) -> typing.List[str]:
    """Extract one run name per non-empty line. Whitespace-tolerant. Skips any header row starting with 'MODEL'."""
    run_names: typing.List[str] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0].startswith("#") or tokens[0] == "MODEL":
                logger.debug("Skipping header row: %s", line.rstrip())
                continue
            run_names.append(tokens[0])
    return run_names


def _filter_run_names_by_datasets(
    run_names: typing.Sequence[str],
    dataset_names: typing.Optional[typing.Sequence[str]],
) -> typing.List[str]:
    """Keep only runs whose data-name prefix is in ``dataset_names``.

    ``None`` means no dataset filtering. Allowed names come from the explicit
    hand-maintained ``ALL_DATASET_NAMES`` list.
    """
    if dataset_names is None:
        return list(run_names)

    invalid = sorted(set(dataset_names) - set(ALL_DATASET_NAMES))
    if invalid:
        raise ValueError(f"Unknown dataset filter(s): {invalid}. Valid options: {list(ALL_DATASET_NAMES)}")

    selected = set(dataset_names)
    filtered: typing.List[str] = []
    for run_name in run_names:
        data_name, _, _ = split_run_name(run_name)
        if data_name in selected:
            filtered.append(run_name)
    return filtered


def _empty_metrics_row(error_msg: str) -> typing.Dict[str, float]:
    """Build an all-NaN metrics dict marked with an ``error`` field."""
    metrics: typing.Dict[str, float] = {}
    for m in ExperimentResults.DISPLAY_METRICS:
        if m in ExperimentResults.NON_BOOTSTRAP_METRICS:
            metrics[m] = float("nan")
        else:
            metrics[f"{m}_mean"] = float("nan")
            metrics[f"{m}_std"] = float("nan")
    for m in ExperimentResults.EXTRA_METRICS:
        if m in ExperimentResults.NON_BOOTSTRAP_METRICS:
            metrics[m] = float("nan")
        else:
            metrics[f"{m}_mean"] = float("nan")
            metrics[f"{m}_std"] = float("nan")
    metrics["error"] = error_msg
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: typing.Sequence[str]) -> argparse.Namespace:
    argv = [a for a in argv if a != "--"]  # strip shell separator passed via exec()
    p = argparse.ArgumentParser(description="Bootstrap-aware recompute over a raw-results file.")
    p.add_argument(
        "--input",
        default=os.path.join(ROOT_DIR, "test", "paper_experiments", "out", "results_raw.txt"),
        help="Raw-results text file listing models to recompute.",
    )
    p.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "test", "paper_experiments", "out", "results_bootstrap.txt"),
        help="Enriched output text file.",
    )
    p.add_argument("--n-bootstraps", type=int, default=None, help="Override bootstrap count B.")
    p.add_argument("--trainer-seed", type=int, default=42, help="Seed for pl.seed_everything before trainer.test.")
    p.add_argument("--fast", action="store_true", help="Reduced-scale local mode (B=5).")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=ALL_DATASET_NAMES,
        help="Only recompute runs whose dataset prefix is in this list. Omit to process every dataset.",
    )
    p.add_argument("--gpu-id", type=int, default=0, help="GPU index to use for trainer.test (default: 0).")
    p.add_argument(
        "--skip-recompute",
        action="store_true",
        help="Resolve and log the target model, then skip checkpoint loading and trainer.test.",
    )
    return p.parse_args(argv)


def main(argv: typing.Sequence[str]) -> int:
    args = parse_args(argv)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    total_start = time.perf_counter()
    logger.info("=" * 72)
    logger.info("  recompute_bootstrap.py  |  started=%s  gpu=%d", start_time, args.gpu_id)
    logger.info("=" * 72)

    settings = BootstrapEvalSettings.fast() if args.fast else BootstrapEvalSettings()
    overrides: typing.Dict[str, typing.Any] = {}
    if args.n_bootstraps is not None:
        overrides["n_bootstraps"] = int(args.n_bootstraps)
    overrides["trainer_seed"] = args.trainer_seed
    settings = replace(settings, **overrides)

    run_names = _filter_run_names_by_datasets(_read_run_names(args.input), args.datasets)
    total_runs = len(run_names)
    logger.info("Recomputing %d models with B=%d (fast=%s)", total_runs, settings.n_bootstraps, args.fast)
    if total_runs:
        logger.critical(
            "############################################################\n"
            "Starting bootstrap recompute for %d models with B=%d\n"
            "############################################################",
            total_runs,
            settings.n_bootstraps,
        )

    rows: typing.List[typing.Dict[str, typing.Any]] = []
    for i, run_name in enumerate(run_names, start=1):
        t0 = time.perf_counter()
        try:
            row = recompute_one_row(run_name, settings, gpu_id=args.gpu_id, skip_recompute=args.skip_recompute)
        except Exception as e:
            logger.exception("Unexpected failure on %s", run_name)
            row = {"model_name": run_name, "metrics": _empty_metrics_row(str(e))}

        error_msg = row.get("metrics", {}).get("error")
        if error_msg is not None:
            # Failure contract: keep the row but write NaN for every metric.
            row = {"model_name": row["model_name"], "metrics": _empty_metrics_row(error_msg)}

        elapsed = time.perf_counter() - t0
        if error_msg is None:
            logger.critical("[%d/%d] %s done in %.1fs", i, total_runs, run_name, elapsed)
        else:
            logger.warning("[%d/%d] %s failed in %.1fs: %s", i, total_runs, run_name, elapsed, error_msg)
        half = total_runs // 2
        if half and i == half:
            logger.critical(
                "############################################################\n"
                "[%d/%d] halfway through bootstrap recompute; latest=%s done in %.1fs\n"
                "############################################################",
                i,
                total_runs,
                run_name,
                elapsed,
            )
        rows.append(row)

    results = ExperimentResults(rows, version="bootstrap")
    base, ext = os.path.splitext(args.output)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stamped_output = f"{base}_{stamp}_gpu{args.gpu_id}{ext}"
    out_dir = os.path.dirname(os.path.abspath(stamped_output))
    os.makedirs(out_dir, exist_ok=True)
    ranked = results.normalize_and_rank()
    ranked._write_txt(stamped_output)
    per_replicate_path = f"{base}_{stamp}_gpu{args.gpu_id}.npz"
    _write_per_replicate_npz(rows, per_replicate_path, settings.n_bootstraps)
    total_elapsed = time.perf_counter() - total_start
    logger.info("Enriched results written to %s", stamped_output)
    logger.info("Total diagnostic runtime: %.1fs", total_elapsed)
    return 0


if __name__ == "__main__":
    # Parameters are defined at the top of the file - edit them there.
    _cli = [a for a in sys.argv[1:] if a != "--"]
    # Any explicit CLI args bypass the hard-coded fallback settings below.
    # The top-of-file defaults (_INPUT/_OUTPUT/_FAST/_DATASETS/_GPU_ID/_SKIP_RECOMPUTE)
    # are only used when the script is launched with no CLI args.
    if _cli:
        # Explicit CLI args take precedence (e.g. launched from terminal with -- --fast ...).
        raise SystemExit(main(_cli))
    _argv = ["--input", _INPUT, "--output", _OUTPUT, "--gpu-id", str(_GPU_ID)]
    if _FAST:
        _argv.append("--fast")
    if _DATASETS is not None:
        _argv += ["--datasets", *_DATASETS]
    if _SKIP_RECOMPUTE:
        _argv.append("--skip-recompute")
    raise SystemExit(main(_argv))
