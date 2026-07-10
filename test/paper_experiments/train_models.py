import gc
import logging
import sys

import torch
from matplotlib import pyplot as plt

from src.logger.init_logger import set_config_logging

# Put here to shut all logs from usual libraries but keep the logs from this project.
set_config_logging()
logger = logging.getLogger(__name__)

from test.paper_experiments.training_helpers import load_experiment_config
from test.paper_experiments.training_runner import run_experiment_config


# Models trained, in order, for the dataset passed as argv[1]. Comment out to skip.
# Resolves to "<dataset>/<model>.yaml" via load_experiment_config (shared root YAML +
# merged configs/<dataset>/experiment.yaml).
_MODELS = ["deter", "sigtpp", "ddpm", "vae", "wgan", "gamma"]

_default_dataset = "poisson_three_marks"
_dataset = sys.argv[1] if len(sys.argv) > 1 else _default_dataset

failed = []
_total_models = len(_MODELS)
for _idx, model in enumerate(_MODELS, start=1):
    config_path = f"{_dataset}/{model}.yaml"
    # Per-model counter only; no global ETA across models (per-config durations
    # are too heterogeneous across model families to extrapolate). Within-model
    # ETA is emitted by log_config_message on each ==== CONFIG a/b ==== line.
    logger.info("=" * 80)
    logger.info("=== MODEL %d/%d %s ===", _idx, _total_models, config_path)
    logger.info("=" * 80)
    try:
        cfg = load_experiment_config(config_path)
        run_experiment_config(cfg)
    except Exception:
        logger.exception("Config %s failed; continuing sweep.", config_path)
        failed.append(config_path)
    finally:
        plt.close('all')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if failed:
    logger.warning("Sweep finished with %d failure(s): %s", len(failed), failed)
else:
    logger.info("Sweep finished cleanly across %d model(s) for dataset '%s'.", len(_MODELS), _dataset)
