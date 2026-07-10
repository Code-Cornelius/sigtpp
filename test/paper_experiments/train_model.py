import argparse
import logging

from matplotlib import pyplot as plt

from src.logger.init_logger import set_config_logging

# Put here to shut all logs from usual libraries but keep the logs from this project.
set_config_logging()
logger = logging.getLogger(__name__)

from test.paper_experiments.training_helpers import load_experiment_config
from test.paper_experiments.training_runner import run_experiment_config


# Pattern: "<dataset>/<model>.yaml" loads the shared root-level configs/<model>.yaml.
# Dataset-specific params (time_max, adjacency, etc.) are merged from configs/<dataset>/experiment.yaml when present.
# Datasets: poisson_three_marks, inh_poisson_three_marks, hawkes, hawkes_3x3, earthquake, stackoverflow, taobao, taxi, yelp_mississauga.
# Models: sigtpp, wgan, ddpm, deter, gamma, vae.

_default_config = "taxi/sigtpp_test.yaml"

_parser = argparse.ArgumentParser()
_parser.add_argument("config", nargs="?", default=_default_config)
_parser.add_argument("--gpu", type=int, default=None)
_args = _parser.parse_args()

cfg = load_experiment_config(_args.config)
if _args.gpu is not None:
    cfg["gpu_id"] = [_args.gpu]
run_experiment_config(cfg)
# plt.show()
