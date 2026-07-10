"""
Experiment factories for the 3x3 marked Hawkes experiment (Tick-backed).

Registers 'hawkes_3x3' in the experiment registry. Model wiring is reused
from settings.poisson.get_model — no duplication of the per-architecture switch.
"""

import logging

logger = logging.getLogger(__name__)

from src.nn.architectures.architecture_types import Architectures
from test.paper_experiments.data.synthetic.hawkes.hawkes_3x3_dataset import Hawkes3x3DataModule
from test.paper_experiments.settings.poisson import get_model
from test.paper_experiments.training_helpers import get_model_name, register_experiment


@register_experiment('hawkes_3x3')
def register_hawkes_3x3_factories():
    def data_factory(cfg):
        return Hawkes3x3DataModule(
            data_size=2_000,
            # Dataset seed is independent of the (multiseed) training seed: every
            # seed in a sweep must train/test against the same data so that
            # cross-seed variance reflects training stochasticity only.
            seed=cfg.get("data_seed", 42),
            time_max=cfg["time_max"],
            baseline=cfg["baseline"],
            adjacency=cfg["adjacency"],
            decays=cfg["decays"],
            batch_size=cfg["parameter_sets"].get("batch_size", None),
        )

    def model_factory(cfg, data, period_plot_val, datamodel_path, logger_custom, checkpoint=None):
        return get_model(
            version=Architectures(cfg["version"]),
            config=cfg["parameter_sets"],
            data=data,
            period_plot_val=period_plot_val,
            datamodel_path=datamodel_path,
            total_epochs=cfg["epochs"],
            model_checkpoint_path=checkpoint,
            logger_custom=logger_custom,
            server_training=cfg.get("server_training", False),
            n_bootstraps=cfg["n_bootstraps"],
        )

    def model_namer(time_max, cfg, custom_file_name=None):
        return get_model_name(
            "hawkes_3x3",
            cfg["version"],
            custom_file_name,
            cfg["parameter_sets"],
            time_max,
            seed=cfg.get("_multiseed_seed_tag"),
        )

    return {
        "data_factory": data_factory,
        "model_factory": model_factory,
        "model_namer": model_namer,
        "loss_metrics_fn": Architectures.get_metrics,
    }
