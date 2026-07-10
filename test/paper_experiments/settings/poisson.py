"""
Experiment-specific factories for Poisson process variants.

All Poisson variants share a single parametric factory.  The experiment name
is registered from this one file:

    poisson_three_marks — reads num_marks / mark_probs from the YAML; defaults to num_marks=3

To add another variant (e.g. five marks), add one entry to the registration
loop at the bottom of this file.  No new file is needed.
"""

import logging
import typing

import numpy as np

logger = logging.getLogger(__name__)

from src.data_types.sigw_loss_data_props import SigWLossDataProps, sigw_loss_data_props_from_config
from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.nn.architectures.architecture_types import Architectures
from src.nn.architectures.tpp_architecture import TPPArchitecture
from test.paper_experiments.data.synthetic.poisson.poisson_dataset import PoissonDataModule
from test.paper_experiments.training_helpers import get_model_name, register_experiment


def get_model(
    version: str,
    config: dict,
    data,
    period_plot_val: int,
    datamodel_path: str,
    total_epochs: int,
    model_checkpoint_path: typing.Optional[str] = None,
    logger_custom=None,
    server_training: bool = False,
    n_bootstraps: int = 1,
) -> TPPArchitecture:
    """
    Initialize or load a model for any registered Poisson experiment variant.

    The variant registered at the bottom of this module is
    ``poisson_three_marks``.

    Args:
        version (Architectures): The version of the model to initialize.
        config (Dict): Configuration dictionary containing model hyperparameters.
        data: Data object containing training and validation sets.
        period_plot_val (int): Period for plotting validation results.
        datamodel_path (str): Output directory path for saving model outputs and plots.
        total_epochs (int): Total number of training epochs.
        model_checkpoint_path (Optional[str]): Path to a checkpoint file to load a pre-trained model. If None, a new model is initialized.
        logger_custom (Optional): Custom logger for tracking errors.
        server_training (bool): If True, adjusts plotting frequency for server environments. Defaults to False.

    Returns:
        TPPArchitecture: Initialized or loaded model.

    Raises:
        ValueError: If the specified version is unknown.
    """
    model_class = Architectures.get_model_class(version)
    if version is Architectures.SIGTPP:
        model_kwargs = {
            "data_train": data.train_in,
            "data_train_lens": data.train_in_len,
            "data_val": data.val_in,
            "data_val_lens": data.val_in_len,
            "period_plot_val": period_plot_val,
            "loss_properties": sigw_loss_data_props_from_config(config, False, True),
            "learning_rate": config["lr_gen"],
            "concentration_factor": config["concentration_factor"],
            "hid_size_rep": config["hid_size_rep"],
            "use_teacher_forcing": config["use_teacher_forcing"],
            "total_epochs": total_epochs,
            "t_max": data.time_max,
            "output_dir": datamodel_path,
            "enable_plot": not server_training,
            "plot_every_n_val_steps": 2 if server_training else 1,
            "terminal_anchor_mode": TerminalAnchorMode(config["terminal_anchor"]),
            "detach_cum_channel": config["detach_cum_channel"],
            "use_lr_scheduler": config.get("use_lr_scheduler", False),
            "num_marks": data.num_marks,
            "train_marks": data.train_marks,
            "val_marks": data.val_marks,
            "mark_loss_weight": config["mark_loss_weight"],
        }

    elif version is Architectures.WGAN:
        model_kwargs = {
            "data_train": data.train_in,
            "data_train_lens": data.train_in_len,
            "data_val": data.val_in,
            "data_val_lens": data.val_in_len,
            "lr_gen": config["lr_gen"],
            "lr_disc": config["lr_disc"],
            "hidden_size_rnn": config["hid_size_rnn"],
            "concentration_factor": config["concentration_factor"],
            "lipschitz_reg": config["lipschitz_reg"],
            "t_max": data.time_max,
            "period_plot_val": period_plot_val,
            "output_dir": datamodel_path,
            "enable_plot": not server_training,
            "plot_every_n_val_steps": 2 if server_training else 1,
            "num_marks": data.num_marks,
            "train_marks": data.train_marks,
            "val_marks": data.val_marks,
        }
    elif version is Architectures.DDPM:
        model_kwargs = {
            "data_train": data.train_in,
            "data_train_lens": data.train_in_len,
            "data_val": data.val_in,
            "data_val_lens": data.val_in_len,
            "lr": config["lr"],
            "hidden_size_rnn": config["hid_size_rnn"],
            "concentration_factor": config["concentration_factor"],
            "num_diff_steps": config["num_diff_steps"],
            "t_max": data.time_max,
            "period_plot_val": period_plot_val,
            "output_dir": datamodel_path,
            "enable_plot": not server_training,
            "plot_every_n_val_steps": 50 if server_training else 10,
            "num_marks": data.num_marks,
            "train_marks": data.train_marks,
            "val_marks": data.val_marks,
        }
    elif version is Architectures.DETER:
        model_kwargs = {
            "data_train": data.train_in,
            "data_train_lens": data.train_in_len,
            "data_val": data.val_in,
            "data_val_lens": data.val_in_len,
            "period_plot_val": period_plot_val,
            "loss_properties": SigWLossDataProps(5, False, True),
            "learning_rate": config["lr_gen"],
            "concentration_factor": 1.0,
            "hid_size_rep": config["hid_size_rep"],
            "t_max": data.time_max,
            "output_dir": datamodel_path,
            "enable_plot": not server_training,
            "plot_every_n_val_steps": 1,
            "num_marks": data.num_marks,
            "train_marks": data.train_marks,
            "val_marks": data.val_marks,
        }
    elif version is Architectures.GAMMA:
        model_kwargs = {
            "data_train": data.train_in,
            "data_train_lens": data.train_in_len,
            "data_val": data.val_in,
            "data_val_lens": data.val_in_len,
            "loss_properties": SigWLossDataProps(5, False, True),
            "learning_rate": config["learning_rate"],
            "t_max": data.time_max,
            "output_dir": datamodel_path,
            "enable_plot": not server_training,
            "period_plot_val": period_plot_val,
            "num_marks": data.num_marks,
            "train_marks": data.train_marks,
            "val_marks": data.val_marks,
        }
    elif version is Architectures.VAE:
        model_kwargs = {
            "data_train": data.train_in,
            "data_train_lens": data.train_in_len,
            "data_val": data.val_in,
            "data_val_lens": data.val_in_len,
            "lr": config["lr"],
            "hidden_size_rnn": config["hid_size_rnn"],
            "concentration_factor": config["concentration_factor"],
            "latent_dim": config["latent_dim"],
            "t_max": data.time_max,
            "period_plot_val": period_plot_val,
            "output_dir": datamodel_path,
            "enable_plot": not server_training,
            "plot_every_n_val_steps": 2 if server_training else 1,
            "kl_anneal_epochs": config["kl_anneal_epochs"],
            "free_bits": config["free_bits"],
            "recon_weight": config["recon_weight"],
            "num_marks": data.num_marks,
            "train_marks": data.train_marks,
            "val_marks": data.val_marks,
        }
    else:
        raise ValueError(f"Unknown version: {version}")

    model_kwargs["n_bootstraps"] = n_bootstraps

    if model_checkpoint_path:
        model = model_class.load_from_checkpoint(checkpoint_path=model_checkpoint_path, **model_kwargs)
    else:
        model = model_class(**model_kwargs)
        if logger_custom is not None and hasattr(model, 'approx_err') and hasattr(model, 'approx_err_histoloss'):
            # Target Err Train (sig baseline) only makes sense when the train loss is itself a sig metric.
            logger_custom.set_base_error(
                model.approx_err if version is Architectures.SIGTPP else None,
                model.approx_err_histoloss,
            )

    return model


def _make_poisson_factories(
    default_num_marks: int,
    namer_prefix: str,
    default_mark_probs: typing.Optional[typing.Sequence[float]] = None,
) -> dict:
    """
    Return a factories dict for a Poisson experiment variant.

    All registered Poisson variants share this implementation.  The only
    per-variant knobs are the default number of marks (overridable from YAML),
    the output directory prefix used by the model namer, and an optional
    default mark probability vector (overridable from YAML via ``mark_probs``).
    When ``default_mark_probs`` is None, marks are equiprobable by default.
    """

    def data_factory(cfg):
        num_marks = int(cfg.get('num_marks', default_num_marks))
        mark_probs_cfg = cfg.get('mark_probs', None)
        if mark_probs_cfg is not None:
            mark_probs = np.array(mark_probs_cfg, dtype=np.float64)
        elif default_mark_probs is not None:
            mark_probs = np.array(default_mark_probs, dtype=np.float64)
        else:
            mark_probs = np.ones(num_marks, dtype=np.float64) / num_marks
        return PoissonDataModule(
            data_size=2_000,
            # Dataset seed is independent of the (multiseed) training seed: every
            # seed in a sweep must train/test against the same data so that
            # cross-seed variance reflects training stochasticity only.
            seed=cfg.get('data_seed', 42),
            use_IHP_or_HP=False,
            batch_size=cfg['parameter_sets'].get('batch_size', None),
            num_marks=num_marks,
            mark_probs=mark_probs,
            zero_marks=cfg.get('zero_marks', False),
        )

    def model_factory(cfg, data, period_plot_val, datamodel_path, logger_custom, checkpoint=None):
        return get_model(
            version=(Architectures(cfg['version'])),
            config=cfg["parameter_sets"],
            data=data,
            period_plot_val=period_plot_val,
            datamodel_path=datamodel_path,
            total_epochs=cfg["epochs"],
            model_checkpoint_path=checkpoint,
            logger_custom=logger_custom,
            server_training=cfg.get('server_training', False),
            n_bootstraps=cfg['n_bootstraps'],
        )

    def model_namer(time_max, cfg, custom_file_name=None):
        return get_model_name(
            namer_prefix,
            cfg['version'],
            custom_file_name,
            cfg['parameter_sets'],
            time_max,
            seed=cfg.get("_multiseed_seed_tag"),
        )

    return {
        "data_factory": data_factory,
        "model_factory": model_factory,
        "model_namer": model_namer,
        "loss_metrics_fn": Architectures.get_metrics,
    }


# ---------------------------------------------------------------------------
# Registration
#
# Each entry is (experiment_type_name, default_num_marks, namer_prefix).
# To add a new Poisson variant, append a row here — no new file required.
# ---------------------------------------------------------------------------
def _register_poisson_variant(name: str, default_num_marks: int, namer_prefix: str) -> None:
    @register_experiment(name)
    def _register():
        return _make_poisson_factories(default_num_marks, namer_prefix)


_register_poisson_variant('poisson_three_marks', 3, 'hp_three_marks')
