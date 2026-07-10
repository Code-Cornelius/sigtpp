"""
Experiment-specific factories: data, model, naming, loss metrics for Hawkes processes.
"""

import typing

from src.data_types.sigw_loss_data_props import SigWLossDataProps, sigw_loss_data_props_from_config
from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.nn.architectures.architecture_types import Architectures
from src.nn.architectures.tpp_architecture import TPPArchitecture
from test.paper_experiments.data.synthetic.hawkes.hawkes_dataset import HawkesDataModule
from test.paper_experiments.training_helpers import get_model_name, register_experiment


@register_experiment('hawkes')
def register_hawkes_factories():
    def data_hawkes(cfg):
        return HawkesDataModule(
            data_size=10_000,
            # Dataset seed is independent of the (multiseed) training seed: every
            # seed in a sweep must train/test against the same data so that
            # cross-seed variance reflects training stochasticity only.
            seed=cfg.get('data_seed', 42),
            mu=0.3,
            alpha=0.4,
            beta=1.0,
            batch_size=cfg['parameter_sets'].get('batch_size', None),
        )

    def model_hawkes(cfg, data, period_plot_val, datamodel_path, logger_custom, checkpoint=None):
        return get_model(
            version=Architectures(cfg['version']),
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
        Initializes or loads a model based on the specified version and checkpoint path.

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
            return model_class.load_from_checkpoint(checkpoint_path=model_checkpoint_path, **model_kwargs)

        model = model_class(**model_kwargs)

        if logger_custom is not None and hasattr(model, 'approx_err') and hasattr(model, 'approx_err_histoloss'):
            logger_custom.set_base_error(
                model.approx_err if version is Architectures.SIGTPP else None,
                model.approx_err_histoloss,
            )

        return model

    def name_hawkes(time_max, cfg, custom_file_name=None):
        return get_model_name(
            'hawkes',
            cfg['version'],
            custom_file_name,
            cfg['parameter_sets'],
            time_max,
            seed=cfg.get("_multiseed_seed_tag"),
        )

    return {
        "data_factory": data_hawkes,
        "model_factory": model_hawkes,
        "model_namer": name_hawkes,
        "loss_metrics_fn": Architectures.get_metrics,
    }
