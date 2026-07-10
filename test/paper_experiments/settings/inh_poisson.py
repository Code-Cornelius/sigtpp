"""
Experiment-specific factories for inhomogeneous Poisson process variants.

All IHP variants share a single parametric factory.  The experiment name
is registered from this one file:

    inh_poisson_three_marks — 3 marks (probs YAML-configurable)

To add another variant (e.g. five marks), add one entry to the registration
loop at the bottom of this file.  No new file is needed.
"""

import logging
import typing

import numpy as np

logger = logging.getLogger(__name__)

from src.nn.architectures.architecture_types import Architectures
from test.paper_experiments.data.synthetic.poisson.poisson_dataset import PoissonDataModule
from test.paper_experiments.training_helpers import get_model_name, register_experiment
from test.paper_experiments.settings.poisson import get_model


def _make_inh_poisson_factories(
    default_num_marks: int,
    namer_prefix: str,
    default_mark_probs: typing.Optional[typing.Sequence[float]] = None,
) -> dict:
    """
    Return a factories dict for an IHP experiment variant.

    All registered IHP variants share this implementation.  The only
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
            data_size=5_000,
            # Dataset seed is independent of the (multiseed) training seed: every
            # seed in a sweep must train/test against the same data so that
            # cross-seed variance reflects training stochasticity only.
            seed=cfg.get('data_seed', 42),
            use_IHP_or_HP=True,
            batch_size=cfg['parameter_sets'].get('batch_size', None),
            num_marks=num_marks,
            mark_probs=mark_probs,
            zero_marks=cfg['zero_marks'],
        )

    def model_factory(cfg, data, period_plot_val, datamodel_path, logger_custom, checkpoint=None):
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
# To add a new IHP variant, append a row here — no new file required.
# ---------------------------------------------------------------------------
def _register_inh_poisson_variant(name: str, default_num_marks: int, namer_prefix: str) -> None:
    @register_experiment(name)
    def _register():
        return _make_inh_poisson_factories(default_num_marks, namer_prefix)


_register_inh_poisson_variant('inh_poisson_three_marks', 3, 'ihp_three_marks')
