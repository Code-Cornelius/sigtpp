# This module acts as a central registry for experiment configurations.
# The register_experiment decorator is used by each experiment module (e.g., poisson.py)
# to register their factory functions upon import.

# Note on circular imports:
# Although experiment modules like `poisson.py` import `register_experiment` from this file,
# and this file (via other modules like trainingmanager.py) imports those same experiment modules
# purely for their side effects (i.e., registration), this does NOT cause circular import issues.
#
# Why it's safe:
# - The decorator function `register_experiment` is defined and available before any imports occur.
# - The experiment modules only *use* this decorator; they do not import or access any symbols from `experiment_registry`
#   beyond the decorator itself.
# - In `train_model.py`, the experiment modules are imported *after* all initial setup is done.
# - Python handles such circular dependencies gracefully as long as no unresolved symbol access occurs during import.
#
# Thus, circular import errors are avoided by careful ordering and minimal mutual dependency.

# After all settings modules have been imported (see trainingmanager.py), the registry
# looks like this (keys are the experiment_type strings used in YAML configs):
#
# EXPERIMENT_REGISTRY = {
#     "poisson_three_marks": {
#         "data_factory":    lambda cfg: PoissonDataModule(...),
#         "model_factory":   lambda cfg, data, period_plot_val, datamodel_path, logger_custom, checkpoint=None: ...,
#         "model_namer":     lambda time_max, cfg, custom_file_name=None: ...,
#         "loss_metrics_fn": Architectures.get_metrics,
#     },
#     "hawkes":             { ... },   # same shape, different defaults
#     "stackoverflow":      { ... },
#     ...
# }
EXPERIMENT_REGISTRY = {}
