import difflib
import logging
import os
import typing
from pathlib import Path
from statistics import median

import yaml

logger = logging.getLogger(__name__)

from config import ROOT_DIR
from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY


def get_experiment_entry(
    experiment_type: str,
    registry: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    logger_override: typing.Optional[logging.Logger] = None,
) -> typing.Dict[str, typing.Any]:
    """Resolve an experiment_type to its registered factories with a readable error."""
    registry = EXPERIMENT_REGISTRY if registry is None else registry
    if experiment_type in registry:
        return registry[experiment_type]

    available = sorted(registry.keys())
    suggestions = difflib.get_close_matches(experiment_type, available, n=3, cutoff=0.5)
    if not suggestions:
        suggestions = [name for name in available if name.startswith(experiment_type)][:3]
    message = f"Unknown experiment_type '{experiment_type}'. " f"Available experiments: {', '.join(available)}."
    if suggestions:
        if len(suggestions) == 1:
            message += f" Did you mean '{suggestions[0]}'?"
        else:
            message += f" Did you mean one of: {', '.join(repr(name) for name in suggestions)}?"
    if logger_override:
        logger_override.error(message)
    raise ValueError(message)


def _format_eta_rounded(seconds: float) -> str:
    """Format a positive duration as ``Xd-Yh`` or ``Xh`` or ``<1h``, rounded to the hour."""
    hours_total = int(round(seconds / 3600.0))
    if hours_total < 1:
        return "<1h"
    if hours_total >= 24:
        days, hours = divmod(hours_total, 24)
        return f"{days}d-{hours}h"
    return f"{hours_total}h"


def log_config_message(
    i: int,
    num_configs: int,
    total_width: int,
    prev_train_times: typing.Optional[typing.Sequence[float]] = None,
) -> None:
    """Log ``=== CONFIG i/num_configs ===`` centered on a line of equals.

    If ``prev_train_times`` is non-empty (per-config wall-clock seconds for completed configs),
    logs a separate ETA line after the banner. Percent is config-count based (``(i-1)/num_configs``)
    so skipped configs count as done. Remaining time uses the median of past durations. ``T`` is
    rounded to the nearest hour: ``<1h``, ``Yh``, or ``Xd-Yh``.
    """
    config_str = f"CONFIG {i}/{num_configs}"

    separator = '=' * total_width
    config_with_spaces = f" {config_str} "

    # Add extra space if needed to ensure symmetric equal signs on both sides
    padding_total = total_width - len(config_with_spaces)
    if padding_total % 2 == 1:
        config_with_spaces += " "
        padding_total -= 1

    left_padding = padding_total // 2
    centered_line = config_with_spaces.rjust(len(config_with_spaces) + left_padding, '=').ljust(total_width, '=')

    logger.info(separator)
    logger.info(centered_line)
    logger.info(separator)

    if prev_train_times:
        median_time = float(median(prev_train_times))
        configs_remaining = max(num_configs - i + 1, 0)
        eta_remaining = median_time * configs_remaining
        percent_done = 100.0 * (i - 1) / num_configs
        logger.info("ETA: %d%% done, %s left", int(percent_done), _format_eta_rounded(eta_remaining))


_VERSION_KEY_ORDERS: typing.Dict[str, typing.Dict[str, typing.List[str]]] = {
    "sigtpp": {
        "first": ["use_teacher_forcing"],
        "priority": ["hid_size_rep"],
        "last": ["terminal_anchor", "detach_cum_channel"],
    }
}


# Module-level forward abbreviation tables. Used by both get_dir_name_from_params (forward)
# and parse_model_dir_to_cfg (inverse). Adding a key here REQUIRES a corresponding entry
# in _INV_KEY_ABBREV / _INV_VALUE_ABBREV below to keep the inverse parser correct.
_KEY_ABBREV_FORWARD: typing.Dict[str, str] = {
    "terminal_anchor": "anch",
    # use_lr_scheduler collides with use_teacher_forcing under the default
    # 4-char truncation ("use_"). Distinct token; changes dir names of future
    # runs only (legacy two-`use_` names were never parseable anyway).
    "use_lr_scheduler": "lrsc",
}
_VALUE_ABBREV_FORWARD: typing.Dict[str, str] = {
    "free_endpoint": "free",
    "residual": "resi",
}

# Inverse abbreviation tables. Keep in sync with the forward tables above.
_INV_KEY_ABBREV: typing.Dict[str, str] = {v: k for k, v in _KEY_ABBREV_FORWARD.items()}
_INV_VALUE_ABBREV: typing.Dict[str, str] = {v: k for k, v in _VALUE_ABBREV_FORWARD.items()}


def get_model_name(
    data_name: str,
    version: str,
    custom_file_name: typing.Optional[str],
    config: typing.Any,
    time_max: float,
    seed: typing.Optional[int] = None,
) -> str:
    if custom_file_name is not None:
        # Single config only: we drop the per-config tokens here, so a grid would
        # collapse onto one name -- colliding checkpoints and losing the `_sig_<d>`
        # ablation token. See _LOCAL_RESULTS_FILE_NAME in training_runner.py.
        base = version + "_" + custom_file_name
        return base if seed is None else f"{base}_seed{int(seed)}"
    key_order = _VERSION_KEY_ORDERS.get(version, {})
    return get_dir_name_from_params(
        data_name,
        version,
        config,
        time_max,
        first_keys=key_order.get("first", []),
        priority_keys=key_order.get("priority", []),
        last_keys=key_order.get("last", []),
        seed=seed,
    )


def get_dir_name_from_params(
    data_name: str,
    version: str,
    config: typing.Any,
    time_max: float,
    first_keys: typing.Sequence[str] = (),
    priority_keys: typing.Sequence[str] = (),
    last_keys: typing.Sequence[str] = (),
    seed: typing.Optional[int] = None,
) -> str:
    """
    Generate a directory name based on version, configuration parameters, and maximum time.

    Parameters:
    - version (str): Version identifier.
    - config (object): An object containing various parameters as attributes.
    - time_max (float): Maximum time value.

    Returns:
    - str: Generated directory name string.
    """

    def format_value(value: typing.Union[float, int, str, typing.Any]) -> str:
        """
        Format the parameter value based on its type and magnitude.

        Rules:
        - Booleans: True -> "T", False -> "F".
        - Floats >= 1: 2 decimal places with trailing zeros stripped.
        - Floats < 1: Scientific notation with 'g' format.
        - Integers: Written as integers.
        - Strings: Lowercased and truncated to first 6 characters.
        - Replace '.' with ',' in float/string representations.
        """
        if isinstance(value, bool):
            return "T" if value else "F"
        elif isinstance(value, float):
            if value >= 1:
                # Example: 12.0 -> "12", 1.5 -> "1,5", 3.14 -> "3,14"
                return format(value, '.2f').rstrip('0').rstrip('.').replace('.', ',')
            else:
                # Example: 0.0001234 -> "1,234e-04"
                return format(value, '.4g').replace('.', ',')
        elif isinstance(value, int):
            # Example: 42 -> "42"
            return str(value)
        elif isinstance(value, str):
            # Example: "Scheduler" -> "sche"
            return value.lower()[:6]
        else:
            return str(value)

    _KEY_ABBREV = _KEY_ABBREV_FORWARD
    _VALUE_ABBREV = _VALUE_ABBREV_FORWARD

    param_parts = [f"TX{format_value(time_max)}"]

    def abbrev_key(k: str) -> str:
        return _KEY_ABBREV.get(k, k[:4].lower())

    def abbrev_value(v) -> str:
        if isinstance(v, str):
            return _VALUE_ABBREV.get(v, format_value(v))
        return format_value(v)

    def sort_key(k: str) -> tuple:
        if k in first_keys:
            return (0, first_keys.index(k))
        elif k in priority_keys:
            return (1, priority_keys.index(k))
        elif k in last_keys:
            return (3, last_keys.index(k))
        else:
            return (2, k)

    for key in sorted(config.keys(), key=sort_key):
        value = config[key]
        if value is not None:
            param_parts.append(f"{abbrev_key(key)}{abbrev_value(value)}")
    if seed is not None:
        # Trailing seed tag is peeled by parse_model_dir_to_cfg before parameter parsing.
        param_parts.append(f"seed{int(seed)}")
    return data_name + "_" + version + "_" + "_".join(param_parts)


def parse_model_dir_to_cfg(
    dir_name: str,
    experiment_type: str,
    configs_root: str,
    ref_cfg_override: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> typing.Dict[str, typing.Any]:
    """
    Reverse of get_dir_name_from_params. Reconstructs cfg from the checkpoint directory name by
    consulting the canonical reference YAML at <configs_root>/<experiment_type>/<version>.yaml for
    parameter ordering and types.

    Args:
        dir_name: Checkpoint directory name.
        experiment_type: Registered experiment name (e.g. ``"hawkes"``).
        configs_root: Root directory holding per-experiment / shared YAMLs.
        ref_cfg_override: Optional preloaded reference config dict. When provided
            (e.g. from :func:`load_experiment_config`), the function skips the
            on-disk YAML lookup and uses this dict to drive the parameter-set
            schema. This lets callers reuse the existing config-resolution
            machinery for repos whose layout is shared YAMLs instead of strict
            ``<experiment>/<version>.yaml`` files.

    Raises:
        ValueError: on key abbreviation collisions or unparseable tokens.
        FileNotFoundError: if the reference YAML is missing and no override is supplied.
    """
    tokens = dir_name.split("_")
    if len(tokens) < 3:
        raise ValueError(f"Unrecognised dir_name: {dir_name!r}")
    version = tokens[1]
    if not tokens[2].startswith("TX"):
        raise ValueError(f"Expected TX<time_max> as third token, got {tokens[2]!r}")

    # Multi-seed runs append a trailing `seed<N>` token (see get_dir_name_from_params).
    # Peel it before parameter parsing so the parser doesn't try to match it against
    # parameter_sets keys.
    parsed_seed: typing.Optional[int] = None
    if len(tokens) > 3:
        last = tokens[-1]
        if last.startswith("seed") and last[4:].isdigit():
            parsed_seed = int(last[4:])
            tokens = tokens[:-1]

    if ref_cfg_override is not None:
        ref_cfg = dict(ref_cfg_override)
    else:
        ref_yaml = os.path.join(configs_root, experiment_type, f"{version}.yaml")
        if not os.path.isfile(ref_yaml):
            raise FileNotFoundError(f"Reference YAML not found: {ref_yaml}")
        with open(ref_yaml, "r") as f:
            ref_cfg = yaml.safe_load(f)

    schema = ref_cfg.get("parameter_sets", {})
    if not isinstance(schema, dict):
        raise ValueError(f"Reference YAML {ref_yaml} has no `parameter_sets` dict")

    abbrev_to_key: typing.Dict[str, str] = {}
    for k in schema.keys():
        abbrev = _KEY_ABBREV_FORWARD.get(k, k[:4].lower())
        if abbrev in abbrev_to_key:
            raise ValueError(f"abbreviation collision: '{abbrev_to_key[abbrev]}' and '{k}' both map to '{abbrev}'")
        abbrev_to_key[abbrev] = k

    recovered: typing.Dict[str, typing.Any] = {}
    # Rejoin the parameter tokens and scan left-to-right so that abbreviations
    # ending with '_' (e.g. "hid_" for "hid_size_rep") are matched correctly.
    # Splitting "hid_32" by "_" gives ["hid", "32"]; rejoining and scanning
    # avoids that ambiguity.
    param_str = "_".join(t for t in tokens[3:] if t)
    sorted_abbrevs = sorted(abbrev_to_key, key=len, reverse=True)
    pos = 0
    while pos < len(param_str):
        while pos < len(param_str) and param_str[pos] == "_":
            pos += 1
        if pos >= len(param_str):
            break

        match = next(
            (a for a in sorted_abbrevs if param_str[pos:].startswith(a)),
            None,
        )
        if match is None:
            raise ValueError(
                f"Cannot resolve parameter string at {param_str[pos:]!r} "
                f"(full: {param_str!r}) against schema keys {list(abbrev_to_key)}"
            )
        pos += len(match)
        full_key = abbrev_to_key[match]
        if full_key in recovered:
            raise ValueError(
                f"Duplicate parameter token for {full_key!r} in {dir_name!r}: legacy "
                "names with two 'use_' tokens cannot be parsed unambiguously."
            )

        # Value ends at the next "_<abbreviation>" boundary.
        value_end = len(param_str)
        for a in sorted_abbrevs:
            idx = param_str.find("_" + a, pos)
            if 0 <= idx < value_end:
                value_end = idx
        raw_val = param_str[pos:value_end]
        pos = value_end

        ref_val = schema[full_key]
        # parameter_sets entries are lists of candidate values; use first element for type dispatch.
        type_sample = ref_val[0] if isinstance(ref_val, list) else ref_val
        if isinstance(type_sample, bool):
            recovered[full_key] = raw_val == "T"
        elif isinstance(type_sample, int):
            recovered[full_key] = int(raw_val)
        elif isinstance(type_sample, float):
            recovered[full_key] = float(raw_val.replace(",", "."))
        elif isinstance(type_sample, str):
            if raw_val in _INV_VALUE_ABBREV:
                recovered[full_key] = _INV_VALUE_ABBREV[raw_val]
            else:
                # Truncated string: validate against the reference, do NOT invent.
                if str(type_sample).lower()[:6] != raw_val:
                    raise ValueError(
                        f"Cannot recover string {full_key}={raw_val!r}; reference YAML has {type_sample!r}"
                    )
                recovered[full_key] = type_sample
        else:
            raise ValueError(f"Unsupported type for {full_key}: {type(type_sample).__name__}")

    out_cfg = dict(ref_cfg)
    out_cfg["parameter_sets"] = recovered
    out_cfg["experiment_type"] = experiment_type
    out_cfg["version"] = version
    if parsed_seed is not None:
        out_cfg["seeds"] = [parsed_seed]
    return out_cfg


def get_model_chkpt_path(datamodel_path: str) -> str:
    """Find the best checkpoint file in the given model directory, by lowest val_epdf."""
    try:
        chkpt_files = [f for f in os.listdir(datamodel_path) if f.startswith("model-") and f.endswith(".ckpt")]
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint directory not found: '{datamodel_path}'")

    if not chkpt_files:
        raise FileNotFoundError(
            f"No checkpoint files found in '{datamodel_path}'. "
            "Ensure the directory exists and contains files matching the pattern 'model-*.ckpt'."
        )

    def _val_epdf_from_name(fname: str) -> float:
        # Filename format: model-epoch=XXXX-val_epdf=Y.YYYY.ckpt
        try:
            return float(fname.split("val_epdf=")[1].replace(".ckpt", ""))
        except (IndexError, ValueError):
            return float("inf")  # unparseable name → sort last

    chkpt_files.sort(key=_val_epdf_from_name)
    best_checkpoint = os.path.join(datamodel_path, chkpt_files[0])
    logger.critical(f"Loading model from checkpoint: {best_checkpoint}")
    return best_checkpoint


def load_experiment_config(config_name: str) -> typing.Dict[str, typing.Any]:
    """Load config from a shared root-level model YAML, falling back to per-dataset config.

    Paths like ``"poisson_three_marks/sigtpp.yaml"`` (2 parts) resolve the shared
    model config at ``configs/sigtpp.yaml``.

    Paths like ``"poisson_three_marks/sigtpp/sigtpp_test.yaml"`` (3+ parts) first try
    the shared root-level config ``configs/sigtpp_test.yaml`` and, if it does not
    exist, fall back to the per-dataset config at
    ``configs/poisson_three_marks/sigtpp/sigtpp_test.yaml``.

    In both cases, if ``configs/<experiment>/experiment.yaml`` exists, its values form
    the base and model config values overwrite on conflict. This is how dataset-specific
    parameters (e.g. ``adjacency``, ``time_max``) are supplied without duplicating them
    in every model YAML.

    The returned config gets ``experiment_type`` injected from the path.
    """
    configs_root = Path(ROOT_DIR) / "test/paper_experiments/configs"
    config_rel = Path(config_name)
    parts = list(config_rel.parts)

    if len(parts) < 2:
        raise ValueError(f"Invalid config path '{config_name}'. Expected '<experiment>/<model_config>.yaml'.")

    experiment_type = parts[0]
    model_cfg_name = config_rel.name
    shared_model_path = configs_root / model_cfg_name

    if shared_model_path.exists():
        cfg = _load_yaml_file(str(shared_model_path))
        cfg["experiment_type"] = experiment_type
        logger.info(f"Loaded shared model config from {shared_model_path} with experiment_type='{experiment_type}'")
    else:
        # Fall back to per-dataset config (e.g. configs/poisson_three_marks/sigtpp/sigtpp_test.yaml)
        per_dataset_path = configs_root / config_rel
        if per_dataset_path.exists():
            cfg = _load_yaml_file(str(per_dataset_path))
            cfg["experiment_type"] = experiment_type
            logger.info(f"Loaded per-dataset config from {per_dataset_path} with experiment_type='{experiment_type}'")
        else:
            raise FileNotFoundError(
                f"No config found for '{config_name}'. "
                f"Tried shared: {shared_model_path}, per-dataset: {per_dataset_path}"
            )

    # Merge experiment.yaml if present (model config values take precedence)
    experiment_yaml_path = configs_root / experiment_type / "experiment.yaml"
    if experiment_yaml_path.exists():
        experiment_cfg = _load_yaml_file(str(experiment_yaml_path))
        experiment_cfg.pop("experiment_type", None)  # already injected from path
        cfg = {**experiment_cfg, **cfg}
        logger.debug(f"Merged experiment config from {experiment_yaml_path}")

    assert "n_bootstraps" in cfg, f"Config must define 'n_bootstraps'. Got keys {list(cfg)}."
    assert "seeds" in cfg, f"Config must define 'seeds', e.g. seeds: [42]. Got keys {list(cfg)}."
    assert (
        isinstance(cfg["seeds"], list) and cfg["seeds"]
    ), f"Config field 'seeds' must be a non-empty list, e.g. seeds: [42]. Got {cfg['seeds']!r}."

    return cfg


def _load_yaml_file(file_path: str) -> typing.Dict[str, typing.Any]:
    """Load and validate a YAML configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found at: {file_path}")

    try:
        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML config from {file_path}: {e}")

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a dictionary. Got: {type(cfg)}")

    return cfg


# Now instead of manually adding each experiment, you can use the decorator to register them.
# For register to work, import in trainingmanager.py
def register_experiment(name: str) -> typing.Callable:
    def decorator(func: typing.Callable) -> typing.Callable:
        EXPERIMENT_REGISTRY[name] = func()
        return func

    return decorator
