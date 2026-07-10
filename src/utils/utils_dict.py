import logging
import typing
from typing import Any, Dict


def _summarize_container(value: Any, max_items: int = 8) -> str:
    """Return a compact representation for large containers in error messages."""
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = ", ".join(repr(key) for key in keys[:max_items])
        if len(keys) > max_items:
            preview += ", ..."
        return f"dict(len={len(value)}, keys=[{preview}])"

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        preview = ", ".join(repr(item) for item in items[:max_items])
        if len(items) > max_items:
            preview += ", ..."
        return f"{type(value).__name__}(len={len(items)}, items=[{preview}])"

    return repr(value)


def verbose_get(
    d: Dict[str, Any], key: str, logger: typing.Optional[logging.Logger] = None, default: typing.Optional[Any] = None
) -> Any:
    """
    Retrieve a value from a dictionary. If the key is missing:
    - log a warning and return default if default is provided
    - raise KeyError if no default is provided

    Parameters:
    - d: the dictionary to retrieve the key from
    - key: the key to look up
    - default: the default value to use if key is missing
    - logger: optional logger to emit a warning

    Returns:
    - The value associated with the key, or the default if provided

    Raises:
    - KeyError if the key is missing and no default is provided
    """
    if key in d:
        return d[key]
    container_summary = _summarize_container(d)
    if default is not None:
        if logger:
            logger.warning(f"Missing key '{key}' from {container_summary}. Using default: {default!r}")
        return default
    if logger:
        logger.error(f"Missing key '{key}' from {container_summary} and no default provided.")
    raise KeyError(f"Required key '{key}' missing from {container_summary} and no default provided.")
