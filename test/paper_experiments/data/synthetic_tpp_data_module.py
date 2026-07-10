import logging
import os
import typing
from abc import ABC
from typing import Optional

import torch

logger = logging.getLogger(__name__)

from config import ROOT_DIR
from src.utils.tpp_utils import atomic_torch_save
from src.utils.utils_os import factory_fct_linked_path
from test.paper_experiments.data.tpp_data_module import TPPDataModule


class SyntheticTPPDataModule(TPPDataModule, ABC):
    """Intermediate base class for all synthetic (simulated) TPP datasets.

    Provides shared infrastructure that is identical across every synthetic
    data module:

    - ``_DATA_LINKER``: resolves paths under ``data/`` relative to ROOT_DIR,
      replaces the module-level ``path2file_linker`` that each subclass used
      to define independently.
    - ``_load_from_cache(filepath)``: try to restore a previously generated
      dataset from disk; return ``None`` on miss or corruption.
    - ``_save_to_cache(filepath, data_dict)``: persist an arbitrary dict of
      tensors/primitives to disk with directory creation and an INFO log.
    - ``_split_60_20_20(inputs, inputs_len)``: slice ``inputs`` and
      ``inputs_len`` into 60 / 20 / 20 train / val / test portions and return
      all six tensors.
    - ``__init__(batch_size)``: apply the 15 000-sample default and store
      ``self.batch_size`` so subclasses don't repeat the pattern.

    Subclasses must still implement:
    - ``format_filename(...)`` — deterministic cache filename from hyperparams
    - ``_get_cache_filename(...)`` — call ``format_filename`` + ``_DATA_LINKER``
    - simulation / generation logic
    - ``__init__`` — call ``super().__init__(batch_size=batch_size)``, then
      orchestrate cache lookup → simulate → split → assign marks
    """

    # Shared path resolver: points at <ROOT_DIR>/data/
    # Replaces the per-file `path2file_linker = factory_fct_linked_path(ROOT_DIR, "data")`
    # that every subclass used to define at module level.
    _DATA_LINKER = factory_fct_linked_path(ROOT_DIR, "data")

    DEFAULT_BATCH_SIZE: int = 15_000

    def __init__(self, batch_size: Optional[int] = None):
        super().__init__()
        self.batch_size = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_from_cache(filepath: str) -> typing.Optional[typing.Dict]:
        """Load a cached dataset dict from *filepath*, or return ``None``.

        Returns ``None`` when the file does not exist or cannot be loaded
        (corrupt / format mismatch), logging a WARNING in the latter case.
        """
        if os.path.exists(filepath):
            try:
                # weights_only=True is intentionally omitted: the parameter was
                # added in PyTorch 1.13 and is not available in torch 1.9.
                # Cache files are written by this codebase only (standard
                # tensors + Python primitives), so loading without the
                # restriction is safe.
                data = torch.load(filepath, map_location="cpu")
                logger.info("Loaded dataset from cache: %s", filepath)
                return data
            except Exception as e:
                logger.warning("Cache %s corrupt or incompatible, regenerating. Error: %s", filepath, e)
                return None
        return None

    @staticmethod
    def _save_to_cache(filepath: str, data_dict: dict) -> None:
        """Save *data_dict* to *filepath*, creating parent directories as needed.

        Uses atomic_torch_save: writes to a same-directory ``.tmp`` file and
        ``os.replace``s it onto *filepath*, so a crash mid-write never leaves a
        truncated/corrupt cache file behind (which `_load_from_cache` would
        otherwise hit on the next run, and which a stale-looking corrupt file
        could get committed and later resurrected via `git checkout`/rollback).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        atomic_torch_save(data_dict, filepath)
        logger.info("Saved dataset cache to %s", filepath)

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------

    @staticmethod
    def _split_60_20_20(
        inputs: torch.Tensor,
        inputs_len: torch.Tensor,
    ) -> typing.Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Split *inputs* and *inputs_len* into 60 / 20 / 20 portions.

        Returns:
            train_in, val_in, test_in, train_in_len, val_in_len, test_in_len
        """
        n = len(inputs)
        train_end = int(0.60 * n)
        val_end = int(0.80 * n)
        return (
            inputs[:train_end],
            inputs[train_end:val_end],
            inputs[val_end:],
            inputs_len[:train_end],
            inputs_len[train_end:val_end],
            inputs_len[val_end:],
        )
