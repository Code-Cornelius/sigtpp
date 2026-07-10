import logging
import math
import os
import typing

import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


from config import ROOT_DIR
from src.utils.utils_os import factory_fct_linked_path
from test.paper_experiments.data.real.real_world_dataset import RealWorldDataModule


path2file_linker = factory_fct_linked_path(ROOT_DIR, "data")


class EasyTPPDataModule(RealWorldDataModule):
    """
    Abstract base class for EasyTPP HuggingFace datasets.

    Subclasses must set DATASET_NAME (e.g. "taxi") and inherit everything else.
    Subclasses may override HF_SPLIT_MAP if the HuggingFace dataset uses
    non-standard split names (e.g. "dev" instead of "validation").
    """

    # Class-level stubs satisfying TPPDataModule abstract properties
    train_in = val_in = test_in = None
    train_in_len = val_in_len = test_in_len = None
    time_max = num_marks = None
    train_marks = val_marks = test_marks = None

    DATASET_NAME: str  # Must be set by each subclass, e.g. "taxi"
    # Hugging Face Hub revision (commit SHA) to pin the dataset to. None means
    # "whatever upstream currently serves on main", which is not reproducible.
    # Subclasses set a full commit SHA so paper numbers are frozen against a
    # specific immutable snapshot even if easytpp re-uploads the data.
    DATASET_REVISION: typing.Optional[str] = None
    # Divide all cumulative times by this factor after loading.
    # Override in a subclass to rescale time units (e.g. TIME_UNIT = 3600 for seconds → hours).
    TIME_UNIT: float = 1.0
    # Maps internal split names to HuggingFace split names.
    # Override in a subclass if the dataset uses e.g. "dev" instead of "validation".
    HF_SPLIT_MAP: typing.Dict[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    DEFAULT_BATCH_SIZE = 10_000

    def __init__(self, *, batch_size: typing.Optional[int] = None, zero_marks: bool = False):
        super().__init__()
        self.batch_size = batch_size if batch_size is not None else EasyTPPDataModule.DEFAULT_BATCH_SIZE
        self.zero_marks = zero_marks
        self._load_data()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_dir(self) -> str:
        return path2file_linker(["easytpp", self.DATASET_NAME])

    def _cache_path(self, split: str) -> str:
        return os.path.join(self._cache_dir(), f"{split}.pt")

    def _load_split_from_cache(self, split: str) -> typing.Optional[typing.Dict]:
        path = self._cache_path(split)
        if os.path.exists(path):
            logger.debug("Loading %s/%s from cache: %s", self.DATASET_NAME, split, path)
            return torch.load(path, map_location="cpu")
        return None

    def _save_split_to_cache(self, split: str, data: typing.Dict) -> None:
        path = self._cache_path(split)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)
        logger.debug("Saved %s/%s to cache: %s", self.DATASET_NAME, split, path)
        return

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_split(
        split_data,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor], int]:
        """
        Convert one EasyTPP split to the project's tensor format.

        Returns
        -------
        inputs      : (N, L+1, 1) float32: cumulative times, constant-padded with last valid value
        inputs_len  : (N,)        long   : sequence length
        inputs_marks: (N, L+1)    long   : event types, or None
        num_marks   : int                : number of distinct event types (dim_process)
        """
        num_sequences = len(split_data)
        max_length = max(sample["seq_len"] for sample in split_data)
        padded_length = max_length + 1  # keep one extra padded slot beyond the longest valid sequence

        inputs = torch.zeros((num_sequences, padded_length, 1), dtype=torch.float32)
        inputs_len = torch.zeros(num_sequences, dtype=torch.long)

        has_marks = "type_event" in split_data[0]
        if has_marks:
            inputs_marks = torch.zeros((num_sequences, padded_length), dtype=torch.long)
            num_marks = int(split_data[0]["dim_process"])
        else:
            inputs_marks = None
            num_marks = 1

        for i, sample in enumerate(split_data):
            seq_len = sample["seq_len"]

            # EasyTPP uses position 0 as the anchor at t=0 (despite being a true event).
            inputs[i, :seq_len, 0] = torch.tensor(sample["time_since_start"], dtype=torch.float32)
            inputs_len[i] = seq_len

            # Propagate last valid cumulative time into padding positions.
            # Guard: if seq_len==0 there are no events, padding stays 0.0.
            if seq_len < padded_length and seq_len > 0:
                last_cum_time = float(sample["time_since_start"][-1])
                inputs[i, seq_len:, 0] = last_cum_time

            if inputs_marks is not None:
                inputs_marks[i, :seq_len] = torch.tensor(sample["type_event"], dtype=torch.long)

        return inputs, inputs_len, inputs_marks, num_marks

    # ------------------------------------------------------------------
    # Main loading routine
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        # Fail early when a dataset subclass does not declare its HuggingFace name.
        if not hasattr(self, "DATASET_NAME") or not isinstance(self.DATASET_NAME, str):
            raise NotImplementedError(f"{type(self).__name__} must define a string class attribute DATASET_NAME.")

        splits = ("train", "validation", "test")
        loaded: typing.Dict[str, typing.Dict] = {}

        missing = []
        for split in splits:
            cached = self._load_split_from_cache(split)
            if cached is not None:
                loaded[split] = cached
            else:
                missing.append(split)

        # Download and convert only the splits not already cached
        if missing:
            logger.info(
                "Cache missing %s for %s: downloading from HuggingFace (revision=%s) and converting.",
                missing,
                self.DATASET_NAME,
                self.DATASET_REVISION,
            )
            dataset = load_dataset(f"easytpp/{self.DATASET_NAME}", revision=self.DATASET_REVISION)
            for split in missing:
                # Map local split names to the corresponding HuggingFace split keys.
                hf_key = self.HF_SPLIT_MAP[split]
                inputs, inputs_len, inputs_marks, num_marks = self._convert_split(dataset[hf_key])
                cache_entry = {
                    "inputs": inputs,
                    "inputs_len": inputs_len,
                    "num_marks": num_marks,
                }
                if inputs_marks is not None:
                    cache_entry["inputs_marks"] = inputs_marks
                self._save_split_to_cache(split, cache_entry)
                loaded[split] = cache_entry

        # --- Populate attributes ---
        def _unpack(split: str):
            d = loaded[split]
            marks = d.get("inputs_marks", None)
            # Handle old cached format where marks had shape (N, L+1, 1) instead of (N, L+1).
            if marks is not None and marks.dim() == 3:
                marks = marks.squeeze(-1)
            return d["inputs"], d["inputs_len"], marks, d["num_marks"]

        self.train_in, self.train_in_len, self.train_marks, self.num_marks = _unpack("train")
        self.val_in, self.val_in_len, self.val_marks, val_num_marks = _unpack("validation")
        self.test_in, self.test_in_len, self.test_marks, test_num_marks = _unpack("test")
        assert (
            len(self.train_in) > 0
        ), f"{type(self).__name__}: training split is empty; training split may be degenerate."

        if val_num_marks != self.num_marks or test_num_marks != self.num_marks:
            raise ValueError(
                f"{self.DATASET_NAME}: num_marks mismatch across splits: "
                f"train={self.num_marks}, val={val_num_marks}, test={test_num_marks}. "
                "Delete cached .pt files and reload."
            )

        self.train_in, self.train_in_len = self.jitter_zero_interarrival_times(
            self.train_in,
            self.train_in_len,
            inputs_marks=self.train_marks,
            seed=42,
        )
        self.val_in, self.val_in_len = self.jitter_zero_interarrival_times(
            self.val_in,
            self.val_in_len,
            inputs_marks=self.val_marks,
            seed=43,
        )
        self.test_in, self.test_in_len = self.jitter_zero_interarrival_times(
            self.test_in,
            self.test_in_len,
            inputs_marks=self.test_marks,
            seed=44,
        )

        if self.TIME_UNIT != 1.0:
            self.train_in = self.train_in / self.TIME_UNIT
            self.val_in = self.val_in / self.TIME_UNIT
            self.test_in = self.test_in / self.TIME_UNIT

        # Use the last valid cumulative time per sequence instead of a global .max(), which
        # would be fragile if padding were ever migrated from constant to NaN.
        last_valid = self.train_in[torch.arange(len(self.train_in)), self.train_in_len - 1, 0]
        self.time_max = float(math.ceil(last_valid.max().item()))
        assert self.time_max > 0, (
            f"{type(self).__name__}: time_max is non-positive " f"({self.time_max}); training split may be degenerate."
        )

        # Ensure marks are never None (3-tuple contract). Create all-zeros when unmarked.
        for attr, in_attr in [
            ("train_marks", "train_in"),
            ("val_marks", "val_in"),
            ("test_marks", "test_in"),
        ]:
            if getattr(self, attr) is None:
                t = getattr(self, in_attr)
                setattr(self, attr, torch.zeros(t.shape[0], t.shape[1], dtype=torch.long))

        # When zero_marks is set, override all marks to zeros and report num_marks=1.
        if self.zero_marks:
            self.num_marks = 1
            self.train_marks = torch.zeros(self.train_in.shape[0], self.train_in.shape[1], dtype=torch.long)
            self.val_marks = torch.zeros(self.val_in.shape[0], self.val_in.shape[1], dtype=torch.long)
            self.test_marks = torch.zeros(self.test_in.shape[0], self.test_in.shape[1], dtype=torch.long)

        logger.info(
            "%s loaded: train=%d, val=%d, test=%d, time_max=%.4f, num_marks=%d",
            self.DATASET_NAME,
            len(self.train_in),
            len(self.val_in),
            len(self.test_in),
            self.time_max,
            self.num_marks,
        )
        return
