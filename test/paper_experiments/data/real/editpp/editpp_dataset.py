import logging
import math
import sys
import typing

import torch

logger = logging.getLogger(__name__)

from config import ROOT_DIR
from src.utils.utils_os import factory_fct_linked_path
from test.paper_experiments.data.real.real_world_dataset import RealWorldDataModule


path2file_linker = factory_fct_linked_path(ROOT_DIR, "data")


class EditTPPDataModule(RealWorldDataModule):
    """
    Abstract base class for EDITPP datasets stored as .pkl files under data/editpp/.

    Each .pkl contains a dict with:
      - 'sequences': list of dicts, each with 'arrival_times' (cumulative, numpy array)
      - 't_max': float
      - 'mean_number_items': tensor

    Subclasses must set DATASET_FILE (the .pkl filename under data/editpp/).
    All EDITPP datasets in this repo are unmarked (num_marks=1).
    Position 0 is a synthetic anchor at t=0, never a real event. This matches
    the synthetic-dataset convention and differs from EasyTPPDataModule, where
    position 0 is the anchor at t=0.
    """

    train_in = val_in = test_in = None
    train_in_len = val_in_len = test_in_len = None
    time_max = num_marks = None
    train_marks = val_marks = test_marks = None

    DATASET_FILE: str  # e.g. "<dataset>.pkl"
    DATASET_NAME: str  # e.g. "<dataset>"

    # 60/20/20 split, matching the shared synthetic dataset convention.
    TRAIN_FRAC: float = 0.60
    VAL_FRAC: float = 0.20
    DEFAULT_BATCH_SIZE: int = 15_000

    # Subclasses may set this to cap sequence length (inclusive of anchor).
    # None means no filtering. Use to prevent OOM during test-phase sampling
    # when the dataset has very long-tailed sequence lengths.
    MAX_SEQ_LEN: typing.Optional[int] = None

    def __init__(self, *, batch_size: typing.Optional[int] = None, seed: int = 42):
        super().__init__()
        self.batch_size = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE
        self._seed = seed
        self._load_data()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_empty_sequences(sequences) -> typing.List[typing.Dict[str, typing.Any]]:
        """Drop anchor-only sequences; they carry no training signal."""
        return [sample for sample in sequences if len(sample["arrival_times"]) > 0]

    @classmethod
    def _split_train_val_test(
        cls,
        inputs: torch.Tensor,
        inputs_len: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split tensors into 60 / 20 / 20 train / val / test partitions."""
        n = inputs.shape[0]
        train_end = int(cls.TRAIN_FRAC * n)
        val_end = train_end + int(cls.VAL_FRAC * n)
        return (
            inputs[:train_end],
            inputs[train_end:val_end],
            inputs[val_end:],
            inputs_len[:train_end],
            inputs_len[train_end:val_end],
            inputs_len[val_end:],
        )

    @staticmethod
    def _convert_sequences(sequences) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list of per-sequence dicts to canonical (N, L+1, 1) tensors.

        Each sequence's arrival_times holds cumulative event times without an anchor.
        We prepend 0.0 as the anchor, matching the project's (N, L+1, 1) contract.

        Returns
        -------
        inputs     : (N, L+1, 1) float32 - cumulative times, constant-padded
        inputs_len : (N,) long          - sequence length including anchor
        """
        num_sequences = len(sequences)
        # +1 for the anchor prepended at position 0
        max_length = max(len(sample["arrival_times"]) for sample in sequences) + 1

        inputs = torch.zeros((num_sequences, max_length, 1), dtype=torch.float32)
        inputs_len = torch.zeros(num_sequences, dtype=torch.long)

        for i, sample in enumerate(sequences):
            arr = torch.tensor(sample["arrival_times"], dtype=torch.float32)
            seq_len = len(arr) + 1  # +1 for anchor at position 0

            inputs[i, 0, 0] = 0.0  # anchor
            inputs[i, 1:seq_len, 0] = arr
            inputs_len[i] = seq_len

            # Constant-pad remaining positions with the last valid cumulative time.
            if seq_len < max_length:
                inputs[i, seq_len:, 0] = arr[-1].item()

        return inputs, inputs_len

    # ------------------------------------------------------------------
    # Main loading routine
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        if not hasattr(self, "DATASET_FILE") or not isinstance(self.DATASET_FILE, str):
            raise NotImplementedError(f"{type(self).__name__} must define a string class attribute DATASET_FILE.")
        if not hasattr(self, "DATASET_NAME") or not isinstance(getattr(self, "DATASET_NAME", None), str):
            # Keep EDITPP report naming aligned with EasyTPP modules without
            # forcing every concrete subclass to repeat the stem manually.
            suffix = ".pkl"
            name = self.DATASET_FILE
            self.DATASET_NAME = name[: -len(suffix)] if name.endswith(suffix) else name

        pkl_path = path2file_linker(["editpp", self.DATASET_FILE])
        raw = torch.load(pkl_path, map_location="cpu")

        sequences = raw["sequences"]

        # Optional length cap: drop sequences longer than MAX_SEQ_LEN (including anchor).
        if self.MAX_SEQ_LEN is not None:
            before = len(sequences)
            sequences = [sample for sample in sequences if len(sample["arrival_times"]) + 1 <= self.MAX_SEQ_LEN]
            logger.info(
                "%s: filtered %d -> %d sequences (MAX_SEQ_LEN=%d)",
                self.DATASET_FILE,
                before,
                len(sequences),
                self.MAX_SEQ_LEN,
            )

        before_non_empty = len(sequences)
        sequences = self._drop_empty_sequences(sequences)
        dropped_empty = before_non_empty - len(sequences)
        if dropped_empty > 0:
            logger.warning("%s: dropped %d anchor-only sequences.", self.DATASET_FILE, dropped_empty)
        assert sequences, (
            f"{type(self).__name__}: no non-empty sequences remain after filtering; "
            "check MAX_SEQ_LEN and the raw dataset contents."
        )

        inputs, inputs_len = self._convert_sequences(sequences)

        generator = torch.Generator().manual_seed(self._seed)
        permutation = torch.randperm(inputs.shape[0], generator=generator)
        inputs = inputs[permutation]
        inputs_len = inputs_len[permutation]

        (
            self.train_in,
            self.val_in,
            self.test_in,
            self.train_in_len,
            self.val_in_len,
            self.test_in_len,
        ) = self._split_train_val_test(inputs, inputs_len)
        assert len(self.train_in) > 0, (
            f"{type(self).__name__}: training split is empty after the 60/20/20 split; "
            "dataset is too small after filtering."
        )

        self.train_in, self.train_in_len = self.jitter_zero_interarrival_times(
            self.train_in,
            self.train_in_len,
            seed=self._seed,
        )
        self.val_in, self.val_in_len = self.jitter_zero_interarrival_times(
            self.val_in,
            self.val_in_len,
            seed=self._seed + 1,
        )
        self.test_in, self.test_in_len = self.jitter_zero_interarrival_times(
            self.test_in,
            self.test_in_len,
            seed=self._seed + 2,
        )

        last_valid = self.train_in[torch.arange(len(self.train_in)), self.train_in_len - 1, 0]
        self.time_max = float(math.ceil(last_valid.max().item()))
        assert self.time_max > 0, (
            f"{type(self).__name__}: time_max is non-positive " f"({self.time_max}); training split may be degenerate."
        )

        self.num_marks = 1
        l_plus_1 = self.train_in.shape[1]
        self.train_marks = torch.zeros(self.train_in.shape[0], l_plus_1, dtype=torch.long)
        self.val_marks = torch.zeros(self.val_in.shape[0], l_plus_1, dtype=torch.long)
        self.test_marks = torch.zeros(self.test_in.shape[0], l_plus_1, dtype=torch.long)

        logger.info(
            "%s loaded: train=%d, val=%d, test=%d, time_max=%.4f, num_marks=%d",
            self.DATASET_FILE,
            len(self.train_in),
            len(self.val_in),
            len(self.test_in),
            self.time_max,
            self.num_marks,
        )


def preview_editpp_datamodule(
    datamodule_cls: typing.Type[EditTPPDataModule],
    *,
    seed: int = 42,
    max_diagnostic_paths: int = 50,
) -> None:
    """Build one EDITPP datamodule and write the shared diagnostic report to disk."""
    from src.logger.init_logger import set_config_logging

    set_config_logging()
    from src.diagnostics.dataset_diagnostics import export_dataset_report

    dm = datamodule_cls(seed=seed)
    print(f"num_marks={dm.num_marks}, time_max={dm.time_max:.2f}")
    export_dataset_report(dm, max_paths=max_diagnostic_paths, fig_format="svg", preview=False)


if __name__ == "__main__":
    from src.logger.init_logger import set_config_logging
    from test.paper_experiments.data.real.editpp.yelp_mississauga_dataset import YelpMississaugaDataModule

    dataset_registry: typing.Dict[str, typing.Type[EditTPPDataModule]] = {
        "yelp_mississauga": YelpMississaugaDataModule,
    }
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "yelp_mississauga"
    if dataset_name not in dataset_registry:
        available = ", ".join(dataset_registry)
        raise ValueError(f"Unknown EDITPP dataset '{dataset_name}'. Available datasets: {available}.")

    set_config_logging()
    preview_editpp_datamodule(dataset_registry[dataset_name])
