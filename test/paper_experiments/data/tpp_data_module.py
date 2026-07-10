import logging
from abc import ABC, abstractmethod

import torch
from pytorch_lightning import LightningDataModule

logger = logging.getLogger(__name__)

from src.utils.fast_tensor_data_loader import FastTensorDataLoader


class TPPDataModule(LightningDataModule, ABC):
    """
    Universal base class for all TPP dataset modules (real-world and synthetic).

    Subclasses must satisfy the 11 abstract properties below by declaring
    class-level stubs and setting the real values during initialisation
    (either in __init__ directly or in a _load_data() method called from __init__):

        class MyDataModule(TPPDataModule):
            train_in = val_in = test_in = None
            train_in_len = val_in_len = test_in_len = None
            time_max = num_marks = None
            train_marks = val_marks = test_marks = None

            def __init__(self):
                super().__init__()
                ...
                self.train_in = ...   # (N, L+1, 1) float32
                self.num_marks = 1    # MUST be int, not None
                ...

    Python's ABC mechanism raises TypeError at instantiation if any stub is missing.

    Important: num_marks must be set to an int (1 for unmarked processes).
    Leaving it as None from the stub will cause downstream TypeErrors.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def train_in(self) -> torch.Tensor:
        """(N, L+1, 1) float32: cumulative times, constant-padded."""
        pass

    @property
    @abstractmethod
    def val_in(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def test_in(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def train_in_len(self) -> torch.Tensor:
        """(N,) long: sequence lengths including anchor."""
        pass

    @property
    @abstractmethod
    def val_in_len(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def test_in_len(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def time_max(self) -> float:
        """Max cumulative time in training data."""
        pass

    @property
    @abstractmethod
    def num_marks(self) -> int:
        """Number of distinct event types (1 if no marks)."""
        pass

    @property
    @abstractmethod
    def train_marks(self) -> torch.Tensor:
        """(N, L+1) long: event types. All-zeros when num_marks == 1."""
        pass

    @property
    @abstractmethod
    def val_marks(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def test_marks(self) -> torch.Tensor:
        pass

    # ------------------------------------------------------------------
    # Dataloaders: shared across all datasets; override only if needed
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return FastTensorDataLoader(self.train_in, self.train_in_len, self.train_marks, batch_size=self.batch_size)

    def val_dataloader(self):
        return FastTensorDataLoader(self.val_in, self.val_in_len, self.val_marks, batch_size=self.batch_size)

    def test_dataloader(self):
        return FastTensorDataLoader(self.test_in, self.test_in_len, self.test_marks, batch_size=int(1e9))

    # ------------------------------------------------------------------
    # Sequence-length utilities: shared across all datasets
    # ------------------------------------------------------------------

    @staticmethod
    def get_sequence_lengths_from_zeros(inter_times: torch.Tensor) -> torch.Tensor:
        """Compute lengths from zero-padded inter-arrival times.

        Args:
            inter_times: (N, L, 1) float: zero marks end of valid data.

        Returns:
            (N,) long: number of valid time steps per sequence.
        """
        assert inter_times.ndimension() == 3, (
            f"Expected 3 dimensions (N, L, 1), got {inter_times.ndimension()}. " f"Full shape: {inter_times.shape}"
        )
        assert inter_times.shape[2] == 1, (
            f"Expected last dimension to be 1, got {inter_times.shape[2]}. " f"Full shape: {inter_times.shape}"
        )
        from src.utils.get_index import set_zero_len2len, index_first_zero_torch

        tensor_lens = index_first_zero_torch(inter_times)
        tensor_lens = set_zero_len2len(tensor_lens, inter_times.shape[1])
        return tensor_lens

    @staticmethod
    def get_sequence_lengths_from_nans(tensor: torch.Tensor) -> torch.Tensor:
        """Compute lengths from NaN-padded cumulative times.

        Args:
            tensor: (N, L, 1) float: NaN marks end of valid data.

        Returns:
            (N,) long: number of valid time steps per sequence.
        """
        assert tensor.ndimension() == 3, (
            f"Expected 3 dimensions (N, L, 1), got {tensor.ndimension()}. " f"Full shape: {tensor.shape}"
        )
        assert tensor.shape[2] == 1, (
            f"Expected last dimension to be 1, got {tensor.shape[2]}. " f"Full shape: {tensor.shape}"
        )
        squeezed = tensor.squeeze(dim=-1)
        lengths = torch.isnan(squeezed).int().argmax(dim=1)
        from src.utils.get_index import set_zero_len2len

        lengths = set_zero_len2len(lengths, tensor.shape[1])
        return lengths

    @staticmethod
    def subset_to_tensor(subset) -> torch.Tensor:
        """Stack a PyTorch Subset (or list of tensors) into a single tensor."""
        return torch.stack([x for x in subset])

    # ------------------------------------------------------------------
    # Plotting and dataset diagnostics live in `src/diagnostics/dataset_diagnostics`.
    # Use:
    #   from src.diagnostics.dataset_diagnostics import (
    #       export_dataset_report, plot_sample_paths, plot_sequence_marks, ...
    #   )
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Validation comparison: shared across synthetic datasets
    # ------------------------------------------------------------------

    def _build_fresh(self, seed: int) -> "TPPDataModule":
        """Build a fresh instance with the same parameters but a different seed.

        Subclasses that support compare_validation_to_fresh must override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _build_fresh(). "
            "Override it to enable compare_validation_to_fresh()."
        )

    def compare_validation_to_fresh(self, other_seed: int) -> dict:
        """Compare training data to a freshly generated dataset with the same parameters.

        Returns dict with keys: mae, mape, mae_len_sorted, mape_len_sorted
        (and placeholder NaN slots for histogram, corr, autocorr, sigW1).
        """
        from src.utils.fix_seq_ends import set_seq_to_nan_from_index
        from src.metrics.lebesgue_loss import get_L1loss, get_perc_L1loss

        fresh = self._build_fresh(other_seed)

        metrics = {
            "mae": float("nan"),
            "mape": float("nan"),
            "mae_len_sorted": float("nan"),
            "mape_len_sorted": float("nan"),
            "histogram": float("nan"),
            "histogram_cum": float("nan"),
            "corr": float("nan"),
            "corr_short": float("nan"),
            "autocorr": float("nan"),
            "autocorr_short": float("nan"),
            "autocorr_it": float("nan"),
            "autocorr_it_short": float("nan"),
            "sigW1": float("nan"),
        }

        x_self = self.train_in.diff(dim=1)[:, :, 0]
        x_fresh = fresh.train_in.diff(dim=1)[:, :, 0]
        len_self = self.train_in_len - 1
        len_fresh = fresh.train_in_len - 1

        # --- (A) Baseline: unsorted ---
        comparable_length = min(x_self.size(1), x_fresh.size(1))
        pair_min_len = torch.minimum(len_self, len_fresh)

        dt_self_m = set_seq_to_nan_from_index(x_self.unsqueeze(-1), pair_min_len - 1)
        dt_fresh_m = set_seq_to_nan_from_index(x_fresh.unsqueeze(-1), pair_min_len - 1)

        metrics["mae"] = get_L1loss(
            dt_self_m[:, :comparable_length],
            dt_fresh_m[:, :comparable_length],
        ).item()
        metrics["mape"] = get_perc_L1loss(
            dt_self_m[:, :comparable_length],
            dt_fresh_m[:, :comparable_length],
        ).item()

        # --- (B) Length-sorted pairing ---
        idx_self = torch.argsort(len_self)
        idx_fresh = torch.argsort(len_fresh)

        x_self_sorted = x_self.index_select(0, idx_self)
        x_fresh_sorted = x_fresh.index_select(0, idx_fresh)
        len_self_sorted = len_self.index_select(0, idx_self)
        len_fresh_sorted = len_fresh.index_select(0, idx_fresh)

        pair_min_len_sorted = torch.minimum(len_self_sorted, len_fresh_sorted)
        dt_self_sorted_m = set_seq_to_nan_from_index(x_self_sorted.unsqueeze(-1), pair_min_len_sorted)
        dt_fresh_sorted_m = set_seq_to_nan_from_index(x_fresh_sorted.unsqueeze(-1), pair_min_len_sorted)

        comparable_length_sorted = min(x_self_sorted.size(1), x_fresh_sorted.size(1))
        metrics["mae_len_sorted"] = get_L1loss(
            dt_self_sorted_m[:, :comparable_length_sorted],
            dt_fresh_sorted_m[:, :comparable_length_sorted],
        ).item()
        metrics["mape_len_sorted"] = get_perc_L1loss(
            dt_self_sorted_m[:, :comparable_length_sorted],
            dt_fresh_sorted_m[:, :comparable_length_sorted],
        ).item()

        return metrics
