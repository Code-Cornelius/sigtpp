import logging
import typing

import numpy as np
import torch

logger = logging.getLogger(__name__)

from test.paper_experiments.data.synthetic_tpp_data_module import SyntheticTPPDataModule
from src.generators import hp, ihp


class PoissonDataModule(SyntheticTPPDataModule):
    """Synthetic Poisson / Hawkes dataset with automatic on-disk caching, optionally with marks."""

    # Class-level stubs satisfying TPPDataModule abstract properties
    train_in = val_in = test_in = None
    train_in_len = val_in_len = test_in_len = None
    time_max = num_marks = None
    train_marks = val_marks = test_marks = None

    MARK_PROBS_DECIMALS = 4

    @staticmethod
    def _canonicalize_mark_probs(
        mark_probs: typing.Optional[np.ndarray], decimals: int = MARK_PROBS_DECIMALS
    ) -> typing.Optional[np.ndarray]:
        if mark_probs is None:
            return None

        probs = np.asarray(mark_probs, dtype=np.float64).copy()
        if probs.ndim != 1:
            raise ValueError(f"mark_probs must be a 1D array, got shape {probs.shape}")
        if np.any(probs < 0):
            raise ValueError(f"mark_probs must be non-negative, got {probs}")
        if probs.sum() <= 0:
            raise ValueError(f"mark_probs must have positive sum, got {probs}")

        rounded = np.round(probs, decimals=decimals)
        if rounded.size > 1:
            rounded[-1] = np.round(1.0 - rounded[:-1].sum(), decimals=decimals)
        if np.any(rounded < 0):
            raise ValueError(
                f"Rounded mark_probs became invalid at {decimals} decimals: {rounded}. " "Use more decimal precision."
            )
        if not np.isclose(rounded.sum(), 1.0):
            raise ValueError(f"Rounded mark_probs must sum to 1, got {rounded.sum()} for {rounded}")
        return rounded

    @staticmethod
    def _format_mark_probs_token(mark_probs: np.ndarray, decimals: int = MARK_PROBS_DECIMALS) -> str:
        probs = PoissonDataModule._canonicalize_mark_probs(mark_probs, decimals=decimals)
        parts = [f"{prob:.{decimals}f}".rstrip("0").rstrip(".").replace(".", "p") for prob in probs]
        return "probs_" + "_".join(parts)

    @staticmethod
    def format_filename(
        seed: int,
        use_IHP_or_HP: bool,
        data_size: int,
        tmax: float,
        num_marks: int = 1,
        mark_probs: typing.Optional[np.ndarray] = None,
        prefix="poisson_one_mark",
        base_intensity: float = 1.0,
    ):
        tmax_str = str(tmax).replace('.', ',')  # 10.0 -> 10,0
        lam_str = str(base_intensity).replace('.', ',')
        if num_marks > 1:
            if mark_probs is not None:
                probs_token = PoissonDataModule._format_mark_probs_token(mark_probs)
                filename = (
                    f"{prefix}_seed{seed}_ihp{use_IHP_or_HP}_size{data_size}"
                    f"_tmax{tmax_str}_lam{lam_str}_marks{num_marks}_{probs_token}.pt"
                )
            else:
                # Uniform distribution (no hash needed)
                filename = f"{prefix}_seed{seed}_ihp{use_IHP_or_HP}_size{data_size}_tmax{tmax_str}_lam{lam_str}_marks{num_marks}.pt"
        else:
            filename = f"{prefix}_seed{seed}_ihp{use_IHP_or_HP}_size{data_size}_tmax{tmax_str}_lam{lam_str}.pt"
        return filename

    @staticmethod
    def _get_cache_filename(
        seed: int,
        use_ihp: bool,
        data_size: int,
        time_max: float,
        num_marks: int = 1,
        mark_probs: typing.Optional[np.ndarray] = None,
        base_intensity: float = 1.0,
    ) -> str:
        prefix = "poisson_one_mark" if num_marks <= 1 else f"poisson_{num_marks}_marks"
        filename = PoissonDataModule.format_filename(
            seed=seed,
            use_IHP_or_HP=use_ihp,
            data_size=data_size,
            tmax=time_max,
            num_marks=num_marks,
            mark_probs=mark_probs,
            base_intensity=base_intensity,
            prefix=prefix,
        )
        return SyntheticTPPDataModule._DATA_LINKER(["synthetic", filename])

    def __init__(
        self,
        *,
        data_size: int,
        seed: int,
        use_IHP_or_HP: bool,
        batch_size: typing.Optional[int] = None,
        num_marks: int = 1,
        mark_probs: typing.Optional[np.ndarray] = None,
        base_intensity: float = 1.0,
        zero_marks: bool = False,
    ):
        super().__init__(batch_size=batch_size)
        self.use_IHP_or_HP = use_IHP_or_HP
        self.time_max = 10.0 if use_IHP_or_HP else 12.0
        self.base_intensity: float = base_intensity
        self.data_size = data_size
        self.zero_marks = zero_marks
        self.num_marks = num_marks
        self.mark_probs = self._canonicalize_mark_probs(mark_probs)

        cache_file = self._get_cache_filename(
            seed,
            self.use_IHP_or_HP,
            self.data_size,
            self.time_max,
            self.num_marks,
            self.mark_probs,
            self.base_intensity,
        )
        cached = self._load_from_cache(cache_file)

        if cached:
            inputs = cached["inputs"]
            inputs_len = cached["inputs_len"]
            inputs_marks = cached.get("inputs_marks", None)
            # Verify num_marks matches if cached
            cached_num_marks = cached.get("num_marks", 1)
            if cached_num_marks != self.num_marks:
                logger.warning(
                    f"Cached dataset has num_marks={cached_num_marks} but requested num_marks={self.num_marks}. "
                    "Using cached value."
                )
                self.num_marks = cached_num_marks
            # Verify mark_probs matches if cached
            cached_mark_probs = cached.get("mark_probs", None)
            if cached_mark_probs is not None:
                cached_mark_probs = cached_mark_probs.numpy()
            if cached_mark_probs is not None and self.mark_probs is not None:
                if not np.allclose(cached_mark_probs, self.mark_probs):
                    logger.warning(
                        f"Cached dataset has different mark_probs. "
                        f"Cached: {cached_mark_probs}, Requested: {self.mark_probs}. "
                        "Using cached value."
                    )
                    self.mark_probs = cached_mark_probs
            # Verify base_intensity matches if cached
            cached_base_intensity = cached.get("base_intensity", 1.0)
            if not np.isclose(cached_base_intensity, self.base_intensity):
                logger.warning(
                    f"Cached dataset has base_intensity={cached_base_intensity} but "
                    f"requested base_intensity={self.base_intensity}. Using cached value."
                )
                self.base_intensity = cached_base_intensity
        else:
            logger.info(f"No cached dataset found - generating afresh the dataset {cache_file}.")
            rng = np.random.default_rng(seed)

            if use_IHP_or_HP:
                # IHP:
                gen_result = ihp.gen(
                    self.data_size,
                    lambda t: 1 * self.base_intensity
                    + 1 * self.base_intensity * np.heaviside(t - self.time_max / 2.0, 0.0),
                    2.0 * self.base_intensity,
                    self.time_max,
                    None,
                    num_marks=self.num_marks,
                    mark_probs=self.mark_probs,
                    rng=rng,
                )
            else:
                # HP:
                gen_result = hp.gen(
                    self.data_size,
                    self.base_intensity,
                    self.time_max,
                    None,
                    num_marks=self.num_marks,
                    mark_probs=self.mark_probs,
                    rng=rng,
                )

            # Handle return value: tuple if marks, array if no marks
            if self.num_marks > 1:
                inter_times, marks = gen_result
                inter_times = torch.from_numpy(inter_times).float()
                marks = torch.from_numpy(marks).long()
            else:
                inter_times = torch.from_numpy(gen_result).float()
                marks = None

            inputs_len = (
                PoissonDataModule.get_sequence_lengths_from_zeros(inter_times) + 1
            )  # + 1 account for 0, that would make troubles in length computation.

            # Concatenate the zero tensor along the second axis (dim=1). That zero represents that the beginning of the sequence is at 0.
            anchor_time_seqs = torch.zeros(inter_times.size(0), 1, inter_times.shape[2])
            inter_times = torch.cat((anchor_time_seqs, inter_times), dim=1)
            inputs = inter_times.cumsum(1)

            # Handle marks: prepend a "0" mark for the anchor time
            # Squeeze last dim: generator returns (N, L, 1) but contract is (N, L+1).
            if marks is not None:
                marks_2d = marks.squeeze(-1)  # (N, L)
                anchor_marks = torch.zeros(marks_2d.size(0), 1, dtype=marks_2d.dtype)
                inputs_marks = torch.cat((anchor_marks, marks_2d), dim=1)  # (N, L+1)
            else:
                inputs_marks = None

            # Assemble cache dict at the call site (keys vary per dataset)
            cache_data: dict = {
                "inputs": inputs,
                "inputs_len": inputs_len,
                "num_marks": self.num_marks,
                "base_intensity": self.base_intensity,
            }
            if inputs_marks is not None:
                cache_data["inputs_marks"] = inputs_marks
            if self.mark_probs is not None:
                cache_data["mark_probs"] = torch.from_numpy(self.mark_probs)
            self._save_to_cache(cache_file, cache_data)

        # Split data into training, validation, and test sets
        (self.train_in, self.val_in, self.test_in, self.train_in_len, self.val_in_len, self.test_in_len) = (
            self._split_60_20_20(inputs, inputs_len)
        )

        # Split marks: real marks when num_marks > 1, all-zeros otherwise (3-tuple contract).
        if inputs_marks is not None:
            n = len(inputs)
            train_end = int(0.60 * n)
            val_end = int(0.80 * n)
            self.train_marks = inputs_marks[:train_end]
            self.val_marks = inputs_marks[train_end:val_end]
            self.test_marks = inputs_marks[val_end:]
        else:
            L_plus_1 = inputs.shape[1]
            self.train_marks = torch.zeros(self.train_in.shape[0], L_plus_1, dtype=torch.long)
            self.val_marks = torch.zeros(self.val_in.shape[0], L_plus_1, dtype=torch.long)
            self.test_marks = torch.zeros(self.test_in.shape[0], L_plus_1, dtype=torch.long)

        # When zero_marks is set, override all marks to zeros and report num_marks=1.
        if self.zero_marks:
            L_plus_1 = inputs.shape[1]
            self.num_marks = 1
            self.train_marks = torch.zeros(self.train_in.shape[0], L_plus_1, dtype=torch.long)
            self.val_marks = torch.zeros(self.val_in.shape[0], L_plus_1, dtype=torch.long)
            self.test_marks = torch.zeros(self.test_in.shape[0], L_plus_1, dtype=torch.long)

    def _build_fresh(self, seed: int) -> "PoissonDataModule":
        return PoissonDataModule(
            data_size=self.data_size,
            seed=seed,
            use_IHP_or_HP=self.use_IHP_or_HP,
            batch_size=self.batch_size,
            base_intensity=self.base_intensity,
            num_marks=self.num_marks,
            mark_probs=self.mark_probs,
            zero_marks=self.zero_marks,
        )


if __name__ == '__main__':
    from src.logger.init_logger import set_config_logging

    set_config_logging()
    from src.diagnostics.dataset_diagnostics import export_dataset_report

    dm = PoissonDataModule(data_size=10_000, seed=142, use_IHP_or_HP=True)
    export_dataset_report(dm, fig_format="svg", preview=True)
