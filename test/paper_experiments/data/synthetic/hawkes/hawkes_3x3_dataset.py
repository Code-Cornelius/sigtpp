import logging
import typing

import numpy as np
import torch

logger = logging.getLogger(__name__)

from test.paper_experiments.data.synthetic_tpp_data_module import SyntheticTPPDataModule


def _format_array_token(arr: np.ndarray) -> str:
    """Format a flat or multi-dimensional array as underscore-joined values, '.' replaced by ','."""
    parts = [str(float(v)).replace(".", ",") for v in np.asarray(arr).ravel()]
    return "_".join(parts)


class Hawkes3x3DataModule(SyntheticTPPDataModule):
    """Tick-backed 3x3 multivariate Hawkes process dataset with on-disk caching.

    K = 3 nodes are simulated using Tick's SimuHawkesExpKernels. All node
    timestamps are merged into one nondecreasing marked stream per sample, with
    the originating node id (0, 1, or 2) as the categorical mark.
    A time anchor at 0.0 and anchor mark 0 are prepended.
    """

    # Class-level stubs satisfying TPPDataModule abstract properties
    train_in = val_in = test_in = None
    train_in_len = val_in_len = test_in_len = None
    time_max = num_marks = None
    train_marks = val_marks = test_marks = None

    K = 3

    @staticmethod
    def format_filename(
        seed: int,
        data_size: int,
        time_max: float,
        baseline: np.ndarray,
        adjacency: np.ndarray,
        decays: np.ndarray,
    ) -> str:
        tmax_str = str(time_max).replace(".", ",")
        bl_str = _format_array_token(baseline)
        adj_str = _format_array_token(adjacency)
        dec_str = _format_array_token(decays)
        return f"hawkes_3x3_seed{seed}_size{data_size}_tmax{tmax_str}_bl{bl_str}_adj{adj_str}_dec{dec_str}.pt"

    @staticmethod
    def _get_cache_filename(
        seed: int,
        data_size: int,
        time_max: float,
        baseline: np.ndarray,
        adjacency: np.ndarray,
        decays: np.ndarray,
    ) -> str:
        filename = Hawkes3x3DataModule.format_filename(seed, data_size, time_max, baseline, adjacency, decays)
        return SyntheticTPPDataModule._DATA_LINKER(["synthetic", filename])

    def _simulate(
        self,
        n_paths: int,
        baseline: np.ndarray,
        adjacency: np.ndarray,
        decays: np.ndarray,
        time_max: float,
        seed: int,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate n_paths 3x3 Hawkes processes with Tick.

        Returns:
            cum_times: (N, L+1, 1) float32  — cumulative times, anchor at 0
            lengths:   (N,)        long     — valid length including anchor
            marks:     (N, L+1)   long     — node id {0,1,2}, anchor mark 0
        """
        try:
            from tick.hawkes import SimuHawkesExpKernels
        except ImportError as exc:
            raise ImportError(
                "tick is required to generate hawkes_3x3 data. " "Install tick or push the cached .pt file."
            ) from exc

        logger.info("Generating %d hawkes_3x3 sequences (seed=%d, tmax=%.1f)", n_paths, seed, time_max)

        all_times: typing.List[np.ndarray] = []
        all_marks_list: typing.List[np.ndarray] = []

        for i in range(n_paths):
            sim = SimuHawkesExpKernels(
                adjacency=adjacency,
                decays=decays,
                baseline=baseline,
                end_time=time_max,
                seed=seed + i,
                verbose=False,
            )
            sim.simulate()
            # sim.timestamps is a list of K arrays, one per node
            node_times: typing.List[np.ndarray] = sim.timestamps  # len = K

            # Merge all node timestamps into one sorted stream
            merged_times = np.concatenate([node_times[k] for k in range(self.K)])
            merged_marks = np.concatenate([np.full(len(node_times[k]), k, dtype=np.int64) for k in range(self.K)])

            # Sort by time; stable sort preserves node order for simultaneous events
            order = np.argsort(merged_times, kind="stable")
            merged_times = merged_times[order]
            merged_marks = merged_marks[order]

            all_times.append(merged_times.astype(np.float32))
            all_marks_list.append(merged_marks)

        lengths = np.array([len(t) for t in all_times], dtype=np.int64)
        max_len = int(lengths.max()) if lengths.max() > 0 else 0

        logger.info(
            "hawkes_3x3: min_len=%d max_len=%d mean_len=%.1f",
            lengths.min(),
            max_len,
            lengths.mean(),
        )

        # Pad times with last valid cumulative time; pad marks with 0
        times_padded = np.zeros((n_paths, max_len), dtype=np.float32)
        marks_padded = np.zeros((n_paths, max_len), dtype=np.int64)
        for i, (t, m) in enumerate(zip(all_times, all_marks_list)):
            L = len(t)
            if L > 0:
                times_padded[i, :L] = t
                times_padded[i, L:] = t[-1]
                marks_padded[i, :L] = m
            # If L == 0: zeros already set

        # Prepend anchor: time 0.0, mark 0
        anchor_t = np.zeros((n_paths, 1), dtype=np.float32)
        anchor_m = np.zeros((n_paths, 1), dtype=np.int64)
        cum_times = np.concatenate([anchor_t, times_padded], axis=1)  # (N, max_len+1)
        marks_arr = np.concatenate([anchor_m, marks_padded], axis=1)  # (N, max_len+1)

        cum_times_t = torch.from_numpy(cum_times).unsqueeze(-1)  # (N, L+1, 1)
        marks_t = torch.from_numpy(marks_arr)  # (N, L+1)
        lengths_t = torch.from_numpy(lengths).long() + 1  # +1 for anchor

        return cum_times_t, lengths_t, marks_t

    def __init__(
        self,
        *,
        data_size: int,
        seed: int,
        time_max: float = 20.0,
        baseline: typing.Sequence[float],
        adjacency: typing.Sequence,
        decays: typing.Sequence,
        batch_size: typing.Optional[int] = None,
    ):
        super().__init__(batch_size=batch_size)

        self._baseline = np.array(baseline, dtype=np.float64)
        self._adjacency = np.array(adjacency, dtype=np.float64)
        self._decays = np.array(decays, dtype=np.float64)

        assert self._baseline.shape == (3,), f"baseline must be shape (3,), got {self._baseline.shape}"
        assert self._adjacency.shape == (3, 3), f"adjacency must be shape (3,3), got {self._adjacency.shape}"
        assert self._decays.shape == (3, 3), f"decays must be shape (3,3), got {self._decays.shape}"

        self.time_max = float(time_max)
        self.data_size = data_size
        self.num_marks = self.K

        cache_file = self._get_cache_filename(
            seed, data_size, self.time_max, self._baseline, self._adjacency, self._decays
        )
        cached = self._load_from_cache(cache_file)

        if cached is not None:
            inputs = cached["inputs"]
            inputs_len = cached["inputs_len"]
            inputs_marks = cached["inputs_marks"]
        else:
            logger.info("No cached dataset found – generating afresh the hawkes_3x3 dataset %s", cache_file)
            inputs, inputs_len, inputs_marks = self._simulate(
                data_size, self._baseline, self._adjacency, self._decays, self.time_max, seed
            )
            self._save_to_cache(
                cache_file,
                {
                    "inputs": inputs,
                    "inputs_len": inputs_len,
                    "inputs_marks": inputs_marks,
                    "num_marks": self.K,
                },
            )

        (self.train_in, self.val_in, self.test_in, self.train_in_len, self.val_in_len, self.test_in_len) = (
            self._split_60_20_20(inputs, inputs_len)
        )

        # Marks share the same 60/20/20 split indices
        n = len(inputs)
        train_end = int(0.60 * n)
        val_end = int(0.80 * n)
        self.train_marks = inputs_marks[:train_end]
        self.val_marks = inputs_marks[train_end:val_end]
        self.test_marks = inputs_marks[val_end:]


if __name__ == "__main__":
    import sys

    from src.logger.init_logger import set_config_logging

    set_config_logging()

    from src.diagnostics.dataset_diagnostics import export_dataset_report
    import test.paper_experiments.settings.hawkes_3x3 as _  # noqa: F401
    from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY
    from test.paper_experiments.training_helpers import load_experiment_config

    _default_config = "hawkes_3x3/deter_test.yaml"
    _config_path = sys.argv[1] if len(sys.argv) > 1 else _default_config
    cfg = load_experiment_config(_config_path)

    dm = EXPERIMENT_REGISTRY[cfg["experiment_type"]]["data_factory"](cfg)
    export_dataset_report(dm, fig_format="pdf", preview=False)
