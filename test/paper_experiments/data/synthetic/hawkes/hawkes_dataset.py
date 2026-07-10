import logging
import typing

import numpy as np
import torch

logger = logging.getLogger(__name__)

from test.paper_experiments.data.synthetic_tpp_data_module import SyntheticTPPDataModule


class HawkesDataModule(SyntheticTPPDataModule):
    """Synthetic univariate Hawkes process dataset with automatic on-disk caching."""

    # Class-level stubs satisfying TPPDataModule abstract properties
    train_in = val_in = test_in = None
    train_in_len = val_in_len = test_in_len = None
    time_max = num_marks = None
    train_marks = val_marks = test_marks = None

    @staticmethod
    def format_filename(seed: int, data_size: int, tmax: float, mu: float, alpha: float, beta: float, prefix="hawkes"):
        tmax_str = str(tmax).replace('.', ',')
        mu_str = str(mu).replace('.', ',')
        alpha_str = str(alpha).replace('.', ',')
        beta_str = str(beta).replace('.', ',')
        return f"{prefix}_seed{seed}_size{data_size}_tmax{tmax_str}_mu{mu_str}_alpha{alpha_str}_beta{beta_str}.pt"

    @staticmethod
    def _get_cache_filename(seed: int, data_size: int, time_max: float, mu: float, alpha: float, beta: float) -> str:
        filename = HawkesDataModule.format_filename(
            seed=seed,
            data_size=data_size,
            tmax=time_max,
            mu=mu,
            alpha=alpha,
            beta=beta,
        )
        return SyntheticTPPDataModule._DATA_LINKER(["synthetic", filename])

    def _generate_hawkes_sequences(
        self, n_paths: int, mu: float, alpha: float, beta: float, T_max: float, seed: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Hawkes process sequences using tick library (single-threaded).

        Returns:
            - cumulative_times: torch.Tensor of shape (N, L+1, 1)
            - sequence_lengths: torch.Tensor of shape (N,)
        """
        try:
            # We only import it here because the server does not have access to tick==0.7.0.1
            from tick.hawkes import SimuHawkesExpKernels
        except ImportError as e:
            raise ImportError(
                "tick is required to generate Hawkes data but is not installed. Install it or push the cached .pt file."
            ) from e

        logger.info(f"Generating {n_paths} Hawkes sequences (mu={mu}, alpha={alpha}, beta={beta}, T_max={T_max})")

        # Sequential generation (single-threaded)
        paths = []
        np.random.seed(seed)
        for i in range(n_paths):
            sim = SimuHawkesExpKernels(
                adjacency=np.array([[alpha]], dtype=float),
                decays=np.array([[beta]], dtype=float),
                baseline=np.array([mu], dtype=float),
                end_time=T_max,
                seed=seed + i,
                verbose=False,
            )
            sim.simulate()
            paths.append(sim.timestamps[0])  # 1D numpy array of event times

        # Compute lengths (number of events in each path)
        lengths = np.array([len(p) for p in paths])
        max_len = int(lengths.max()) if len(lengths) > 0 and lengths.max() > 0 else 0

        logger.info(
            f"Generated sequences - min length: {lengths.min()}, max length: {max_len}, mean: {lengths.mean():.2f}"
        )

        # Pad and stack into array
        cum_times_padded = np.zeros((n_paths, max_len), dtype=np.float32)
        for i, path in enumerate(paths):
            L = len(path)
            if L > 0:
                cum_times_padded[i, :L] = path
                # Pad with last cumulative time (marking invalid data)
                if L < max_len:
                    cum_times_padded[i, L:] = path[-1]
            else:
                # Handle empty sequences (no events)
                cum_times_padded[i, :] = 0.0

        # Prepend anchor zeros: shape becomes (N, max_len+1)
        anchor_zeros = np.zeros((n_paths, 1), dtype=np.float32)
        cum_times_with_anchor = np.concatenate([anchor_zeros, cum_times_padded], axis=1)

        # Reshape to (N, L+1, 1)
        cum_times_tensor = torch.from_numpy(cum_times_with_anchor).unsqueeze(-1)

        # Lengths include the anchor: +1
        lengths_tensor = torch.from_numpy(lengths).long() + 1

        return cum_times_tensor, lengths_tensor

    def __init__(
        self,
        *,
        data_size: int,
        seed: int,
        mu: float = 0.5,
        alpha: float = 0.5,
        beta: float = 1.0,
        batch_size: typing.Optional[int] = None,
    ):
        super().__init__(batch_size=batch_size)
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.time_max = 20.0
        self.data_size = data_size
        self.num_marks = 1

        cache_file = self._get_cache_filename(seed, self.data_size, self.time_max, mu, alpha, beta)
        cached = self._load_from_cache(cache_file)

        if cached:
            inputs = cached["inputs"]
            inputs_len = cached["inputs_len"]
        else:
            logger.info(f"No cached dataset found - generating afresh the dataset {cache_file}.")
            inputs, inputs_len = self._generate_hawkes_sequences(self.data_size, mu, alpha, beta, self.time_max, seed)
            self._save_to_cache(cache_file, {"inputs": inputs, "inputs_len": inputs_len})

        (self.train_in, self.val_in, self.test_in, self.train_in_len, self.val_in_len, self.test_in_len) = (
            self._split_60_20_20(inputs, inputs_len)
        )

        # Trivial marks: all-zeros tensors of shape (N, L+1) for the 3-tuple contract.
        L_plus_1 = inputs.shape[1]
        self.train_marks = torch.zeros(self.train_in.shape[0], L_plus_1, dtype=torch.long)
        self.val_marks = torch.zeros(self.val_in.shape[0], L_plus_1, dtype=torch.long)
        self.test_marks = torch.zeros(self.test_in.shape[0], L_plus_1, dtype=torch.long)

    def _build_fresh(self, seed: int) -> "HawkesDataModule":
        return HawkesDataModule(
            data_size=self.data_size,
            seed=seed,
            mu=self.mu,
            alpha=self.alpha,
            beta=self.beta,
            batch_size=self.batch_size,
        )


if __name__ == '__main__':
    import sys

    from src.logger.init_logger import set_config_logging

    set_config_logging()
    from src.diagnostics.dataset_diagnostics import export_dataset_report
    import test.paper_experiments.settings.hawkes as _  # noqa: F401
    from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY
    from test.paper_experiments.training_helpers import load_experiment_config

    _default_config = "hawkes/deter_test.yaml"
    _config_path = sys.argv[1] if len(sys.argv) > 1 else _default_config
    cfg = load_experiment_config(_config_path)

    dm = EXPERIMENT_REGISTRY[cfg["experiment_type"]]["data_factory"](cfg)
    export_dataset_report(dm, fig_format="svg", preview=True)
