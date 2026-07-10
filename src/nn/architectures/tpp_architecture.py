"""
TPPArchitecture: Unified base class for Temporal Point Process architectures.

This class merges the functionality from BaseArchitecture and ContinuousTimeArchitecture
into a single coherent base class for all TPP models.
"""

import logging
import math
import os
import time
import typing
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule, seed_everything
from tqdm import tqdm

logger = logging.getLogger(__name__)
from src.metrics.anchors.terminal_anchor_strategy import TerminalAnchorStrategy, make_anchor_strategy
from src.nn.architectures.mark_prediction_utils import (
    MARK_IGNORE_INDEX,
    MarkEvalTensors,
    compute_mark_accuracy_metrics,
    prepare_next_mark_prediction_tensors,
)

from src.data_transformations.expscaler import ExpScaler, ScalingStrategy
from src.data_transformations.standardscaler import StandardScaler
from src.data_transformations.statscompute import variable_len_standard_stats
from src.data_types.bootstrap_eval import (
    aggregate_bootstrap_metrics,
    build_per_replicate_matrix,
    generate_bootstrap_indices,
)
from src.data_types.samplingresult import UnconditionalSamplingResult, ConditionalSamplingResult
from src.data_types.tppmetrics import TPPMetricsConfig, TPPMetrics, DatasetSplitType
from src.metrics.corrloss import CorrLoss
from src.metrics.sigw1_degree_detector import SigW1DegreeDetector
from src.metrics.totalvar import total_var
from src.nn.embeddings.event import EventEmbedding
from src.nn.embeddings.mark import MarkEmbedding
from src.nn.embeddings.time import TimeEmbedding, TrigoTimeEmbedding
from src.nn.nn.mark_predictor import MarkPredictor
from src.nn.sampler.categoricalsampler import CategoricalSampler
from src.utils.fix_seq_ends import (
    _replace_from_index_with_value_torch,
    set_seq_to_nan_from_index,
    set_seq_to_cst_val_from_index,
    to_cst_val_gr,
)
from src.utils import tpp_utils
from src.utils.utils_os import savefig
from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.plot.tpp_plots import diagnostic_plots_tpp
from src.plot.mark_plots import diagnostic_plots_marks, mask_mark_sequences

VERBOSE_TESTING = False


class TPPArchitecture(LightningModule, metaclass=ABCMeta):
    """
    Base architecture class for Temporal Point Process models.

    This class provides:
    - Data preprocessing and scaling for TPP sequences
    - Signature-based loss computation setup
    - TPP metrics initialization and computation
    - Utility methods for sampling, plotting, and evaluation
    """

    # Class constants
    NUM_PLOT_SEQS_DURING_VAL: typing.Final[int] = 8_000
    NUM_PLOT_SEQS_DURING_TEST: typing.Final[int] = 8_000
    NUM_SAMPLES_TEST: typing.Final[int] = 100_000
    NUM_REPEAT_PER_SEQ_FOR_TEST_METRIC: typing.Final[int] = 100
    EXACT_SAMPLING_OVERSAMPLING_FACTOR: typing.Final[float] = 1.25
    TIME_EMB_SIZE: typing.Final[int] = 64
    MARK_EMB_SIZE: typing.Final[int] = 64

    # Diagnostic figure/axes attributes — set by _set_eval_plots(), declared here
    # so that the type checker resolves them as plt.Figure / plt.Axes rather than
    # nn.Module.__getattr__'s Union[Tensor, Module].
    hist_fig: plt.Figure
    hist_ax: typing.Any  # ndarray of Axes, shape (2, 2)
    acf_fig: plt.Figure
    acf_ax: typing.Any  # ndarray of Axes, shape (1, 2)
    intensity_fig: plt.Figure
    intensity_ax: typing.Any  # ndarray of Axes, shape (1, 2)
    cov_err_fig: plt.Figure
    cov_err_ax: plt.Axes
    temporal_plot_fig: plt.Figure
    temporal_plot_ax: plt.Axes
    mark_marginal_fig: plt.Figure
    mark_marginal_ax: plt.Axes
    mark_conditional_fig: plt.Figure
    mark_conditional_axes: typing.Sequence[plt.Axes]

    # section ######################################################################
    #  #############################################################################
    #  Abstract Methods (must be implemented by subclasses)
    #
    #  Subclasses must also set `self.sigw_loss_properties` (a SigWLossDataProps)
    #  BEFORE calling super().__init__(), as it is consumed during base initialisation.

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def sample(
        self,
        *,
        num_seq: typing.Optional[int] = None,
        starting_times: typing.Optional[torch.Tensor] = None,
        log_inter_arr_times: typing.Optional[torch.Tensor] = None,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        """
        Generate scaled inter-arrival time sequences.

        Either num_seq (unconditional) or log_inter_arr_times (conditional) must be provided.

        Args:
            marks: (N, L) integer mark types for conditional sampling. Ignored by non-mark architectures.

        Returns:
            Tuple of (scaled_samples, latent_rep_history, generated_marks):
                - scaled_samples: shape (N, L, D)
                - latent_rep_history: shape (N, L-1, H)
                - generated_marks: shape (N, L) for unconditional marked learnable models,
                  None for non-mark models and conditional teacher-forced paths.
        """
        pass

    @staticmethod
    def filter_patho_seqs(
        tensor1: torch.Tensor, lens_for_masking: torch.Tensor, tensor2: torch.Tensor = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Filter out pathological sequences (e.g. length 0 or 1) produced during sampling.

        Args:
            tensor1: Sampled sequences, shape (N, L, D).
            lens_for_masking: Sequence lengths, shape (N,).
            tensor2: Optional auxiliary tensor to filter in lockstep (e.g. latent history).

        Returns:
            Filtered (tensor1, lens_for_masking, tensor2).
        """
        mask_valid: torch.Tensor = lens_for_masking > 1
        tensor1, tensor2 = tpp_utils.apply_mask(tensor1, mask_valid, tensor2)
        lens_for_masking = lens_for_masking[mask_valid]
        return tensor1, lens_for_masking, tensor2

    @property
    def _ce_normalizer(self) -> float:
        """log(num_marks) normaliser so mark CE is scale-free across datasets."""
        return math.log(self.num_marks) if self.num_marks > 1 else 1.0

    # section ######################################################################
    #  #############################################################################
    #  Static Methods

    @staticmethod
    def _get_extrema_for_clamping(data_train: torch.Tensor, use_10_or_30: bool) -> typing.Tuple[float, float]:
        # Fix the extremas. Works only for univariate data because of our function calls.
        # Easily changeable.
        # Use 10% magnifying for exponential data, 30% for linear.
        # Returns plain floats so .clamp(min=..., max=...) is device-agnostic.
        # Store as Python floats (not tensors) so .clamp(min=..., max=...) works on any
        # device. Storing raw tensors here would keep them on CPU after PL moves the
        # model to GPU, breaking .clamp() in PyTorch >= 1.12.
        assert (
            len(data_train.shape) == 3
        ), f"Expected 3 dimensions for the data but got {len(data_train.shape)} dimensions."
        assert (
            data_train.shape[2] == 1
        ), f"This function is only for univariate data, but got {data_train.shape[2]} dimensions."

        if use_10_or_30:
            MAGNIFYING_FACTOR = 1.1
            REDUCTION_FACTOR = 0.9
        else:
            MAGNIFYING_FACTOR = 1.3
            REDUCTION_FACTOR = 0.7

        max_scaled_data = data_train.max().item()
        min_scaled_data = data_train.min().item()
        if max_scaled_data > 0.0:
            max_scaled_data *= MAGNIFYING_FACTOR
        else:
            max_scaled_data *= REDUCTION_FACTOR
        if min_scaled_data > 0.0:
            min_scaled_data *= REDUCTION_FACTOR
        else:
            min_scaled_data *= MAGNIFYING_FACTOR
        return min_scaled_data, max_scaled_data

    @staticmethod
    def log_results_comparison(metrics):
        # Prefer bootstrap ``<name>_mean`` keys when present, but fall back to
        # plain scalar metrics for values computed outside the bootstrap loop.
        display_names = [
            f"{name}_mean" if f"{name}_mean" in metrics else name for name in TPPMetrics.DISPLAY_METRIC_NAMES
        ]
        table = tpp_utils.format_metrics_table(metrics, display_names)
        logger.info("Test Results:\n" + table + "\n")
        return

    # section ######################################################################
    #  #############################################################################
    #  Initialization

    def __init__(
        self,
        time_max: int,
        num_marks: int,
        data_train: torch.Tensor,
        data_train_lens: torch.Tensor,
        data_val: torch.Tensor,
        data_val_lens: torch.Tensor,
        train_marks: torch.Tensor,
        val_marks: torch.Tensor,
        concentration_factor: float,
        ########################################################################
        output_dir: str = None,
        enable_plot: bool = False,
        period_plot_val: int = 1000,
        plot_every_n_val_steps: int = 1,
        # Gamma baseline disables the exponential scaler and works directly in raw inter-arrival space.
        disable_scaler=False,
        n_bootstraps: int = 1,
    ):
        """
        Initialize the TPP architecture.

        Args:
            time_max: Maximum time horizon for sequences
            num_marks: Number of mark categories (1 = unmarked)
            data_train: Training cumulative times with anchor, shape (N, L+1, D)
            data_train_lens: Training sequence lengths
            data_val: Validation cumulative times with anchor, shape (N, L+1, D)
            data_val_lens: Validation sequence lengths
            concentration_factor: Factor for exponential scaling
            output_dir: Directory for saving outputs (plots, samples)
            enable_plot: Whether to enable plotting during training
            period_plot_val: Period for validation plotting
            disable_scaler: Whether to disable the exponential scaler
            n_bootstraps: Number of bootstrap replicates for test metrics (1 = no bootstrap).
        """
        super().__init__()

        # Mark configuration — set early so subclass __init__ code can use self.use_marks.
        self.num_marks: int = num_marks
        self.use_marks: bool = num_marks > 1

        # Mark data. Always tensors (all-zeros when the dataset has no marks).
        # Keep as non-persistent buffers so they move with the model device but
        # are not serialized into checkpoints (these can be dataset-sized).
        self.register_buffer('train_marks', train_marks, persistent=False)
        self.register_buffer('val_marks', val_marks, persistent=False)

        # Base configuration
        self.output_dir: typing.Final[str] = output_dir
        self.enable_plot: typing.Final[bool] = enable_plot
        self.period_validation_eval_plots: typing.Final[int] = period_plot_val
        self.plot_every_n_val_steps: int = plot_every_n_val_steps

        # Embeddings
        self.time_emb: typing.Final[TimeEmbedding] = TrigoTimeEmbedding(self.TIME_EMB_SIZE, min_time=0.0, max_time=1.0)

        # Time max for sequence truncation
        self.time_max: typing.Final[int] = time_max

        # Initialize plots if enabled
        if self.enable_plot:
            self._set_eval_plots()

        #####################################
        # Process training and validation data
        #####################################
        """
        Start point, X0,  X1,  X2,  X3
        [ -- 75 --, 100, 105, 108, 120]
            From start point we get X0,
            From the X0 (so we pass the inter time and the history), we get X1...
            From Xi we get Xi+1.
        For N points (4 above), we get 4 intertimes and do 3 steps (last step omitted because we can't compare it).
        To get X0, we do not need to encode. It is only about decoding from current time + history.
        If start point is not know, it is the same problem actually, but predict X1.

        encoder(X_i-1) -> h_i; decoder(h_i, X_i-1) -> X_i

        Date for time should be stored as cumulative time.

        --------------- Conclusions
        Change data with TPP such that it starts with the starting value of the sequence.
        So for example, it would be beginning of observations.
        You get then [0, 15,16,24]. From that you get the I.T.s.
        """
        # Separate the first value from the rest of the sequence.
        # data_train of shape (N, L+1, D).
        self.full_data_train_dts = data_train.diff(dim=1)  # shape (N, L, D).
        # Register lengths as buffers so they auto-move to GPU with the model.
        self.register_buffer('full_data_train_dt_lens', data_train_lens - 1, persistent=False)
        self.train_anchor_times = data_train[:, 0:1, :]

        self.train_first_it = self.full_data_train_dts[:, 0:1, :]
        self.data_train_dts = self.full_data_train_dts[:, 1:, :]

        self.full_data_val_dts = data_val.diff(dim=1)
        self.register_buffer('full_data_val_dt_lens', data_val_lens - 1, persistent=False)
        self.data_val_dts = self.full_data_val_dts[:, 1:, :]

        self.num_dim_seqs: typing.Final[int] = self.data_train_dts.shape[2]

        logger.debug("Data for training (I.T.) has maximal length %s", self.full_data_train_dt_lens.max())
        logger.debug("The data has %s elements in the sequences.", self.full_data_train_dts.shape[1])
        logger.debug("Targets before processing %s", self.data_train_dts)
        data_dts_not_nan = self.data_train_dts[~torch.isnan(self.data_train_dts)]
        num_small_values_in_data = (data_dts_not_nan < 1e-7).sum().item()
        num_values_in_data = data_dts_not_nan.numel()
        perc_small_values_in_data = (
            (num_small_values_in_data / num_values_in_data * 100) if num_values_in_data > 0 else float('nan')
        )
        logger.debug(
            "Number of values in the training dataset below 1e-7: %d out of %d (%.2f%%)",
            num_small_values_in_data,
            num_values_in_data,
            perc_small_values_in_data,
        )

        #####################################
        # Transform the data for all metrics.
        #####################################
        if disable_scaler:

            class IdentityScaler:
                def __call__(self, data):
                    return data

                def unscale(self, data):
                    return data

            self.scaler_exp: typing.Final[typing.Union[IdentityScaler, ExpScaler]] = IdentityScaler()
        else:
            self.scaler_exp: typing.Final[ExpScaler] = ExpScaler(
                self.full_data_train_dts,
                self.full_data_train_dt_lens,
                concentration_factor,
                1e-8,
                ScalingStrategy.NAIVE,
            )

        self._anchor_strategy: TerminalAnchorStrategy = make_anchor_strategy(
            TerminalAnchorMode.FREE_ENDPOINT, scaler_exp=self.scaler_exp
        )

        data_train_scaled_dts, data_train_cum, _ = self._preprocess_dataset_for_metrics(data_train, data_train_lens)
        data_val_scaled_dts, data_val_cum, _ = self._preprocess_dataset_for_metrics(data_val, data_val_lens)

        logger.log(5, "Scaled targets (log) %s", data_train_scaled_dts)
        logger.log(5, "Cum time targets     %s", data_train_cum)

        self.anchor_times_sampler: typing.Final[CategoricalSampler] = CategoricalSampler(self.train_anchor_times, True)

        train_first_it_scaled = self.scaler_exp(self.train_first_it)
        # Stores scaled τ₁ only (float32). Marks are kept separately in train_marks (long).
        # Use return_indices=True when sampling so the matching mark can be fetched by index.
        self.first_value_ts_sampler: typing.Final[CategoricalSampler] = CategoricalSampler(train_first_it_scaled, True)

        self.set_scaler_paths_for_sig(
            torch.cat([data_train_scaled_dts, data_train_cum], axis=2),
            self.full_data_train_dt_lens - 1,
        )

        # Scaler to scale the cumulative time. Use the full training data.
        mean_cum, std_cum = variable_len_standard_stats(data_train, self.full_data_train_dt_lens, True)
        self.scaler_cumsum_value_for_generator: typing.Final[StandardScaler] = StandardScaler(
            means=mean_cum, stds=std_cum
        )

        train_sig_loss_seqs = self.scale_paths_pre_sig(
            torch.cat([data_train_scaled_dts, data_train_cum], axis=2),
            seq_lens=self.full_data_train_dt_lens - 1,
        )
        val_sig_loss_seqs = self.scale_paths_pre_sig(
            torch.cat([data_val_scaled_dts, data_val_cum], axis=2),
            seq_lens=self.full_data_val_dt_lens - 1,
        )
        logger.log(5, "Target sequences for metric %s", train_sig_loss_seqs)

        # Register extrema of the scaled training data to set clamping bounds.
        (self.MIN_SCALED_DATA, self.MAX_SCALED_DATA) = TPPArchitecture._get_extrema_for_clamping(
            data_train_scaled_dts, True
        )

        # Initialize TPPMetrics BEFORE _set_target_losses (metrics are needed to create signature loss)
        self._initialize_tpp_metrics(
            data_train,
            data_train_lens,
            data_val,
            data_val_lens,
            train_sig_loss_seqs,
            val_sig_loss_seqs,
            n_bootstraps=n_bootstraps,
        )

        self._set_target_losses()

        # Compute baseline approximation errors from training data (measures inherent metric variability).
        # Used as reference for model-generated samples during training/validation.
        self.approx_err, self.approx_err_histoloss = self._compute_approx_errors(data_train_scaled_dts, data_train_cum)

        # Result: PL's trainer.test() return value cannot carry raw arrays, so external
        # callers (trainingmanager.py, recompute_bootstrap.py) read results directly off the
        # model instance after trainer.test() returns.
        # metrics_test      — mean/std scalar dict populated by test_step.
        # _bootstrap_per_replicate — raw (B,) arrays per metric, populated by
        #                     _run_bootstrap_metrics; used by recompute_bootstrap.py
        #                     to write the per-replicate .npz for downstream paired tests.
        self.metrics_test = None
        # Contract: each value is a 1-D float ndarray of length n_bootstraps (B).
        # Tiled to (B,) even for metrics computed outside the bootstrap loop (mark metrics),
        # so downstream consumers can treat every key uniformly.
        self._bootstrap_per_replicate: typing.Dict[str, np.ndarray] = {}
        return

    # section ######################################################################
    #  #############################################################################
    #  Init Helpers

    def _initialize_tpp_metrics(
        self,
        data_train: torch.Tensor,
        data_train_lens: torch.Tensor,
        data_val: torch.Tensor,
        data_val_lens: torch.Tensor,
        train_sig_loss_seqs: torch.Tensor,
        val_sig_loss_seqs: torch.Tensor,
        n_bootstraps: int = 1,
    ):
        """
        Initialize TPPMetrics for train and validation datasets.

        Args:
            data_train: Training cumulative times with anchor, shape (N, L+1, D)
            data_train_lens: Training sequence lengths
            data_val: Validation cumulative times with anchor, shape (N, L+1, D)
            data_val_lens: Validation sequence lengths
            train_sig_loss_seqs: Preprocessed training sequences for signature metric
            val_sig_loss_seqs: Preprocessed validation sequences for signature metric
            n_bootstraps: Passed through to TPPMetricsConfig.
        """
        metrics_config = TPPMetricsConfig(
            sig_degree=self.sigw_loss_properties.sig_degree,
            scale_high_degrees=self.sigw_loss_properties.scale_high_degrees,
            standardise_sig=self.sigw_loss_properties.standardise_sig,
            use_float64_signature=self.sigw_loss_properties.use_float64_signature,
            time_max=self.time_max,
            n_bootstraps=n_bootstraps,
        )
        # Store config for lazy test metrics initialization
        self._metrics_config = metrics_config

        # Initialize train and validation metrics with signature data
        self.metrics_train = TPPMetrics(
            data_train,
            data_train_lens,
            self.scaler_exp,
            metrics_config,
            train_sig_loss_seqs,
            self.scale_paths_pre_sig,
            split=DatasetSplitType.TRAIN,
        )

        self.metrics_val = TPPMetrics(
            data_val,
            data_val_lens,
            self.scaler_exp,
            metrics_config,
            val_sig_loss_seqs,
            self.scale_paths_pre_sig,
            split=DatasetSplitType.VAL,
        )

    def _set_target_losses(self):
        """
        Set up signature metrics for training loss computation.

        This method obtains signature metrics from TPPMetrics instances.
        Requires that self.metrics_train and self.metrics_val exist (call _initialize_tpp_metrics first).
        """
        effective_train_degree = None
        if self.sigw_loss_properties.use_degree_detector:
            detector = SigW1DegreeDetector(
                self.metrics_train.sig_loss_seqs,
                self.sigw_loss_properties.sig_degree,
            )
            effective_train_degree = self.sigw_loss_properties.resolve_detected_sig_degree(
                detector.effective_sig_degree
            )

        self.sigw1metric_train = self.metrics_train.create_and_get_signature_metrics(
            effective_sig_degree=effective_train_degree
        )
        self.sigw1metric_val = self.metrics_val.create_and_get_signature_metrics(
            effective_sig_degree=effective_train_degree
        )
        return

    def _scale_paths_pre_sig_for_train_proxy(
        self,
        input_data_to_compute_loss: torch.Tensor,
        seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Preprocess paths for approx_err so it matches the training loss pipeline.

        Default behavior uses the metrics/validation preprocessing.
        Architectures with a dedicated training preprocessing pipeline should override this.
        """
        return self.scale_paths_pre_sig(input_data_to_compute_loss, seq_lens=seq_lens)

    def _compute_approx_errors(
        self,
        data_train_scaled_dts: torch.Tensor,
        data_train_cum: torch.Tensor,
        dt_lens: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[float, float]:
        """
        Compute baseline approximation errors by comparing training data against itself.

        Measures inherent metric variability when comparing real samples to real samples.
        Uses TRAINING data intentionally (not validation) to provide a reference baseline.

        Args:
            data_train_scaled_dts: Scaled inter-arrival times from training set, const-ended (N, L, D)
            data_train_cum: Cumulative times from training set, const-ended (N, L, D)
            dt_lens: Inter-arrival lengths (incl. tau_1) matching the passed data, shape (N,).
                Defaults to ``self.full_data_train_dt_lens`` (full-length training data). Pass the
                capped lengths when the reference data has been truncated (train_seq_cap), so the
                proxy lengths match the truncated width rather than the full-length lengths.
                This is the customization point: the base (never-capped) architecture relies on the
                default, while ``ArchitectureOneToOne`` supplies its capped reference width through
                this argument rather than overriding the method (parameterization, not subclassing).

        Returns:
            Tuple of (approx_err_sig, approx_err_histloss): baseline metric values
        """
        logger.debug("Computing approximation errors using TPPMetrics.")

        if dt_lens is None:
            dt_lens = self.full_data_train_dt_lens

        # Take first half of training data as "generated" samples
        half_idx = len(data_train_scaled_dts) // 2
        gen_samples_scaled = data_train_scaled_dts[:half_idx]
        gen_samples_cum = data_train_cum[:half_idx]

        train_seq_lens = dt_lens[:half_idx] - 1
        gen_out_sigloss = self._scale_paths_pre_sig_for_train_proxy(
            torch.cat([gen_samples_scaled, gen_samples_cum], axis=2),
            seq_lens=train_seq_lens,
        )
        approx_error_sig = self.sigw1metric_train(gen_out_sigloss).item()

        gen_samples_naned = set_seq_to_nan_from_index(
            gen_samples_scaled[:, :, :1], dt_lens[:half_idx] - 2  # Only first feature
        )
        gen_samples_cum_naned = set_seq_to_nan_from_index(gen_samples_cum, dt_lens[:half_idx] - 2)

        # Compute histogram metrics
        hist_metrics = self.metrics_train.compute_histogram_metrics(gen_samples_naned, gen_samples_cum_naned)

        # Average the two histogram losses (as in original code)
        approx_error_hist_loss = (hist_metrics['hist_it'] + hist_metrics['hist_int']) / 2.0

        logger.info(
            "Approximation errors - signature: %.4f, histogram: %.4f",
            approx_error_sig,
            approx_error_hist_loss,
        )

        return approx_error_sig, approx_error_hist_loss

    def set_scaler_paths_for_sig(
        self,
        target_seqs,
        target_seq_lens=None,
    ) -> None:
        # First add the zero like in the scaling procedure.
        target_seqs = tpp_utils.insert_zero_beg(target_seqs)
        target_seqs = self._anchor_strategy.append(target_seqs, self.time_max, seq_lens=target_seq_lens)
        if target_seq_lens is not None:
            effective_lens = (
                target_seq_lens + 1 + self._anchor_strategy.terminal_anchor_extra_len()
            )  # +1 for zero anchor
            mean_paths_scaled, std_paths_scaled = variable_len_standard_stats(target_seqs, effective_lens, True)
        else:
            mean_paths_scaled, std_paths_scaled = target_seqs.mean((0, 1)), target_seqs.std((0, 1))

        # Remove the mean-removal procedure for the cumsum.
        # This is to ensure that when adding a zero at the beginning of the cum times, the increasing order is preserved.
        mean_paths_scaled[1] = 0.0
        self.scaler_std: StandardScaler = StandardScaler(means=mean_paths_scaled, stds=std_paths_scaled)
        scaled_targets = self.scaler_std(target_seqs)

        # Calculated after any augmentation, for signature balancing.
        with torch.no_grad():
            self.total_vars = total_var(scaled_targets).mean().item()
        logger.debug("Total vars of the paths %s", self.total_vars)
        return

    def scale_paths_pre_sig(
        self,
        input_data_to_compute_loss: torch.Tensor,
        seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Add a zero to the inputs. Then scale.
        The way it is scaled is [tau_j - mu_tau / sigma_tau ]. In this case, because we add a zero first,
        The mean includes that zero. What we need is that the cumulative time do start from zero (so we do not remove the mean).
        We also need to anchor the first coordinate, so we can add 0 and remove the mean (shifting).
        But because it is a deterministic transformation, it works as intended.

        Args:
            input_data_to_compute_loss: (N, L, D) paths with channels [scaled_tau, cum_time], constant-padded.
            seq_lens: Per-sequence valid lengths of input_data_to_compute_loss, shape (N,).
                Required when terminal_anchor_mode is RESIDUAL; ignored otherwise.
        """
        input_data_to_compute_loss = tpp_utils.insert_zero_beg(input_data_to_compute_loss)
        input_data_to_compute_loss = self._anchor_strategy.append(
            input_data_to_compute_loss, self.time_max, seq_lens=seq_lens
        )
        logger.log(5, "Preprocessed targets %s", input_data_to_compute_loss)
        input_data_to_compute_loss = self.scaler_std(input_data_to_compute_loss)
        input_data_to_compute_loss /= self.total_vars

        logger.log(5, "Scaled targets (by std) %s", input_data_to_compute_loss)
        return input_data_to_compute_loss

    def configure_optimizers(self):
        assert self.lr is not None, "Learning rate is not defined, or redefine configure_optimizers."
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)

    def _sample_first_event(self, num_seq: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Co-sample the first inter-arrival time and its mark from the same training sequence.

        Uses `first_value_ts_sampler` with `return_indices=True` so the sampled τ₁
        and the corresponding first-event mark come from the same training sequence,
        preserving the joint (time, mark) dependence.

        Args:
            num_seq: Number of sequences to sample.

        Returns:
            first_it: (num_seq, 1, D) scaled first inter-arrival time.
            first_mark: (num_seq,) long tensor of first-event marks.
                Returns zeros when marks are not used.
        """
        dummy = torch.zeros(num_seq, device=self.device)
        # return_indices ties each sampled τ₁ back to its training sequence so the mark lookup below is consistent.
        first_it, indices = self.first_value_ts_sampler.sample(dummy, return_indices=True)
        first_it = first_it[:, 0]  # (N, 1, D) -> (N, D) to match samples[:, 0] assignment

        # Fetch the first-event mark from the same draw: indices[i] is the row in train_marks that produced
        # first_it[i], preserving the empirical (τ₁, mark₁) joint distribution.
        # train_marks[:, 0] is the anchor; column 1 is the first real event mark.
        if self.train_marks is not None and self.train_marks.shape[1] > 1:
            first_mark = self.train_marks[:, 1][indices]  # (num_seq,)
        else:
            first_mark = torch.zeros(num_seq, dtype=torch.long, device=self.device)

        return first_it, first_mark

    # section ######################################################################
    #  #############################################################################
    #  Forward Pass

    def forward(
        self, num_seq: int, include_first_it: bool
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        """
        Generate samples in unscaled format.

        Args:
            num_seq: Number of sequences to generate.
            include_first_it: If False (training/eval path), strips τ₁ and returns
                (N, L-1, D) with lens = number of ITs excluding τ₁, preserving the
                original contract.  If True (plotting path), keeps τ₁ and returns
                (N, L, D) with lens = number of ITs including τ₁.

        Returns:
            Tuple of (inter_arrival_times, sequence_lengths, generated_marks).
            generated_marks is (N, L) when include_first_it=True and marks are available,
            (N, L-1) when include_first_it=False and marks are available, or None.
        """
        scaled_samples, _, gen_marks = self.sample(num_seq=num_seq)
        samples = self.scaler_exp.unscale(scaled_samples)
        const_samples, samples_lens = to_cst_val_gr(samples, samples.cumsum(axis=1), self.time_max)

        if not include_first_it:
            # Remove τ₁ which is the anchor/seed sampled from data.
            const_samples = const_samples[:, 1:]
            samples_lens = samples_lens - 1
            if gen_marks is not None:
                gen_marks = gen_marks[:, 1:]

        gen_marks = self._mask_generated_marks_tail(gen_marks, samples_lens)
        return const_samples, samples_lens, gen_marks

    # section ######################################################################
    #  #############################################################################
    #  Sampling Methods

    def sample_for_a_fixed_batch_and_fix(
        self,
        batch_history: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        num_samples_per_seq: int,
        name_phase4logger: str,
        exact_num_sampling: bool = False,
    ) -> ConditionalSamplingResult:
        """Sample sequences from a canonical 3-tuple batch and prepare evaluation tensors.

        Both code paths perform *conditional* sampling: the real inter-arrival times are
        always passed as conditioning history (log_inter_arr_times).  This function never
        calls sample_and_fix_seqs with num_seq (the unconditional path).

        The value of num_samples_per_seq selects one of two structurally different paths:

        num_samples_per_seq == 1  — single-sample path
            _sample_with_retry is called once on the whole batch (N sequences together).
            filter_patho_seqs may discard sequences whose generated sample is degenerate,
            so the returned batch can shrink: N → N' ≤ N.
            seq_lens comes from result.seq_lens (the post-filter lengths, length N').
            Output shape: (N', L-1, D).

        num_samples_per_seq > 1  — multi-sample path
            _sample_batch_loop iterates over each of the N sequences individually and
            calls _sample_with_retry per sequence, so pathological retries happen at the
            per-sequence level.  Every sequence always produces exactly num_samples_per_seq
            survivors; N never shrinks.
            seq_lens is set to data_dts_lens - 1 (the nominal batch lengths, length N).
            Output shape: (S, N, L-1, D) where S = num_samples_per_seq.

        Args:
            batch_history:
                Tuple containing:
                  - Tensor of shape (N, L+1, D): the batch of input sequences.
                  - Tensor of shape (N,): the sequence lengths.
                  - Tensor of shape (N, L+1): marks (event types).
            num_samples_per_seq (int):
                Number of generated samples per real sequence.
                Pass 1 for the single-sample path (may return N' < N sequences).
                Pass > 1 for the multi-sample path (always returns N sequences).
            name_phase4logger (str):
                Name identifier for logging purposes.
            exact_num_sampling (bool):
                If True, uses oversampling + a top-up pass to guarantee exactly
                num_samples_per_seq survivors per sequence (multi-sample path only).

        Expects `get_num_needed_resample`, `self.scaler_exp`, `cum_times_to_log_inter_times`, `sample_and_fix_seqs`.

        Returns:
            ConditionalSamplingResult containing:
                - its_scaled_cst: Scaled ITs, constant-padded (filtered).
                - cum_abs_cst: Cumulative absolute times, constant-padded (filtered).
                - ref_its_nan: Unscaled real inter-arrival times with NaN masking.
                - gen_its_tf_nan: Unscaled generated inter-arrivals (given true history) with NaN masking.
                - seq_lens: Sequence lengths, shape (N',) for S=1 or (N,) for S>1.
            Tensor shapes: (S, N, L-1, D) for S>1, or (N', L-1, D) for S=1.

            L-1 arises because: L+1 cumulative times → L inter-arrivals after differencing,
            then τ₁ (the seed sampled from data) is dropped, leaving L-1 predicted steps.
        """
        if len(batch_history) != 3:
            raise ValueError("batch_history must be a canonical 3-tuple: (data, lengths, marks).")

        data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch_history, self.scaler_exp)
        batch_size, seq_len, feature_dim = data_dts_scaled.shape
        data_starting_times = batch_history[0][:, :1]
        marks = batch_history[2][:, 1:]

        if num_samples_per_seq > 1:
            oversampling_factor = self.EXACT_SAMPLING_OVERSAMPLING_FACTOR if exact_num_sampling else 1.0
            oversampled_num_per_seq = int(num_samples_per_seq * oversampling_factor)

            # Repeat inputs to (S, N, L, D) for per-sequence sampling.
            data_dts_scaled = data_dts_scaled.unsqueeze(0).repeat(oversampled_num_per_seq, 1, 1, 1)
            data_starting_times = data_starting_times.unsqueeze(0).repeat(oversampled_num_per_seq, 1, 1, 1)
            # Only need the exact number of repetitions for the lengths tensor.
            data_dts_lens = data_dts_lens.unsqueeze(0).repeat(num_samples_per_seq, 1, 1, 1).flatten()
            marks = marks.unsqueeze(0).repeat(oversampled_num_per_seq, 1, 1)

            logger.info(
                "Sampling %d times per sequence, oversampling factor %.2f.",
                num_samples_per_seq,
                oversampling_factor,
            )

            # --- first pass: sample oversampled_num_per_seq copies per sequence -------
            its_scaled_cst_list, cum_abs_cst_list, its_scaled_raw_list, cond_its_scaled_list = self._sample_batch_loop(
                data_starting_times=data_starting_times,
                data_dts_scaled=data_dts_scaled,
                num_copies=num_samples_per_seq,
                batch_size=batch_size,
                name_phase4logger=name_phase4logger,
                marks=marks,
            )

            # --- top-up pass: resample the deficit based on worst-case survival rate ---
            needed_resample = tpp_utils.get_num_needed_resample(
                num_samples_per_seq, oversampled_num_per_seq, oversampling_factor, its_scaled_cst_list
            )

            if needed_resample > 0:
                logger.info("Resampling %d sequences to ensure enough samples for each sequence.", needed_resample)
                marks_resample = marks[:needed_resample]
                its_scaled_cst_list_sec, cum_abs_cst_list_sec, _, _ = self._sample_batch_loop(
                    data_starting_times=data_starting_times[:needed_resample],
                    data_dts_scaled=data_dts_scaled[:needed_resample],
                    num_copies=needed_resample,
                    batch_size=batch_size,
                    name_phase4logger=name_phase4logger,
                    marks=marks_resample,
                )
            else:
                its_scaled_cst_list_sec, cum_abs_cst_list_sec = None, None

            # Merge first + second pass to exactly num_samples_per_seq survivors per sequence.
            final_its_scaled_cst_list = tpp_utils.concat_two_samples_together(
                its_scaled_cst_list, its_scaled_cst_list_sec, num_samples_per_seq
            )
            final_cum_abs_cst_list = tpp_utils.concat_two_samples_together(
                cum_abs_cst_list, cum_abs_cst_list_sec, num_samples_per_seq
            )

            # Stack to (S, N, L-1, D); raw outputs flatten to (S*N, L-1, D).
            final_its_scaled_cst = torch.stack(final_its_scaled_cst_list, dim=1)
            final_cum_abs_cst = torch.stack(final_cum_abs_cst_list, dim=1)
            cond_its_scaled_final = torch.stack(cond_its_scaled_list, dim=1).flatten(0, 1)
            its_scaled_raw_final = torch.stack(its_scaled_raw_list, dim=1).flatten(0, 1)
        else:
            # Single-sample path (num_samples_per_seq == 1).
            # The entire batch is passed to _sample_with_retry at once.
            # filter_patho_seqs may remove degenerate sequences at the batch level,
            # so the result can have N' < N rows.  We must use result.seq_lens
            # (length N') rather than data_dts_lens - 1 (length N) to avoid a shape
            # mismatch downstream (e.g. ResidualStrategy._last_real_event).
            result = self._sample_with_retry(
                starting_times=data_starting_times,
                log_inter_arr_times=data_dts_scaled,
                name_phase4logger=name_phase4logger,
                marks=marks,
            )
            final_its_scaled_cst = result.its_scaled_cst
            final_cum_abs_cst = result.cum_abs_cst
            its_scaled_raw_final = result.its_scaled_raw
            cond_its_scaled_final = result.cond_its_scaled
            filtered_seq_lens = result.seq_lens  # N' ≤ N after pathological-sequence removal

        ### We unscale to ensure that the metrics are comparable across experiments.
        # If the stats/concentration factor are different, it would lead to different results!

        # We use -2 because we removed the first value, and we set the constant from the index.
        ref_its_nan = self.scaler_exp.unscale(set_seq_to_nan_from_index(cond_its_scaled_final, data_dts_lens - 2))
        gen_its_tf_nan = self.scaler_exp.unscale(set_seq_to_nan_from_index(its_scaled_raw_final, data_dts_lens - 2))

        if num_samples_per_seq > 1:
            rep_samples_shape = (num_samples_per_seq, batch_size, seq_len - 1, feature_dim)
            ref_its_nan = ref_its_nan.view(rep_samples_shape)
            gen_its_tf_nan = gen_its_tf_nan.view(rep_samples_shape)
            # Multi-sample path: N never shrinks (per-sequence retries keep all N sequences),
            # so nominal batch lengths are correct.
            filtered_seq_lens = data_dts_lens - 1

        return ConditionalSamplingResult(
            its_scaled_cst=final_its_scaled_cst,
            cum_abs_cst=final_cum_abs_cst,
            ref_its_nan=ref_its_nan,
            gen_its_tf_nan=gen_its_tf_nan,
            seq_lens=filtered_seq_lens,
        )

    def _sample_with_retry(
        self,
        *,
        starting_times: torch.Tensor,
        log_inter_arr_times: torch.Tensor,
        name_phase4logger: str,
        marks: typing.Optional[torch.Tensor] = None,
        max_attempts: int = 10,
    ) -> UnconditionalSamplingResult:
        """Call sample_and_fix_seqs, retrying when filter_patho_seqs removes all sequences.

        filter_patho_seqs can eliminate every sequence in two ways:
          1. Subclasses that route through _apply_mask raise ValueError.
          2. A custom override may simply return an empty tensor.
        Both are caught here and retried up to max_attempts times.
        """
        for attempt in range(max_attempts):
            try:
                result = self.sample_and_fix_seqs(
                    starting_times=starting_times,
                    log_inter_arr_times=log_inter_arr_times,
                    name_phase4logger=name_phase4logger,
                    marks=marks,
                )
            except ValueError:
                # _apply_mask raises when all sequences are filtered as pathological.
                logger.warning(
                    "All samples filtered as pathological (%s), retry %d/%d.",
                    name_phase4logger,
                    attempt + 1,
                    max_attempts,
                )
                continue

            if result.its_scaled_cst.shape[0] > 0:
                return result

            logger.warning(
                "All samples filtered as pathological (%s), retry %d/%d.",
                name_phase4logger,
                attempt + 1,
                max_attempts,
            )

        raise RuntimeError(
            f"All samples were filtered as pathological after {max_attempts} attempts ({name_phase4logger}). "
            "Check the model for degenerate outputs."
        )

    def _sample_batch_loop(
        self,
        data_starting_times: torch.Tensor,
        data_dts_scaled: torch.Tensor,
        num_copies: int,
        batch_size: int,
        name_phase4logger: str,
        marks: torch.Tensor,
    ) -> typing.Tuple[
        typing.List[torch.Tensor],
        typing.List[torch.Tensor],
        typing.List[torch.Tensor],
        typing.List[torch.Tensor],
    ]:
        """Sample for each batch element with retry logic.

        Args:
            data_starting_times: Starting times, shape (S, N, 1, D).
            data_dts_scaled: Scaled inter-arrival times, shape (S, N, L, D).
            num_copies: Number of copies per sequence (for truncation of unfiltered outputs).
            batch_size: Number of sequences in batch.
            name_phase4logger: Name for logging.
            marks: Marks, shape (S, N, L) — anchor already stripped (event types).

        Returns:
            Tuple of (its_scaled_cst_list, cum_abs_cst_list, its_scaled_raw_list, cond_its_scaled_list).
        """
        its_scaled_cst_list = []
        cum_abs_cst_list = []
        its_scaled_raw_list = []
        cond_its_scaled_list = []

        _prev_log_level = logger.level
        logger.setLevel(logging.WARNING)
        for n in tqdm(range(batch_size), disable=not VERBOSE_TESTING):
            marks_n = marks[:, n]
            result = self._sample_with_retry(
                starting_times=data_starting_times[:, n],
                log_inter_arr_times=data_dts_scaled[:, n, :, :],
                name_phase4logger=f"{name_phase4logger}_batch_{n + 1}",
                marks=marks_n,
            )
            its_scaled_cst_list.append(result.its_scaled_cst)
            cum_abs_cst_list.append(result.cum_abs_cst)
            its_scaled_raw_list.append(result.its_scaled_raw[:num_copies])
            cond_its_scaled_list.append(result.cond_its_scaled[:num_copies])

        logger.setLevel(_prev_log_level)
        return its_scaled_cst_list, cum_abs_cst_list, its_scaled_raw_list, cond_its_scaled_list

    def sample_and_fix_seqs(
        self,
        *,
        num_seq: typing.Optional[int] = None,
        starting_times: typing.Optional[torch.Tensor] = None,
        log_inter_arr_times: typing.Optional[torch.Tensor] = None,
        name_phase4logger: str = "training",
        marks: typing.Optional[torch.Tensor] = None,
    ) -> UnconditionalSamplingResult:
        """Generate samples and apply post-processing (filtering, constant endings).

        Either num_seq (unconditional) or log_inter_arr_times (conditional) must be provided.

        Args:
            num_seq: Number of sequences to sample unconditionally.
            starting_times: Starting times for conditional sampling.
            log_inter_arr_times: Scaled inter-arrival times for conditional sampling, shape (N, L, D).
            name_phase4logger: Name for logging.
            marks: Optional marks for conditional sampling, shape (N, L) — anchor already stripped.

        Returns:
            UnconditionalSamplingResult containing filtered and unfiltered outputs with shapes (N, L-1, D).
        """
        gen_out, latent_rep_history, gen_marks = self.sample(
            num_seq=num_seq, starting_times=starting_times, log_inter_arr_times=log_inter_arr_times, marks=marks
        )
        logger.debug("Generated paths for %s %s", name_phase4logger, gen_out)

        unscaled_gen_out = self.scaler_exp.unscale(gen_out)
        logger.log(5, "Unscaled paths %s", unscaled_gen_out)

        with torch.no_grad():
            gen_out_cum = unscaled_gen_out.cumsum(axis=1)
        its_scaled_cst, seq_lens = to_cst_val_gr(gen_out, gen_out_cum, self.time_max)

        # Compute the pathological-sequence mask before filtering so we can apply
        # it to gen_marks in lockstep.
        patho_mask = seq_lens > 1
        its_scaled_cst, seq_lens, latent_rep_history = self.filter_patho_seqs(
            its_scaled_cst, seq_lens, latent_rep_history
        )

        # Filter gen_marks in lockstep with pathological-sequence removal.
        if gen_marks is not None:
            gen_marks = gen_marks[patho_mask]

        # We remove the first value, which for now is handled differently.
        # Remove the value, while we keep that the place where we fixed the sequence the same.
        # In other words, now when computing the cumulative value, it gives T_max - epsilon ~ Poisson.
        gen_out = gen_out[:, 1:, :]
        its_scaled_cst = its_scaled_cst[:, 1:, :]
        if log_inter_arr_times is not None:
            log_inter_arr_times = log_inter_arr_times[:, 1:, :]
        seq_lens = seq_lens - 1

        # Strip first mark in lockstep with τ₁.
        if gen_marks is not None:
            gen_marks = gen_marks[:, 1:]
            gen_marks = self._mask_generated_marks_tail(gen_marks, seq_lens)

        # Recompute cumsum.
        unscaled_cst = self.scaler_exp.unscale(its_scaled_cst)
        cum_abs = unscaled_cst.cumsum(axis=1)
        logger.log(5, "Samples CUM. %s", cum_abs)
        logger.log(5, "lengths: %s", seq_lens)
        cum_abs_cst = set_seq_to_cst_val_from_index(cum_abs, seq_lens - 1)

        # Use its_scaled_cst which has been filtered and same B size as seq_lens.
        its_scaled_nan = set_seq_to_nan_from_index(its_scaled_cst, seq_lens - 1)
        cum_rel_nan = set_seq_to_nan_from_index(cum_abs_cst, seq_lens - 1)

        logger.log(5, "Fixed endings samples (missing 1st one) I.T. %s", its_scaled_cst)
        logger.log(5, "Fixed endings samples (missing 1st one) CUM. %s", cum_abs_cst)
        assert self._check_tail_contract(its_scaled_nan, cum_rel_nan, gen_marks, seq_lens)
        return UnconditionalSamplingResult(
            its_scaled_cst=its_scaled_cst,
            cum_abs_cst=cum_abs_cst,
            its_scaled_nan=its_scaled_nan,
            cum_rel_nan=cum_rel_nan,
            its_scaled_raw=gen_out,
            cond_its_scaled=log_inter_arr_times,
            seq_lens=seq_lens,
            gen_marks=gen_marks,
        )

    @staticmethod
    def _mask_generated_marks_tail(
        gen_marks: typing.Optional[torch.Tensor], seq_lens: torch.Tensor
    ) -> typing.Optional[torch.Tensor]:
        """Replace generated-mark positions beyond each sequence length with -1."""
        if gen_marks is None:
            return None
        assert gen_marks.ndim == 2, f"Expected generated marks to be 2D, got shape {tuple(gen_marks.shape)}."
        marks_3d = gen_marks.unsqueeze(-1)
        fill_value = marks_3d.new_full((marks_3d.shape[0], 1, 1), -1)
        masked_marks = _replace_from_index_with_value_torch(marks_3d, seq_lens.to(gen_marks.device) - 1, fill_value)
        return masked_marks.squeeze(-1)

    @staticmethod
    def _check_tail_contract(
        its_scaled_nan: torch.Tensor,
        cum_rel_nan: torch.Tensor,
        gen_marks: typing.Optional[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> bool:
        """Verify that padded tails are NaN (times) or -1 (marks). Returns True or raises."""
        L = its_scaled_nan.shape[1]
        # (N, L) bool mask: True at padded positions
        tail_mask = torch.arange(L, device=seq_lens.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        if tail_mask.any():
            tail_mask_3d = tail_mask.unsqueeze(-1)  # (N, L, 1)
            assert torch.isnan(
                its_scaled_nan[tail_mask_3d.expand_as(its_scaled_nan)]
            ).all(), "its_scaled_nan has non-NaN values in padded tail"
            assert torch.isnan(
                cum_rel_nan[tail_mask_3d.expand_as(cum_rel_nan)]
            ).all(), "cum_rel_nan has non-NaN values in padded tail"
            if gen_marks is not None:
                assert (gen_marks[tail_mask] == -1).all(), "gen_marks has values != -1 in padded tail"
        return True

    def _get_fake_real_samples(
        self, num_plot_seq: int, dataset_split: DatasetSplitType, include_first_it: bool
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        """
        Get generated and real samples for plotting/comparison.

        Args:
            num_plot_seq: Number of sequences to generate.
            dataset_split: Which dataset to use for real samples (TRAIN or VAL only).
            include_first_it: Forwarded to forward(). When True, τ₁ is included in
                both generated and real samples (plotting path). When False, τ₁ is
                stripped.

        Returns:
            Tuple of (generated_samples, generated_lens, real_samples, real_lens, gen_marks).
            Lens = number of valid ITs; whether τ₁ is counted depends on include_first_it.
            gen_marks is (N, L) or None.

        Note:
            TEST split is not supported. Test data flows through test_step and is not
            stored in the architecture.
        """
        gen_out, gen_out_lens, gen_marks = self(num_plot_seq, include_first_it=include_first_it)

        if include_first_it:
            if dataset_split is DatasetSplitType.TRAIN:
                targets = self.full_data_train_dts
                targets_lens = self.full_data_train_dt_lens
            elif dataset_split is DatasetSplitType.VAL:
                targets = self.full_data_val_dts
                targets_lens = self.full_data_val_dt_lens
            elif dataset_split is DatasetSplitType.TEST:
                raise ValueError(
                    "TEST split not supported in _get_fake_real_samples(). "
                    "Test data should not be stored in the architecture. "
                    "Use TRAIN or VAL splits only."
                )
            else:
                raise ValueError(f"Unknown dataset_split: {dataset_split}. Must be TRAIN or VAL.")
        else:
            if dataset_split is DatasetSplitType.TRAIN:
                targets = self.data_train_dts
                targets_lens = self.full_data_train_dt_lens - 1
            elif dataset_split is DatasetSplitType.VAL:
                targets = self.data_val_dts
                targets_lens = self.full_data_val_dt_lens - 1
            elif dataset_split is DatasetSplitType.TEST:
                raise ValueError(
                    "TEST split not supported in _get_fake_real_samples(). "
                    "Test data should not be stored in the architecture. "
                    "Use TRAIN or VAL splits only."
                )
            else:
                raise ValueError(f"Unknown dataset_split: {dataset_split}. Must be TRAIN or VAL.")
        return gen_out, gen_out_lens, targets, targets_lens, gen_marks

    def on_train_end(self) -> None:
        """Close all training diagnostic figures when training completes."""
        if not self.enable_plot:
            return
        for fig_attr in (
            'hist_fig',
            'acf_fig',
            'intensity_fig',
            'cov_err_fig',
            'temporal_plot_fig',
            'mark_marginal_fig',
            'mark_conditional_fig',
        ):
            fig = getattr(self, fig_attr, None)
            if fig is not None:
                plt.close(fig)
        return

    def _compute_validation_epoch_end_metrics(self) -> typing.Dict[str, torch.Tensor]:
        """Collect extra validation metrics to log once per epoch.

        Base architectures now log mark metrics directly from ``validation_step()``,
        so this hook is intentionally empty here. It is still called by
        ``on_validation_epoch_end()`` and kept as an override point for
        architectures that genuinely need extra epoch-end metrics.

        ``Architecture_DDPM`` overrides this to add histogram-style metrics
        (``epdf``, ``hist_it``, ``hist_int``) that are computed on a periodic
        plot/eval schedule rather than on every validation batch. Mark metrics
        are not logged here, they are logged per-batch in ``validation_step()``.
        """
        return {}

    def _run_validation_epoch_end_plots(self) -> None:
        """Run the default periodic validation plotting schedule."""
        if self.enable_plot and (self.current_epoch + 1) % self.period_validation_eval_plots == 0:
            effective_plot_period = self.period_validation_eval_plots * self.plot_every_n_val_steps
            if (self.current_epoch + 1) % effective_plot_period == 0:
                try:
                    self.sample_and_plot(name_plot4save=str(self.current_epoch + 1))
                except Exception as e:
                    logger.warning("Plotting at epoch %d failed: %s", self.current_epoch + 1, e)

    def on_validation_epoch_end(self):
        metrics2log = self._compute_validation_epoch_end_metrics()
        if metrics2log:
            self._log_all_metrics(metrics2log, "val_")
        self._run_validation_epoch_end_plots()

    # section ######################################################################
    #  #############################################################################
    #  Plotting

    def _set_eval_plots(self):
        """Create the standard 5 diagnostic figures. Subclasses can override and call super() to add extra figures."""
        self.hist_fig, self.hist_ax = plt.subplots(2, 2)
        self.hist_ax[1, 0].get_shared_y_axes().join(self.hist_ax[1, 0], self.hist_ax[1, 1])
        self.hist_fig.canvas.manager.set_window_title('Hist Values')

        self.acf_fig, self.acf_ax = plt.subplots(1, 2)
        self.acf_fig.canvas.manager.set_window_title('ACF')

        self.intensity_fig, self.intensity_ax = plt.subplots(1, 2)
        self.intensity_ax[0].get_shared_y_axes().join(self.intensity_ax[0], self.intensity_ax[1])
        self.intensity_fig.canvas.manager.set_window_title('Intensity Plot')

        self.cov_err_fig, self.cov_err_ax = plt.subplots()
        self.cov_err_fig.canvas.manager.set_window_title('Covariance Error')

        self.temporal_plot_fig, self.temporal_plot_ax = plt.subplots()
        self.temporal_plot_fig.canvas.manager.set_window_title('Temporal Point Process Samples')

        self.mark_marginal_fig, self.mark_marginal_ax = plt.subplots()
        self.mark_marginal_fig.canvas.manager.set_window_title('Mark Marginal Class Fit')

        self.mark_conditional_fig, self.mark_conditional_axes = plt.subplots(1, 2, sharey=True)
        self.mark_conditional_fig.canvas.manager.set_window_title('Mark Conditional Structure')

    def _clear_all_axes(self):
        """Clear all base diagnostic axes. Subclasses override and call super() to clear extras."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Attempt to set non-positive ylim on a log-scaled axis will be ignored."
                )
                for ax in self.hist_ax.flat:
                    ax.clear()
            for ax in self.acf_ax.flat:
                ax.clear()
            for ax in self.intensity_ax.flat:
                ax.clear()
            # Recreate cov_err axes to properly remove the colorbar
            for ax in self.cov_err_fig.axes:
                self.cov_err_fig.delaxes(ax)
            self.cov_err_ax = self.cov_err_fig.subplots(1, 1)
            self.temporal_plot_ax.clear()
            self.mark_marginal_ax.clear()
            for ax in self.mark_conditional_fig.axes:
                self.mark_conditional_fig.delaxes(ax)
            self.mark_conditional_axes = self.mark_conditional_fig.subplots(1, 2, sharey=True)
        except AttributeError as e:
            logger.error("Error clearing plots: %s", e)
            raise ValueError("The plots have not been initialised but requested to clear them.") from e

    def _save_eval_plots(self, post_str: str, include_mark_plots: bool = False):
        """Save all base diagnostic figures. Subclasses can override and call super() to save extras."""
        if self.output_dir is None:
            return
        savefig(self.hist_fig, f'{self.output_dir}hist_{post_str}.png')
        savefig(self.acf_fig, f'{self.output_dir}acf_{post_str}.png')
        savefig(self.intensity_fig, f'{self.output_dir}intens_{post_str}.png')
        savefig(self.cov_err_fig, f'{self.output_dir}cov_{post_str}.png')
        savefig(self.temporal_plot_fig, f'{self.output_dir}samp_path_plot_{post_str}.png')
        if include_mark_plots:
            savefig(self.mark_marginal_fig, f'{self.output_dir}mk_marg_{post_str}.png')
            savefig(self.mark_conditional_fig, f'{self.output_dir}mk_cond_{post_str}.png')

    def _pre_diagnostic_plots_hook(self) -> None:
        """
        Hook called before diagnostic_plots_tpp.
        Override in subclasses for custom pre-plotting (e.g., forward/backward trajectories in ddpm).
        """
        pass

    def _compute_mark_logits(
        self,
        *,
        marks_with_anchor: torch.Tensor,
        marks_full: torch.Tensor,
        dts: torch.Tensor,
        dt_lens: torch.Tensor,
        current_targets: torch.Tensor,
    ) -> typing.Optional[torch.Tensor]:
        """Return mark logits for validation/test mark evaluation.

        Shared evaluation flow:
        - skip entirely when marks are disabled,
        - let simple baseline models provide logits directly,
        - otherwise ask the architecture for latent history aligned to the
          next-mark targets and apply the shared mark predictor once.

        Architectures should normally override one of the smaller hooks below
        rather than overriding this method directly.
        """
        if not self.use_marks:
            return None

        if self._use_mark_eval_no_history:
            return self._compute_mark_logits_no_history(
                marks_with_anchor=marks_with_anchor,
                marks_full=marks_full,
                dts=dts,
                dt_lens=dt_lens,
                current_targets=current_targets,
            )

        latent_rep_history = self._compute_mark_latent_history_for_eval(
            marks_with_anchor=marks_with_anchor,
            marks_full=marks_full,
            dts=dts,
            dt_lens=dt_lens,
            current_targets=current_targets,
        )
        if latent_rep_history is None:
            # No learnable mark-evaluation path is implemented for this architecture.
            return None

        return self.mark_predictor(latent_rep_history)

    @property
    def _use_mark_eval_no_history(self) -> bool:
        """Return True when this architecture evaluates marks without latent history."""
        return False

    def _compute_mark_latent_history_for_eval(
        self,
        *,
        marks_with_anchor: torch.Tensor,
        marks_full: torch.Tensor,
        dts: torch.Tensor,
        dt_lens: torch.Tensor,
        current_targets: torch.Tensor,
    ) -> typing.Optional[torch.Tensor]:
        """Learnable-model hook for mark evaluation.

        Return latent history already aligned to ``current_targets`` so the base
        implementation can apply ``self.mark_predictor(...)`` once in a shared
        place. Return ``None`` when the architecture does not use this path.
        """
        return None

    def _compute_mark_logits_no_history(
        self,
        *,
        marks_with_anchor: torch.Tensor,
        marks_full: torch.Tensor,
        dts: torch.Tensor,
        dt_lens: torch.Tensor,
        current_targets: torch.Tensor,
    ) -> typing.Optional[torch.Tensor]:
        """Baseline-model hook for mark evaluation.

        Return logits directly when the architecture does not use latent
        history plus ``self.mark_predictor`` for validation/test mark metrics.
        """
        return None

    def _build_mark_tensors_for_validation(
        self,
        *,
        dts: typing.Optional[torch.Tensor] = None,
        dt_lens: typing.Optional[torch.Tensor] = None,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Optional[MarkEvalTensors]:
        """Single entry point for all mark evaluation (plots and metrics).

        Resolves source tensors, derives shared inputs, delegates to the subclass
        hook for logits, applies the single-class fallback, masks padding, and
        returns ready-to-consume evaluation tensors.

        When called without arguments, uses the stored validation tensors.
        When called with explicit tensors (e.g. from on_test_end), uses those instead.
        """
        if not self.use_marks:
            return None

        # 1. Source resolution
        if marks is not None:
            marks_with_anchor = marks
        elif hasattr(self, 'val_marks'):
            marks_with_anchor = self.val_marks
            dts = self.full_data_val_dts
            dt_lens = self.full_data_val_dt_lens
        else:
            return None

        # Stored validation inter-arrival tensors are plain attributes, so normalize all
        # mark-eval inputs to the model device once here instead of in every subclass hook.
        dts = dts.to(self.device)
        dt_lens = dt_lens.to(self.device)
        marks_with_anchor = marks_with_anchor.to(self.device)

        # 2. Shared preparation
        marks_full = marks_with_anchor[:, 1:]
        previous_marks, current_targets, valid_lengths = prepare_next_mark_prediction_tensors(
            marks_with_anchor,
            dt_lens,
        )

        # 3. Architecture-specific logits
        logits = self._compute_mark_logits(
            marks_with_anchor=marks_with_anchor,
            marks_full=marks_full,
            dts=dts,
            dt_lens=dt_lens,
            current_targets=current_targets,
        )

        # 4. Require model logits for marked evaluation.
        if logits is None:
            return None

        # 5. Mask padding
        previous_marks_masked, current_targets_masked = mask_mark_sequences(
            previous_marks,
            current_targets,
            valid_lengths,
        )
        return MarkEvalTensors(logits, previous_marks_masked, current_targets_masked)

    def _plot_validation_mark_diagnostics(
        self,
        *,
        dts: typing.Optional[torch.Tensor] = None,
        dt_lens: typing.Optional[torch.Tensor] = None,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Optional[MarkEvalTensors]:
        """Draw mark diagnostic plots for the current validation epoch.

        Returns the evaluation tensors when plots were produced (caller can
        reuse them for metrics), or None when no plots were drawn.
        """
        mark_eval_data = self._build_mark_tensors_for_validation(dts=dts, dt_lens=dt_lens, marks=marks)
        if mark_eval_data is None:
            return None

        # Same guard as in _compute_mark_metrics_from_eval_tensors.
        if not (mark_eval_data.current_targets != MARK_IGNORE_INDEX).any():
            return None

        diagnostic_plots_marks(
            mark_eval_data.logits,
            mark_eval_data.previous_marks,
            mark_eval_data.current_targets,
            fig_marginal=self.mark_marginal_fig,
            ax_marginal=self.mark_marginal_ax,
            fig_conditional=self.mark_conditional_fig,
            axes_conditional=self.mark_conditional_axes,
        )
        return mark_eval_data

    @property
    def _include_top3_mark_accuracy(self) -> bool:
        """Hook for metric policy.

        Default: report top-3 when there are at least 3 classes.
        Baselines can override to return False.
        """
        return self.num_marks >= 3

    def _count_valid_next_mark_targets(self, lengths: typing.Optional[torch.Tensor]) -> int:
        """Return the number of valid next-mark targets represented by a batch.
        using lengths such that it is independent of the marks encoding of wrong.
        """
        # Use the batch contract (lengths -> valid next-mark targets) as the single
        # weighting rule so train/val logging stays consistent across all call sites.
        if lengths is None:
            return 0
        return int((lengths - 1).clamp_min(0).sum().item())

    def _build_mark_eval_tensors_from_logits(
        self,
        *,
        mark_logits: typing.Optional[torch.Tensor],
        marks: typing.Optional[torch.Tensor],
        lengths: typing.Optional[torch.Tensor],
    ) -> typing.Optional[MarkEvalTensors]:
        """Build masked mark-eval tensors from caller-provided logits.

        Architectural invariant: τ₁ is always seeded from training data, so the
        model is evaluated on events 2..L only.  Targets are extracted internally
        as marks[:, 1:] (shape (N, L-1)).  The caller MUST provide mark_logits of
        shape (N, L-1, num_marks) — one logit per next-mark target.  Passing logits
        of shape (N, L, ...) will produce a shape crash in top_k_accuracy and
        cross_entropy downstream.

        Args:
            mark_logits: (N, L-1, num_marks) — pre-aligned logits.
            marks:       (N, L) — anchor-stripped marks (batch[2][:, 1:]).
            lengths:     (N,) — number of valid inter-arrival times (= L per sequence).
        """
        if mark_logits is None or marks is None or lengths is None:
            return None

        valid_lengths = (lengths - 1).clamp_min(0)
        if self._count_valid_next_mark_targets(lengths) == 0:
            return None

        previous_marks_masked, current_targets_masked = mask_mark_sequences(
            marks[:, :-1].clone(),
            marks[:, 1:].clone(),
            valid_lengths,
        )
        return MarkEvalTensors(mark_logits, previous_marks_masked, current_targets_masked)

    def _compute_mark_metrics_from_eval_tensors(
        self,
        *,
        mark_eval_data: typing.Optional[MarkEvalTensors],
        include_ce: bool = False,
        include_accuracy: bool = True,
        detach_ce: bool = False,
        precomputed_ce: typing.Optional[torch.Tensor] = None,
    ) -> typing.Optional[typing.Dict[str, typing.Union[torch.Tensor, float]]]:
        """Compute mark metrics from masked evaluation tensors."""
        if mark_eval_data is None:
            return None
        # Guard: generator can collapse to single-event sequences (lengths<=1), leaving no valid targets.
        # In which case we cant compute metrics.
        if not (mark_eval_data.current_targets != MARK_IGNORE_INDEX).any():
            return None

        mark_metrics: typing.Dict[str, typing.Union[torch.Tensor, float]] = {}
        if include_accuracy:
            mark_metrics.update(
                compute_mark_accuracy_metrics(
                    mark_eval_data.logits,
                    mark_eval_data.current_targets,
                    include_top3=self._include_top3_mark_accuracy,
                )
            )
        if include_ce:
            ce_loss = precomputed_ce
            if ce_loss is None:
                ce_loss = torch.nn.functional.cross_entropy(
                    mark_eval_data.logits.reshape(-1, mark_eval_data.logits.shape[-1]),
                    mark_eval_data.current_targets.reshape(-1),
                    ignore_index=MARK_IGNORE_INDEX,
                )
            mark_metrics['mark_ce'] = ce_loss.detach().cpu().item() if detach_ce else ce_loss

        return mark_metrics

    def _compute_and_log_mark_metrics_from_logits(
        self,
        *,
        mark_logits: typing.Optional[torch.Tensor],
        marks: typing.Optional[torch.Tensor],
        lengths: typing.Optional[torch.Tensor],
        prefix: str,
        include_ce: bool = False,
        include_accuracy: bool = True,
        precomputed_ce: typing.Optional[torch.Tensor] = None,
    ) -> typing.Optional[typing.Dict[str, typing.Union[torch.Tensor, float]]]:
        """Build, compute, and log mark metrics from caller-provided logits and targets."""
        mark_eval_data = self._build_mark_eval_tensors_from_logits(
            mark_logits=mark_logits,
            marks=marks,
            lengths=lengths,
        )
        mark_metrics = self._compute_mark_metrics_from_eval_tensors(
            mark_eval_data=mark_eval_data,
            include_ce=include_ce,
            include_accuracy=include_accuracy,
            precomputed_ce=precomputed_ce,
        )
        if mark_metrics is None:
            return None

        # Use the batch contract (lengths -> valid next-mark targets) as the single
        # weighting rule so train/val logging stays consistent across all call sites.
        self._log_all_metrics(
            mark_metrics,
            prefix,
            batch_size=self._count_valid_next_mark_targets(lengths),
        )
        return mark_metrics

    def _compute_mark_metrics_from_inputs(
        self,
        *,
        dts: typing.Optional[torch.Tensor] = None,
        dt_lens: typing.Optional[torch.Tensor] = None,
        marks: typing.Optional[torch.Tensor] = None,
        include_ce: bool = False,
        mark_eval_data: typing.Optional[MarkEvalTensors] = None,
    ) -> typing.Optional[typing.Dict[str, typing.Union[torch.Tensor, float]]]:
        """Return mark metrics from resolved model inputs or prebuilt eval tensors.

        When ``include_ce`` is True, also computes cross-entropy loss (used by ``on_test_end``).
        When ``mark_eval_data`` is provided, skips the build step (avoids a redundant forward pass).
        """
        if mark_eval_data is None:
            mark_eval_data = self._build_mark_tensors_for_validation(dts=dts, dt_lens=dt_lens, marks=marks)
        return self._compute_mark_metrics_from_eval_tensors(
            mark_eval_data=mark_eval_data,
            include_ce=include_ce,
            include_accuracy=True,
            detach_ce=True,
        )

    def sample_and_plot(
        self,
        name_plot4save: str = "",
        use_more_samples: bool = False,
        real_targets_and_lens: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        dts: typing.Optional[torch.Tensor] = None,
        dt_lens: typing.Optional[torch.Tensor] = None,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Generate samples and create diagnostic plots comparing generated vs real data.

        Args:
            name_plot4save: Suffix for saved plot filenames
            use_more_samples: If True, generate 10x more samples for plotting
            real_targets_and_lens: Optional (targets, lens) tuple to override default VAL data.
                                   Used during test phase to plot against test data.

        Returns:
            Dictionary of computed losses from the diagnostic plots.
        """
        if not self.enable_plot:
            return {}

        with torch.no_grad():
            num_seq = self.NUM_PLOT_SEQS_DURING_VAL * 10 if use_more_samples else self.NUM_PLOT_SEQS_DURING_VAL
            (gen_samples, gen_samples_lens, targets, targets_lens, _) = self._get_fake_real_samples(
                num_seq, DatasetSplitType.VAL, include_first_it=True
            )
            if real_targets_and_lens is not None:
                targets, targets_lens = real_targets_and_lens

            self._clear_all_axes()

            # Hook for subclass-specific pre-plotting (e.g., forward/backward trajectories)
            self._pre_diagnostic_plots_hook()

            # Detach and move to CPU
            gen_samples = gen_samples.detach().cpu()
            gen_samples_lens = gen_samples_lens.detach().cpu()
            targets = targets.detach().cpu()
            targets_lens = targets_lens.detach().cpu()

            # Set NaNs for values beyond sequence lengths
            gen_samples = set_seq_to_nan_from_index(gen_samples, gen_samples_lens - 1)
            targets4plot = set_seq_to_nan_from_index(targets, targets_lens - 1)

            num_elements_to_plot = min(gen_samples.shape[0], targets4plot.shape[0])

            try:
                # Moment metrics follow the TPPMetrics convention: drop the first inter-arrival (tau_1) and work on the remaining raw inter-arrival sequence (tau_2+). We slice from the already NaN-masked tensors, so the tail masking stays correct and does not need to be applied a second time.
                corr_loss = CorrLoss(targets4plot[:num_elements_to_plot, 1:, :])(
                    gen_samples[:num_elements_to_plot, 1:, :]
                )
            except Exception as e:
                logger.error("Failed to compute moment losses: %s", e)
                corr_loss = None

            diagnostic_plots_tpp(
                gen_samples[:num_elements_to_plot],
                targets4plot[:num_elements_to_plot],
                gen_samples_lens[:num_elements_to_plot],
                targets_lens[:num_elements_to_plot],
                self.hist_fig,
                self.intensity_ax,
                self.acf_fig,
                self.cov_err_fig,
                corr_loss=corr_loss,
                ax_temporal_plot=self.temporal_plot_ax,
                time_max=self.time_max,
            )
            mark_eval_tensors = self._plot_validation_mark_diagnostics(dts=dts, dt_lens=dt_lens, marks=marks)

            self._save_eval_plots(name_plot4save, include_mark_plots=mark_eval_tensors is not None)
            plt.pause(0.1)
            return

    # section ######################################################################
    #  #############################################################################
    #  Test Step

    def test_step(self, batch, batch_idx):
        """Lightning TEST hook: thin wrapper over :meth:`evaluate_split`.

        Kept only to satisfy the Lightning test loop, which is used for the
        single final pass on the validation-selected winner (and reruns in
        ``recompute_bootstrap``). Validation diagnostics do NOT come through
        here: the training manager calls :meth:`evaluate_split` directly with
        ``split=VAL``, so the split is always an explicit argument and the model
        carries no hidden evaluation-mode state.

        ``batch_idx > 0`` is rejected here because the Lightning loop is the only
        caller that batches; see :meth:`evaluate_split` for why a single batch is
        required.

        Args:
            batch: canonical ``(data, data_lens, marks)`` test batch
            batch_idx: Batch index (must be 0)
        """
        if batch_idx > 0:
            raise RuntimeError(
                "Multi-batch testing is not supported. "
                "Use a single batch containing all test data (or a representative subsample). "
                "See evaluate_split for rationale."
            )
        if len(batch) != 3:
            raise ValueError("test_step expects a canonical 3-tuple batch: (data, lengths, marks).")

        # Stage marker for experiment-log monitoring; test_step now runs only on
        # the real test pass, so this reliably signals the winner's evaluation.
        logger.info("Testing the model.")
        data, data_lens, marks = batch[0], batch[1], batch[2]
        self.evaluate_split(data, data_lens, marks, split=DatasetSplitType.TEST)
        return

    def evaluate_split(
        self,
        data: torch.Tensor,
        data_lens: torch.Tensor,
        marks: typing.Optional[torch.Tensor],
        *,
        split: DatasetSplitType = DatasetSplitType.TEST,
    ) -> typing.Dict[str, float]:
        """Run the full diagnostic suite on one split's data and return its metrics.

        Single source of truth for both evaluation passes; ``split`` is an
        explicit argument, so there is no hidden evaluation-mode state on the
        model:
          - ``test_step`` calls it with ``split=TEST``. The Lightning test loop
            provides device placement / ``eval`` / ``no_grad``, and afterwards
            ``on_test_end`` saves samples and plots.
          - The training manager calls it directly with ``split=VAL`` for
            per-config hyperparameter ranking, supplying device / ``eval`` /
            ``no_grad`` itself (``on_test_end`` never runs, so no artifacts are
            written for validation).
        The metric code is identical for both, so validation ranking and the
        final test report are directly comparable.

        Design note: single batch
        --------------------------
        ``data`` must hold the whole split in one batch. The metrics (signature
        W1, histogram, correlation, energy distance, Wasserstein) are
        distribution-based: they compare the generated distribution against the
        reference distribution and cannot be decomposed across batches and
        averaged, e.g. ``ED(gen, b1) + ED(gen, b2) != ED(gen, full)``. For very
        large splits, pass a representative subsample (~2000-5000 sequences give
        sufficient coverage, the same principle ``_estimate_metric_with_sampling``
        uses for ED/W1).

        Args:
            data: cumulative times with anchor, shape ``(N, L+1, D)``.
            data_lens: sequence lengths, shape ``(N,)``.
            marks: optional marks, shape ``(N, L+1)``; ``None`` when unmarked.
            split: which split this data belongs to (tags the metrics and labels
                the logs); ``TEST`` by default to preserve the test path.

        Returns:
            The aggregated ``*_mean`` / ``*_std`` metric dict. Also stored on
            ``self.metrics_test`` (read by the caller) and cached in
            ``self._test_batch_cache`` for ``on_test_end``.
        """
        logger.info("Running diagnostics on the %s split.", split.value)

        # Generate unconditional samples (produced once and reused across bootstrap replicates).
        uncond_result = self.sample_and_fix_seqs(num_seq=TPPArchitecture.NUM_SAMPLES_TEST)

        assert (
            TPPArchitecture.NUM_REPEAT_PER_SEQ_FOR_TEST_METRIC > 1
        ), "NUM_REPEAT_PER_SEQ_FOR_TEST_METRIC should be greater than 1 to ensure enough samples for the metrics."
        # Generate conditional samples for MAPE metrics (produced once for the full split).
        # The third argument is only a logging phase label; it follows the split
        # ("val"/"test") so log lines name the correct pass.
        cond_result = self.sample_for_a_fixed_batch_and_fix(
            (data, data_lens, marks),
            TPPArchitecture.NUM_REPEAT_PER_SEQ_FOR_TEST_METRIC,
            split.value,
            exact_num_sampling=True,
        )

        logger.debug("In testing of the current model.")
        logger.log(5, "The ref_its_nan are %s", cond_result.ref_its_nan.flatten(0, 1))
        logger.log(5, "The gen_its_tf_nan are %s", cond_result.gen_its_tf_nan.flatten(0, 1))

        metrics = self._run_bootstrap_metrics(
            data=data,
            data_lens=data_lens,
            marks=marks,
            uncond_result=uncond_result,
            cond_result=cond_result,
            split=split,
        )

        # Cache the split data for on_test_end() (plots + saving samples). Only the
        # Lightning test loop calls on_test_end, so this is consumed on TEST only.
        self._test_batch_cache = {
            'metrics': metrics,
            'data': data,
            'data_lens': data_lens,
            'marks': marks,
        }

        # Store results (read by the caller, e.g. trainingmanager via metrics_test).
        self.metrics_test = metrics

        return metrics

    def evaluate_split_no_grad(
        self,
        data: torch.Tensor,
        data_lens: torch.Tensor,
        marks: typing.Optional[torch.Tensor],
        *,
        split: DatasetSplitType = DatasetSplitType.TEST,
    ) -> typing.Dict[str, float]:
        """Run :meth:`evaluate_split` outside the Lightning Trainer loop.

        Supplies the ``eval()`` / ``no_grad()`` / device-placement boilerplate that
        ``trainer.test`` provides for the test pass. Callers (e.g. the training
        manager's direct validation diagnostics) must call ``self.to(device)``
        first; this method moves ``data``/``data_lens``/``marks`` onto
        ``self.device`` to match.
        """
        self.eval()
        data = data.to(self.device)
        data_lens = data_lens.to(self.device)
        marks = marks.to(self.device) if marks is not None else None
        with torch.no_grad():
            return self.evaluate_split(data, data_lens, marks, split=split)

    def _run_bootstrap_metrics(
        self,
        data: torch.Tensor,
        data_lens: torch.Tensor,
        marks: typing.Optional[torch.Tensor],
        uncond_result: UnconditionalSamplingResult,
        cond_result: ConditionalSamplingResult,
        split: DatasetSplitType = DatasetSplitType.TEST,
    ) -> typing.Dict[str, float]:
        """Compute B bootstrap replicates of the split's metrics, returning ``*_mean`` / ``*_std`` keys.

        Runs for either split (``VAL`` for ranking, ``TEST`` for the final
        report); "the split" below means whichever was passed.

        Bootstrap design
        ----------------
        The bootstrap unit is the evaluated sequence: each replicate resamples N
        sequences with replacement from the N split sequences, then evaluates all
        metrics on that resampled set. Generated samples (unconditional pool and
        conditional per-sequence samples) are produced *once* before the loop and
        reused across all B replicates, so bootstrap variance reflects uncertainty
        in the reference set only, not noise from re-sampling the model.

        Paired-testing invariant
        ------------------------
        The local CPU generator is re-seeded to 42 on every call. Because the generator
        is private (not shared with global torch/numpy/random state), two models evaluated
        on the same split of size N will draw the *identical* sequence of resampling
        indices for b = 0 ... B-1. This is the pre-condition for paired statistical tests
        (paired t / Wilcoxon / Diebold-Mariano) across models: the difference
        metric_A(b) - metric_B(b) is meaningful only when both are evaluated on the same
        resampled reference for replicate b.

        Per-replicate output
        --------------------
        The raw per-replicate metric vectors are stored in ``self._bootstrap_per_replicate``
        (a dict of (B,) arrays, one per metric) alongside the aggregated mean/std dict
        returned here. The per-replicate .npz written by recompute_bootstrap preserves these
        vectors so that downstream notebooks can run paired tests without re-running
        the full bootstrap loop.

        With B == 1 the loop degenerates to a single deterministic pass (std == 0).
        """
        cfg = self._metrics_config
        num_replicates = cfg.n_bootstraps

        N = data.shape[0]
        device = data.device

        # Pre-compute the full (B, N) index matrix with a private CPU generator so
        # that every call on the same split draws the same sequence of indices.
        # Required for paired comparisons across independently trained models; see
        # tests_src/test_bootstrap_pairing.py for the invariant lock.
        bootstrap_indices = generate_bootstrap_indices(N, num_replicates, seed=42)

        logger.info(
            "Bootstrap metrics: %d replicate(s), N=%d sequences.",
            num_replicates,
            N,
        )
        t_bootstrap_start = time.perf_counter()
        replicate_times: typing.List[float] = []

        metrics_per_replicate: typing.List[typing.Dict[str, float]] = []
        for b in range(num_replicates):
            t_rep_start = time.perf_counter()

            idx = bootstrap_indices[b].to(device)

            boot_data = data.index_select(0, idx)
            boot_lens = data_lens.index_select(0, idx)
            # boot_marks = marks.index_select(0, idx) if marks is not None else None

            # Pass the active split so the resulting TPPMetrics is tagged val/test;
            # this is what keeps the val-ranking columns and the test-report
            # columns namespaced apart downstream.
            boot_metrics = self._create_metrics_from_batch(boot_data, boot_lens, split=split)

            # Conditional samples are paired with the split's sequences along dim=1 of shape (S, N, L, D).
            cond_gen_b = cond_result.gen_its_tf_nan.index_select(1, idx)
            cond_ref_b = cond_result.ref_its_nan.index_select(1, idx)

            # Unconditional samples are intentionally the same fixed pool for every replicate:
            # all bootstrap variance comes from resampling the reference set, not from
            # regenerating samples (consistent with standard bootstrap evaluation design).
            metrics_b = boot_metrics.compute_all_metrics(
                uncond_result.its_scaled_cst,
                uncond_result.cum_abs_cst,
                uncond_result.its_scaled_nan,
                uncond_result.cum_rel_nan,
                cond_gen_b,
                cond_ref_b,
                uncond_result.seq_lens,
            )

            # Mark losses excluded from bootstrap replicates for speed.
            # boot_targets = boot_data.diff(dim=1)
            # boot_target_lens = boot_lens - 1
            # mark_metrics_b = self._compute_mark_metrics_from_inputs(
            #     dts=boot_targets,
            #     dt_lens=boot_target_lens,
            #     marks=boot_marks,
            #     include_ce=True,
            # )
            # if mark_metrics_b is not None:
            #     metrics_b.update(mark_metrics_b)

            metrics_per_replicate.append({k: float(v) for k, v in metrics_b.items()})

            replicate_times.append(time.perf_counter() - t_rep_start)
            # Log ~10 progress lines for any B (B=1 logs once; B<10 logs every replicate).
            log_step = max(1, num_replicates // 10)
            if (b + 1) % log_step == 0:
                elapsed = time.perf_counter() - t_bootstrap_start
                logger.info(
                    "Bootstrap progress: %d/%d replicates done | avg %.2fs/rep | est. %.1fs remaining.",
                    b + 1,
                    num_replicates,
                    elapsed / (b + 1),
                    elapsed / (b + 1) * (num_replicates - b - 1),
                )

        total_bootstrap_time = time.perf_counter() - t_bootstrap_start
        logger.info(
            "Bootstrap complete: %d replicates in %.2fs (avg %.2fs/rep).",
            num_replicates,
            total_bootstrap_time,
            total_bootstrap_time / num_replicates,
        )

        # Persist the raw per-replicate vectors before aggregation discards them.
        # recompute_bootstrap reads _bootstrap_per_replicate to write the per-replicate .npz;
        # without it, only mean/std survive and paired tests are impossible downstream.
        self._bootstrap_per_replicate = build_per_replicate_matrix(metrics_per_replicate)
        aggregated = aggregate_bootstrap_metrics(metrics_per_replicate)

        # Mark metrics are excluded from the bootstrap loop for speed; compute once
        # and save as plain scalar columns, not as bootstrap mean/std fields.
        targets = data.diff(dim=1)
        targets_lens = data_lens - 1
        mark_metrics = self._compute_mark_metrics_from_inputs(
            dts=targets,
            dt_lens=targets_lens,
            marks=marks,
            include_ce=True,
        )
        if mark_metrics is not None:
            for k, v in mark_metrics.items():
                aggregated[k] = float(v)
        return aggregated

    def _create_metrics_from_batch(
        self,
        data_split: torch.Tensor,
        data_split_lens: torch.Tensor,
        split: DatasetSplitType = DatasetSplitType.TEST,
    ) -> TPPMetrics:
        """
        Create TPPMetrics for a diagnostic batch without storing the split data.

        This method creates metrics on-the-fly, ensuring the architecture does
        not store evaluation data as instance attributes. It serves both the
        validation split (cross-config ranking) and the test split (final
        reporting); the same preprocessing and the model's train-fitted scaler
        are reused so the two are directly comparable.

        Args:
            data_split: Cumulative times with anchor, shape (B, L+1, D)
            data_split_lens: Sequence lengths, shape (B,)
            split: Which split this batch belongs to (default ``TEST`` to
                preserve existing test/bootstrap callers).

        Returns:
            TPPMetrics object for this batch, tagged with ``split``.
        """
        data_scaled_dts, data_cum, full_data_dt_lens = self._preprocess_dataset_for_metrics(data_split, data_split_lens)

        sig_loss_seqs = self.scale_paths_pre_sig(
            torch.cat([data_scaled_dts, data_cum], axis=2),
            seq_lens=full_data_dt_lens - 1,
        )

        metrics = TPPMetrics(
            data_split,
            data_split_lens,
            self.scaler_exp,
            self._metrics_config,
            sig_loss_seqs,
            self.scale_paths_pre_sig,
            split=split,
        )

        return metrics

    def _init_mark_components(self, *, history_size: int) -> int:
        """Initialize learnable mark components and return the history-encoder input size.

        Relies on self.num_marks / self.use_marks already set by the base class __init__.
        """
        if not self.use_marks:
            self.mark_emb = None
            self.event_emb = None
            self.mark_predictor = None
            return self.TIME_EMB_SIZE

        self.mark_emb = MarkEmbedding(self.num_marks, self.MARK_EMB_SIZE)
        self.event_emb = EventEmbedding(self.time_emb, self.mark_emb)
        self.mark_predictor = MarkPredictor(history_size, self.num_marks)
        return self.event_emb.embed_size

    def _preprocess_dataset_for_metrics(
        self,
        data: torch.Tensor,
        data_lens: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared preprocessing for train, val, and test datasets.

        Args:
            data: (N, L+1, D) cumulative times with zero anchor prepended.
            data_lens: (N,) sequence lengths (number of cumulative-time entries, including anchor).

        Returns:
            scaled_dts:      (N, L-1, D) exp-scaled inter-arrivals, constant-padded at seq end.
            cum:             (N, L-1, D) cumulative times, constant-padded at seq end.
            dt_lens:         (N,) = data_lens - 1.
        """
        full_dts = data.diff(dim=1)
        dt_lens = data_lens - 1
        dts = full_dts[:, 1:, :]

        cum = dts.cumsum(dim=1)

        scaled_dts = self.scaler_exp(dts)

        cum = set_seq_to_cst_val_from_index(cum, dt_lens - 2)
        scaled_dts = set_seq_to_cst_val_from_index(scaled_dts, dt_lens - 2)

        return scaled_dts, cum, dt_lens

    def on_test_end(self):
        """Save the final generated samples and plots after the test pass.

        This Lightning hook fires only on the ``trainer.test`` path. Validation
        diagnostics run through :meth:`evaluate_split` directly (never
        ``trainer.test``), so this hook only ever runs for the single
        validation-selected winner: artifacts are impossible for validation by
        control flow, with no split flag to check.
        """
        # Generate samples for saving to disk (τ₁ included, matches test targets contract).
        # Pin all RNG sources so samples_gen.pth is reproducible for the same checkpoint
        # regardless of prior RNG history (training steps, retry sampling, etc.).
        # No need to restore afterwards: sample_and_plot below is visualization-only.
        seed_everything(42, workers=True)
        completely_generated_samples, completely_generated_samples_lens, _, _, gen_marks_for_save = (
            self._get_fake_real_samples(self.NUM_PLOT_SEQS_DURING_TEST, DatasetSplitType.VAL, include_first_it=True)
        )
        # Process test data locally (not stored in self). τ₁ included to match plotting contract.
        targets = self._test_batch_cache['data'].diff(dim=1)  # shape (N, L, D); τ₁ at col 0
        targets_lens = self._test_batch_cache['data_lens'] - 1  # number of ITs including τ₁
        target_marks = self._test_batch_cache['marks']
        if self.output_dir is not None:
            tpp_utils.save_samples(
                inter_times=completely_generated_samples,
                lengths=completely_generated_samples_lens,
                path=f'{self.output_dir}samples_gen.pth',
                marks=gen_marks_for_save,
            )
            # Save target marks aligned with target times (skip anchor column).
            target_marks_for_save = target_marks[:, 1:] if target_marks is not None else None
            # The test split is identical (and deterministically chosen) across every
            # model trained for this experiment, so samples_tgt.pth used to be a
            # byte-identical copy in every model's output directory. Save it once at
            # the experiment level (one directory above models/<model_name>/), keyed
            # by a content fingerprint so repeated runs don't keep rewriting it.
            experiment_dir = os.path.dirname(os.path.dirname(os.path.normpath(self.output_dir)))
            tpp_utils.save_test_targets_once(
                experiment_dir=experiment_dir,
                inter_times=targets,
                lengths=targets_lens,
                marks=target_marks_for_save,
            )

        self.sample_and_plot(
            name_plot4save="test",
            use_more_samples=True,
            real_targets_and_lens=(targets, targets_lens),
            dts=targets,
            dt_lens=targets_lens,
            marks=self._test_batch_cache['marks'],
        )
        plt.pause(1)

        try:
            self.log_results_comparison(self.metrics_test)
        except Exception as e:
            logger.error("Could not log the results due to %s.", e)

        # Clean up cache
        del self._test_batch_cache

        return

    def on_test_epoch_start(self):
        self._test_start = time.time()
        return

    def on_test_epoch_end(self):
        total = time.time() - self._test_start
        logger.info("Total time for testing: %.2f seconds", total)
        return

    # section ######################################################################
    #  #############################################################################
    #  Helper Methods

    def _log_all_metrics(
        self,
        metrics: typing.Dict[str, float],
        prefix: str,
        batch_size: typing.Optional[int] = None,
    ):
        """Log all metrics with an optional explicit batch-size weight."""
        for name, value in metrics.items():
            log_kwargs = {
                'name': prefix + name,
                'value': value,
                'prog_bar': True,
                'on_step': False,
                'on_epoch': True,
            }
            if batch_size is not None:
                log_kwargs['batch_size'] = batch_size
            self.log(**log_kwargs)
        return

    def register_gradient_clipping(self):
        CLIP_VALUE: float = 0.1
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: torch.clamp(grad, -CLIP_VALUE, CLIP_VALUE))
        return

    # section ######################################################################
    #  #############################################################################
    #  Load/Save State

    def load_state_dict(self, state_dict, strict=True):
        # Metric statistics (histograms, autocorr, etc.) are stored as non-persistent buffers
        # and therefore never appear in state_dict: no key filtering needed.
        super().load_state_dict(state_dict, strict=False)
