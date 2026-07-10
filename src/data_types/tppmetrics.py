"""
TPP Metrics - Temporal Point Process Metrics

This module provides a unified interface for computing metrics on temporal point process models.
It separates metric computation from model architecture, improving testability and reusability.
"""

import logging
import typing
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Dict, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)
from src.metrics.sigw1metric_exp import SigW1MetricExp

from src.utils.fix_seq_ends import set_seq_to_cst_val_from_index, set_seq_to_nan_from_index
from src.utils.utils_os import suppress_logging
from src.data_transformations.statscompute import nanmean, nanmedian_numpy
from src.metrics.crps import get_crps_loss_weighted_by_targets
from src.metrics.energy_distance_tpp import energy_distance_tpp
from src.metrics.histogram_loss import HistogramLoss
from src.metrics.lebesgue_loss import (
    get_L1loss_weighted_by_targets,
    get_L1loss_conditional_weighted_by_targets,
    get_L2loss_weighted_by_targets,
)
from src.metrics.corrloss import CorrLoss
from src.metrics.autocorr_loss import AutoCorrLoss
from src.metrics.wasstpp import w1_between_processes_via_tpp_norm


class DatasetSplitType(str, Enum):
    """Enum for dataset split types."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class TPPMetricsConfig:
    """Configuration for TPP metrics computation."""

    # Signature metric parameters
    sig_degree: int = 4
    scale_high_degrees: bool = True
    standardise_sig: bool = True
    # Run the signature metric's internal computation (and backward) in float64
    # while keeping the float32 boundary. See SigW1MetricExp for the rationale.
    use_float64_signature: bool = False

    # Histogram parameters
    num_bins: Optional[int] = None  # If None, use Freedman-Diaconis rule

    # Distance metric sampling parameters
    energy_distance_sample_size: int = 2000
    wasserstein_sample_size: int = 2000

    # CRPS parameters
    crps_weight_by_targets: bool = True

    # General
    time_max: float = 12.0

    # Extra metrics (test-only, not ranked, not in norm_score)
    save_extra_metrics: bool = True

    # Bootstrap evaluation (test step). When > 1, the test step resamples test
    # sequences with replacement and reports metric_mean / metric_std across
    # replicates. Generated samples are produced once and reused; only the
    # reference test set is bootstrapped.
    n_bootstraps: int = 1


class TPPMetrics(nn.Module):
    """
    Metrics computation for Temporal Point Process models.

    Internal metric objects:
        - histogram_loss_it: HistogramLoss for inter-arrival times
        - histogram_loss_cum: HistogramLoss for cumulative times
        - sigw_metric_high_order: Signature W1 metric (high order, standardized)
        - sigw_metric_low_order: Signature W1 metric (low order, not standardized)
        - corr_loss: Correlation loss on raw inter-arrival times
        - corr_short_loss: Correlation loss on a short-lag temporal window
        - autocorr_it_loss: Autocorrelation loss on raw inter-arrival times
        - autocorr_it_short_loss: Short-lag autocorrelation loss on raw inter-arrival times
        - autocorr_loss: Autocorrelation loss on raw cumulative times
        - autocorr_short_loss: Short-lag autocorrelation loss on raw cumulative times
    """

    DISPLAY_METRIC_NAMES: typing.ClassVar[typing.List[str]] = [
        "sigW_loword_notstd",
        "hist_it",
        "hist_int",
        "ED",
        "W1",
        "CRPS",
        "corr",
        "corr_short",
        "autocorr_it",
        "autocorr_it_short",
        "autocorr",
        "autocorr_short",
        "MAE_proper",
        "MSE_proper",
        "MAE",
        "mark_ce",
        "top1_mark_acc",
        "top3_mark_acc",
    ]

    def __init__(
        self,
        reference_data: torch.Tensor,  # (N, L+1, D) - cumulative times with anchor
        reference_lens: torch.Tensor,  # (N,) - sequence lengths
        scaler,  # ExpScaler or similar with forward() and unscale() methods
        config: TPPMetricsConfig,
        sig_loss_seqs: torch.Tensor,  # Preprocessed sequences for signature metric
        scale_paths_pre_sig: Callable[
            ..., torch.Tensor
        ],  # Callable to scale paths before signature computation (accepts optional seq_lens)
        split: DatasetSplitType = DatasetSplitType.TRAIN,
    ):
        """
        Initialize metrics with reference data.

        Args:
            reference_data: Cumulative times including anchor point, shape (N, L+1, D)
            reference_lens: Sequence lengths, shape (N,)
            scaler: Scaler object (must have forward() and unscale() methods)
            config: Configuration for metrics
            sig_loss_seqs: Preprocessed sequences for signature metric computation (scaled inter-arrivals + cumulative times)
            scale_paths_pre_sig: Callable[[torch.Tensor], torch.Tensor] - scales paths before signature computation
            split: Dataset split (TRAIN, VAL, or TEST)
        """
        super().__init__()
        self.config = config
        self.split = split
        self.scaler = scaler
        self.scale_paths_pre_sig = scale_paths_pre_sig

        logger.debug(f"Initializing TPPMetrics for {split.value} dataset")

        # Register sig_loss_seqs as non-persistent buffer so it moves with .to(device)
        # but is excluded from state_dict.
        self.register_buffer('sig_loss_seqs', sig_loss_seqs, persistent=False)

        # Process reference data
        self._process_reference_data(reference_data, reference_lens)

        logger.debug(
            f"TPPMetrics initialized for {split.value} with {self.reference_data_cum_naned.shape[0]} sequences"
        )

    @property
    def device(self):
        """Device where the module's tensors are located."""
        return self.reference_data_cum_naned.device

    def _process_reference_data(
        self,
        reference_data: torch.Tensor,
        reference_lens: torch.Tensor,
    ):
        """Process reference data into required formats."""

        # Convert cumulative to inter-arrival times
        full_data_dts = reference_data.diff(dim=1)  # (N, L, D)
        full_data_lens = reference_lens - 1

        # Remove first inter-arrival time (handled separately in architecture)
        reference_data_dts = full_data_dts[:, 1:, :]  # (N, L-1, D)
        reference_lens_processed = full_data_lens - 1

        # Compute relative cumulative times (τ₂, τ₂+τ₃, …), τ₁ excluded.
        # This is the same for ALL anchor modes, ensuring INT/ED/W1 measure
        # the same random object and are comparable across strategies.
        reference_data_cum = reference_data_dts.cumsum(axis=1)

        # Scale the data
        reference_data_scaled = self.scaler(reference_data_dts)

        # Set sequences to constant value from length index (for signature computation)
        # -1 because we removed first value, -1 because we set from the index
        reference_data_cum_const = set_seq_to_cst_val_from_index(reference_data_cum, reference_lens_processed - 1)
        reference_data_scaled_const = set_seq_to_cst_val_from_index(reference_data_scaled, reference_lens_processed - 1)

        # NaN-masked versions for metric computation
        reference_data_naned = set_seq_to_nan_from_index(reference_data_scaled_const, reference_lens_processed - 1)
        reference_data_cum_naned = set_seq_to_nan_from_index(reference_data_cum_const, reference_lens_processed - 1)
        reference_data_it_raw_naned = set_seq_to_nan_from_index(reference_data_dts, reference_lens_processed - 1)

        # Register buffers needed after init (for logging and device movement).
        # Non-persistent: excluded from state_dict (data-derived statistics, not model weights).
        self.register_buffer('reference_data_naned', reference_data_naned, persistent=False)
        self.register_buffer('reference_data_it_raw_naned', reference_data_it_raw_naned, persistent=False)
        self.register_buffer('reference_data_cum_naned', reference_data_cum_naned, persistent=False)

        # Initialize metric objects with local data
        self._initialize_metric_objects(
            reference_data_dts,
            reference_data_naned,
            reference_data_it_raw_naned,
            reference_data_cum_naned,
        )

    def _initialize_metric_objects(
        self,
        reference_data_dts: torch.Tensor,
        reference_data_naned: torch.Tensor,
        reference_data_it_raw_naned: torch.Tensor,
        reference_data_cum_naned: torch.Tensor,
    ):
        """Initialize metric objects with reference data."""
        # Histogram losses
        num_bins = self.config.num_bins or HistogramLoss.num_bins_freedman_diaconis_rule(reference_data_dts.shape[0])

        self.histogram_loss_it = HistogramLoss(reference_data_naned, num_bins)

        self.histogram_loss_cum = HistogramLoss(reference_data_cum_naned, num_bins)

        # Flattened HistogramLoss: pool (N, L, D) into one 1-D bag of (N*L*D, 1, 1).
        # Reuses the existing machinery; for a 1-cell input _weigh_by_sample_size is trivial.
        ref_naned_flat = reference_data_naned.reshape(-1, 1, 1)
        ref_cum_flat = reference_data_cum_naned.reshape(-1, 1, 1)
        self.histogram_loss_it_flat = HistogramLoss(ref_naned_flat, num_bins)
        self.histogram_loss_cum_flat = HistogramLoss(ref_cum_flat, num_bins)

        # Correlation losses
        self.corr_loss = CorrLoss(reference_data_it_raw_naned)
        short_lag = self._compute_short_lag(reference_data_it_raw_naned.shape[1])
        short_corr_window = max(1, min(short_lag, reference_data_it_raw_naned.shape[1]))
        self.corr_short_loss = CorrLoss(reference_data_it_raw_naned[:, :short_corr_window, :])

        # Autocorrelation loss - compute max lag similar to BaseArchitecture
        max_lag = self._compute_default_autocorr_max_lag(reference_data_it_raw_naned.shape[1])
        self.autocorr_it_loss = AutoCorrLoss(reference_data_it_raw_naned, max_lag, True)
        self.autocorr_loss = AutoCorrLoss(reference_data_cum_naned, max_lag, True)
        max_lag_short = min(short_lag, reference_data_it_raw_naned.shape[1] - 1)
        self.autocorr_it_short_loss = AutoCorrLoss(reference_data_it_raw_naned, max_lag_short, True)
        self.autocorr_short_loss = AutoCorrLoss(reference_data_cum_naned, max_lag_short, True)

        # Initialize signature metrics (now always called since sig_loss_seqs is required)
        self._initialize_signature_metrics()

    @staticmethod
    def _compute_default_autocorr_max_lag(seq_len: int) -> int:
        """Main ACF lag budget used historically (roughly half-sequence, capped at 50)."""
        return max(0, min(seq_len // 2 - 1, 50))

    @staticmethod
    def _compute_short_lag(seq_len: int) -> int:
        """
        Short-lag budget for additional correlation diagnostics.
        - 5 lags for sequences up to length 100
        - 10 lags for sequences longer than 100
        """
        requested = 10 if seq_len > 100 else 5
        return max(0, min(requested, seq_len - 1))

    def _initialize_signature_metrics(self):
        with suppress_logging():
            self.sigw_metric_low_order = SigW1MetricExp(
                self.sig_loss_seqs,
                sig_degree=3,
                scale_high_degrees=False,
                standardise=False,
            )
        logger.debug(f"Signature metrics initialized for {self.split.value} dataset")

    def create_and_get_signature_metrics(self, effective_sig_degree: Optional[int] = None) -> SigW1MetricExp:
        """Factory method to create and return a signature metric."""
        return SigW1MetricExp(
            self.sig_loss_seqs,
            sig_degree=self.config.sig_degree,
            scale_high_degrees=self.config.scale_high_degrees,
            standardise=self.config.standardise_sig,
            effective_sig_degree=effective_sig_degree,
            use_float64_signature=self.config.use_float64_signature,
        )

    # ========================================================================
    # DISTRIBUTION METRICS (Unconditional)
    # ========================================================================

    def compute_signature_metrics(
        self,
        generated_samples_scaled: torch.Tensor,  # (N, L, D) - ExpScaler-scaled inter-arrivals, const-ended
        generated_samples_cum: torch.Tensor,  # (N, L, D) - unscaled cumulative times, const-ended
        seq_lens: Optional[torch.Tensor] = None,  # (N,) - per-sequence valid lengths, needed for RESIDUAL
    ) -> Dict[str, float]:
        """
        Compute signature-based distribution metrics.
        generated_samples_scaled: Inter-arrival times scaled with ExpScaler (const-ended), shape (N, L, D)
        generated_samples_cum: Unscaled cumulative times (const-ended), shape (N, L, D)
        seq_lens: Per-sequence valid lengths (needed for RESIDUAL anchor mode)

        Returns:
            Dictionary with keys:
                - 'sigW_loword_notstd': Low-order non-standardized signature W1 distance
        """
        # Combine ExpScaler-scaled inter-arrivals with unscaled cumulative times (same as old code)
        gen_sig_input = torch.cat([generated_samples_scaled, generated_samples_cum], axis=2)

        # Apply scale_paths_pre_sig for signature computation
        gen_sig_scaled = self.scale_paths_pre_sig(gen_sig_input, seq_lens)

        # Log after cat+scale so both are in the same space as sig_loss_seqs (which is also cat+scaled)
        logger.debug("Signature metrics input  gen_sig_scaled: %s", gen_sig_scaled)
        logger.debug("Signature metrics target sig_seqs:       %s", self.sig_loss_seqs)

        return {
            'sigW_loword_notstd': self.sigw_metric_low_order(gen_sig_scaled).item(),
        }

    def compute_histogram_metrics(
        self,
        generated_samples: torch.Tensor,  # (N, L, D) - SCALED inter-arrivals (NaN-masked)
        generated_samples_cum: torch.Tensor,  # (N, L, D) - cumulative times
    ) -> Dict[str, float]:
        """
        Compute histogram-based distribution metrics.

        Args:
            generated_samples: Inter-arrival times (NaN-masked), shape (N, L, D)
            generated_samples_cum: Cumulative times (NaN-masked), shape (N, L, D)

        Returns:
            Dictionary with keys:
                - 'hist_it': Histogram distance for SCALED inter-arrival times
                - 'hist_int': Histogram distance for cumulative times (intensity)
        """
        logger.debug("Histogram metrics input  it:  %s", generated_samples)
        logger.debug("Histogram metrics input  cum: %s", generated_samples_cum)
        logger.debug("Histogram metrics target it:  %s", self.reference_data_naned)
        logger.debug("Histogram metrics target cum: %s", self.reference_data_cum_naned)
        return {
            'hist_it': self.histogram_loss_it(generated_samples, []).item(),
            'hist_int': self.histogram_loss_cum(generated_samples_cum, []).item(),
        }

    def compute_histogram_metrics_flat(
        self,
        generated_samples: torch.Tensor,  # (N, L, D) - SCALED inter-arrivals (NaN-masked)
        generated_samples_cum: torch.Tensor,  # (N, L, D) - cumulative times (NaN-masked)
    ) -> Dict[str, float]:
        """Flattened histogram metrics. Pools (N, L, D) into a 1-D bag, bins once, weighted reduction."""
        gen_it_flat = generated_samples.reshape(-1, 1, 1)
        gen_cum_flat = generated_samples_cum.reshape(-1, 1, 1)
        return {
            'hist_it_flat': self.histogram_loss_it_flat(gen_it_flat, []).item(),
            'hist_int_flat': self.histogram_loss_cum_flat(gen_cum_flat, []).item(),
        }

    def compute_correlation_metrics(
        self,
        generated_samples_it_raw: torch.Tensor,  # (N, L, D) - raw inter-arrivals, NaN-masked
        generated_samples_cum_raw: torch.Tensor,  # (N, L, D) - raw cumulative times, NaN-masked
    ) -> Dict[str, float]:
        """
        Compute correlation-based metrics.

        Args:
            generated_samples_it_raw: Raw inter-arrival times (NaN-masked), shape (N, L, D)
            generated_samples_cum_raw: Raw cumulative times (NaN-masked), shape (N, L, D)

        Returns:
            Dictionary with keys:
                - 'corr': Correlation loss on raw inter-arrival times
                - 'corr_short': Correlation loss on raw inter-arrival times (first short-lag window)
                - 'autocorr_it': Autocorrelation loss on raw inter-arrival times
                - 'autocorr_it_short': Short-lag autocorrelation loss on raw inter-arrival times
                - 'autocorr': Autocorrelation loss on raw cumulative times
                - 'autocorr_short': Short-lag autocorrelation loss on raw cumulative times
        """
        logger.debug("Correlation metrics input  it_raw:  %s", generated_samples_it_raw)
        logger.debug("Correlation metrics input  cum_raw: %s", generated_samples_cum_raw)
        logger.debug("Correlation metrics target it_raw:  %s", self.reference_data_it_raw_naned)
        logger.debug("Correlation metrics target cum_raw: %s", self.reference_data_cum_naned)
        corr_short_window = int(self.corr_short_loss.slice_t.item())
        return {
            'corr': self.corr_loss.loss(generated_samples_it_raw).item(),
            'corr_short': self.corr_short_loss.loss(generated_samples_it_raw[:, :corr_short_window, :]).item(),
            'autocorr_it': self.autocorr_it_loss.loss(generated_samples_it_raw).item(),
            'autocorr_it_short': self.autocorr_it_short_loss.loss(generated_samples_it_raw).item(),
            'autocorr': self.autocorr_loss.loss(generated_samples_cum_raw).item(),
            'autocorr_short': self.autocorr_short_loss.loss(generated_samples_cum_raw).item(),
        }

    def compute_all_unconditional(
        self,
        generated_samples_scaled: torch.Tensor,  # (N, L, D) - scaled, const-ended
        generated_samples_cum: torch.Tensor,  # (N, L, D) - cumulative, const-ended
        generated_samples_naned: torch.Tensor,  # (N, L, D) - scaled, NaN-masked
        generated_samples_cum_naned: torch.Tensor,  # (N, L, D) - cumulative, NaN-masked
        seq_lens: Optional[torch.Tensor] = None,  # (N,) - for RESIDUAL
    ) -> Dict[str, float]:
        """
        Compute all unconditional distribution metrics.

        These metrics measure how well the generated distribution matches
        the reference distribution, without conditioning on specific histories.

        Args:
            generated_samples_scaled: Scaled inter-arrivals (const-ended)
            generated_samples_cum: Cumulative times (const-ended)
            generated_samples_naned: Scaled inter-arrivals (NaN-masked)
            generated_samples_cum_naned: Cumulative times (NaN-masked)
            seq_lens: Per-sequence valid lengths (needed for RESIDUAL anchor mode)

        Returns:
            Dictionary containing all unconditional metrics
        """
        metrics = {}

        # Signature metrics
        metrics.update(self.compute_signature_metrics(generated_samples_scaled, generated_samples_cum, seq_lens))

        # Histogram metrics
        metrics.update(self.compute_histogram_metrics(generated_samples_naned, generated_samples_cum_naned))

        # Correlation metrics are computed on raw quantities, not on ExpScaler-transformed values.
        generated_samples_it_raw_naned = self.scaler.unscale(generated_samples_naned)
        metrics.update(self.compute_correlation_metrics(generated_samples_it_raw_naned, generated_samples_cum_naned))

        return metrics

    # ========================================================================
    # CONDITIONAL METRICS (Given History)
    # ========================================================================

    def compute_pointwise_metrics(
        self,
        generated_samples: torch.Tensor,  # (S, N, L, D) or (N, L, D) - NaN-masked, unscaled
        target_samples: torch.Tensor,  # (S, N, L, D) or (N, L, D) - NaN-masked, unscaled
    ) -> Dict[str, float]:
        """
        Compute pointwise prediction metrics (MAE_proper, MSE_proper).

        These metrics measure how well the model predicts the next inter-arrival time
        given a history, compared to the true next inter-arrival time.

        For multi-sample predictions (S > 1), the "proper" metrics aggregate across
        the S samples for each of the N sequences before computing the error.

        Args:
            generated_samples: Generated inter-arrivals, shape (S, N, L, D), should be NaN-masked and unscaled.
            target_samples: Target inter-arrivals, shape (S, N, L, D)
            For multi-sample, targets are replicated S times (same target for each sample).

        Returns:
            Dictionary with keys:
                - 'MAE_proper': MAE using median of samples per sequence
                - 'MSE_proper': MSE using mean of samples per sequence
        """
        logger.debug("Pointwise metrics input  gen[0]:    %s", generated_samples[0])
        logger.debug("Pointwise metrics input  target[0]: %s", target_samples[0])

        # Aggregate across S dimension for each sequence: (S, N, L, D) -> (N, L, D)
        # MAE_proper uses median, MSE_proper uses mean
        # Extract unique target (first sample, since all S samples have same target)
        # (S, N, L, D) -> (N, L, D)
        tgt = target_samples[0]

        # Aggregate across S (dim=0) per (N, L, D)
        # Use torch nan-safe reductions; if you already have your own helpers, swap them in.
        pred_median = nanmedian_numpy(generated_samples, dim=0)  # (N, L, D)
        pred_mean = nanmean(generated_samples, dim=0)  # (N, L, D)
        mae_proper = get_L1loss_weighted_by_targets(pred_median, tgt)
        mse_proper = get_L2loss_weighted_by_targets(pred_mean, tgt)

        return {
            'MAE_proper': mae_proper.item(),
            'MSE_proper': mse_proper.item(),
        }

    def compute_crps(
        self,
        generated_samples: torch.Tensor,  # (S, N, L, D) - multiple samples per sequence
        target_samples: torch.Tensor,  # (S, N, L, D) - targets replicated S times
    ) -> float:
        """
        Compute Continuous Ranked Probability Score (CRPS).

        CRPS measures the quality of probabilistic predictions.

        Args:
            generated_samples: Generated samples, shape (S, N, L, D)
                              S samples per sequence for probabilistic evaluation
            target_samples: Target samples, shape (S, N, L, D)
                           Targets replicated S times (all S copies are identical)

        Returns:
            CRPS score (float)
        """
        # CRPS expects (S, N, L) shape and (N, L) target
        # Transpose to (N, S, L) if needed
        if generated_samples.ndim == 4 and generated_samples.shape[-1] == 1:
            gen_transposed = generated_samples[:, :, :, 0].transpose(0, 1).transpose(1, 2)
            tgt = target_samples[0, :, :, 0]  # Take first target (they should all be the same)
        else:
            raise ValueError(f"CRPS expects 4D input with D=1, got shape {generated_samples.shape}")

        return get_crps_loss_weighted_by_targets(gen_transposed, tgt).item()

    def compute_mae_conditional(
        self,
        generated_samples: torch.Tensor,  # (S, N, L, D)
        target_samples: torch.Tensor,  # (S, N, L, D)
    ) -> float:
        """
        Compute MAE from conditional samples: E_S[|sample - target|], then weighted reduce.

        Same input convention as CRPS: S samples per sequence, targets replicated S times.
        """
        if generated_samples.ndim == 4 and generated_samples.shape[-1] == 1:
            gen_transposed = generated_samples[:, :, :, 0].transpose(0, 1)  # (S,N,L) → (N,S,L)
            tgt = target_samples[0, :, :, 0]  # (N, L)
        else:
            raise ValueError(f"MAE expects 4D input with D=1, got shape {generated_samples.shape}")

        return get_L1loss_conditional_weighted_by_targets(gen_transposed, tgt).item()

    def compute_all_conditional(
        self,
        generated_samples: torch.Tensor,
        target_samples: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute all conditional metrics.

        Args:
            generated_samples: Generated samples, shape (S, N, L, D).
            target_samples: Target samples, same shape as generated

        Returns:
            Dictionary containing all conditional metrics
        """
        metrics = {}
        metrics.update(self.compute_pointwise_metrics(generated_samples, target_samples))
        metrics['CRPS'] = self.compute_crps(generated_samples, target_samples)
        metrics['MAE'] = self.compute_mae_conditional(generated_samples, target_samples)
        return metrics

    # ========================================================================
    # DISTANCE METRICS (Distribution Comparison)
    # ========================================================================

    def compute_energy_distance(
        self,
        generated_samples_cum: torch.Tensor,  # (N, L, D) - cumulative times, NaN-masked
    ) -> float:
        """
        Compute Energy Distance between generated and reference distributions.

        Args:
            generated_samples_cum: Generated cumulative times, shape (N, L, D)

        Returns:
            Mean distance across sampling rounds.
        """
        return self._estimate_metric_with_sampling(
            generated_samples_cum,
            self.reference_data_cum_naned,
            energy_distance_tpp,
            sample_size=self.config.energy_distance_sample_size,
        )

    def compute_wasserstein_distance(
        self,
        generated_samples_cum: torch.Tensor,  # (N, L, D) - cumulative times, NaN-masked
        reg: float = None,
    ) -> float:
        """
        Compute Wasserstein-1 Distance (W1) between generated and reference distributions.
        Uses exact EMD (Earth Mover's Distance) via POT's ot.emd2 with chunked pairwise cost matrix.
        If reg > 0, uses Sinkhorn entropic regularization instead.

        Args:
            generated_samples_cum: Generated cumulative times, shape (N, L, D)
            reg: Regularization parameter. None → exact EMD; >0 → Sinkhorn (blur_frac).

        Returns:
            Mean distance across sampling rounds.
        """
        return self._estimate_metric_with_sampling(
            generated_samples_cum,
            self.reference_data_cum_naned,
            partial(w1_between_processes_via_tpp_norm, reg=reg),
            sample_size=self.config.wasserstein_sample_size,
        )

    def compute_sliced_energy_distance(
        self,
        generated_samples_cum: torch.Tensor,  # (N, L, D) - cumulative times, NaN-masked
        num_projections: int = 1000,
    ) -> float:
        """
        Compute Sliced Energy Distance. O(num_projections*N*log N) alternative to standard ED.
        """
        from src.metrics.sliced_energy_distance import sliced_energy_distance_tpp

        return sliced_energy_distance_tpp(
            generated_samples_cum,
            self.reference_data_cum_naned,
            T=self.config.time_max,
            num_projections=num_projections,
        )

    def compute_sliced_wasserstein_distance(
        self,
        generated_samples_cum: torch.Tensor,  # (N, L, D) - cumulative times, NaN-masked
        num_projections: int = 1000,
    ) -> float:
        """
        Compute Sliced Wasserstein-1 Distance. O(num_projections*N*log N) alternative to standard W1.
        """
        from src.metrics.sliced_wasserstein_tpp import sliced_wasserstein_tpp

        return sliced_wasserstein_tpp(
            generated_samples_cum,
            self.reference_data_cum_naned,
            T=self.config.time_max,
            num_projections=num_projections,
        )

    def _estimate_metric_with_sampling(
        self,
        model_samples: torch.Tensor,
        target_samples: torch.Tensor,
        metric_fn: Callable,
        sample_size: int = 2000,
    ) -> float:
        """
        Estimate distance metrics using Monte Carlo subsampling to manage memory constraints.

        This method computes expensive distance metrics (Energy Distance, Wasserstein Distance)
        between large datasets by:
        1. Taking random subsamples of size B from both datasets
        2. Computing the metric on these subsamples
        3. Repeating multiple times and averaging results

        Mathematical Justification:
        ------------------------
        For Energy Distance (ED):
            - ED is defined as an expectation over all pairs: ED(P,Q) = 2E[d(X,Y)] - E[d(X,X')] - E[d(Y,Y')]
            - Computing ED on i.i.d. subsamples gives unbiased estimates of the true ED
            - Averaging K independent estimates reduces variance by factor of K
            - This is standard Monte Carlo estimation

        For Wasserstein Distance (W1):
            - W1 on subsamples approximates the true W1 (not necessarily unbiased)
            - As subsample size increases, W1_subsample → W1_full (consistency)
            - Averaging multiple subsample estimates reduces variance and improves stability

        Memory Management:
        -----------------
        Computing metrics on full datasets requires O(M*N) memory for pairwise distances.
        For M=N=10,000 sequences, this requires ~100M distance computations.
        By subsampling to B=2000, we reduce this to 4M computations per round.

        Sampling Strategy:
        -----------------
        - If B <= dataset_size: Use torch.randperm (sampling without replacement)
        - If B > dataset_size: Use torch.randint (sampling with replacement)

        The number of rounds is chosen to provide reasonable coverage:
        - If B >= min(M,N): Single round (subsample already covers smallest dataset)
        - Otherwise: ceil(min(M,N) / B) rounds to ensure multiple independent estimates

        Args:
            model_samples: Generated samples, shape (M, L, D)
                          M = number of generated sequences
            target_samples: Reference samples, shape (N, L, D)
                           N = number of reference sequences
            metric_fn: Distance metric function with signature:
                      metric_fn(batch1: Tensor, batch2: Tensor, T: float) -> float
                      Examples: energy_distance_tpp, w1_between_processes_via_tpp_norm
            sample_size: Maximum number of samples per round (default: 2000)
                        Controls memory usage vs. estimation quality tradeoff

        Returns:
            Mean of metric estimates across all rounds.
        """
        # Use device from input tensors
        device = model_samples.device

        # Determine dataset sizes and batch size
        M = model_samples.shape[0]  # Number of generated sequences
        N = target_samples.shape[0]  # Number of reference sequences
        B = min(sample_size, M, N)  # Actual batch size (limited by smallest dataset)
        min_len = min(M, N)

        # Calculate number of sampling rounds for Monte Carlo estimation
        # If B >= min_len: one batch covers the entire smaller dataset → 1 round sufficient
        # If B < min_len: perform ceil(min_len / B) rounds to get multiple independent estimates
        num_rounds = 1 if B >= min_len else (min_len + B - 1) // B

        results_per_batch = []
        for _ in range(num_rounds):
            # Sample B indices from model_samples (M total)
            if B > M:
                # Need more samples than available → sample with replacement
                midx = torch.randint(0, M, (B,), device=device)
            else:
                # Can sample without replacement → use random permutation (avoids duplicates)
                midx = torch.randperm(M, device=device)[:B]

            # Sample B indices from target_samples (N total)
            if B > N:
                # Need more samples than available → sample with replacement
                tidx = torch.randint(0, N, (B,), device=device)
            else:
                # Can sample without replacement → use random permutation (avoids duplicates)
                tidx = torch.randperm(N, device=device)[:B]

            # Extract subsamples
            mb = model_samples[midx]  # (B, L, D)
            tb = target_samples[tidx]  # (B, L, D)

            # Compute metric on this subsample pair
            v = metric_fn(mb, tb, T=self.config.time_max)
            v = float(v)
            results_per_batch.append(v)

        # Aggregate results across rounds
        results_per_batch = torch.tensor(results_per_batch, dtype=torch.float64, device=device)
        return results_per_batch.mean().item()

    def compute_all_metrics(
        self,
        generated_samples_scaled: torch.Tensor,  # (N, L, D) - scaled, const-ended
        generated_samples_cum: torch.Tensor,  # (N, L, D) - cumulative, const-ended
        generated_samples_naned: torch.Tensor,  # (N, L, D) - scaled, NaN-masked
        generated_samples_cum_naned: torch.Tensor,  # (N, L, D) - cumulative, NaN-masked
        generated_samples_conditional: torch.Tensor,  # (S, N, L, D) - for conditional metrics
        target_samples_conditional: torch.Tensor,  # (S, N, L, D) - for conditional metrics
        seq_lens: Optional[torch.Tensor] = None,  # (N,) - for RESIDUAL
    ) -> Dict[str, float]:
        """
        Comprehensive helper to compute all available metrics in one call.

        This method consolidates all metric computations to avoid calling multiple methods
        in different places throughout the codebase. All inputs are required.

        Args:
            generated_samples_scaled: Scaled inter-arrivals (const-ended), shape (N, L, D)
            generated_samples_cum: Cumulative times (const-ended), shape (N, L, D)
            generated_samples_naned: Scaled inter-arrivals (NaN-masked), shape (N, L, D)
            generated_samples_cum_naned: Cumulative times (NaN-masked), shape (N, L, D)
            generated_samples_conditional: Generated samples for conditional metrics, shape (S, N, L, D)
            target_samples_conditional: Target samples for conditional metrics, shape (S, N, L, D)
            seq_lens: Per-sequence valid lengths (needed for RESIDUAL anchor mode)

        Returns:
            Dictionary containing all computed metrics:
                - Unconditional: sigW_loword_notstd, hist_it, hist_int,
                  corr, corr_short, autocorr_it, autocorr_it_short, autocorr, autocorr_short
                - Conditional: MAE_proper, MSE_proper, CRPS
                - Distance: ED, W1
        """
        logger.debug(
            "\n\n%s\n  Computing all metrics for %s split\n%s\n",
            "=" * 70,
            self.split.value,
            "=" * 70,
        )
        metrics = {}

        # Unconditional metrics
        metrics.update(
            self.compute_all_unconditional(
                generated_samples_scaled,
                generated_samples_cum,
                generated_samples_naned,
                generated_samples_cum_naned,
                seq_lens,
            )
        )

        # Test-only flattened histogram metrics (not displayed/ranked).
        if self.split == DatasetSplitType.TEST and self.config.save_extra_metrics:
            metrics.update(
                self.compute_histogram_metrics_flat(
                    generated_samples_naned,
                    generated_samples_cum_naned,
                )
            )

        metrics.update(
            self.compute_all_conditional(
                generated_samples_conditional,
                target_samples_conditional,
            )
        )

        try:
            ed = self.compute_energy_distance(generated_samples_cum_naned)
            logger.debug("Energy distance: %.4f", ed)
            w1 = self.compute_wasserstein_distance(generated_samples_cum_naned, reg=None)
            logger.debug("Wasserstein distance (exact): %.4f", w1)
        except Exception as e:
            logger.error("Failed to compute ED/W1 (setting to NaN): %s", e)
            ed = w1 = float('nan')
        metrics.update(
            {
                'ED': ed,
                'W1': w1,
            }
        )

        return metrics
