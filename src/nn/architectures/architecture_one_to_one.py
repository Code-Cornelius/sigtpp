import logging
import math
import typing

import torch

logger = logging.getLogger(__name__)
from src.data_transformations.standardscaler import StandardScaler
from src.data_transformations.statscompute import variable_len_standard_stats
from src.nn.architectures.tpp_architecture import TPPArchitecture
from src.nn.architectures.mark_prediction_utils import (
    extract_marks_without_anchor_from_batch,
)

from src.utils import tpp_utils
from src.metrics.sigw1_degree_detector import SigW1DegreeDetector
from src.metrics.sigw1metric_exp import SigW1MetricExp
from src.metrics.totalvar import total_var

from src.data_types.exceptions import SkipConfig
from src.data_types.sigw_loss_data_props import SigWLossDataProps
from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.metrics.anchors.terminal_anchor_strategy import make_anchor_strategy
from src.nn.nn.sigwgan_modules.rnn_sampling_generator_tpp import RNNSamplingGeneratorTPP


class ArchitectureOneToOne(TPPArchitecture):
    """Autoregressive one-step generator with optional scheduled sampling during training."""

    # Scheduled sampling curriculum constants. Teacher Forcing, where we use dataset for conditioning.
    CURRICULUM_DECAY_FRACTION = 0.35  # Use 35% of training epochs for curriculum learning
    CURRICULUM_MAX_EPOCHS = 5000  # Cap curriculum duration at 5000 epochs maximum
    COSINE_SCHEDULER_MIN_LR_RATIO = 0.1  # Keep a non-zero LR floor for late adversarial fine-tuning.

    @property
    def _is_train_sig_pipeline_ready(self) -> bool:
        """True once train-time signature preprocessing objects are fully configured."""
        return (
            self._anchor_strategy_train is not None
            and self._scaler_std_train is not None
            and self._total_vars_train is not None
        )

    @staticmethod
    def _truncate_to_cap(
        data: torch.Tensor,
        lens: torch.Tensor,
        cap: int,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Truncate cumulative-time data to the anchor + first ``cap`` real events.

        Shape change: ``(N, L+1, D) -> (N, min(cap, L)+1, D)``. The t=0 anchor at
        column 0 is preserved; columns hold anchor + first ``cap`` cumulative times.
        ``lens`` (number of cumulative entries including anchor) is clamped to
        ``cap + 1``. Sequences with <= cap events are unchanged. Faithful to the
        constant-padding convention used by the EDITPP data modules.

        Args:
            data: (N, L+1, D) cumulative times with the zero anchor at column 0.
            lens: (N,) cumulative-entry counts including the anchor.
            cap: maximum number of real events (anchor excluded) to keep.

        Returns:
            (data_capped, lens_capped) with the shapes/clamping described above.
        """
        keep = min(cap + 1, data.shape[1])  # +1 for the t=0 anchor column
        data_capped = data[:, :keep, :].contiguous()
        lens_capped = torch.clamp(lens, max=keep)
        return data_capped, lens_capped

    def _effective_rollout_dts_len(self) -> int:
        """Number of post-tau_1 inter-arrival steps to roll out unconditionally.

        Capped to ``train_seq_cap - 1`` during training (``self.training`` True), so the
        generated path holds the first ``train_seq_cap`` events to match the capped
        signature reference. Validation/test/plotting (``self.training`` False) use the
        full length. ``data_train_dts`` excludes tau_1, hence the ``-1`` on the cap.
        """
        full = self.data_train_dts.shape[1]
        if self.training and self.train_seq_cap is not None:
            return max(1, min(self.train_seq_cap - 1, full))
        return full

    def _cap_training_batch(
        self,
        batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Truncate a (data, lens, marks) training batch to the first ``train_seq_cap`` events.

        Keeps the teacher-forced path's generated length aligned with the capped
        signature reference. Only called from ``training_step``; validation/test
        batches are never capped.
        """
        data, lens, marks = batch[0], batch[1], batch[2]
        data_capped, lens_capped = self._truncate_to_cap(data, lens, self.train_seq_cap)
        keep = data_capped.shape[1]
        marks_capped = marks[:, :keep] if marks is not None else marks
        return data_capped, lens_capped, marks_capped

    def __init__(
        self,
        data_train: torch.Tensor,
        data_train_lens: torch.Tensor,
        data_val: torch.Tensor,
        data_val_lens: torch.Tensor,
        train_marks: torch.Tensor,
        val_marks: torch.Tensor,
        period_plot_val: int,
        ############################################
        loss_properties: SigWLossDataProps,
        learning_rate: float,
        concentration_factor: float,
        hid_size_rep: int,
        use_teacher_forcing: bool,
        t_max: int,
        num_marks: int,
        total_epochs: int,
        ############################################
        output_dir: str = None,
        enable_plot: bool = False,
        plot_every_n_val_steps: int = 1,
        detach_cum_channel: bool = False,
        terminal_anchor_mode: TerminalAnchorMode = TerminalAnchorMode.FREE_ENDPOINT,
        use_lr_scheduler: bool = False,
        ############################################
        mark_loss_weight: float = 1.0,
        n_bootstraps: int = 1,
        train_seq_cap: typing.Optional[int] = None,
        **kwargs,
    ):
        """
        Data given is of shape (N, L+1, D). There are L inter-arrival times available.
        We use that naming convention throughout the file.

        train_seq_cap: optional length cap (number of real events, anchor excluded) applied
            ONLY to the training path -- the autoregressive rollout and the signature
            reference are truncated to the first ``train_seq_cap`` events for memory/throughput
            on long-sequence datasets. Validation/test always run on full-length sequences,
            and the dataset itself is never capped. ``None`` disables capping (default).
            See docs/seq_length_capping/06-15-15_FEAT_seq_length_capping.md.
        """
        if terminal_anchor_mode == TerminalAnchorMode.RESIDUAL and not detach_cum_channel:
            raise SkipConfig("RESIDUAL anchor requires detach_cum_channel=True (training collapses otherwise).")

        self.sigw_loss_properties: SigWLossDataProps = loss_properties
        self.use_teacher_forcing = use_teacher_forcing
        self.total_epochs = total_epochs
        self.use_lr_scheduler = use_lr_scheduler

        # Train-only sequence-length cap (sigwgan/SigTPP). Set before super().__init__ and the
        # training-pipeline setup below, both of which consult it. None => no cap.
        if train_seq_cap is not None and (not isinstance(train_seq_cap, int) or train_seq_cap < 2):
            raise ValueError(f"train_seq_cap must be an int >= 2 (real events) or None, got {train_seq_cap!r}.")
        self.train_seq_cap: typing.Optional[int] = train_seq_cap
        # Training-pipeline attributes are initialized eagerly to avoid attribute-order issues
        # while parent __init__ computes baseline proxies.
        self._anchor_strategy_train = None
        self._scaler_std_train = None
        self._total_vars_train = None

        # Calculate curriculum decay period using class constants
        self.curriculum_decay_epochs = min(
            int(total_epochs * self.CURRICULUM_DECAY_FRACTION), self.CURRICULUM_MAX_EPOCHS
        )

        # Training loss anchor mode; parent (metrics/validation) always uses FREE_ENDPOINT.
        self._terminal_anchor_mode_train = terminal_anchor_mode

        super().__init__(
            t_max,
            num_marks,
            data_train,
            data_train_lens,
            data_val,
            data_val_lens,
            train_marks,
            val_marks,
            concentration_factor,
            output_dir,
            enable_plot,
            period_plot_val,
            plot_every_n_val_steps,
            n_bootstraps=n_bootstraps,
        )
        self.detach_cum_channel: bool = detach_cum_channel

        # Set up training-specific anchor pipeline (may differ from metrics pipeline).
        self._setup_training_anchor_pipeline(data_train, data_train_lens)

        self.mark_loss_weight: float = mark_loss_weight
        self._init_mark_components(history_size=hid_size_rep)

        # Parameters model
        self.hid_size_rep: int = hid_size_rep
        self.lr = learning_rate
        self.generator = RNNSamplingGeneratorTPP(
            self.time_emb,
            1,
            self.hid_size_rep,
            0.0,
            deterministic_model=False,
            event_emb=self.event_emb if self.use_marks else None,
            mark_predictor=self.mark_predictor if self.use_marks else None,
        )

        ## Clip the gradient.
        self.register_gradient_clipping()
        return

    def configure_optimizers(self) -> typing.Union[torch.optim.Optimizer, typing.Dict]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
        if not self.use_lr_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.total_epochs),
            eta_min=self.lr * self.COSINE_SCHEDULER_MIN_LR_RATIO,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "cosine_lr",
            },
        }

    def _setup_training_anchor_pipeline(self, data_train: torch.Tensor, data_train_lens: torch.Tensor):
        """Set up a separate anchor strategy + scaler for the training loss.

        The base class uses FREE_ENDPOINT for metrics/validation. We build a parallel
        training pipeline and override sigw1metric_train whenever the training path must
        diverge from the metrics path -- either because the training anchor mode differs
        (e.g. RESIDUAL) or because the training reference is length-capped (train_seq_cap).
        Reference and generated training data are then processed consistently, while the
        full-length validation/test pipeline is left intact.
        """
        self._anchor_strategy_train = make_anchor_strategy(self._terminal_anchor_mode_train, scaler_exp=self.scaler_exp)

        free_endpoint = self._terminal_anchor_mode_train == TerminalAnchorMode.FREE_ENDPOINT
        if free_endpoint and self.train_seq_cap is None:
            # Training uses the same anchor and full-length reference as metrics: reuse existing pipeline.
            self._scaler_std_train = self.scaler_std
            self._total_vars_train = self.total_vars
            return

        # A separate training pipeline is needed when the training anchor differs from the
        # metrics anchor (RESIDUAL) and/or the training reference is length-capped
        # (train_seq_cap). When capped, the reference is built on the first train_seq_cap
        # events only; the validation/test signature pipeline (self.scaler_std / total_vars /
        # sigw1metric_val) is left untouched by the base class and stays full-length.
        if self.train_seq_cap is not None:
            data_train, data_train_lens = self._truncate_to_cap(data_train, data_train_lens, self.train_seq_cap)

        # Recompute reference data for the training anchor (and/or cap). Lengths are derived
        # from the (possibly truncated) data_train_lens so they track the cap; in the uncapped
        # case this equals self.full_data_train_dt_lens - 1, preserving the original behaviour.
        data_train_scaled_dts, data_train_cum, _ = self._preprocess_dataset_for_metrics(data_train, data_train_lens)
        target_seqs = torch.cat([data_train_scaled_dts, data_train_cum], axis=2)
        ref_dt_lens = data_train_lens - 1  # inter-arrival count incl. tau_1, matches full_data_train_dt_lens
        target_seq_lens = ref_dt_lens - 1

        # Compute training-specific scaler (mirrors set_scaler_paths_for_sig logic).
        seqs = tpp_utils.insert_zero_beg(target_seqs)
        seqs = self._anchor_strategy_train.append(seqs, self.time_max, seq_lens=target_seq_lens)
        effective_lens = target_seq_lens + 1 + self._anchor_strategy_train.terminal_anchor_extra_len()
        mean_paths_scaled, std_paths_scaled = variable_len_standard_stats(seqs, effective_lens, True)
        mean_paths_scaled[1] = 0.0
        self._scaler_std_train = StandardScaler(means=mean_paths_scaled, stds=std_paths_scaled)
        scaled_targets = self._scaler_std_train(seqs)
        with torch.no_grad():
            self._total_vars_train = total_var(scaled_targets).mean().item()

        # Process training reference data through the training pipeline.
        train_sig_loss_seqs = self._scale_paths_pre_sig_train(target_seqs, seq_lens=target_seq_lens)

        # Override sigw1metric_train with one calibrated on training-anchor-processed data.
        effective_train_degree = None
        if self.sigw_loss_properties.use_degree_detector:
            detector = SigW1DegreeDetector(
                train_sig_loss_seqs,
                self.sigw_loss_properties.sig_degree,
            )
            effective_train_degree = self.sigw_loss_properties.resolve_detected_sig_degree(
                detector.effective_sig_degree
            )

        self.sigw1metric_train = SigW1MetricExp(
            train_sig_loss_seqs,
            sig_degree=self.sigw_loss_properties.sig_degree,
            scale_high_degrees=self.sigw_loss_properties.scale_high_degrees,
            standardise=self.sigw_loss_properties.standardise_sig,
            effective_sig_degree=effective_train_degree,
            use_float64_signature=self.sigw_loss_properties.use_float64_signature,
        )
        # ref_dt_lens matches the (possibly capped) reference width, so approx_err proxy
        # lengths line up with the truncated data instead of the full-length lengths.
        self.approx_err, self.approx_err_histoloss = self._compute_approx_errors(
            data_train_scaled_dts, data_train_cum, dt_lens=ref_dt_lens
        )
        return

    def _scale_paths_pre_sig_for_train_proxy(
        self,
        input_data_to_compute_loss: torch.Tensor,
        seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Approx proxy preprocessing must match the training-loss pipeline."""
        if not self._is_train_sig_pipeline_ready:
            # During parent __init__, training pipeline may not be configured yet.
            return self.scale_paths_pre_sig(input_data_to_compute_loss, seq_lens=seq_lens)
        return self._scale_paths_pre_sig_train(input_data_to_compute_loss, seq_lens=seq_lens)

    def _scale_paths_pre_sig_train(
        self,
        input_data: torch.Tensor,
        seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Scale paths using the training pipeline (training anchor + training scaler)."""
        if not self._is_train_sig_pipeline_ready:
            # Fallback used only before training-specific pipeline is configured.
            return self.scale_paths_pre_sig(input_data, seq_lens=seq_lens)
        input_data = tpp_utils.insert_zero_beg(input_data)
        input_data = self._anchor_strategy_train.append(input_data, self.time_max, seq_lens=seq_lens)
        input_data = self._scaler_std_train(input_data)
        input_data /= self._total_vars_train
        return input_data

    def get_teacher_forcing_ratio(self) -> float:
        """
        Calculate the current teacher forcing probability using a cosine decay schedule.

        Schedule: ratio = 0.5 * (1 + cos(π * t / T))
        - At t=0: ratio = 0.5 * (1 + 1) = 1.0 (always teacher forcing)
        - At t=T: ratio = 0.5 * (1 - 1) = 0.0 (never teacher forcing)
        - Smooth cosine curve between them

        After decay period completes, ratio stays at 0.0.

        Returns:
            float: Probability of using teacher forcing (0.0 to 1.0)
        """
        if not self.use_teacher_forcing:
            return 0.0

        # PyTorch Lightning attribute (0-indexed)
        if self.current_epoch >= self.curriculum_decay_epochs:
            return 0.0  # Curriculum complete, no more teacher forcing

        # Cosine decay: smooth transition from 1.0 to 0.0
        progress = self.current_epoch / self.curriculum_decay_epochs
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    def sample(
        self,
        *,
        num_seq: typing.Optional[int] = None,
        starting_times: typing.Optional[torch.Tensor] = None,
        log_inter_arr_times: typing.Optional[torch.Tensor] = None,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        """Sample marked or unmarked sequences.

        Branch semantics:
        - Conditional branch (`log_inter_arr_times` provided): teacher-forced sampling.
          The generator consumes real history and real previous marks from the batch.
          This path is for conditional prediction and returns `None` for generated marks.
        - Unconditional branch (`num_seq` provided): free generation.
          We first co-sample (`tau_1`, `mark_1`) from the same training sequence, store
          `mark_1` in `gen_marks[:, 0]`, then the shared generator uses that seeded mark
          as the previous-mark state for the first recurrent step. After that, marks are
          sampled autoregressively and written into `gen_marks[:, i]` in lockstep with
          the generated times.

        Returns:
            Tuple of `(samples, latent_rep_history, gen_marks)` with shapes
            `(N, L, 1)`, `(N, L-1, H_in)`, and optional `(N, L)`.
        """
        assert (num_seq is not None) ^ (log_inter_arr_times is not None), (
            f"Invalid input: "
            f"{'Both num_seq and history are None' if num_seq is None and log_inter_arr_times is None else 'Both num_seq and history are provided'}."
            f" Got num_seq={num_seq} and history={log_inter_arr_times}. Provide one or the other, but not both or neither."
        )

        assert (starting_times is None) or (log_inter_arr_times is not None), (
            f"Invalid input: starting_times was provided without log_inter_arr_times. "
            f"starting_times can only be used when log_inter_arr_times is provided. "
            f"Got starting_times={starting_times}, log_inter_arr_times={log_inter_arr_times}."
        )

        gen_marks = None

        # Conditional / teacher-forced branch: use real previous events and marks from the batch.
        if log_inter_arr_times is not None:
            assert isinstance(log_inter_arr_times, torch.Tensor), f"Expected a tensor, got {type(log_inter_arr_times)}."
            assert len(log_inter_arr_times.shape) == 3, f"Expected 3D tensor, got {log_inter_arr_times.shape}."
            samples, latent_rep_history = self.generator.generate_with_history(
                starting_times,
                log_inter_arr_times[:, :-1],
                self.MIN_SCALED_DATA,
                self.MAX_SCALED_DATA,
                self.scaler_exp,
                self.scaler_cumsum_value_for_generator,
                marks=marks[:, :-1] if marks is not None else None,
            )

        else:
            # Unconditional branch: create a full synthetic rollout.
            # We do max num elements samples. The first value corresponds to the beginning of the sequence:
            #   - either an actual event time,
            #   - or the starting time of recording.

            # Do 1 more, where first value is the starting time for the sequence and then we iterate over the whole sequence.
            # H_n does not need additional value because first value does not require it. Then, every value needs it.
            # Rollout length: full for val/test, capped to train_seq_cap during training.
            rollout_dts_len = self._effective_rollout_dts_len()
            L_full = rollout_dts_len + 1
            samples = torch.zeros((num_seq, rollout_dts_len + 1, self.num_dim_seqs), device=self.device)
            latent_rep_history = torch.zeros((num_seq, rollout_dts_len, self.hid_size_rep), device=self.device)

            # Seed the generated trajectory with a coherent first event.
            # The shared generator then uses this stored mark_1 as the previous-mark
            # state for the first recurrent step that predicts tau_2 / mark_2.
            first_it_seqs, first_mark = self._sample_first_event(num_seq)

            # Allocate mark history for unconditional generation.
            if self.use_marks:
                gen_marks = torch.full((num_seq, L_full), -1, dtype=torch.long, device=self.device)
                gen_marks[:, 0] = first_mark

            samples, latent_rep_history, gen_marks = self.generator.generate(
                self.anchor_times_sampler.sample(samples[:, 0, 0]),
                first_it_seqs.unsqueeze(1),  # (N, D) -> (N, 1, D) for initial_intertimes
                samples,
                latent_rep_history,
                self.MIN_SCALED_DATA,
                self.MAX_SCALED_DATA,
                self.scaler_exp,
                self.scaler_cumsum_value_for_generator,
                gen_marks=gen_marks,
            )

        return samples, latent_rep_history, gen_marks

    def _compute_mark_latent_history_for_eval(
        self,
        *,
        marks_with_anchor: torch.Tensor,
        marks_full: torch.Tensor,
        dts: torch.Tensor,
        dt_lens: torch.Tensor,
        current_targets: torch.Tensor,
    ) -> typing.Optional[torch.Tensor]:
        """Build next-mark-aligned latent history for shared mark evaluation."""
        with torch.no_grad():
            val_dts_scaled = self.scaler_exp(dts)
            return self.generator.encode_history(val_dts_scaled[:, :-1], marks_full[:, :-1])

    # section ######################################################################
    #  #############################################################################
    #  Training

    def training_step(self, batch, batch_nb: int):
        self.zero_grad()
        logger.debug("\n================== New Batch ==================")
        # Train-only sequence cap: truncate the batch to the first train_seq_cap events so the
        # teacher-forced path, the unconditional rollout, and the signature reference are all
        # capped consistently. Validation/test never reach this and stay full-length.
        if self.train_seq_cap is not None:
            batch = self._cap_training_batch(batch)
        # Randomly decide whether to use teacher forcing
        # Sample based on teacher forcing decision
        if torch.rand(1).item() < self.get_teacher_forcing_ratio():
            # Conditional sampling with true history (teacher forcing)
            cond_result = self.sample_for_a_fixed_batch_and_fix(batch, 1, "training")
            its_scaled_cst = cond_result.its_scaled_cst
            cum_abs_cst = cond_result.cum_abs_cst
            seq_lens = cond_result.seq_lens
        else:
            uncond_result = self.sample_and_fix_seqs(num_seq=batch[0].shape[0])
            its_scaled_cst = uncond_result.its_scaled_cst
            cum_abs_cst = uncond_result.cum_abs_cst
            seq_lens = uncond_result.seq_lens

        cum = cum_abs_cst.detach() if self.detach_cum_channel else cum_abs_cst
        gen_out_sigloss = self._scale_paths_pre_sig_train(
            torch.cat([its_scaled_cst, cum], axis=2),
            seq_lens=seq_lens,
        )
        logger.debug("Sequences (scaled) for metric %s", gen_out_sigloss)
        time_loss = self.sigw1metric_train(gen_out_sigloss)

        marks = extract_marks_without_anchor_from_batch(batch)
        ce_loss = None
        total_loss = time_loss
        metrics_to_log = {'sigW': time_loss}

        if self.use_marks:
            data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
            # Mark CE is teacher-forced on the real history; encode it directly instead of
            # routing back through the full sampling path.
            latent_rep_history = self.generator.encode_history(data_dts_scaled[:, :-1], marks[:, :-1])
            mark_logits = self.mark_predictor(latent_rep_history)
            mark_metrics = self._compute_and_log_mark_metrics_from_logits(
                mark_logits=mark_logits,
                marks=marks,
                lengths=data_dts_lens,
                prefix="train_",
                include_ce=True,
                include_accuracy=False,
            )
            if mark_metrics is not None:
                ce_loss = mark_metrics['mark_ce']
                total_loss = time_loss + (self.mark_loss_weight / self._ce_normalizer) * ce_loss

        self._log_all_metrics(metrics_to_log, "train_")
        return total_loss

    def validation_step(self, batch, batch_nb: int):
        logger.debug("\n================== New Batch ==================")
        # Always use unconditional sampling for validation to measure real-world performance
        uncond_result = self.sample_and_fix_seqs(num_seq=batch[0].shape[0])

        cum = uncond_result.cum_abs_cst.detach() if self.detach_cum_channel else uncond_result.cum_abs_cst
        gen_out_sigloss = self.scale_paths_pre_sig(
            torch.cat([uncond_result.its_scaled_cst, cum], axis=2),
            seq_lens=uncond_result.seq_lens,
        )
        logger.debug("Sequences (scaled) for metric %s", gen_out_sigloss)
        loss = self.sigw1metric_val(gen_out_sigloss)

        hist_it = self.metrics_val.histogram_loss_it(uncond_result.its_scaled_nan, [])
        hist_int = self.metrics_val.histogram_loss_cum(uncond_result.cum_rel_nan, [])

        metrics_to_log = {
            'sigW': loss,
            'epdf': (hist_it + hist_int) / 2.0,
            'hist_it': hist_it,
            'hist_int': hist_int,
        }

        marks = extract_marks_without_anchor_from_batch(batch)
        if self.use_marks:
            data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
            latent_rep_history = self.generator.encode_history(data_dts_scaled[:, :-1], marks[:, :-1])
            mark_logits = self.mark_predictor(latent_rep_history)
            self._compute_and_log_mark_metrics_from_logits(
                mark_logits=mark_logits,
                marks=marks,
                lengths=data_dts_lens,
                prefix="val_",
                include_ce=True,
            )

        self._log_all_metrics(metrics_to_log, "val_")
        return
