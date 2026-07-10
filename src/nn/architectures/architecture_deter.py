import logging
import typing

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
from src.nn.architectures.tpp_architecture import TPPArchitecture
from src.nn.architectures.mark_prediction_utils import (
    compute_majority_class,
    build_majority_class_logits,
    prepare_next_mark_prediction_tensors,
)

from src.utils import tpp_utils

from src.data_types.sigw_loss_data_props import SigWLossDataProps
from src.utils.fix_seq_ends import (
    set_seq_to_zero_from_index,
)

from src.nn.nn.sigwgan_modules.rnn_sampling_generator_tpp import RNNSamplingGeneratorTPP


class ArchitectureDeter(TPPArchitecture):
    """Deterministic next-inter-arrival baseline built on the shared TPP architecture."""

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
        ############################################
        t_max: int,
        num_marks: int,
        output_dir: str = None,
        enable_plot: bool = False,
        plot_every_n_val_steps: int = 1,
        n_bootstraps: int = 1,
        **kwargs,
    ):
        """
        Data given is of shape (N, L+1, D). There are L inter-arrival times available.
        We use that naming convention throughout the file.
        """
        self.sigw_loss_properties: SigWLossDataProps = loss_properties

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
            output_dir=output_dir,
            enable_plot=enable_plot,
            period_plot_val=period_plot_val,
            plot_every_n_val_steps=plot_every_n_val_steps,
            n_bootstraps=n_bootstraps,
        )

        # Parameters model
        self.hid_size_rep: int = hid_size_rep
        self.lr = learning_rate
        self.generator = RNNSamplingGeneratorTPP(self.time_emb, 1, self.hid_size_rep, 0.0, deterministic_model=True)

        ## Clip the gradient.
        self.register_gradient_clipping()

        self.mse_loss_fn = nn.MSELoss()

        # Mark classifier: majority-class baseline.
        if self.use_marks:
            self._majority_class = compute_majority_class(train_marks, data_train_lens, num_marks)
            logger.info("Deter mark classifier: majority class = %d", self._majority_class)
        else:
            self._majority_class = None
        return

    def sample(
        self,
        *,
        num_seq: typing.Optional[int] = None,
        starting_times: typing.Optional[torch.Tensor] = None,
        log_inter_arr_times: typing.Optional[torch.Tensor] = None,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        # Expect either a number of sequence to sample, or the inter arrival times.
        # If log_inter_arr_times provided, it should be of shape (N, L, D).
        # Returns (N, L, 1), (N, L-1, H_in), and None (deterministic baseline has no mark generation).
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

        # Case where the history is fixed.
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
            )

        else:
            # We do max num elements samples. The first value corresponds to the beginning of the sequence:
            #   - either an actual event time,
            #   - or the starting time of recording.

            # Do 1 more, where first value is the starting time for the sequence and then we iterate over the whole sequence.
            # H_n does not need additional value because first value does not require it. Then, every value needs it.
            samples = torch.zeros((num_seq, self.data_train_dts.shape[1] + 1, self.num_dim_seqs), device=self.device)
            latent_rep_history = torch.zeros(
                (num_seq, self.data_train_dts.shape[1], self.hid_size_rep), device=self.device
            )

            samples, latent_rep_history, _ = self.generator.generate(
                self.anchor_times_sampler.sample(samples[:, 0, 0]),
                self.first_value_ts_sampler.sample(samples[:, 0, 0]),
                samples,
                latent_rep_history,
                self.MIN_SCALED_DATA,
                self.MAX_SCALED_DATA,
                self.scaler_exp,
                self.scaler_cumsum_value_for_generator,
            )
        return samples, latent_rep_history, None

    # section ######################################################################
    #  #############################################################################
    #  Training

    def training_step(self, batch, batch_nb: int):
        self.zero_grad()
        logger.debug("\n================== New Batch ==================")
        data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
        gen_out, _, _ = self.sample(starting_times=batch[0][:, :1], log_inter_arr_times=data_dts_scaled)

        gen_out = set_seq_to_zero_from_index(gen_out, data_dts_lens - 1)
        data_dts_scaled = set_seq_to_zero_from_index(data_dts_scaled, data_dts_lens - 1)
        loss = self.mse_loss_fn(gen_out, data_dts_scaled)
        self._log_all_metrics(
            {
                'MSE': loss,
            },
            "train_",
        )
        return loss

    def validation_step(self, batch, batch_nb: int):
        logger.debug("\n================== New Batch ==================")
        # Always use unconditional sampling for validation to measure real-world performance
        uncond_result = self.sample_and_fix_seqs(num_seq=batch[0].shape[0])

        gen_out_sigloss = self.scale_paths_pre_sig(
            torch.cat([uncond_result.its_scaled_cst, uncond_result.cum_abs_cst], axis=2),
            seq_lens=uncond_result.seq_lens,
        )
        logger.debug("Sequences (scaled) for metric %s", gen_out_sigloss)
        loss_sigW = self.sigw1metric_val(gen_out_sigloss)

        hist_it = self.metrics_val.histogram_loss_it(uncond_result.its_scaled_nan, [])
        hist_int = self.metrics_val.histogram_loss_cum(uncond_result.cum_rel_nan, [])

        # MSE loss uses teacher forcing to measure conditional prediction accuracy
        data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
        gen_out, _, _ = self.sample(starting_times=batch[0][:, :1], log_inter_arr_times=data_dts_scaled)
        gen_out = set_seq_to_zero_from_index(gen_out, data_dts_lens - 1)
        data_dts_scaled = set_seq_to_zero_from_index(data_dts_scaled, data_dts_lens - 1)
        loss = self.mse_loss_fn(gen_out, data_dts_scaled)
        self._log_all_metrics(
            {
                'MSE': loss,
                'sigW': loss_sigW,
                'epdf': (hist_it + hist_int) / 2.0,
                'hist_it': hist_it,
                'hist_int': hist_int,
            },
            "val_",
        )
        if self.use_marks:
            marks_with_anchor = batch[2]
            _, current_targets, _ = prepare_next_mark_prediction_tensors(marks_with_anchor, data_dts_lens)
            mark_logits = build_majority_class_logits(current_targets, self.num_marks, self._majority_class)
            self._compute_and_log_mark_metrics_from_logits(
                mark_logits=mark_logits,
                marks=marks_with_anchor[:, 1:],  # (N, L) anchor-stripped, method contract; targets sliced internally
                lengths=data_dts_lens,
                prefix="val_",
                include_ce=False,
            )
        return

    @property
    def _include_top3_mark_accuracy(self) -> bool:
        return False

    @property
    def _use_mark_eval_no_history(self) -> bool:
        return True

    def _compute_mark_logits_no_history(
        self,
        *,
        marks_with_anchor: torch.Tensor,
        marks_full: torch.Tensor,
        dts: torch.Tensor,
        dt_lens: torch.Tensor,
        current_targets: torch.Tensor,
    ) -> typing.Optional[torch.Tensor]:
        """Return direct majority-class logits for shared mark evaluation."""
        if not self.use_marks:
            return None
        return build_majority_class_logits(current_targets, self.num_marks, self._majority_class)
