import itertools
import logging
import sys
import typing

import numpy as np
import torch
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

from src.nn.nn.score_modules.decoder_for_score_matching import Decoder4ScoreMatching

from src.utils import tpp_utils
from src.utils.utils_os import savefig

from src.differentialequations.diffusionprocess_continuous import ContinuousDiffusionProcess
from src.differentialequations.sdetype import SDEType

from src.nn.architectures.tpp_architecture import TPPArchitecture
from src.nn.architectures.mark_prediction_utils import (
    extract_marks_without_anchor_from_batch,
)

from src.utils.fix_seq_ends import (
    set_seq_to_zero_from_index,
)
from src.nn.rnn.recurrent_nn import Recurrent_nn, RNNType

from src.data_types.sigw_loss_data_props import SigWLossDataProps
import torch.nn.functional as F


class Architecture_DDPM(TPPArchitecture):
    """Score-based TPP model integrated with the repository's training and metric pipeline."""

    def __init__(
        self,
        data_train: torch.Tensor,
        data_train_lens: torch.Tensor,
        data_val: torch.Tensor,
        data_val_lens: torch.Tensor,
        train_marks: torch.Tensor,
        val_marks: torch.Tensor,
        ############################################
        lr: float,
        hidden_size_rnn: int,
        concentration_factor: float,
        num_diff_steps: int,
        ############################################
        t_max: int,
        num_marks: int,
        period_plot_val: int,
        output_dir: str = None,
        enable_plot: bool = False,
        plot_every_n_val_steps: int = 10,
        mark_loss_weight: float = 1.0,
        n_bootstraps: int = 1,
        **kwargs,
    ):
        """
        Data given is of shape (N, L+1, D). There are L inter-arrival times available.
        We use that naming convention throughout the file.
        """
        self.sigw_loss_properties: SigWLossDataProps = SigWLossDataProps(5, False, True)

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
        self.num_diff_steps = num_diff_steps
        self.diffusion_process = ContinuousDiffusionProcess(
            total_steps=self.num_diff_steps,
            sde_type=SDEType.VP,
        )
        # Remove this error for the plot: We do not use the sigW loss for training in this architecture.
        self.approx_err = torch.tensor(0.0)

        self.lr: float = lr
        self.hid_size_rep: int = hidden_size_rnn
        num_layers_rnn: int = 1
        dropout: float = 0.0

        # Learnable mark head (CE-head pattern, same as sigtpp/wgan).
        self.mark_loss_weight: float = mark_loss_weight
        rnn_input_size = self._init_mark_components(history_size=self.hid_size_rep)

        self.enc_rnn: Recurrent_nn = Recurrent_nn(
            rnn_input_size, self.hid_size_rep, num_layers_rnn, False, dropout, sys.maxsize, RNNType.LSTM, True
        )

        self.score_net = Decoder4ScoreMatching(
            self.hid_size_rep, self.hid_size_rep * 2, 3, self.hid_size_rep, dim_timeseries_embed=self.TIME_EMB_SIZE
        )

        ## Clip the gradient.
        self.register_gradient_clipping()
        self.L2_loss = torch.nn.MSELoss()

        self.seq_len = data_train.shape[1] - 1  # I.T.

        return

    def configure_optimizers(self):
        param_groups = [self.enc_rnn.parameters(), self.score_net.parameters()]
        if self.use_marks:
            param_groups.extend([self.event_emb.parameters(), self.mark_predictor.parameters()])
        else:
            param_groups.append(self.time_emb.parameters())
        optim_gen = torch.optim.Adam(itertools.chain(*param_groups), lr=self.lr, weight_decay=0.0)
        return optim_gen

    def sample(
        self,
        *,
        num_seq: typing.Optional[int] = None,
        starting_times: typing.Optional[torch.Tensor] = None,
        log_inter_arr_times: typing.Optional[torch.Tensor] = None,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        # Either sample the next value given the history or sample full sequences in a recursive manner.
        # The sequences returned are the inter-arrival times, scaled (log format) of shape (N, L, 1).
        # will need to be masked outside. Cannot be masked here because we do not have access to the scaler.
        # Inputs are expected to be scaled.

        # log_inter_arr_times is of shape (N, L, 1), where L is the length.
        # We predict L-1 values (excluding the prediction of the first one).
        # We still return the right shape (N,L,1) where the first value is sampled from targets.

        # if num_seq, we will just iteratively create new sequences. The returned sequences are of shape (N, L-1, 1).

        ###############################################
        # Returns (N, L, 1), (N, L-1, H_in), and Optional (N, L) tensors.
        ###############################################

        assert (num_seq is not None) ^ (log_inter_arr_times is not None), (
            f"Invalid input: "
            f"{'Both num_seq and history are None' if num_seq is None and log_inter_arr_times is None else 'Both num_seq and history are provided'}."
            f" Got num_seq={num_seq} and history={log_inter_arr_times}. Provide one or the other, but not both or neither."
        )

        assert (starting_times is None) == (log_inter_arr_times is None), (
            f"Invalid input: starting_times was provided without log_inter_arr_times. "
            f"starting_times can only be used when log_inter_arr_times is provided. "
            f"Got starting_times={starting_times}, log_inter_arr_times={log_inter_arr_times}."
        )

        gen_marks = None

        if log_inter_arr_times is not None:
            assert isinstance(log_inter_arr_times, torch.Tensor), f"Expected a tensor, got {type(log_inter_arr_times)}."
            assert len(log_inter_arr_times.shape) == 3, f"Expected 3D tensor, got {log_inter_arr_times.shape}."
            samples, latent_rep_history = self.all_step_scorenet_one_iter(
                starting_times, log_inter_arr_times, marks=marks
            )
        # If we do not have access to the history, we sample the first value and then iteratively predict the next ones.
        else:
            logger.debug("Sampling full sequence which is slow.")
            # Do 1 more, where first value is the starting time (no event) and then we iterate over the whole sequence.
            # We do as many elements as in the data.
            # It is sliced at the end to remove the starting time.
            samples = torch.zeros((num_seq, self.data_train_dts.shape[1] + 1, self.num_dim_seqs), device=self.device)
            latent_rep_history = torch.zeros(
                (num_seq, self.data_train_dts.shape[1], self.enc_rnn.hidden_size), device=self.device
            )
            starting_time_sequences = self.anchor_times_sampler.sample(latent_rep_history[:, 0, 0])

            # Co-sample first IT and first mark from the same training sequence.
            first_it_seqs, first_mark = self._sample_first_event(num_seq)
            samples[:, 0] = first_it_seqs

            # Allocate mark history for unconditional generation.
            if self.num_marks > 1:
                gen_marks = torch.full(
                    (num_seq, (self.full_data_train_dts.shape[1])), -1, dtype=torch.long, device=self.device
                )
                gen_marks[:, 0] = first_mark

            samples, latent_rep_history, gen_marks = self._all_step_scorenet_all_iter_with_marks(
                samples, starting_time_sequences, latent_rep_history, gen_marks
            )
        return samples, latent_rep_history, gen_marks

    # section ######################################################################
    #  #############################################################################
    #  Training

    def _embed_history(
        self, log_inter_arr_times: torch.Tensor, marks: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_marks:
            if marks is None:
                marks = torch.zeros(
                    log_inter_arr_times.shape[0],
                    log_inter_arr_times.shape[1],
                    dtype=torch.long,
                    device=log_inter_arr_times.device,
                )
            return self.event_emb(log_inter_arr_times, marks)
        return self.time_emb(log_inter_arr_times)

    def _encode_history(
        self, log_inter_arr_times: torch.Tensor, marks: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        inter_arr_times_emb = self._embed_history(log_inter_arr_times, marks)
        hn, cn = self.enc_rnn.get_first_hidden_state(log_inter_arr_times.shape[0])
        latent_rep_history, _ = self.enc_rnn(inter_arr_times_emb, (hn, cn))
        return latent_rep_history

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
            latent_rep_history = self._encode_history(val_dts_scaled, marks_full)
            return latent_rep_history[:, :-1]

    def training_step(self, batch, batch_nb: int):
        data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
        logger.debug("\n================== New Batch ==================")
        if self.use_marks:
            marks = extract_marks_without_anchor_from_batch(batch)
            time_loss = self._compute_score_matching_loss(
                log_inter_arr_times=data_dts_scaled,
                lengths=data_dts_lens,
                marks=marks,
            )
        else:
            time_loss = self._compute_score_matching_loss(
                log_inter_arr_times=data_dts_scaled,
                lengths=data_dts_lens,
            )

        # Mark CE loss
        ce_loss = None
        total_loss = time_loss
        if self.use_marks:
            latent_rep_history = self._encode_history(data_dts_scaled, marks)
            mark_logits = self.mark_predictor(latent_rep_history[:, :-1])
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

        metrics_to_log = {'score': time_loss}

        self._log_all_metrics(metrics_to_log, "train_")
        return total_loss

    def validation_step(self, batch, batch_nb: int):
        data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
        if self.use_marks:
            marks = extract_marks_without_anchor_from_batch(batch)
            time_loss = self._compute_score_matching_loss(
                log_inter_arr_times=data_dts_scaled,
                lengths=data_dts_lens,
                marks=marks,
            )
        else:
            time_loss = self._compute_score_matching_loss(
                log_inter_arr_times=data_dts_scaled,
                lengths=data_dts_lens,
            )
        ce_loss = None
        total_loss = time_loss
        if self.use_marks:
            latent_rep_history = self._encode_history(data_dts_scaled, marks)
            mark_logits = self.mark_predictor(latent_rep_history[:, :-1])
            mark_metrics = self._compute_and_log_mark_metrics_from_logits(
                mark_logits=mark_logits,
                marks=marks,
                lengths=data_dts_lens,
                prefix="val_",
                include_ce=True,
            )
            if mark_metrics is not None:
                ce_loss = mark_metrics['mark_ce']
                total_loss = time_loss + (self.mark_loss_weight / self._ce_normalizer) * ce_loss
        logger.debug("\n================== New Batch ==================")
        metrics2log = {'score': time_loss}
        self._log_all_metrics(metrics2log, "val_")
        return total_loss

    def _compute_validation_epoch_end_metrics(self) -> typing.Dict[str, torch.Tensor]:
        metrics2log = super()._compute_validation_epoch_end_metrics()
        metrics2log['epdf'] = torch.tensor(float('nan'), device=self.device)
        if self.enable_plot and not (self.current_epoch + 1) % self.period_validation_eval_plots:
            uncond_result = self.sample_and_fix_seqs(
                # Get the length of one batch, which potentially is larger than the total size in which case we fallback to that.
                num_seq=min(self.trainer.val_dataloaders[0].batch_size, self.trainer.val_dataloaders[0].dataset_len)
            )
            hist_metrics = self.metrics_val.compute_histogram_metrics(
                uncond_result.its_scaled_nan, uncond_result.cum_rel_nan
            )
            metrics2log.update(
                {
                    'epdf': (hist_metrics['hist_it'] + hist_metrics['hist_int']) / 2.0,
                    'hist_it': hist_metrics['hist_it'],
                    'hist_int': hist_metrics['hist_int'],
                }
            )
        return metrics2log

    def _run_validation_epoch_end_plots(self) -> None:
        # Plot every (period_validation_eval_plots x plot_every_n_val_steps) epochs.
        if self.enable_plot and not (self.current_epoch + 1) % self.period_validation_eval_plots:
            effective_plot_period = self.period_validation_eval_plots * self.plot_every_n_val_steps
            if (self.current_epoch + 1) % effective_plot_period == 0:
                self.sample_and_plot(name_plot4save=str(self.current_epoch + 1))
        return

    # section ######################################################################
    #  #############################################################################
    #  Sampling/Plotting

    def _pre_diagnostic_plots_hook(self) -> None:
        """Plot forward and backward diffusion trajectories before main diagnostics."""
        frw, bckw, _ = self.get_forward_and_backward(self.scaler_exp(self.data_train_dts.to(self.device)))
        bckw = bckw.flip(0)
        self.plot_for_back_ward_trajectories(
            bckw.flatten(2, 3).transpose(0, 1),
            frw.flatten(2, 3).transpose(0, 1),
        )

    def _compute_score_matching_loss(
        self,
        log_inter_arr_times: torch.Tensor,
        lengths: torch.Tensor,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # We learn how to denoise the log inter-arrival times (scaled i.t.s).
        time_step_diffusion = torch.randint(1, self.num_diff_steps + 1, (1,), device=self.device)
        _, diffusion = self.diffusion_process._compute_drift_and_diffusion(
            torch.zeros_like(log_inter_arr_times), time_step_diffusion
        )
        # # We diffuse the embedding of the log inter arr times.
        # inter_arr_times_emb: torch.Tensor = self.time_emb(log_inter_arr_times)
        # Not doing that as we would decode embeddings and not times.

        mean, std = self.diffusion_process._perturbation_kernel(log_inter_arr_times, time_step_diffusion)
        noise = torch.randn_like(log_inter_arr_times)
        emb_perturbed_noise = mean + std * noise

        pred_score = self.one_step_scorenet_one_iter(
            log_inter_arr_times,
            emb_perturbed_noise,
            time_step_diffusion,
            marks=marks,
        )

        # NCSN score matching objective function (x_tilda - x) / sigma^2
        #### WE TRUNCATED AND BELOW REMOVE ONE OF LENGTH
        targets = (-noise / std)[:, 1:]

        # Mask the pred score such that only the ones within the length are considered.
        masked_scores = set_seq_to_zero_from_index(pred_score, lengths - 2)
        masked_targets = set_seq_to_zero_from_index(targets, lengths - 2)

        loss_gen_score_matching = diffusion * diffusion * self.L2_loss(masked_scores, masked_targets)
        return loss_gen_score_matching

    def one_step_scorenet_one_iter(
        self,
        log_inter_arr_times,
        perturbed_values_into_noise,
        time_step_diffusion,
        marks: typing.Optional[torch.Tensor] = None,
    ):
        # Pass the log-inter-times (N, L, D) and their perturbed versions.
        latent_rep_history = self._encode_history(log_inter_arr_times, marks)
        starting_times = self.anchor_times_sampler.sample(log_inter_arr_times[:, 0, 0])
        #######################################
        # Here, to be comparable / because we will never sample the first value, we truncate the first value out.
        #######################################
        running_cum_time = starting_times + self.scaler_exp.unscale(log_inter_arr_times).cumsum(1)
        scaled_running_cum_time = self.scaler_cumsum_value_for_generator(running_cum_time)
        pred_score = self.score_net(
            None,  # Not passing previous values bc we don't predict the next value
            latent_rep_history[:, :-1],
            scaled_running_cum_time[:, :-1].detach(),
            perturbed_values_into_noise[:, 1:],
            time_step_diffusion,
        )
        return pred_score

    def all_step_scorenet_one_iter(
        self,
        starting_time_sequences,
        log_inter_arr_times,
        marks: typing.Optional[torch.Tensor] = None,
    ):
        noise_start = torch.randn_like(log_inter_arr_times)  # We will use for all data points one starting noise.
        latent_rep_history = self._encode_history(log_inter_arr_times, marks)
        running_cum_time = starting_time_sequences + self.scaler_exp.unscale(log_inter_arr_times).cumsum(1)
        scaled_running_cum_time = self.scaler_cumsum_value_for_generator(running_cum_time)
        traj_back = self.diffusion_process.backward_sample(
            noise_start,
            latent_rep_history,
            scaled_running_cum_time.detach(),
            self.score_net,
        )[-1].clamp(min=self.MIN_SCALED_DATA, max=self.MAX_SCALED_DATA)
        return traj_back, latent_rep_history

    def all_step_scorenet_all_iter(self, samples, starting_time_sequences, latent_rep_history):
        """
        samples is the container for the sampled inter-arrival times. First value is the starting i.t..
        latent_rep_history is the container for the latent representations of the inter-arrival times.
        Inherently slow.
        """
        samples, latent_rep_history, _ = self._all_step_scorenet_all_iter_with_marks(
            samples, starting_time_sequences, latent_rep_history, gen_marks=None
        )
        return samples, latent_rep_history

    def _all_step_scorenet_all_iter_with_marks(self, samples, starting_time_sequences, latent_rep_history, gen_marks):
        """Like all_step_scorenet_all_iter but also collects generated marks into gen_marks."""

        noise_start = torch.randn_like(samples)

        (hn, cn) = self.enc_rnn.get_first_hidden_state(samples.shape[0])
        running_cum_time = starting_time_sequences

        # Track running marks for autoregressive mark-conditioned encoding.
        if self.num_marks > 1:
            running_marks = (
                gen_marks[:, 0]
                if gen_marks is not None
                else torch.zeros(samples.shape[0], dtype=torch.long, device=self.device)
            )
        else:
            running_marks = None

        for i in range(1, self.seq_len):
            running_cum_time += self.scaler_exp.unscale(samples[:, i - 1 : i])
            scaled_running_cum_time = self.scaler_cumsum_value_for_generator(running_cum_time).clamp(max=5.0)

            if running_marks is not None:
                # Use event embedding (time + mark) instead of just time embedding.
                inter_arr_times_emb = self.event_emb(samples[:, i - 1 : i], running_marks.unsqueeze(1))
            else:
                inter_arr_times_emb = self.time_emb(samples[:, i - 1 : i])
            latent_rep_history[:, i - 1 : i], (hn, cn) = self.enc_rnn(inter_arr_times_emb, (hn, cn))

            samples[:, i : i + 1] = self.diffusion_process.backward_sample(
                noise_start[:, i : i + 1, :],
                latent_rep_history[:, i - 1 : i],
                scaled_running_cum_time.detach(),
                self.score_net,
            )[-1].clamp(min=self.MIN_SCALED_DATA, max=self.MAX_SCALED_DATA)

            # Sample next mark autoregressively from the mark predictor.
            if running_marks is not None:
                mark_logits = self.mark_predictor(latent_rep_history[:, i - 1 : i])
                mark_probs = F.softmax(mark_logits.squeeze(1), dim=-1)
                running_marks = torch.multinomial(mark_probs, 1).squeeze(-1)
                if gen_marks is not None:
                    gen_marks[:, i] = running_marks

        return samples, latent_rep_history, gen_marks

    def get_forward_and_backward(self, data_to_noise_and_denoise: torch.Tensor):
        num_seq = data_to_noise_and_denoise.shape[0]

        forw_paths = self._get_forward_path(data_to_noise_and_denoise)  # S, N, L, D

        # # Assuming forw_paths is your tensor of shape [101, 2000, 26, 1]
        # data = forw_paths.squeeze(-1)  # Shape: [101, 2000, 26]
        #
        # feature_means = data.mean(dim=(0, 1))
        # feature_stds = data.std(dim=(0, 1))
        #
        # # Print them out
        # for i, (mean, std) in enumerate(zip(feature_means, feature_stds)):
        #     print(f"Feature {i:2d} - Mean: {mean.item():.4f}, Std: {std.item():.4f}")
        #
        # # Move features to first axis for easier access: [26, 101, 2000]
        # data = data.permute(2, 0, 1)  # [26, 101, 2000]
        #
        # num_groups = 4
        # group_size = 3
        # sample_indices = [i for i in range(0, data.shape[2], 10)]
        #
        # for PLOT_SEP_NUM in range(2):
        #     fig, axes = plt.subplots(num_groups, group_size, figsize=(18, 12))
        #     fig.tight_layout(pad=3.0)
        #
        #     for i in range(num_groups):
        #         for j in range(group_size):
        #             dim_idx = i * group_size + j + 12 * PLOT_SEP_NUM
        #             if dim_idx >= data.shape[0]:
        #                 axes[i, j].axis('off')
        #                 continue
        #
        #             ax = axes[i, j]
        #             dim_data = data[dim_idx]  # shape: [101, 2000]
        #
        #             # Mean trajectory
        #             mean_traj = dim_data.mean(dim=1).cpu().numpy()
        #             ax.plot(mean_traj, label='Mean', color='black', linewidth=2)
        #
        #             # Plot a few sample trajectories
        #             for k in sample_indices:
        #                 sample_traj = dim_data[:, k].cpu().numpy()
        #                 ax.plot(sample_traj, linestyle='--', alpha=0.6)
        #
        #             ax.set_title(f'Dimension {dim_idx}')
        #             ax.grid(True)
        #             ax.set_xlabel("Trajectory Step")
        #             ax.set_ylabel("Value")
        #
        #     plt.suptitle('Trajectories Across 24 Dimensions (Mean + Samples)', fontsize=16)
        #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()

        samples = torch.zeros_like(forw_paths)[-1]
        full_back_traj = torch.zeros_like(forw_paths)
        latent_rep_history = torch.zeros(
            (num_seq, self.data_train_dts.shape[1], self.enc_rnn.hidden_size), device=self.device
        )
        starting_time_sequences = self.anchor_times_sampler.sample(latent_rep_history[:, 0, 0])

        samples[:, 0] = data_to_noise_and_denoise[:, 0]

        noise_start = forw_paths[-1]
        (hn, cn) = self.enc_rnn.get_first_hidden_state(samples.shape[0])
        running_cum_time = starting_time_sequences

        # Track running marks for mark-conditioned encoding (mirrors all_step_scorenet_all_iter).
        if self.num_marks > 1:
            running_marks = torch.zeros(samples.shape[0], dtype=torch.long, device=self.device)

        ## We use the samples (along L) as they appear in the real data to ensure that the way we denoise
        # is comparable to the way we added noise.
        for i in range(1, self.seq_len):
            running_cum_time += self.scaler_exp.unscale(data_to_noise_and_denoise[:, i - 1 : i])
            scaled_running_cum_time = self.scaler_cumsum_value_for_generator(running_cum_time)
            if self.num_marks > 1:
                inter_arr_times_emb = self.event_emb(
                    data_to_noise_and_denoise[:, i - 1 : i], running_marks.unsqueeze(1)
                )
            else:
                inter_arr_times_emb = self.time_emb(data_to_noise_and_denoise[:, i - 1 : i])
            latent_rep_history[:, i - 1 : i], (hn, cn) = self.enc_rnn(inter_arr_times_emb, (hn, cn))

            traj = self.diffusion_process.backward_sample(
                noise_start[:, i - 1 : i, :],
                latent_rep_history[:, i - 1 : i],
                scaled_running_cum_time.detach(),
                self.score_net,
            )
            full_back_traj[:, :, i : i + 1] = traj
            samples[:, i : i + 1] = traj[-1]

            # Update running marks autoregressively for the next step.
            if self.num_marks > 1:
                mark_logits = self.mark_predictor(latent_rep_history[:, i - 1 : i])
                mark_probs = F.softmax(mark_logits.squeeze(1), dim=-1)
                running_marks = torch.multinomial(mark_probs, 1).squeeze(-1)

        # Backward, first I.T. is sampled from the starting time and so not sampled in this manner.
        return forw_paths[:, :, 1:], full_back_traj[:, :, 1:], samples

    # section ######################################################################
    #  #############################################################################
    #  Plots

    # On top of the other plots, we should have the trajectories for some samples.
    def _set_eval_plots(self):
        super()._set_eval_plots()
        self.plot_diffusion_fig, self.plot_diffusion_axes = plt.subplots(5, 2, figsize=(10, 7))
        for row_axes in self.plot_diffusion_axes:
            row_axes[0].get_shared_y_axes().join(*row_axes)

    def _clear_all_axes(self):
        super()._clear_all_axes()
        for ax_row in self.plot_diffusion_axes:
            for ax in ax_row:
                ax.clear()

    def _save_eval_plots(self, post_str: str, include_mark_plots: bool = False):
        super()._save_eval_plots(post_str, include_mark_plots=include_mark_plots)
        savefig(self.plot_diffusion_fig, f'{self.output_dir}diff_{post_str}.png')

    def on_train_end(self):
        """Close the score-specific diffusion figure in addition to the base diagnostics."""
        super().on_train_end()
        fig = getattr(self, 'plot_diffusion_fig', None)
        if fig is not None:
            plt.close(fig)

    def _get_forward_path(
        self,
        starting_data: torch.Tensor,
    ) -> torch.Tensor:
        # To get the totally noised data, use: output[-1, :, :, :]
        assert (
            len(starting_data.shape) == 3
        ), f"Incorrect shape for starting_data: Expected 3 dimensions (N, L, D) but got {len(starting_data.shape)} dimensions with shape {starting_data.shape}. Make sure the tensor is correctly reshaped or initialized."

        diffused_starting_data = self.diffusion_process.forward_sample(starting_data)

        # Shape (S, N, L, D). This shape makes sense because we are interested in the tensor N,L,D by slices over S-dim.
        return diffused_starting_data

    def plot_for_back_ward_trajectories(self, denoised_diffused_targets, diffused_targets):
        # Expect shape N, S, L * D
        assert (
            len(denoised_diffused_targets.shape) == 3
        ), f"Expected 3 dimensions but got {len(denoised_diffused_targets.shape)} dimensions. Possible there is an extra feature dimension?"

        assert denoised_diffused_targets.shape == diffused_targets.shape, (
            f"Expected denoised_diffused_targets and diffused_targets to have the same shape, "
            f"but got {denoised_diffused_targets.shape} and {diffused_targets.shape}."
        )

        denoised_diffused_targets = denoised_diffused_targets.detach().cpu().numpy()
        diffused_targets = diffused_targets.detach().cpu().numpy()

        diffusion_forw_steps = np.arange(diffused_targets.shape[1])
        diffusion_back_steps = np.arange(denoised_diffused_targets.shape[1])

        threshold = -10
        for ax_plot_idx, row_idx in enumerate(
            # Start at 0 and plot every nth. The reason to multiply by n is because we have a step, and we are looking
            # for the minimum between the number of sequences and the maximal index of the plotted sequences.
            range(0, min(diffused_targets.shape[2], 5 * len(self.plot_diffusion_axes)), 5)
        ):
            # Cap the maximum number of trajectories at 100
            max_trajs_for_this_plot = 100

            # Build a mask of which trajectories have first-value ≥ threshold
            valid_seqs = diffused_targets[:, 0, row_idx] >= threshold
            valid_indices = np.where(valid_seqs)[0]

            # Sample up to max_trajs_for_this_plot from valid sequences
            if len(valid_indices) > max_trajs_for_this_plot:
                sampled_indices = np.random.choice(valid_indices, max_trajs_for_this_plot, replace=False)
            else:
                sampled_indices = valid_indices

            # --- Forward Path ---
            ax_fwd = self.plot_diffusion_axes[ax_plot_idx, 0]
            for element_dataset in sampled_indices:
                ax_fwd.plot(
                    diffusion_forw_steps,
                    diffused_targets[element_dataset, :, row_idx],
                    linewidth=1.0,
                )
            ax_fwd.set_title(f"Forward Diffusion: Time Index {row_idx} (n={len(sampled_indices)})")
            if ax_plot_idx == len(self.plot_diffusion_axes) - 1:
                ax_fwd.set_xlabel("Diffusion Steps")

            # --- Backward Path ---
            ax_bwd = self.plot_diffusion_axes[ax_plot_idx, 1]
            for element_dataset in sampled_indices:
                ax_bwd.plot(
                    diffusion_back_steps,
                    denoised_diffused_targets[element_dataset, :, row_idx],
                    linewidth=1.0,
                )
            ax_bwd.invert_xaxis()  # reverse the x-axis
            ax_bwd.set_title(f"Backward Diffusion: Time Index {row_idx} (n={len(sampled_indices)})")
            if ax_plot_idx == len(self.plot_diffusion_axes) - 1:
                ax_bwd.set_xlabel("Diffusion Steps (reversed)")

        # Set figure title and layout
        self.plot_diffusion_fig.suptitle(
            f"Diffusion Process Trajectories: Forward vs Backward Denoising (n={diffused_targets.shape[0]} sequences)",
        )
        self.plot_diffusion_fig.tight_layout()

        # savefig(
        #     self.plot_diffusion_fig,
        #     self.output_dir_images + f"trajectories_{str(self.current_epoch + 1)}.png",
        # )
        return
