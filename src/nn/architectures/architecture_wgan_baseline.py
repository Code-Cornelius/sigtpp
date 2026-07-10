import itertools
import logging
import sys
import typing

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
from src.nn.architectures.tpp_architecture import TPPArchitecture
from src.nn.architectures.mark_prediction_utils import (
    extract_marks_without_anchor_from_batch,
    compute_mark_ce_from_logits,
)
from src.utils import tpp_utils
from src.utils.fix_seq_ends import (
    set_seq_to_nan_from_index,
)


from src.nn.rnn.recurrent_nn import Recurrent_nn, RNNType

from src.data_types.sigw_loss_data_props import SigWLossDataProps

from src.nn.nn.gan_modules.gan_discriminator_baseline import GANDiscriminatorBaseline
from src.nn.nn.sigwgan_modules.decoder_for_gan_full_input import Decoder4GANFullInput


class Architecture_wgan_baseline(TPPArchitecture):
    """WGAN baseline adapted to the shared TPP training and evaluation pipeline."""

    def __init__(
        self,
        data_train: torch.Tensor,
        data_train_lens: torch.Tensor,
        data_val: torch.Tensor,
        data_val_lens: torch.Tensor,
        train_marks: torch.Tensor,
        val_marks: torch.Tensor,
        ############################################
        lr_gen: float,
        lr_disc: float,
        hidden_size_rnn: int,
        concentration_factor: float,
        lipschitz_reg: float,
        ############################################
        t_max: int,
        num_marks: int,
        period_plot_val: int,
        output_dir: str = None,
        enable_plot: bool = False,
        plot_every_n_val_steps: int = 1,
        ############################################
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
        # Remove this error for the plot: We do not use the sigW loss for training in this architecture.
        self.approx_err = torch.tensor(0.0)

        self.lr_gen: float = lr_gen
        self.lr_disc: float = lr_disc
        hidden_size_rnn: int = hidden_size_rnn
        num_layers_rnn: int = 1
        dropout: float = 0.0

        self.mark_loss_weight: float = mark_loss_weight
        rnn_input_size = self._init_mark_components(history_size=hidden_size_rnn)

        # We perform backprop manually.
        self.automatic_optimization = False

        self.enc_rnn: Recurrent_nn = Recurrent_nn(
            rnn_input_size, hidden_size_rnn, num_layers_rnn, False, dropout, sys.maxsize, RNNType.LSTM, True
        )

        self.decoder: Decoder4GANFullInput = Decoder4GANFullInput(
            hidden_size_rnn, hidden_size_rnn * 2, 3, hidden_size_rnn
        )
        self.nn_num_layers: int = 3
        self.discriminator: GANDiscriminatorBaseline = GANDiscriminatorBaseline(
            hidden_size_rnn, self.nn_num_layers, hidden_size_rnn
        )
        self.lipschitz_reg = lipschitz_reg

        ## Clip the gradient.
        self.register_gradient_clipping()
        return

    def configure_optimizers(self):
        gen_params = itertools.chain(self.enc_rnn.parameters(), self.decoder.parameters())
        if self.use_marks:
            gen_params = itertools.chain(gen_params, self.event_emb.parameters(), self.mark_predictor.parameters())
        optim_gen = torch.optim.Adam(gen_params, lr=self.lr_gen, weight_decay=0.0)
        optim_discr = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc, weight_decay=0.0)
        return [optim_gen, optim_discr], []

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

        # Case where the history is fixed.
        if log_inter_arr_times is not None:
            assert isinstance(log_inter_arr_times, torch.Tensor), f"Expected a tensor, got {type(log_inter_arr_times)}."
            assert len(log_inter_arr_times.shape) == 3, f"Expected 3D tensor, got {log_inter_arr_times.shape}."
            running_cum_time = starting_times + self.scaler_exp.unscale(log_inter_arr_times[:, :-1]).cumsum(1)
            scaled_running_cum_time = self.scaler_cumsum_value_for_generator(running_cum_time)
            if self.use_marks and marks is not None:
                inter_arr_times_emb: torch.Tensor = self.event_emb(log_inter_arr_times[:, :-1], marks[:, :-1])
            else:
                inter_arr_times_emb: torch.Tensor = self.time_emb(log_inter_arr_times[:, :-1])
            latent_rep_history, _ = self.enc_rnn(inter_arr_times_emb)
            samples = self.decoder(
                inter_arr_times_emb,
                latent_rep_history,
                scaled_running_cum_time.detach(),
            ).clamp(min=self.MIN_SCALED_DATA, max=self.MAX_SCALED_DATA)
            ### Adding the first value to fit in the framework/interface.
            samples = torch.cat((log_inter_arr_times[:, 0:1], samples), dim=1)

        # If we do not have access to the history, we sample the first value and then iteratively predict the next ones.
        else:
            L_full = self.full_data_train_dts.shape[1]
            samples = torch.zeros((num_seq, self.data_train_dts.shape[1] + 1, self.num_dim_seqs), device=self.device)
            latent_rep_history = torch.zeros(
                (num_seq, self.data_train_dts.shape[1], self.enc_rnn.hidden_size), device=self.device
            )
            starting_times = self.anchor_times_sampler.sample(latent_rep_history[:, 0, 0])

            # Co-sample first IT and first mark from the same training sequence.
            first_it_seqs, first_mark = self._sample_first_event(num_seq)
            samples[:, 0] = first_it_seqs

            (hn, cn) = self.enc_rnn.get_first_hidden_state(samples.shape[0])

            running_cum_time = starting_times

            # Allocate mark history for unconditional generation.
            if self.use_marks:
                gen_marks = torch.full((num_seq, L_full), -1, dtype=torch.long, device=self.device)
                gen_marks[:, 0] = first_mark
                running_marks_unc = first_mark
            else:
                running_marks_unc = None

            for i in range(1, L_full):
                running_cum_time += self.scaler_exp.unscale(samples[:, i - 1 : i])
                scaled_running_cum_time = self.scaler_cumsum_value_for_generator(running_cum_time)
                if self.use_marks:
                    inter_arr_times_emb: torch.Tensor = self.event_emb(
                        samples[:, i - 1 : i], running_marks_unc.unsqueeze(1)
                    )
                else:
                    inter_arr_times_emb: torch.Tensor = self.time_emb(samples[:, i - 1 : i])
                latent_rep_history[:, i - 1 : i], (hn, cn) = self.enc_rnn(inter_arr_times_emb, (hn, cn))
                # Clamping here for stability
                samples[:, i : i + 1, 0] = self.decoder(
                    inter_arr_times_emb, latent_rep_history[:, i - 1 : i], scaled_running_cum_time.detach()
                ).clamp(min=self.MIN_SCALED_DATA, max=self.MAX_SCALED_DATA)
                # Sample next mark autoregressively (stochastic, not argmax).
                if running_marks_unc is not None:
                    mark_logits_unc = self.mark_predictor(latent_rep_history[:, i - 1 : i])  # (N, 1, M)
                    mark_probs_unc = F.softmax(mark_logits_unc.squeeze(1), dim=-1)  # (N, M)
                    running_marks_unc = torch.multinomial(mark_probs_unc, 1).squeeze(-1)  # (N,)
                    gen_marks[:, i] = running_marks_unc

        return samples, latent_rep_history, gen_marks

    def sample_and_get_loss(self, log_inter_arr_times, lengths, mode, marks=None):
        if mode == "g":
            gen_out, latent_rep_history, _ = self.sample(
                starting_times=self.anchor_times_sampler.sample(log_inter_arr_times[:, 0, 0]),
                log_inter_arr_times=log_inter_arr_times,
                marks=marks,
            )
        else:
            with torch.no_grad():
                gen_out, latent_rep_history, _ = self.sample(
                    starting_times=self.anchor_times_sampler.sample(log_inter_arr_times[:, 0, 0]),
                    log_inter_arr_times=log_inter_arr_times,
                    marks=marks,
                )
        logger.log(
            5,
            "\nOutputs %s\n\nTargets %s\n\nLatent Representation %s",
            gen_out,
            log_inter_arr_times,
            latent_rep_history,
        )
        """
            Pass the whole sequences fake_samples and true_samples. We shift the sequences here:
                - Remove the last value for the fake_samples,
                - Remove the first value for the true_samples.
            If the true values were [1,2,3,4,5], we would want to predict: [2,3,4,5, (6)], where 6 in this example cannot be compared to any true value.
            """
        gen_out = gen_out[:, 1:, :]
        log_inter_arr_times = log_inter_arr_times[:, 1:, :]
        logger.debug("\nOutputs %s\n\nTargets %s\n\n Lengths %s", gen_out, log_inter_arr_times, lengths.view(-1, 1, 1))
        loss = self.discriminator(
            gen_out,
            log_inter_arr_times,
            # lengths - 1 because we removed one value.
            lengths - 1,
            latent_rep_history,
            nu=self.lipschitz_reg,
            mode=mode,
        )
        return gen_out, log_inter_arr_times, loss

    # section ######################################################################
    #  #############################################################################
    #  Training

    def _embed_history(
        self,
        log_inter_arr_times: torch.Tensor,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed a teacher-forced history using time-only or time+mark embeddings."""
        if self.use_marks and marks is not None:
            return self.event_emb(log_inter_arr_times[:, :-1], marks[:, :-1])
        return self.time_emb(log_inter_arr_times[:, :-1])

    def _encode_history(
        self,
        log_inter_arr_times: torch.Tensor,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode teacher-forced history without decoding the next inter-arrival times."""
        inter_arr_times_emb = self._embed_history(log_inter_arr_times, marks)
        latent_rep_history, _ = self.enc_rnn(inter_arr_times_emb)
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
            return self._encode_history(val_dts_scaled, marks_full)

    def training_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor], batch_nb: int):
        data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
        marks = extract_marks_without_anchor_from_batch(batch)

        optim_gen, optim_discr = self.optimizers()
        logger.debug("\n================== New Batch ==================")
        self.discriminator.eval()
        gen_loss, ce_loss = self._training_step_gen(optim_gen, data_dts_scaled, data_dts_lens, marks)
        self.discriminator.train()

        loss_disc_fake = self._training_step_disc(optim_discr, data_dts_scaled, data_dts_lens, marks)

        metrics_to_log = {
            'wasserstein': gen_loss,
            'lip_loss': gen_loss + loss_disc_fake,
        }
        # ce_loss is already computed and backpropagated inside _training_step_gen,
        # so _compute_and_log_mark_metrics_from_logits is not needed here (unlike DDPM,
        # which computes CE inside that helper).  Logging ce_loss directly has no effect
        # on training — it only skips the per-target batch-size weighting on the logged value.
        if ce_loss is not None:
            metrics_to_log['mark_ce'] = ce_loss
        self._log_all_metrics(metrics_to_log, "train_")
        return

    def _training_step_gen(self, optim_gen, log_inter_arr_times: torch.Tensor, lengths: torch.Tensor, marks=None):
        # Zero grad all parameters.
        self.zero_grad()
        _, _, gen_loss = self.sample_and_get_loss(log_inter_arr_times, lengths, "g", marks)
        ce_loss = None
        mark_logits = None
        total_loss = gen_loss
        if self.use_marks:
            # Mark CE is teacher-forced on the real history; encode it directly instead of
            # routing back through the full conditional sampling path.
            latent_rep_history = self._encode_history(log_inter_arr_times, marks)
            mark_logits = self.mark_predictor(latent_rep_history)
            ce_loss = compute_mark_ce_from_logits(mark_logits, marks, lengths)
            total_loss = gen_loss + (self.mark_loss_weight / self._ce_normalizer) * ce_loss
        self.manual_backward(total_loss)
        optim_gen.step()
        return gen_loss, ce_loss

    def _training_step_disc(self, optim_discr, log_inter_arr_times: torch.Tensor, lengths: torch.Tensor, marks=None):
        self.zero_grad()
        gen_out, log_inter_arr_times, loss_disc = self.sample_and_get_loss(log_inter_arr_times, lengths, "d", marks)
        self.manual_backward(loss_disc)
        optim_discr.step()
        return loss_disc

    def validation_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor], batch_nb: int):
        #  But there should be another computation where we construct new sequences and check for the metrics on that.
        data_dts_lens, data_dts_scaled = tpp_utils.cum_times_to_log_inter_times(batch, self.scaler_exp)
        marks = extract_marks_without_anchor_from_batch(batch)
        shifted_gen_out, _, loss_gen = self.sample_and_get_loss(data_dts_scaled, data_dts_lens, "g", marks)

        const_gen_out = set_seq_to_nan_from_index(shifted_gen_out, data_dts_lens - 2)
        const_gen_out_cum = set_seq_to_nan_from_index(
            self.scaler_exp.unscale(const_gen_out.clone()).cumsum(axis=1), data_dts_lens - 1
        )

        hist_metrics = self.metrics_val.compute_histogram_metrics(const_gen_out, const_gen_out_cum)

        metrics_to_log = {
            'wasserstein': loss_gen,
            'epdf': (hist_metrics['hist_it'] + hist_metrics['hist_int']) / 2.0,
            'hist_it': hist_metrics['hist_it'],
            'hist_int': hist_metrics['hist_int'],
        }
        if self.use_marks:
            latent_rep_history = self._encode_history(data_dts_scaled, marks)
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
