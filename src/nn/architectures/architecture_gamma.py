import logging
import typing

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
from src.nn.architectures.tpp_architecture import TPPArchitecture
from src.nn.architectures.mark_prediction_utils import (
    build_empirical_mark_logits,
    prepare_next_mark_prediction_tensors,
    count_valid_mark_targets,
)

from src.data_types.sigw_loss_data_props import SigWLossDataProps


class ArchitectureGamma(TPPArchitecture):
    def __init__(
        self,
        data_train: torch.Tensor,
        data_train_lens: torch.Tensor,
        data_val: torch.Tensor,
        data_val_lens: torch.Tensor,
        train_marks: torch.Tensor,
        val_marks: torch.Tensor,
        loss_properties: SigWLossDataProps,
        learning_rate: float,
        t_max: int,
        num_marks: int,
        output_dir: str = None,
        enable_plot: bool = False,
        period_plot_val: int = 25,
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
            1.0,  # concentration_factor
            output_dir=output_dir,
            enable_plot=enable_plot,
            period_plot_val=period_plot_val,
            plot_every_n_val_steps=plot_every_n_val_steps,
            disable_scaler=True,
            n_bootstraps=n_bootstraps,
        )

        # Parameters model
        self.lr = learning_rate

        # initialize from method-of-moments
        diff = data_train.diff(dim=1)
        diff_filter = diff[diff > 1e-10]

        # Use per-sequence mean (not grand mean) to avoid length-weighting bias from
        # truncated sequences: longer seqs have more but smaller ITs, so pooling all
        # ITs biases the grand mean down.  Per-sequence mean = elapsed_time / diff_lens.
        diff_lens = (data_train_lens - 1).long()
        valid = diff_lens > 0
        # data_train is (N, L+1, D) cumulative times; index 0 is the anchor, index
        # diff_lens[i] is the last valid entry.  Subtract the anchor to get elapsed time.
        anchor_times = data_train[:, 0, 0]
        last_cumtimes = data_train[torch.arange(data_train.shape[0]), diff_lens.clamp(min=0), 0] - anchor_times
        self.scaling_factor = (last_cumtimes[valid] / diff_lens[valid].float()).mean()
        diff_filter /= self.scaling_factor
        # Pooled mean is biased by length-weighting; true scaled mean is 1.0
        # by construction of scaling_factor as per-sequence mean.
        x_mean = diff_filter.new_tensor(1.0)
        # Std around the true mean (not the biased pooled mean) for consistent MoM.
        x_std = ((diff_filter - x_mean) ** 2).mean().sqrt()

        # Method of moments for Gamma(k, rate=theta): k = mean²/var, rate = mean/var.
        # clamp instead of max(float, tensor): Python's max() returns a plain float when the
        # tensor value is smaller, and float.log() does not exist.  Bursty datasets
        # have high variance so k < 0.1, which triggered this on every run.
        k_init = (x_mean * x_mean / (x_std * x_std)).clamp(min=0.1)
        theta_init = (x_mean / (x_std * x_std)).clamp(min=1e-6)

        self.log_k = nn.Parameter(k_init.log())
        self.log_theta = nn.Parameter(theta_init.log())

        # Divisor set on the first training step so that the logged NLL starts at 10 on plots.
        self._nll_log_scale = None

        # Mark classifier: random baseline using empirical mark probabilities.
        if self.use_marks:
            _, current_targets, valid_lengths = prepare_next_mark_prediction_tensors(train_marks, data_train_lens - 1)
            counts = count_valid_mark_targets(current_targets, valid_lengths, num_marks).float()
            # Register as buffer so it moves with model to GPU and is saved in checkpoints.
            self.register_buffer('_mark_probs', counts / counts.sum().clamp(min=1.0))
            logger.info("Gamma mark classifier: empirical probs = %s", self._mark_probs.tolist())
        else:
            self._mark_probs = None
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
        # Returns (N, L, 1), (N, L-1, H_in), and None (gamma baseline has no mark generation).
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
        dist = torch.distributions.Gamma(concentration=torch.exp(self.log_k), rate=torch.exp(self.log_theta))
        if log_inter_arr_times is not None:
            num_seq = log_inter_arr_times.shape[0]
        samples = dist.sample((num_seq, self.data_train_dts.shape[1] + 1, self.num_dim_seqs))
        samples *= self.scaling_factor
        return samples, torch.zeros_like(samples), None

    # section ######################################################################
    #  #############################################################################
    #  Training

    def training_step(self, batch, batch_nb: int):
        self.zero_grad()
        logger.debug("\n================== New Batch ==================")

        data_dts = batch[0].diff(dim=1)
        data_dts = data_dts[data_dts > 1e-10]
        data_dts_scaled = data_dts / self.scaling_factor
        # Log-likelihood of Gamma(k, rate=θ) summed over n i.i.d. scaled inter-arrival times.
        #   log p(x; k, θ) = (k-1) log x  -  θ x  +  k log θ  -  log Γ(k)
        #   log L = (k-1) Σ log(x_i)  -  θ Σ x_i  +  n (k log θ - log Γ(k))
        # Here k = exp(log_k), θ = exp(log_theta) (rate, not scale), x_i = data_dts / scaling_factor.
        k = self.log_k.exp()
        # lgamma(k) on CPU: on Ada GPUs (sm_89) with a torch built against CUDA <11.8 (e.g.
        # 1.12.1+cu116), torch JIT-compiles the lgamma kernel via nvrtc, whose toolkit does not
        # recognise sm_89 -> "nvrtc: invalid value for --gpu-architecture". k is a scalar, so the
        # CPU round-trip is free; autograd flows back through .to()/.cpu() (its backward digamma
        # also runs on CPU, dodging the same nvrtc wall).
        lgamma_k = torch.lgamma(k.cpu()).to(k.device)
        log_likelihood = (
            (k - 1) * torch.log(data_dts_scaled).sum()  # (k-1) Σ log(x_i)
            - (data_dts_scaled * self.log_theta.exp()).sum()  # - θ Σ x_i
            + data_dts_scaled.shape[0]
            * (k * self.log_theta - lgamma_k)  # + n (k log θ - log Γ(k))
        )
        # NLL is what we minimise; also the only positive quantity, so it renders on the log-scale plot.
        nll = -log_likelihood

        # On the first call, fix the divisor so the logged value starts at 10.
        if self._nll_log_scale is None:
            self._nll_log_scale = nll.detach() / 10.0

        self._log_all_metrics(
            {
                'nll': nll / self._nll_log_scale,
            },
            "train_",
        )

        return nll

    def validation_step(self, batch, batch_nb: int):
        logger.debug("\n================== New Batch ==================")
        uncond_result = self.sample_and_fix_seqs(num_seq=batch[0].shape[0])
        gen_out_sigloss = self.scale_paths_pre_sig(
            torch.cat([(uncond_result.its_scaled_cst), (uncond_result.cum_abs_cst)], axis=2),
            seq_lens=uncond_result.seq_lens,
        )
        logger.debug("Sequences (scaled) for metric %s", gen_out_sigloss)
        loss_sigW = self.sigw1metric_val(gen_out_sigloss)

        hist_it = self.metrics_val.histogram_loss_it(uncond_result.its_scaled_nan, [])
        hist_int = self.metrics_val.histogram_loss_cum(uncond_result.cum_rel_nan, [])

        self._log_all_metrics(
            {
                'sigW': loss_sigW,
                'epdf': (hist_it + hist_int) / 2.0,
                'hist_it': hist_it,
                'hist_int': hist_int,
            },
            "val_",
        )
        if self.use_marks and self._mark_probs is not None:
            marks = batch[2][:, 1:]
            mark_logits = build_empirical_mark_logits(self._mark_probs, marks.shape[0], max(marks.shape[1] - 1, 0))
            # batch[1] - 1 == data_dts_lens by definition (cum_times_to_log_inter_times
            # returns data_lens - 1), so this is consistent with all other call sites.
            self._compute_and_log_mark_metrics_from_logits(
                mark_logits=mark_logits,
                marks=marks,
                lengths=batch[1] - 1,
                prefix="val_",
                include_ce=False,
            )
        return

    @property
    def _include_top3_mark_accuracy(self) -> bool:
        return True

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
        """Return direct empirical-distribution logits for shared mark evaluation."""
        if not self.use_marks or self._mark_probs is None:
            return None
        return build_empirical_mark_logits(self._mark_probs, current_targets.shape[0], current_targets.shape[1])
