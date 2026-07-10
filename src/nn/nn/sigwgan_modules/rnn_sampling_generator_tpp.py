import logging
import sys
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from src.nn.embeddings.time import TimeEmbedding
from src.nn.rnn.recurrent_nn import Recurrent_nn, RNNType
from src.nn.nn.sigwgan_modules.decoder_for_gan_full_input import Decoder4GANFullInput


class RNNSamplingGeneratorTPP(nn.Module):
    """
    Generator for TPPs leveraging an RNN structure.
    This is only adapted for tpps where time is located as the FIRST feature (along third axis).

    When event_emb and mark_predictor are provided (mark-aware mode), the generator
    uses joint time+mark embeddings (Eq. 4, GNTPP) and samples marks autoregressively.
    """

    @staticmethod
    def validate_inputs(initial_value: torch.Tensor, latent_rep_history: torch.Tensor, sequences: torch.Tensor) -> None:
        assert len(sequences.shape) == 3, f"Expected 3D tensor, got {sequences.shape}."
        assert len(initial_value.shape) == 3, f"Expected 3D tensor, got {initial_value.shape}."
        assert (
            initial_value.shape[0] == sequences.shape[0]
        ), f"Expected same batch size, got {initial_value.shape[0]} and {sequences.shape[0]}."
        assert len(latent_rep_history.shape) == 3, f"Expected 3D tensor, got {latent_rep_history.shape}."
        assert torch.all(torch.abs(sequences) < 1e-6), f"Expected sequences to be zero, got {sequences}."
        return

    @staticmethod
    def compute_inputs_dec(
        sequences: torch.Tensor,
        intensity_lambdas_scaled: torch.Tensor,
        outputs_hidden: torch.Tensor,
        scaler_paths: typing.Callable[[torch.Tensor], torch.Tensor],
        i: int,
        len_history_input: int,
    ) -> torch.Tensor:
        assert len(sequences.shape) == 3, f"Expected 3D tensor, got {sequences.shape}."
        assert len(outputs_hidden.shape) == 2, f"Expected 2D tensor, got {outputs_hidden.shape}."
        assert (
            outputs_hidden.shape[0] == sequences.shape[0]
        ), f"Expected same batch size, got {outputs_hidden.shape[0]} and {sequences.shape[0]}."

        #### Manage the case of having too few previous values.
        #### If needed, we pad the beginning of intensity_scaled4rnn and values_scaled4rnn with zeros, representing the absence of previous values.
        beg_seq_idx = max(0, i - len_history_input)
        # Extract intensity and values history, then pad if necessary
        intensity_scaled4rnn = intensity_lambdas_scaled[:, beg_seq_idx:i]
        values_scaled4rnn = scaler_paths(sequences[:, beg_seq_idx:i])

        logger.log(2, "intensity_scaled4rnn: %s", intensity_scaled4rnn.unsqueeze(-1))
        logger.log(2, "values_scaled4rnn: %s", values_scaled4rnn.unsqueeze(-1))

        hidden_with_inputs = torch.cat(
            (
                # h_i, previous hidden state; N, 1, hidden_size_RNN
                outputs_hidden,
                # x_i, previous values; N, len_history_input, num_dim_seqs
                intensity_scaled4rnn,
                # x_i, previous values; N, len_history_input, num_dim_seqs
                values_scaled4rnn,
            ),
            dim=1,
        )
        return hidden_with_inputs

    def __init__(
        self,
        time_emb: TimeEmbedding,
        num_layers_rnn: int,
        hidden_size_RNN: int,
        dropout: float,
        deterministic_model: bool = False,
        event_emb=None,
        mark_predictor=None,
    ) -> None:
        super().__init__()
        self.time_emb: TimeEmbedding = time_emb
        self.event_emb = event_emb  # Optional EventEmbedding (mark-aware mode)
        self.mark_predictor = mark_predictor  # Optional MarkPredictor

        # Use event_emb input size when mark-aware, otherwise time_emb size.
        if self.event_emb is not None:
            INPUT_SIZE: int = self.event_emb.embed_size
        else:
            INPUT_SIZE: int = self.time_emb.embed_size
        self.OUTPUT_SIZE: typing.Final[int] = 1

        # Why was init state not taken from the recurrent nn?
        self.recurrent_unit: Recurrent_nn = Recurrent_nn(
            INPUT_SIZE, hidden_size_RNN, num_layers_rnn, False, 0.0, sys.maxsize, RNNType.LSTM, True
        )
        self.decoder: Decoder4GANFullInput = Decoder4GANFullInput(
            hidden_size_RNN, hidden_size_RNN * 2, 3, hidden_size_RNN, deterministic_model=deterministic_model
        )
        return

    def _embed(self, times: torch.Tensor, marks: typing.Optional[torch.Tensor]) -> torch.Tensor:
        """Embed a single step. Uses event_emb when marks-aware, else time_emb."""
        if self.event_emb is not None and marks is not None:
            return self.event_emb(times, marks)
        return self.time_emb(times)

    def _embed_history(
        self,
        log_inter_arr_times: torch.Tensor,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed a full teacher-forced history, defaulting missing marks to zeros."""
        N, L, _ = log_inter_arr_times.shape
        if self.event_emb is not None:
            if marks is None:
                marks_input = torch.zeros(N, L, dtype=torch.long, device=log_inter_arr_times.device)
            else:
                marks_input = marks
            return self.event_emb(log_inter_arr_times, marks_input)
        return self.time_emb(log_inter_arr_times)

    def encode_history(
        self,
        log_inter_arr_times: torch.Tensor,
        marks: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a teacher-forced history without decoding the next inter-arrival times."""
        inter_arr_times_emb = self._embed_history(log_inter_arr_times, marks)
        hn, cn = self.recurrent_unit.get_first_hidden_state(log_inter_arr_times.shape[0])
        latent_rep_history, _ = self.recurrent_unit(inter_arr_times_emb, (hn, cn))
        return latent_rep_history

    def generate(
        self,
        starting_time_sequences: torch.Tensor,
        # initial value should not be passed scaled.
        initial_intertimes: torch.Tensor,
        sequences: torch.Tensor,
        latent_rep_history: torch.Tensor,
        min_value: torch.Tensor,
        max_value: torch.Tensor,
        scaling_output: typing.Callable[[torch.Tensor], torch.Tensor],
        scaling_cumsum: typing.Callable[[torch.Tensor], torch.Tensor],
        marks: typing.Optional[torch.Tensor] = None,
        gen_marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]:
        """
        The sequences are sampled iteratively.
        The length of sequences is the length of the time series, including initial value that will be set as the first value.
        Do not create any data tensor, they are coming from outside.

        Get as input [log tau_1] and return [log tau_1, hat log tau_2, ..., hat log tau_T].

        Args:
            marks: (N, L) integer marks for conditional / teacher-forced generation.
                   When provided and event_emb is set, the first recurrent step uses
                   `marks[:, 0]` as the previous mark, then later marks are sampled
                   autoregressively from mark_predictor.
            gen_marks: Optional pre-allocated (N, L) long tensor to collect generated marks into.
                   In unconditional marked generation, `gen_marks[:, 0]` must already
                   contain the seeded first-event mark. The first recurrent step then
                   uses that stored mark as the previous-mark state before generating
                   `tau_2` / `mark_2`, and later autoregressively sampled marks are
                   written into positions 1..L-1.

        Returns:
            Tuple of (sequences, latent_rep_history, gen_marks).
        """
        self.validate_inputs(initial_intertimes, latent_rep_history, sequences)
        assert not (
            marks is not None and gen_marks is not None
        ), "generate() expects either conditional history marks or unconditional generated-mark storage, not both."

        # Sequences used for shape.
        (hn, cn) = self.recurrent_unit.get_first_hidden_state(sequences.shape[0])

        running_cum_time = starting_time_sequences

        # Initialise running marks for the first recurrent step.
        # Conditional generation uses the provided history marks.
        # Unconditional marked generation must instead seed from the generated first mark
        # stored in gen_marks[:, 0] so the tau_1/mark_1 pair is used consistently.
        if self.event_emb is not None and self.mark_predictor is not None:
            if marks is not None:
                running_marks = marks[:, 0]
            elif gen_marks is not None:
                # Clone to avoid keeping a view into gen_marks that is updated in-place later.
                running_marks = gen_marks[:, 0].clone()
            else:
                running_marks = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
        else:
            running_marks = None

        # Collect per-step outputs in lists instead of writing into pre-allocated
        # tensors in-place.  In-place slice assignment (e.g. tensor[:, i:i+1] = ...)
        # bumps the base tensor's version counter, which makes earlier autograd
        # snapshots stale — PyTorch >= 1.12 raises RuntimeError on backward.
        prev_step = initial_intertimes  # (N, 1, D)
        seq_steps = [initial_intertimes]
        latent_steps = []

        for i in range(1, sequences.shape[1]):
            running_cum_time = running_cum_time + scaling_output.unscale(prev_step)
            # The cumsum value can be arbitrarily large. Considering we do not clamp before the end of the for loop,
            # the cumsum can achieve values as large as length * max. This can lead to instabilities in the training,
            # despite that these values are not used in practice (because OOB). Clamping below is in fact enough.
            scaled_running_cum_time = scaling_cumsum(running_cum_time)
            # The value 5 is large. Remember, scaled_running_cum_time that is in the positive reel domain, but also has been STDed.
            scaled_running_cum_time = scaled_running_cum_time.clamp(max=5.0)

            step_marks = running_marks.unsqueeze(1) if running_marks is not None else None  # (N, 1)
            inter_arr_times_emb: torch.Tensor = self._embed(prev_step, step_marks)
            latent_step, (hn, cn) = self.recurrent_unit(inter_arr_times_emb, (hn, cn))
            latent_steps.append(latent_step)

            decoded = self.decoder(
                inter_arr_times_emb,
                latent_step,
                scaled_running_cum_time.detach(),
                # Clamping here for stability
            ).clamp(min=min_value, max=max_value)
            prev_step = decoded.unsqueeze(-1)  # (N, 1) -> (N, 1, 1)
            seq_steps.append(prev_step)

            # Sample next mark autoregressively (stochastic, not argmax).
            if running_marks is not None:
                mark_logits = self.mark_predictor(latent_step)  # (N, 1, M)
                mark_probs = F.softmax(mark_logits.squeeze(1), dim=-1)  # (N, M)
                running_marks = torch.multinomial(mark_probs, 1).squeeze(-1)  # (N,)
                if gen_marks is not None:
                    gen_marks[:, i] = running_marks

        sequences = torch.cat(seq_steps, dim=1)
        latent_rep_history = torch.cat(latent_steps, dim=1)
        # Shapes: sequences [N, L, D], latent_rep_history [N, L-1, H]
        return sequences, latent_rep_history, gen_marks

    def generate_with_history(
        self,
        starting_times: torch.Tensor,
        log_inter_arr_times: torch.Tensor,
        min_value: torch.Tensor,
        max_value: torch.Tensor,
        scaling_output: typing.Callable[[torch.Tensor], torch.Tensor],
        scaling_cumsum: typing.Callable[[torch.Tensor], torch.Tensor],
        marks: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        The sequences are sampled iteratively.
        The length of sequences is the length of the time series, including initial value that will be set as the first value.
        Do not create any data tensor, they are coming from outside.

        Get as input [log tau_1, log tau_2, ..., log tau_T-1] and return [log tau_2, hat log tau_3, ..., hat log tau_T].
        The starting times represent the temporal beginning of the sequences, used for the decoder cumulative entry.

        Args:
            marks: (N, L) integer marks for the input history (aligns with log_inter_arr_times).
                   Used when event_emb is set. When None and event_emb is set, defaults to zeros.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Generated sequences and latent representations.
        """
        running_cum_time = starting_times + scaling_output.unscale(log_inter_arr_times).cumsum(1)
        scaled_running_cum_time = scaling_cumsum(running_cum_time)
        # # The value 5 is large. Remember, scaled_running_cum_time that is in the positive reel domain, but also has been STDed.
        scaled_running_cum_time = scaled_running_cum_time.clamp(max=5.0)
        inter_arr_times_emb = self._embed_history(log_inter_arr_times, marks)
        latent_rep_history = self.encode_history(log_inter_arr_times, marks)

        sequences = self.decoder(
            inter_arr_times_emb,
            latent_rep_history,
            scaled_running_cum_time.detach(),
        ).clamp(min=min_value, max=max_value)

        # Add the first value back.
        sequences = torch.cat((log_inter_arr_times[:, 0:1], sequences), dim=1)

        # Shapes: sequences [N, L, D], latent_rep_history [N, L-1, H]
        return sequences, latent_rep_history
