import logging
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from src.nn.rnn.doublestate_recurrent import DoubleStateRecurrent
from src.nn.rnn.singlestate_recurrent import SingleStateRecurrent


class RNNType(str, Enum):
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"

    @classmethod
    def requires_two_hidden_states(cls, rnn_type: 'RNNType') -> bool:
        """
        Determines whether the given RNN type requires two hidden states (like LSTM).
        """
        return rnn_type == cls.LSTM


class Recurrent_nn(nn.Module):
    """
    A recurrent neural network (RNN) with flexible initialization of hidden states (h0).
    This class allows selection between different RNN types (LSTM, GRU, RNN), access to the hidden state (h0),
    and is compatible with bidirectional RNNs.

    Args:
        input_dim (int, optional): The number of expected features in the input `x`. Default is 1.
        num_layers (int, optional): Number of recurrent layers. E.g., setting `num_layers=2` would mean stacking two LSTMs together
            to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default is 1.
        bidirectional (bool, optional): If `True`, becomes a bidirectional RNN. Default is `False`.
        nb_output_consider (int, optional): Number of elements to consider from the output sequence for the final output. Default is 1.
        hidden_size (int, optional): The number of features in the hidden state `h`. Default is 150.
        dropout (float, optional): If non-zero, introduces a `dropout` layer on the outputs of each RNN layer except the last layer, with dropout probability equal to `dropout`. Default is 0.0.
        rnn_class (torch.nn.Module, optional): The RNN module to use. Can be `torch.nn.LSTM`, `torch.nn.GRU`, or `torch.nn.RNN`. Default is `torch.nn.LSTM`.
        init_hidden_states_0 (bool): Whether to initialize hidden states to zero at the beginning of training.

    References:
        - PyTorch LSTM documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        - PyTorch GRU documentation: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        - PyTorch RNN documentation: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    """

    @staticmethod
    def init_weights(layer: nn.Module) -> None:
        # This function handles custom initialization for recurrent layers, like LSTM, GRU, and RNN.
        # The goal is to set the weights and biases in a way that ensures stable and effective training.

        # For LSTM and GRU layers, we need to consider the gates these architectures use.
        if isinstance(layer, (nn.LSTM, nn.GRU)):
            # Loop over all the parameters in the layer (both weights and biases).
            for name, param in layer.named_parameters():

                # 'weight_ih' stands for input-to-hidden weights. These weights are used to process the input features.
                if 'weight_ih' in name:
                    # LSTM has 4 gates (input, forget, cell, output), while GRU has 3 gates (reset, update, new memory).
                    # We divide the weight matrix into blocks, each corresponding to a gate.
                    num_gates = (
                        layer.num_gates if hasattr(layer, 'num_gates') else (4 if isinstance(layer, nn.LSTM) else 3)
                    )
                    gate_size = param.shape[0] // num_gates  # Calculate the size of each gate's weight block.

                    # We initialize each gate's weights using Xavier Uniform initialization.
                    # This method sets weights to values that help maintain a balance between the input and output signal magnitudes.
                    # It helps avoid issues like vanishing or exploding gradients, which can occur when training deep networks.
                    for i in range(num_gates):
                        nn.init.xavier_uniform_(param.data[i * gate_size : (i + 1) * gate_size])

                # 'weight_hh' stands for hidden-to-hidden weights. These are used to process the hidden state at the previous time step.
                elif 'weight_hh' in name:
                    # Just like 'weight_ih', the hidden-to-hidden weights are also split into blocks for each gate.
                    num_gates = (
                        layer.num_gates if hasattr(layer, 'num_gates') else (4 if isinstance(layer, nn.LSTM) else 3)
                    )
                    gate_size = param.shape[0] // num_gates  # Calculate the size of each gate's weight block.

                    for i in range(num_gates):
                        # If the weight matrix is square (i.e., the same number of input and output units), we use Orthogonal Initialization.
                        # Orthogonal Initialization ensures that the weight matrix has orthogonal rows and columns, preserving the magnitude
                        # of the gradient over long sequences, which is important for avoiding gradient vanishing/exploding.
                        if param.shape[0] == param.shape[1]:
                            nn.init.orthogonal_(param.data[i * gate_size : (i + 1) * gate_size])
                        else:
                            # If the matrix is not square, fallback to Xavier Uniform initialization to avoid poor initialization for non-square matrices.
                            nn.init.xavier_uniform_(param.data[i * gate_size : (i + 1) * gate_size])

                # 'bias' refers to the bias terms associated with each gate.
                elif 'bias' in name:
                    # Initialize all biases to zero. This prevents any initial bias from skewing the behavior of the network early in training.
                    nn.init.zeros_(param.data)

                    # Special handling for LSTM layers: the forget gate has a unique role in deciding what to "forget" at each step.
                    # By setting its bias to 1.0, we encourage the forget gate to initially "remember" more information.
                    # This strategy helps stabilize training, especially in tasks that require the model to remember information over long sequences.
                    # This technique is suggested by:
                    # - Gers et al. (1999) in "Learning to Forget: Continual Prediction with LSTM"
                    # - Jozefowicz et al. (2015) in "An Empirical Exploration of Recurrent Network Architectures"
                    if isinstance(layer, nn.LSTM):
                        # LSTM has 4 gates, so we calculate the size of the block corresponding to each gate.
                        num_gates = 4
                        gate_size = param.shape[0] // num_gates

                        # The forget gate is the second gate in LSTM (index 1 when considering 0-based indexing).
                        forget_gate_index = gate_size  # The second block corresponds to the forget gate.

                        # Set the bias for the forget gate to 1.0 to encourage the network to "remember" more during initial training.
                        nn.init.constant_(param.data[forget_gate_index : forget_gate_index + gate_size], 1.0)

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        nb_output_consider: int = 1,
        rnn_type: RNNType = RNNType.RNN,
        init_hidden_states_0: bool = False,
    ) -> None:
        """
        Initializes the RNN model with the provided parameters.

        Args:
            input_dim (int): The number of input features per time step.
            hidden_size (int): The number of hidden units in each RNN layer.
            num_layers (int): The number of RNN layers.
            bidirectional (bool): If True, makes the RNN bidirectional.
            dropout (float): Dropout rate applied to the RNN layers.
            nb_output_consider (int): Number of elements to consider from the output sequence. To consider them all, use nb_output_consider = sys.maxsize.
            rnn_type (RNNType): The type of RNN (RNN, LSTM, GRU).
            init_hidden_states_0 (bool): Whether to initialize hidden states to zero at the beginning of training.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.nb_output_consider = nb_output_consider
        self.rnn_type = rnn_type

        self.init_hidden_states_0 = init_hidden_states_0

        # Choose the appropriate RNN class (RNN, LSTM, GRU)
        self.stacked_rnn = self._create_rnn_module()

        if self.init_hidden_states_0:
            if RNNType.requires_two_hidden_states(self.rnn_type):
                self.hidden_state_parameters = DoubleStateRecurrent(num_layers, bidirectional, hidden_size)
            else:
                self.hidden_state_parameters = SingleStateRecurrent(num_layers, bidirectional, hidden_size)

        # Apply weight initialization
        self.apply(self.init_weights)
        return

    def _create_rnn_module(self) -> nn.Module:
        """
        Creates the appropriate RNN module based on the rnn_type.
        """
        rnn_classes = {RNNType.LSTM: nn.LSTM, RNNType.GRU: nn.GRU, RNNType.RNN: nn.RNN}
        rnn_class = rnn_classes.get(self.rnn_type, None)

        if rnn_class is None:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")

        return rnn_class(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(
        self, seqs: torch.Tensor, h0: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass through the RNN. If h0 is not provided and init_hidden_states_0 is set to True, it initializes with learned parameters.

        Args:
            seqs (torch.Tensor): Input sequence of shape `(batch_size, seq_len, input_dim)`.
            h0 (Optional[Tuple[torch.Tensor]]): Initial hidden state.
                If using LSTM, h0 will be a tuple (h0, c0).
                If `None` and init_hidden_states_0 is False, h0 is initialized automatically by PyTorch.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor]]: The output sequence and the final hidden state.
                        torch.Tensor: Output sequence of shape `(batch_size, seq_len, hidden_size * num_directions)`.
                        torch.Tensor: Hidden state after the last time step. If using LSTM, the hidden state is a tuple (h_n, c_n).
        """
        batch_size = seqs.shape[0]

        if h0 is None and self.init_hidden_states_0:
            h0 = self.get_first_hidden_state(batch_size)
        elif h0 is None:
            zeros = torch.zeros(
                self.num_layers * (int(self.bidirectional) + 1), batch_size, self.hidden_size, device=seqs.device
            )
            if RNNType.requires_two_hidden_states(self.rnn_type):
                h0 = (zeros, zeros.clone())
            else:
                h0 = zeros

        out, hn = self.stacked_rnn(seqs, h0)

        if self.bidirectional:
            # The shape of `out` is (N,  L, hidden_size * num_directions).
            # We extract nb_output_consider elements: h_n, h_{n-1}, ...
            # The second dimension is reversed for the other direction.
            out = torch.cat(
                (
                    out[:, -self.nb_output_consider :, : self.hidden_size],
                    out[:, : self.nb_output_consider, self.hidden_size :],
                ),
                dim=1,
            )
        else:
            # `out` is of shape (batch size, nb_output_consider, hidden_size)
            out = out[:, -self.nb_output_consider :, : self.hidden_size]

        return out, hn

    @property
    def output_len(self) -> int:
        """
        Compute the output length based on hidden size, number of directions, and
        number of output steps considered.

        Returns:
            int: The length of the output (number of features in the output layer).
        """
        return self.hidden_size * (int(self.bidirectional) + 1) * self.nb_output_consider

    def get_first_hidden_state(self, batch_size: int) -> torch.Tensor:
        """
        Get the first hidden state of the RNN.

        Returns:
            torch.Tensor: The first hidden state of the RNN.
        """
        if self.init_hidden_states_0:
            return self.hidden_state_parameters.get_hidden_states(batch_size)
        logger.error("Hidden states are not initialized.")
        return torch.zeros(self.num_layers * (int(self.bidirectional) + 1), batch_size, self.hidden_size)


if __name__ == "__main__":
    input_dim = 3
    hidden_size = 5
    num_layers = 1
    batch_size = 4
    sequence_length = 10

    input_sequence = torch.randn(batch_size, sequence_length, input_dim)

    # Test Case 1: RNN with single hidden state
    rnn_model = Recurrent_nn(input_dim=input_dim, hidden_size=hidden_size, rnn_type=RNNType.RNN)
    output, hidden = rnn_model(input_sequence)
    print(f"RNN output shape: {output.shape}, hidden state shape: {hidden.shape}")

    # Test Case 2: LSTM with double hidden states
    lstm_model = Recurrent_nn(input_dim=input_dim, hidden_size=hidden_size, rnn_type=RNNType.LSTM)
    output, (h0, c0) = lstm_model(input_sequence)
    print(f"LSTM output shape: {output.shape}, hidden state shape: {h0.shape}, cell state shape: {c0.shape}")

    # Test Case 3: GRU with single hidden state
    gru_model = Recurrent_nn(input_dim=input_dim, hidden_size=hidden_size, rnn_type=RNNType.GRU)
    output, hidden = gru_model(input_sequence)
    print(f"GRU output shape: {output.shape}, hidden state shape: {hidden.shape}")

    # Test Case 4: Passing hidden states explicitly
    init_hidden = torch.randn(num_layers, batch_size, hidden_size)
    output, hidden = gru_model(input_sequence, h0=init_hidden)
    print(f"GRU with explicit hidden state: {output.shape}, hidden state shape: {hidden.shape}")
