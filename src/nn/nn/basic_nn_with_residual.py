from typing import List, Callable

import torch
from torch import nn


class BasicNNWithResiduals(nn.Module):
    """
    Feedforward neural network with optional residual connections for layers with matching input/output sizes.
    """

    module_linearity = nn.Linear

    def __init__(
        self,
        input_size: int,
        list_hidden_sizes: List[int],
        output_size: int,
        biases: List[bool],
        activation_functions: List[Callable],
        dropout: float,
    ):
        super().__init__()

        if not isinstance(input_size, int) or not isinstance(output_size, int):
            raise TypeError("input_size and output_size must be integers.")
        if not isinstance(dropout, float) or not (0 <= dropout < 1):
            raise ValueError("dropout must be a float between 0 and 1.")
        if len(biases) != len(list_hidden_sizes) + 1:
            raise ValueError("Length of biases must match the number of layers (hidden + output).")
        if len(activation_functions) != len(list_hidden_sizes) and list_hidden_sizes:
            raise ValueError("Length of activation_functions must match the number of hidden layers.")

        self.input_size = input_size
        self.list_hidden_sizes = list_hidden_sizes
        self.output_size = output_size
        self.biases = biases
        self.activation_functions = activation_functions
        self.dropout = dropout

        self._layers = nn.ModuleList()
        self._apply_dropout = nn.Dropout(p=self.dropout)
        self._residual_connections = []  # Track which layers allow residuals

        self.set_layers()

    def set_layers(self):
        """Defines the layers of the network based on the initialization parameters."""
        layer_sizes = [self.input_size] + self.list_hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            self._layers.append(self.module_linearity(layer_sizes[i], layer_sizes[i + 1], self.biases[i]))
            self._residual_connections.append(layer_sizes[i] == layer_sizes[i + 1])  # True if residual is possible

        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections when possible.
        """
        assert x.shape[-1] == self.input_size, f"Input shape {x.shape} does not match input_size {self.input_size}."

        for i, layer in enumerate(self._layers[:-1]):  # Exclude output layer
            identity = x
            x = layer(x)
            x = self.activation_functions[i](x)
            x = self._apply_dropout(x)
            if self._residual_connections[i]:
                x = x + identity

        x = self._layers[-1](x)
        return x

    @staticmethod
    def init_weights(layer: nn.Module):
        """
        Applies Xavier initialization to layers.
        """
        if isinstance(layer, nn.Linear) and layer.weight.requires_grad:
            gain = nn.init.calculate_gain("relu")
            torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None and layer.bias.requires_grad:
                layer.bias.data.fill_(0.0)
