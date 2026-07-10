from typing import List, Callable

import torch
from torch import nn


class BasicNN(nn.Module):
    """
    A basic feedforward neural network with configurable hidden layers, activation functions, and dropout.

    The architecture consists of:
    - Input layer
    - Sequence of hidden layers, each followed by an activation function and optionally a dropout layer
    - Output layer

    Args:
        input_size (int): The size of the input features.
        list_hidden_sizes (List[int]): A list containing the sizes of the hidden layers. If empty, the model becomes a linear model.
        output_size (int): The size of the output layer.
        biases (List[bool]): A list indicating whether to use bias for each layer. Must match the number of layers (hidden + output).
        activation_functions (List[Callable]): A list of activation functions to apply after each hidden layer.
        dropout (float): Dropout probability to apply after each hidden layer, between 0 and 1.

    Raises:
        TypeError: If input_size or output_size is not an integer.
        ValueError: If dropout is not between 0 and 1.
        ValueError: If the length of biases does not match the number of layers.
        ValueError: If the length of activation_functions does not match the number of hidden layers.
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

        # Input validation
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

        self.set_layers()
        return

    def set_layers(self):
        """Defines the layers of the network based on the initialization parameters."""
        layer_sizes = [self.input_size] + self.list_hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            self._layers.append(self.module_linearity(layer_sizes[i], layer_sizes[i + 1], self.biases[i]))

        # Initialize weights using Xavier initialization
        self.apply(self.init_weights)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        # Noticed how sometimes input shape may be incorrecly input but still the forward works, only backward fails.
        assert x.shape[-1] == self.input_size, f"Input shape {x.shape} does not match input_size {self.input_size}."

        for i, layer in enumerate(self._layers[:-1]):  # Exclude output layer
            x = layer(x)

            x = self.activation_functions[i](x)
            x = self._apply_dropout(x)

        x = self._layers[-1](x)
        return x

    @staticmethod
    def init_weights(layer: nn.Module):
        """
        Applies Xavier initialization to layers.

        Args:
            layer (nn.Module): A layer in the neural network.
        """
        if isinstance(layer, nn.Linear) and layer.weight.requires_grad:
            gain = nn.init.calculate_gain("relu")
            torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None and layer.bias.requires_grad:
                # The way to do it to change data without touching the gradient.
                layer.bias.data.fill_(0.0)
