import torch
import torch.nn as nn


class DoubleStateRecurrent(nn.Module):
    def __init__(self, num_layers, bidirectional: bool, hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.hidden_state_0 = nn.Parameter(
            torch.randn(
                self.num_layers * (int(self.bidirectional) + 1),
                1,  # repeated later to have batch size
                self.hidden_size,
            ),
            requires_grad=True,
        )  # parameters are moved to device and learn.

        self.hidden_cell_0 = nn.Parameter(
            torch.randn(
                self.num_layers * (int(self.bidirectional) + 1),
                1,  # repeated later to have batch size
                self.hidden_size,
            ),
            requires_grad=True,
        )  # parameters are moved to device and learn.

    def get_hidden_states(self, batch_size):
        # See usage in forward.
        repeat_pattern_h0 = (1, batch_size, 1)
        return self.hidden_state_0.repeat(repeat_pattern_h0), self.hidden_cell_0.repeat(repeat_pattern_h0)

    def forward(self, x):
        return (x, self.get_hidden_states(x.shape[0]))
