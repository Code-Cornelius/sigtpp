"""Tests for BasicNN feedforward network (src/nn/nn/basic_nn.py)."""

import pytest
import torch
from torch import nn

from src.nn.nn.basic_nn import BasicNN


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------
class TestBasicNNValidation:
    def test_non_int_input_size_raises(self):
        with pytest.raises(TypeError, match="integers"):
            BasicNN(
                input_size=3.0,
                list_hidden_sizes=[8],
                output_size=2,
                biases=[True, True],
                activation_functions=[torch.relu],
                dropout=0.0,
            )

    def test_non_int_output_size_raises(self):
        with pytest.raises(TypeError, match="integers"):
            BasicNN(
                input_size=3,
                list_hidden_sizes=[8],
                output_size=2.0,
                biases=[True, True],
                activation_functions=[torch.relu],
                dropout=0.0,
            )

    def test_dropout_out_of_range_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            BasicNN(
                input_size=3,
                list_hidden_sizes=[8],
                output_size=2,
                biases=[True, True],
                activation_functions=[torch.relu],
                dropout=1.0,
            )

    def test_dropout_negative_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            BasicNN(
                input_size=3,
                list_hidden_sizes=[8],
                output_size=2,
                biases=[True, True],
                activation_functions=[torch.relu],
                dropout=-0.1,
            )

    def test_biases_wrong_length_raises(self):
        with pytest.raises(ValueError, match="biases"):
            BasicNN(
                input_size=3,
                list_hidden_sizes=[8, 16],
                output_size=2,
                biases=[True],  # should be len 3
                activation_functions=[torch.relu, torch.relu],
                dropout=0.0,
            )

    def test_activation_wrong_length_raises(self):
        with pytest.raises(ValueError, match="activation_functions"):
            BasicNN(
                input_size=3,
                list_hidden_sizes=[8, 16],
                output_size=2,
                biases=[True, True, True],
                activation_functions=[torch.relu],  # should be len 2
                dropout=0.0,
            )


# ---------------------------------------------------------------------------
# Forward pass shapes
# ---------------------------------------------------------------------------
class TestBasicNNForward:
    def test_output_shape_with_hidden_layers(self):
        torch.manual_seed(0)
        model = BasicNN(
            input_size=5,
            list_hidden_sizes=[16, 8],
            output_size=3,
            biases=[True, True, True],
            activation_functions=[torch.relu, torch.relu],
            dropout=0.0,
        )
        x = torch.randn(4, 5)
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_shape_batched_3d(self):
        """BasicNN should work with 3D inputs (N, L, D) due to nn.Linear broadcasting."""
        torch.manual_seed(0)
        model = BasicNN(
            input_size=3,
            list_hidden_sizes=[8],
            output_size=2,
            biases=[True, True],
            activation_functions=[torch.relu],
            dropout=0.0,
        )
        x = torch.randn(4, 10, 3)
        out = model(x)
        assert out.shape == (4, 10, 2)

    def test_linear_model_no_hidden_layers(self):
        """Empty hidden list should create a single linear layer."""
        torch.manual_seed(0)
        model = BasicNN(
            input_size=4,
            list_hidden_sizes=[],
            output_size=2,
            biases=[True],
            activation_functions=[],
            dropout=0.0,
        )
        x = torch.randn(3, 4)
        out = model(x)
        assert out.shape == (3, 2)

    def test_wrong_input_dim_raises(self):
        torch.manual_seed(0)
        model = BasicNN(
            input_size=5,
            list_hidden_sizes=[8],
            output_size=2,
            biases=[True, True],
            activation_functions=[torch.relu],
            dropout=0.0,
        )
        x = torch.randn(4, 3)  # last dim is 3, not 5
        with pytest.raises(AssertionError, match="input_size"):
            model(x)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestBasicNNInit:
    def test_biases_initialized_to_zero(self):
        torch.manual_seed(0)
        model = BasicNN(
            input_size=3,
            list_hidden_sizes=[8],
            output_size=2,
            biases=[True, True],
            activation_functions=[torch.relu],
            dropout=0.0,
        )
        for layer in model._layers:
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                assert torch.allclose(
                    layer.bias.data, torch.zeros_like(layer.bias.data)
                ), "Biases should be initialized to 0"

    def test_weights_are_nonzero(self):
        """After Xavier init, weights should not all be zero."""
        torch.manual_seed(0)
        model = BasicNN(
            input_size=3,
            list_hidden_sizes=[8],
            output_size=2,
            biases=[True, True],
            activation_functions=[torch.relu],
            dropout=0.0,
        )
        for layer in model._layers:
            if isinstance(layer, nn.Linear):
                assert not torch.allclose(
                    layer.weight, torch.zeros_like(layer.weight)
                ), "Weights should not be all zeros after Xavier init"

    def test_no_bias_layers(self):
        torch.manual_seed(0)
        model = BasicNN(
            input_size=3,
            list_hidden_sizes=[8],
            output_size=2,
            biases=[False, False],
            activation_functions=[torch.relu],
            dropout=0.0,
        )
        for layer in model._layers:
            if isinstance(layer, nn.Linear):
                assert layer.bias is None


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------
class TestBasicNNGradient:
    def test_gradient_flows_through(self):
        torch.manual_seed(0)
        model = BasicNN(
            input_size=3,
            list_hidden_sizes=[8, 4],
            output_size=2,
            biases=[True, True, True],
            activation_functions=[torch.relu, torch.relu],
            dropout=0.0,
        )
        x = torch.randn(2, 3, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 3)
