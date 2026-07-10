import torch
import pytest
from src.nn.rnn.recurrent_nn import Recurrent_nn, RNNType


class TestRecurrentNnOutputShapes:
    """Verify output shapes for each RNN type using init_hidden_states_0=True
    to avoid the LSTM h0 bug (line 204: single tensor instead of tuple)."""

    @pytest.fixture(params=[RNNType.RNN, RNNType.LSTM, RNNType.GRU])
    def rnn_type(self, request):
        return request.param

    def test_output_shape(self, rnn_type):
        input_dim, hidden_size, nb_out = 3, 8, 1
        model = Recurrent_nn(
            input_dim=input_dim, hidden_size=hidden_size,
            nb_output_consider=nb_out, rnn_type=rnn_type,
            init_hidden_states_0=True,
        )
        x = torch.randn(4, 10, input_dim)
        out, _ = model(x)
        assert out.shape == (4, nb_out, hidden_size)

    def test_output_shape_multiple_outputs(self, rnn_type):
        input_dim, hidden_size, nb_out = 2, 6, 3
        model = Recurrent_nn(
            input_dim=input_dim, hidden_size=hidden_size,
            nb_output_consider=nb_out, rnn_type=rnn_type,
            init_hidden_states_0=True,
        )
        x = torch.randn(2, 5, input_dim)
        out, _ = model(x)
        assert out.shape == (2, nb_out, hidden_size)


class TestRecurrentNnDefaultH0:
    """Regression tests for the h0 default initialization (h0=None, init_hidden_states_0=False).
    Previously LSTM failed because a single tensor was passed instead of (h0, c0) tuple.
    """

    def test_rnn_default_h0(self):
        model = Recurrent_nn(input_dim=3, hidden_size=8, rnn_type=RNNType.RNN)
        out, _ = model(torch.randn(2, 5, 3))
        assert out.shape == (2, 1, 8)

    def test_gru_default_h0(self):
        model = Recurrent_nn(input_dim=3, hidden_size=8, rnn_type=RNNType.GRU)
        out, _ = model(torch.randn(2, 5, 3))
        assert out.shape == (2, 1, 8)

    def test_lstm_default_h0(self):
        """LSTM with init_hidden_states_0=False and h0=None should produce (h0, c0) tuple."""
        model = Recurrent_nn(input_dim=3, hidden_size=8, rnn_type=RNNType.LSTM)
        out, (hn, cn) = model(torch.randn(2, 5, 3))
        assert out.shape == (2, 1, 8)
        assert hn.shape[1] == 2
        assert cn.shape[1] == 2


class TestRecurrentNnBidirectional:
    def test_bidirectional_output_shape(self):
        input_dim, hidden_size, nb_out = 3, 8, 2
        model = Recurrent_nn(
            input_dim=input_dim, hidden_size=hidden_size,
            nb_output_consider=nb_out, rnn_type=RNNType.GRU,
            bidirectional=True,
        )
        x = torch.randn(4, 10, input_dim)
        out, _ = model(x)
        # Bidirectional concatenates both directions along nb_output_consider
        assert out.shape == (4, 2 * nb_out, hidden_size)

    def test_output_len_property_bidirectional(self):
        model = Recurrent_nn(
            input_dim=2, hidden_size=10,
            nb_output_consider=3, rnn_type=RNNType.LSTM,
            bidirectional=True,
        )
        # output_len = hidden_size * num_directions * nb_output_consider = 10 * 2 * 3
        assert model.output_len == 60


class TestRecurrentNnOutputLenProperty:
    def test_output_len_unidirectional(self):
        model = Recurrent_nn(
            input_dim=2, hidden_size=16,
            nb_output_consider=2, rnn_type=RNNType.RNN,
        )
        assert model.output_len == 16 * 1 * 2


class TestRecurrentNnInitHiddenStates:
    def test_init_hidden_states_are_parameters(self):
        """When init_hidden_states_0=True, hidden states should be learnable parameters."""
        model = Recurrent_nn(
            input_dim=2, hidden_size=8,
            rnn_type=RNNType.LSTM,
            init_hidden_states_0=True,
        )
        param_names = [n for n, _ in model.named_parameters()]
        assert any('hidden_state_parameters' in n for n in param_names)

    def test_no_init_hidden_states_no_extra_params(self):
        model = Recurrent_nn(
            input_dim=2, hidden_size=8,
            rnn_type=RNNType.LSTM,
            init_hidden_states_0=False,
        )
        param_names = [n for n, _ in model.named_parameters()]
        assert not any('hidden_state_parameters' in n for n in param_names)

    def test_forward_with_init_hidden_states_full_shape(self):
        """Forward pass works when init_hidden_states_0=True, with full shape check."""
        input_dim, hidden_size, nb_out = 3, 8, 1
        model = Recurrent_nn(
            input_dim=input_dim, hidden_size=hidden_size,
            nb_output_consider=nb_out,
            rnn_type=RNNType.LSTM,
            init_hidden_states_0=True,
        )
        x = torch.randn(2, 5, input_dim)
        out, (hn, cn) = model(x)
        assert out.shape == (2, nb_out, hidden_size)
        assert hn.shape[1] == 2  # batch size
        assert cn.shape[1] == 2


class TestRecurrentNnLSTMForgetGateBias:
    def test_forget_gate_bias_initialized_to_one(self):
        """After init_weights, the LSTM forget gate bias should be 1.0 (Gers et al.)."""
        model = Recurrent_nn(
            input_dim=3, hidden_size=8,
            rnn_type=RNNType.LSTM,
        )
        for name, param in model.stacked_rnn.named_parameters():
            if 'bias' in name:
                gate_size = param.shape[0] // 4
                forget_gate_bias = param.data[gate_size:2 * gate_size]
                assert torch.allclose(forget_gate_bias, torch.ones_like(forget_gate_bias)), \
                    f"Forget gate bias should be 1.0, got {forget_gate_bias}"

    def test_non_forget_gate_biases_are_zero(self):
        """All biases except the forget gate should be initialized to 0."""
        model = Recurrent_nn(
            input_dim=3, hidden_size=8,
            rnn_type=RNNType.LSTM,
        )
        for name, param in model.stacked_rnn.named_parameters():
            if 'bias' in name:
                gate_size = param.shape[0] // 4
                # Gates: input (0), forget (1), cell (2), output (3)
                for gate_idx in [0, 2, 3]:
                    gate_bias = param.data[gate_idx * gate_size:(gate_idx + 1) * gate_size]
                    assert torch.allclose(gate_bias, torch.zeros_like(gate_bias)), \
                        f"Gate {gate_idx} bias should be 0.0, got {gate_bias}"


class TestRecurrentNnForwardWithExplicitH0:
    def test_explicit_h0_gru(self):
        input_dim, hidden_size = 3, 8
        model = Recurrent_nn(
            input_dim=input_dim, hidden_size=hidden_size,
            rnn_type=RNNType.GRU,
        )
        x = torch.randn(2, 5, input_dim)
        h0 = torch.randn(1, 2, hidden_size)
        out, _ = model(x, h0=h0)
        assert out.shape == (2, 1, hidden_size)

    def test_explicit_h0_lstm(self):
        input_dim, hidden_size = 3, 8
        model = Recurrent_nn(
            input_dim=input_dim, hidden_size=hidden_size,
            rnn_type=RNNType.LSTM,
        )
        x = torch.randn(2, 5, input_dim)
        h0 = torch.randn(1, 2, hidden_size)
        c0 = torch.randn(1, 2, hidden_size)
        out, _ = model(x, h0=(h0, c0))
        assert out.shape == (2, 1, hidden_size)
