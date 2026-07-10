"""Tests for embedding modules: TrigoTimeEmbedding, MarkEmbedding, EventEmbedding, PositionEmbedding."""

import math

import pytest
import torch

from src.nn.embeddings.time import TrigoTimeEmbedding
from src.nn.embeddings.mark import MarkEmbedding
from src.nn.embeddings.event import EventEmbedding
from src.nn.embeddings.position import PositionEmbedding


# ---------------------------------------------------------------------------
# TrigoTimeEmbedding
# ---------------------------------------------------------------------------
class TestTrigoTimeEmbedding:
    def test_output_shape(self):
        torch.manual_seed(0)
        emb = TrigoTimeEmbedding(embed_size=8, min_time=0.0, max_time=10.0)
        times = torch.rand(4, 12, 1) * 10  # (N, L, 1)
        out = emb(times)
        assert out.shape == (4, 12, 8)

    def test_odd_embed_size_raises(self):
        with pytest.raises(AssertionError):
            TrigoTimeEmbedding(embed_size=7)

    def test_zero_embed_size_raises(self):
        with pytest.raises(AssertionError):
            TrigoTimeEmbedding(embed_size=0)

    def test_min_geq_max_raises(self):
        with pytest.raises(AssertionError):
            TrigoTimeEmbedding(embed_size=4, min_time=5.0, max_time=5.0)

    def test_min_gt_max_raises(self):
        with pytest.raises(AssertionError):
            TrigoTimeEmbedding(embed_size=4, min_time=10.0, max_time=5.0)

    def test_output_scaled_by_sqrt_embed_size(self):
        """Output magnitude should be bounded by 1/sqrt(embed_size) * sqrt(embed_size) = 1
        since sin/cos are bounded by 1."""
        torch.manual_seed(0)
        embed_size = 16
        emb = TrigoTimeEmbedding(embed_size=embed_size, learnable_weights=False)
        times = torch.tensor([[[0.5]]])
        out = emb(times)
        # Each element is sin or cos divided by sqrt(embed_size), so |elem| <= 1/sqrt(embed_size)
        bound = 1.0 / math.sqrt(embed_size)
        assert (out.abs() <= bound + 1e-6).all(), f"Elements exceed bound {bound}"

    def test_sin_cos_split(self):
        """First half of output should be sin-derived, second half cos-derived."""
        torch.manual_seed(0)
        embed_size = 8
        emb = TrigoTimeEmbedding(embed_size=embed_size, learnable_weights=False)
        times = torch.tensor([[[0.25]]])  # t=0.25, rescaled to 0.25
        out = emb(times).squeeze()
        # With non-learnable weights, phi = times_rescaled * weights
        # pe = [sin(phi * 2π), cos(phi * 2π)] / sqrt(embed_size)
        # Just verify the two halves differ (sin vs cos at same phase produce different values generically)
        first_half = out[: embed_size // 2]
        second_half = out[embed_size // 2 :]
        assert not torch.allclose(first_half, second_half), "sin and cos halves should differ"

    def test_set_extrema_times(self):
        torch.manual_seed(0)
        emb = TrigoTimeEmbedding(embed_size=4, min_time=0.0, max_time=1.0)
        emb.set_extrema_times(0.0, 100.0)
        assert emb.min_time == 0.0
        assert emb.max_time == 100.0

    def test_learnable_vs_nonlearnable_shapes_match(self):
        torch.manual_seed(0)
        emb_learn = TrigoTimeEmbedding(embed_size=6, learnable_weights=True)
        emb_fixed = TrigoTimeEmbedding(embed_size=6, learnable_weights=False)
        times = torch.rand(2, 5, 1)
        assert emb_learn(times).shape == emb_fixed(times).shape

    def test_deterministic_nonlearnable(self):
        """Non-learnable embedding should be deterministic for same input."""
        emb = TrigoTimeEmbedding(embed_size=4, learnable_weights=False)
        times = torch.tensor([[[0.3]], [[0.7]]])
        out1 = emb(times)
        out2 = emb(times)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# MarkEmbedding
# ---------------------------------------------------------------------------
class TestMarkEmbedding:
    def test_output_shape(self):
        torch.manual_seed(0)
        emb = MarkEmbedding(num_mark_types=5, mark_emb_size=8)
        marks = torch.randint(0, 5, (3, 10))  # (N, L)
        out = emb(marks)
        assert out.shape == (3, 10, 8)

    def test_same_mark_same_embedding(self):
        torch.manual_seed(0)
        emb = MarkEmbedding(num_mark_types=3, mark_emb_size=4)
        marks = torch.tensor([[0, 1, 0, 2, 1]])
        out = emb(marks)
        assert torch.allclose(out[0, 0], out[0, 2]), "Same mark type should produce same embedding"
        assert torch.allclose(out[0, 1], out[0, 4]), "Same mark type should produce same embedding"

    def test_different_marks_different_embeddings(self):
        torch.manual_seed(0)
        emb = MarkEmbedding(num_mark_types=3, mark_emb_size=4)
        marks = torch.tensor([[0, 1, 2]])
        out = emb(marks)
        # After random init, different indices should almost surely have different embeddings
        assert not torch.allclose(out[0, 0], out[0, 1])
        assert not torch.allclose(out[0, 1], out[0, 2])

    def test_out_of_range_mark_raises(self):
        emb = MarkEmbedding(num_mark_types=3, mark_emb_size=4)
        marks = torch.tensor([[3]])  # index 3 is out of range for num_mark_types=3
        with pytest.raises(IndexError):
            emb(marks)


# ---------------------------------------------------------------------------
# EventEmbedding
# ---------------------------------------------------------------------------
class TestEventEmbedding:
    def test_output_shape(self):
        torch.manual_seed(0)
        time_emb = TrigoTimeEmbedding(embed_size=6)
        mark_emb = MarkEmbedding(num_mark_types=4, mark_emb_size=8)
        event_emb = EventEmbedding(time_emb, mark_emb)
        times = torch.rand(2, 5, 1)
        marks = torch.randint(0, 4, (2, 5))
        out = event_emb(times, marks)
        assert out.shape == (2, 5, 6 + 8)

    def test_embed_size_property(self):
        torch.manual_seed(0)
        time_emb = TrigoTimeEmbedding(embed_size=10)
        mark_emb = MarkEmbedding(num_mark_types=3, mark_emb_size=6)
        event_emb = EventEmbedding(time_emb, mark_emb)
        assert event_emb.embed_size == 16

    def test_concatenation_order(self):
        """First columns should be time embeddings, last columns should be mark embeddings."""
        torch.manual_seed(0)
        time_emb = TrigoTimeEmbedding(embed_size=4, learnable_weights=False)
        mark_emb = MarkEmbedding(num_mark_types=3, mark_emb_size=6)
        event_emb = EventEmbedding(time_emb, mark_emb)
        times = torch.tensor([[[0.5]]])
        marks = torch.tensor([[1]])
        out = event_emb(times, marks)
        # Verify by computing each part independently
        t_part = time_emb(times)
        m_part = mark_emb(marks)
        expected = torch.cat([t_part, m_part], dim=-1)
        assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# PositionEmbedding
# ---------------------------------------------------------------------------
class TestPositionEmbedding:
    def test_output_shape_from_sequence(self):
        torch.manual_seed(0)
        pos_emb = PositionEmbedding(embed_size=8, max_length=100)
        seq = torch.randn(3, 10)  # (N, L)
        out = pos_emb(sequence=seq)
        assert out.shape == (3, 10, 8)

    def test_output_shape_from_positions(self):
        torch.manual_seed(0)
        pos_emb = PositionEmbedding(embed_size=6, max_length=50)
        positions = torch.arange(5, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, 5, 1)
        out = pos_emb(positions=positions)
        assert out.shape == (1, 5, 6)

    def test_both_inputs_raises(self):
        pos_emb = PositionEmbedding(embed_size=4, max_length=10)
        seq = torch.randn(1, 5)
        positions = torch.arange(5, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with pytest.raises(AssertionError):
            pos_emb(sequence=seq, positions=positions)

    def test_neither_input_raises(self):
        pos_emb = PositionEmbedding(embed_size=4, max_length=10)
        with pytest.raises(AssertionError):
            pos_emb()

    def test_embed_size_property(self):
        pos_emb = PositionEmbedding(embed_size=12, max_length=50)
        assert pos_emb.embed_size == 12

    def test_same_length_same_embeddings(self):
        """Position embeddings should be identical for sequences of the same length regardless of content."""
        torch.manual_seed(0)
        pos_emb = PositionEmbedding(embed_size=6, max_length=20)
        seq1 = torch.randn(1, 8)
        seq2 = torch.randn(1, 8)
        out1 = pos_emb(sequence=seq1)
        out2 = pos_emb(sequence=seq2)
        assert torch.allclose(out1, out2), "Position embeddings should not depend on sequence content"

    def test_different_positions_different_embeddings(self):
        """Different positions should (almost surely) produce different embeddings."""
        torch.manual_seed(0)
        pos_emb = PositionEmbedding(embed_size=6, max_length=100)
        positions = torch.tensor([[[0.0], [1.0], [2.0]]])
        out = pos_emb(positions=positions)
        assert not torch.allclose(out[0, 0], out[0, 1])
        assert not torch.allclose(out[0, 1], out[0, 2])
