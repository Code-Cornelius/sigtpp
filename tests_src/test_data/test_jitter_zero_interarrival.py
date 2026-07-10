"""Unit tests for jitter_zero_interarrival_times in RealWorldDataModule."""

import torch
import pytest

from test.paper_experiments.data.real.real_world_dataset import RealWorldDataModule

jitter = RealWorldDataModule.jitter_zero_interarrival_times


class TestNoMarksPath:
    """Tests for the no-marks (jitter-only) path."""

    def test_clean_sequences_unchanged(self):
        """Sequences with no zero inter-arrivals should pass through unchanged."""
        torch.manual_seed(0)
        inputs = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]]).float()
        inputs_len = torch.tensor([4], dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, seed=0)

        assert out_len[0].item() == 4
        assert torch.allclose(out, inputs)

    def test_zero_inter_arrivals_are_jittered(self):
        """Zero inter-arrivals should be replaced with positive values."""
        torch.manual_seed(0)
        # Two events at time 1.0 (zero inter-arrival at position 2)
        inputs = torch.tensor([[[0.0], [1.0], [1.0], [2.0]]]).float()
        inputs_len = torch.tensor([4], dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, seed=42)

        # inter-arrivals should all be >= jitter_min
        cum = out[0, :4, 0]
        inter = cum[1:] - cum[:-1]
        assert (inter >= 1e-6).all(), f"Expected all inter >= 1e-6, got {inter}"

    def test_jitter_is_random_not_deterministic(self):
        """Two calls with different seeds should produce different jitter values."""
        inputs = torch.tensor([[[0.0], [1.0], [1.0], [2.0]]]).float()
        inputs_len = torch.tensor([4], dtype=torch.long)

        out1, _ = jitter(inputs, inputs_len, seed=1)
        out2, _ = jitter(inputs, inputs_len, seed=2)

        assert not torch.allclose(out1, out2), "Different seeds should produce different jitter"

    def test_jitter_reproducible_with_same_seed(self):
        """Same seed should produce identical results."""
        inputs = torch.tensor([[[0.0], [1.0], [1.0], [2.0]]]).float()
        inputs_len = torch.tensor([4], dtype=torch.long)

        out1, _ = jitter(inputs, inputs_len, seed=42)
        out2, _ = jitter(inputs, inputs_len, seed=42)

        assert torch.allclose(out1, out2), "Same seed should produce identical results"

    def test_padding_updated_correctly(self):
        """Padding beyond seq_len should reflect the last valid cumulative time."""
        # seq_len=3 in a padded_length=5 tensor
        inputs = torch.tensor([[[0.0], [1.0], [1.0], [1.0], [1.0]]]).float()
        inputs_len = torch.tensor([3], dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, seed=0)

        last_valid = out[0, 2, 0].item()
        assert out[0, 3, 0].item() == pytest.approx(last_valid)
        assert out[0, 4, 0].item() == pytest.approx(last_valid)

    def test_monotonicity_preserved(self):
        """Cumulative times must remain monotonically non-decreasing after jitter."""
        torch.manual_seed(0)
        inputs = torch.tensor([[[0.0], [1.0], [1.0], [1.0], [3.0]]]).float()
        inputs_len = torch.tensor([5], dtype=torch.long)

        out, _ = jitter(inputs, inputs_len, seed=7)

        cum = out[0, :5, 0]
        inter = cum[1:] - cum[:-1]
        assert (inter >= 0).all(), f"Monotonicity violated: {inter}"


class TestSameMarkPath:
    """Tests for mark-aware duplicate removal."""

    def test_same_mark_zero_inter_removed(self):
        """Same-mark events at the same time should be removed, not jittered."""
        inputs = torch.tensor([[[0.0], [1.0], [1.0], [2.0]]]).float()
        inputs_len = torch.tensor([4], dtype=torch.long)
        # marks: event at pos 1 and 2 have same mark (5)
        marks = torch.tensor([[0, 5, 5, 3]], dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, inputs_marks=marks, seed=0)

        # One event removed → seq_len decreases by 1
        assert out_len[0].item() == 3
        # Remaining times: [0.0, 1.0, 2.0]
        assert out[0, 0, 0].item() == pytest.approx(0.0)
        assert out[0, 1, 0].item() == pytest.approx(1.0)
        assert out[0, 2, 0].item() == pytest.approx(2.0)

    def test_chain_of_same_mark_duplicates(self):
        """Chain of 3 same-mark events at same time → only one survives."""
        inputs = torch.tensor([[[0.0], [1.0], [1.0], [1.0], [2.0]]]).float()
        inputs_len = torch.tensor([5], dtype=torch.long)
        marks = torch.tensor([[0, 5, 5, 5, 3]], dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, inputs_marks=marks, seed=0)

        # Two removals → seq_len = 3
        assert out_len[0].item() == 3
        assert out[0, 0, 0].item() == pytest.approx(0.0)
        assert out[0, 1, 0].item() == pytest.approx(1.0)
        assert out[0, 2, 0].item() == pytest.approx(2.0)


class TestDifferentMarkPath:
    """Tests for different-mark events at the same time (should be jittered, not removed)."""

    def test_different_mark_zero_inter_jittered(self):
        """Different-mark events at same time should be jittered, not removed."""
        inputs = torch.tensor([[[0.0], [1.0], [1.0], [2.0]]]).float()
        inputs_len = torch.tensor([4], dtype=torch.long)
        marks = torch.tensor([[0, 3, 7, 3]], dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, inputs_marks=marks, seed=42)

        # No removal: seq_len stays the same
        assert out_len[0].item() == 4
        # But inter-arrivals should now all be positive
        cum = out[0, :4, 0]
        inter = cum[1:] - cum[:-1]
        assert (inter >= 1e-6).all(), f"Expected all inter >= 1e-6, got {inter}"


class TestCleanSequenceFastPath:
    """Sequences with no zeros should be untouched (fast path)."""

    def test_batch_with_mixed_clean_and_dirty(self):
        """Only dirty sequences should be modified."""
        torch.manual_seed(0)
        inputs = torch.tensor(
            [
                [[0.0], [1.0], [2.0], [3.0]],  # clean
                [[0.0], [1.0], [1.0], [2.0]],  # dirty
            ]
        ).float()
        inputs_len = torch.tensor([4, 4], dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, seed=0)

        # Clean sequence should be unchanged
        assert torch.allclose(out[0], inputs[0])
        assert out_len[0].item() == 4

    def test_short_sequence_skipped(self):
        """Sequence with seq_len <= 1 should be skipped."""
        inputs = torch.tensor([[[0.0], [0.0]]]).float()
        inputs_len = torch.tensor([1], dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, seed=0)

        assert out_len[0].item() == 1


class TestReturnTypes:
    """Verify return shapes and dtypes."""

    def test_return_shapes(self):
        torch.manual_seed(0)
        N, L_plus_1 = 5, 10
        inputs = torch.rand(N, L_plus_1, 1).cumsum(dim=1)
        inputs_len = torch.full((N,), L_plus_1, dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, seed=0)

        assert out.shape == inputs.shape
        assert out.dtype == torch.float32
        assert out_len.shape == (N,)
        assert out_len.dtype == torch.long

    def test_return_shapes_with_marks(self):
        torch.manual_seed(0)
        N, L_plus_1 = 3, 6
        inputs = torch.rand(N, L_plus_1, 1).cumsum(dim=1)
        inputs_len = torch.full((N,), L_plus_1, dtype=torch.long)
        marks = torch.randint(0, 5, (N, L_plus_1), dtype=torch.long)

        out, out_len = jitter(inputs, inputs_len, inputs_marks=marks, seed=0)

        assert out.shape == inputs.shape
        assert out_len.shape == (N,)
