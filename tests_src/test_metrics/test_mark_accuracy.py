"""Tests for mark accuracy and mark padding masking in baseline classifiers."""

import torch
import pytest

from src.metrics.mark_accuracy import top_k_accuracy
from src.nn.architectures.mark_prediction_utils import MARK_IGNORE_INDEX, compute_majority_class


class TestTopKAccuracy:
    """Tests for the top_k_accuracy function."""

    def test_top1_perfect(self):
        logits = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])  # (1, 2, 2)
        targets = torch.tensor([[0, 1]])  # (1, 2)
        assert top_k_accuracy(logits, targets, k=1) == 1.0

    def test_top1_wrong(self):
        logits = torch.tensor([[[10.0, 0.0], [10.0, 0.0]]])  # always predicts 0
        targets = torch.tensor([[1, 1]])  # all targets are 1
        assert top_k_accuracy(logits, targets, k=1) == 0.0

    def test_ignore_index_excludes_padding(self):
        logits = torch.tensor([[[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]]])  # (1, 3, 2)
        targets = torch.tensor([[0, MARK_IGNORE_INDEX, MARK_IGNORE_INDEX]])  # only position 0 is valid
        assert top_k_accuracy(logits, targets, k=1) == 1.0

    def test_all_ignored(self):
        logits = torch.tensor([[[10.0, 0.0]]])
        targets = torch.tensor([[MARK_IGNORE_INDEX]])
        assert top_k_accuracy(logits, targets, k=1) == 0.0


class TestMarkPaddingMasking:
    """Verify that padding positions (mark=0) are correctly excluded
    from both bincount and accuracy computations in baseline classifiers.

    This is the test for Critical-1 and Critical-2 from code review:
    padded mark positions have value 0 (not -1), so they must be excluded
    via sequence-length masking before any computation.
    """

    def test_bincount_excludes_padding(self):
        """Simulates the bincount pattern used in Deter/Gamma __init__:
        only valid (non-padded) positions should be counted."""
        num_marks = 3
        # marks_with_anchor: (N=3, L+1=5) — anchor at pos 0, then 4 event positions
        # Sequence lengths (including anchor): [4, 3, 2]
        # So valid event positions: seq0 has 3, seq1 has 2, seq2 has 1
        marks_with_anchor = torch.tensor([
            [0, 1, 2, 1, 0],  # seq 0: valid marks at pos 1-3 → [1, 2, 1]; pos 4 is padding (0)
            [0, 2, 2, 0, 0],  # seq 1: valid marks at pos 1-2 → [2, 2]; pos 3-4 are padding (0)
            [0, 1, 0, 0, 0],  # seq 2: valid mark at pos 1 → [1]; pos 2-4 are padding (0)
        ])
        data_lens = torch.tensor([4, 3, 2])  # includes anchor

        # Correct masking approach (what the fixed code does)
        marks_events = marks_with_anchor[:, 1:]  # (3, 4)
        L = marks_events.shape[1]
        pos = torch.arange(L).unsqueeze(0)
        valid = pos < (data_lens - 1).unsqueeze(1)
        valid_marks_flat = marks_events[valid]
        counts = torch.bincount(valid_marks_flat, minlength=num_marks)

        # Valid marks: [1, 2, 1, 2, 2, 1] → counts: {0: 0, 1: 3, 2: 3}
        assert counts[0].item() == 0, "Category 0 should have zero count (no valid mark is 0)"
        assert counts[1].item() == 3
        assert counts[2].item() == 3

        # BUG (what the old code would do): reshape(-1) includes padding
        bad_counts = torch.bincount(marks_events.reshape(-1), minlength=num_marks)
        assert bad_counts[0].item() > 0, "Without masking, category 0 is inflated by padding"

    def test_accuracy_excludes_padding(self):
        """Verify that mark accuracy only counts valid (non-padded) positions."""
        num_marks = 3
        # marks_with_anchor: (N=2, L+1=4)
        marks_with_anchor = torch.tensor([
            [0, 1, 2, 0],  # seq 0: valid marks at pos 1-2 → [1, 2]; pos 3 is padding (0)
            [0, 0, 0, 0],  # seq 1: valid mark at pos 1 → [0]; pos 2-3 are padding (0)
        ])
        data_lens = torch.tensor([3, 2])  # includes anchor

        mark_targets = marks_with_anchor[:, 1:].clone()  # (2, 3)
        N, L = mark_targets.shape
        pos = torch.arange(L).unsqueeze(0)
        valid = pos < (data_lens - 1).unsqueeze(1)
        mark_targets[~valid] = MARK_IGNORE_INDEX

        # Majority class classifier predicts 1 everywhere
        logits = torch.zeros(N, L, num_marks)
        logits[:, :, 1] = 1.0

        acc = top_k_accuracy(logits, mark_targets, k=1)
        # Valid positions: seq0 pos0=1 (correct), seq0 pos1=2 (wrong), seq1 pos0=0 (wrong)
        # 1 correct out of 3 valid = 1/3
        assert abs(acc - 1 / 3) < 1e-6

    def test_accuracy_without_masking_gives_wrong_result(self):
        """Show that without masking, accuracy is inflated by padding positions."""
        num_marks = 3
        marks_with_anchor = torch.tensor([
            [0, 1, 2, 0],  # padding at pos 3 (mark=0)
            [0, 0, 0, 0],  # padding at pos 2-3 (mark=0)
        ])
        data_lens = torch.tensor([3, 2])

        mark_targets_unmasked = marks_with_anchor[:, 1:]  # (2, 3) — padding stays as 0
        N, L = mark_targets_unmasked.shape

        # Predict class 0 everywhere
        logits = torch.zeros(N, L, num_marks)
        logits[:, :, 0] = 1.0

        acc_unmasked = top_k_accuracy(logits, mark_targets_unmasked, k=1)
        # Without masking: all 6 positions evaluated, mark is 0 at positions
        # [1, 2, 0, 0, 0, 0] — predicting 0 gets: [wrong, wrong, right, right, right, right] = 4/6
        assert abs(acc_unmasked - 4 / 6) < 1e-6

        # With masking: only 3 valid positions
        mark_targets_masked = mark_targets_unmasked.clone()
        pos = torch.arange(L).unsqueeze(0)
        valid = pos < (data_lens - 1).unsqueeze(1)
        mark_targets_masked[~valid] = MARK_IGNORE_INDEX

        acc_masked = top_k_accuracy(logits, mark_targets_masked, k=1)
        # Valid positions: [1, 2, 0] — predicting 0 gets: [wrong, wrong, right] = 1/3
        assert abs(acc_masked - 1 / 3) < 1e-6

        # The masked accuracy is different (and correct)
        assert acc_unmasked != acc_masked


class TestComputeMajorityClass:
    """Tests for the compute_majority_class algorithm used in
    baseline mark classifiers (deter, gamma)."""

    def test_normal_case_returns_most_frequent(self):
        """Majority class is the one with the most valid occurrences."""
        torch.manual_seed(0)
        # marks_with_anchor: (N=3, L+1=5)
        # Sequence lengths (including anchor): [5, 4, 3]
        # Valid event marks: seq0=[1,2,1,1], seq1=[2,1,1], seq2=[1,2]
        # Counts: {0:0, 1:5, 2:3} → majority = 1
        marks = torch.tensor([
            [0, 1, 2, 1, 1],
            [0, 2, 1, 1, 0],  # pos 3 is padding (len=4, so 3 events)
            [0, 1, 2, 0, 0],  # pos 2-3 are padding (len=3, so 2 events)
        ])
        lens = torch.tensor([5, 4, 3])
        result = compute_majority_class(marks, lens, num_marks=3)
        assert result == 1

    def test_single_mark_class(self):
        """When only one mark class exists, it is returned."""
        marks = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        lens = torch.tensor([4, 3])
        result = compute_majority_class(marks, lens, num_marks=1)
        assert result == 0

    def test_tie_returns_first(self):
        """When counts are tied, argmax returns the lowest index."""
        # Valid marks: [0, 1] → counts: {0:1, 1:1} → argmax → 0
        marks = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],  # pos 1 padding (len=2 → 1 event)
        ])
        lens = torch.tensor([3, 2])
        result = compute_majority_class(marks, lens, num_marks=2)
        assert result == 0

    def test_empty_valid_marks_returns_zero(self):
        """Edge case: all sequences have length <= 1 → no valid marks.
        bincount returns all zeros, argmax returns 0."""
        marks = torch.tensor([
            [0, 0],
            [0, 0],
        ])
        lens = torch.tensor([1, 1])  # 0 inter-arrival times → no valid event marks
        result = compute_majority_class(marks, lens, num_marks=3)
        assert result == 0  # argmax of all-zeros returns 0

    def test_device_consistency(self):
        """Verify helper works when marks are on a specific device (CPU)."""
        marks = torch.tensor([[0, 2, 2, 0]], device='cpu')
        lens = torch.tensor([3], device='cpu')
        result = compute_majority_class(marks, lens, num_marks=3)
        # marks[:, 1:] = [2, 2, 0], valid positions: pos < (3-1)=2 → [2, 2]
        # counts: {0:0, 1:0, 2:2} → majority = 2
        assert result == 2

    def test_padding_zeros_excluded(self):
        """Single-event sequences contribute no next-mark targets."""
        # Seq of len 2 (1 event) has no next-mark prediction target.
        marks = torch.tensor([[0, 1, 0, 0, 0]])
        lens = torch.tensor([2])
        result = compute_majority_class(marks, lens, num_marks=2)
        assert result == 0  # all-zero counts -> argmax returns 0
