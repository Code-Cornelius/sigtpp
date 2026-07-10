import torch
import torch.nn.functional as F

from src.nn.architectures.mark_prediction_utils import (
    MARK_IGNORE_INDEX,
    prepare_next_mark_targets,
    compute_mark_ce_from_logits,
    compute_next_mark_accuracy_metrics,
    build_majority_class_logits,
)
from src.plot.mark_plots import mask_mark_sequences


def test_prepare_next_mark_targets_masks_padding():
    marks = torch.tensor(
        [
            [1, 2, 0, 1],
            [2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([4, 2], dtype=torch.long)

    targets = prepare_next_mark_targets(marks, lengths)

    expected = torch.tensor(
        [
            [2, 0, 1],
            [1, MARK_IGNORE_INDEX, MARK_IGNORE_INDEX],
        ],
        dtype=torch.long,
    )
    assert torch.equal(targets, expected)


def test_compute_mark_ce_from_logits_uses_masked_targets():
    logits = torch.tensor(
        [
            [[0.1, 3.0, 0.1], [2.5, 0.2, 0.1], [0.2, 0.1, 2.8]],
            [[0.1, 2.7, 0.1], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    marks = torch.tensor(
        [
            [0, 1, 0, 2],
            [0, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([4, 2], dtype=torch.long)

    loss = compute_mark_ce_from_logits(logits, marks, lengths)

    expected_targets = torch.tensor(
        [
            [1, 0, 2],
            [1, MARK_IGNORE_INDEX, MARK_IGNORE_INDEX],
        ],
        dtype=torch.long,
    )
    expected_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        expected_targets.reshape(-1),
        ignore_index=MARK_IGNORE_INDEX,
    )
    assert torch.isclose(loss, expected_loss)


def test_compute_next_mark_accuracy_metrics_preserves_keys_and_masking():
    logits = torch.tensor(
        [
            [[5.0, 1.0, 0.0], [1.0, 5.0, 0.0], [0.0, 1.0, 5.0]],
            [[0.0, 5.0, 1.0], [1.0, 0.0, 5.0], [5.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    marks = torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([4, 2], dtype=torch.long)

    metrics = compute_next_mark_accuracy_metrics(logits, marks, lengths)

    assert set(metrics) == {"top1_mark_acc", "top3_mark_acc"}
    assert metrics["top1_mark_acc"] == 1.0
    assert metrics["top3_mark_acc"] == 1.0


def test_prepare_mark_plot_payload_masks_only_mark_tensors():
    logits = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    previous_marks = torch.tensor([[0, 1, 0]], dtype=torch.long)
    current_targets = torch.tensor([[1, 0, 1]], dtype=torch.long)
    valid_lengths = torch.tensor([2], dtype=torch.long)

    out_logits, out_previous, out_targets = (
        logits,
        *mask_mark_sequences(
            previous_marks,
            current_targets,
            valid_lengths,
        ),
    )

    assert torch.equal(out_logits, logits)
    assert torch.equal(
        out_previous,
        torch.tensor([[0, 1, MARK_IGNORE_INDEX]], dtype=torch.long),
    )
    assert torch.equal(
        out_targets,
        torch.tensor([[1, 0, MARK_IGNORE_INDEX]], dtype=torch.long),
    )


def test_build_majority_class_logits_softmax_is_concentrated():
    mark_targets = torch.zeros((2, 3), dtype=torch.long)
    logits = build_majority_class_logits(mark_targets, num_marks=3, majority_class=1)
    probs = torch.softmax(logits, dim=-1)

    assert torch.allclose(probs[..., 1], torch.ones_like(probs[..., 1]))
    assert torch.allclose(probs[..., 0], torch.zeros_like(probs[..., 0]))
    assert torch.allclose(probs[..., 2], torch.zeros_like(probs[..., 2]))
