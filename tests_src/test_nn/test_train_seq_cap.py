"""Unit tests for the sigwgan/SigTPP train-only sequence-length cap (``train_seq_cap``).

Covers the train-capped / test-uncapped design in
docs/seq_length_capping/06-15-15_FEAT_seq_length_capping.md:
  - ``_truncate_to_cap`` keeps the anchor + first C events and clamps lengths.
  - The unconditional rollout is capped only in train mode; eval/test stay full-length.
  - A capped (or RESIDUAL) training reference builds a separate signature pipeline,
    while the full-length validation pipeline is left intact.
  - ``training_step`` produces a finite loss with a cap on both the unconditional and
    teacher-forced paths.
"""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import torch

from src.data_types.sigw_loss_data_props import SigWLossDataProps
from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.nn.architectures.architecture_one_to_one import ArchitectureOneToOne

# Synthetic dataset geometry: N sequences, each with EVENTS real events (no padding),
# so data_train is (N, EVENTS+1, 1) and the full unconditional rollout width is EVENTS.
N = 48
EVENTS = 12
CAP = 5  # < EVENTS so the cap actually bites


def _make_cumulative_times(n: int, events: int, seed: int) -> torch.Tensor:
    """(n, events+1, 1) cumulative times with a t=0 anchor at column 0."""
    g = torch.Generator().manual_seed(seed)
    inter = torch.rand(n, events, 1, generator=g) + 0.05  # strictly positive inter-arrivals
    cum = inter.cumsum(dim=1)
    anchor = torch.zeros(n, 1, 1)
    return torch.cat([anchor, cum], dim=1)


def _build_model(
    *,
    train_seq_cap=None,
    anchor=TerminalAnchorMode.FREE_ENDPOINT,
    detach_cum_channel=False,
):
    torch.manual_seed(0)
    data_train = _make_cumulative_times(N, EVENTS, seed=1)
    data_val = _make_cumulative_times(N, EVENTS, seed=2)
    lens = torch.full((N,), EVENTS + 1, dtype=torch.long)
    marks = torch.zeros(N, EVENTS + 1, dtype=torch.long)
    t_max = float(torch.ceil(data_train[:, -1, 0].max()).item()) + 1.0
    return ArchitectureOneToOne(
        data_train=data_train,
        data_train_lens=lens,
        data_val=data_val,
        data_val_lens=lens.clone(),
        train_marks=marks,
        val_marks=marks.clone(),
        period_plot_val=9999,
        loss_properties=SigWLossDataProps(4, False, True),
        learning_rate=1e-4,
        concentration_factor=1.0,
        hid_size_rep=8,
        use_teacher_forcing=False,
        t_max=t_max,
        num_marks=1,
        total_epochs=10,
        enable_plot=False,
        terminal_anchor_mode=anchor,
        detach_cum_channel=detach_cum_channel,
        train_seq_cap=train_seq_cap,
    )


def test_truncate_to_cap_keeps_anchor_and_clamps_lengths():
    data = _make_cumulative_times(6, EVENTS, seed=7)
    lens = torch.tensor([EVENTS + 1, EVENTS + 1, 3, EVENTS + 1, 2, EVENTS + 1], dtype=torch.long)

    capped, lens_capped = ArchitectureOneToOne._truncate_to_cap(data, lens, CAP)

    assert capped.shape == (6, CAP + 1, 1)  # anchor + CAP events
    assert capped.dtype == data.dtype
    assert torch.all(capped[:, 0, 0] == 0.0)  # t=0 anchor preserved
    assert torch.allclose(capped, data[:, : CAP + 1, :])  # first CAP events kept verbatim
    # Lengths clamped to CAP+1; sequences already shorter than the cap are unchanged.
    assert torch.equal(lens_capped, torch.clamp(lens, max=CAP + 1))
    assert lens_capped.tolist() == [CAP + 1, CAP + 1, 3, CAP + 1, 2, CAP + 1]


def test_truncate_to_cap_above_max_length_is_noop():
    data = _make_cumulative_times(4, EVENTS, seed=8)
    lens = torch.full((4,), EVENTS + 1, dtype=torch.long)

    capped, lens_capped = ArchitectureOneToOne._truncate_to_cap(data, lens, cap=EVENTS + 50)

    assert capped.shape == data.shape  # nothing dropped when cap >= max length
    assert torch.equal(lens_capped, lens)


def test_rollout_length_capped_in_train_mode_full_in_eval():
    model = _build_model(train_seq_cap=CAP)
    full_width = model.data_train_dts.shape[1] + 1  # full unconditional rollout width

    model.train()
    with torch.no_grad():
        train_samples, _, _ = model.sample(num_seq=10)

    model.eval()
    with torch.no_grad():
        eval_samples, _, _ = model.sample(num_seq=10)

    assert train_samples.shape[1] == CAP  # capped: anchor-seed + (CAP-1) generated steps
    assert eval_samples.shape[1] == full_width  # uncapped at eval/test/sampling time
    assert full_width > CAP  # sanity: the cap genuinely shortens training rollouts


def test_uncapped_model_rolls_out_full_length_in_train_mode():
    model = _build_model(train_seq_cap=None)
    full_width = model.data_train_dts.shape[1] + 1

    model.train()
    with torch.no_grad():
        samples, _, _ = model.sample(num_seq=10)

    assert samples.shape[1] == full_width


def test_capped_free_endpoint_builds_separate_training_pipeline():
    capped = _build_model(train_seq_cap=CAP, anchor=TerminalAnchorMode.FREE_ENDPOINT)
    uncapped = _build_model(train_seq_cap=None, anchor=TerminalAnchorMode.FREE_ENDPOINT)

    # Uncapped + FREE_ENDPOINT reuses the metrics pipeline (fast path): the training scaler
    # and total-variation are the very objects built for the full-length metrics pipeline.
    assert uncapped._scaler_std_train is uncapped.scaler_std
    assert uncapped._total_vars_train == uncapped.total_vars

    # Capped builds a dedicated training reference scaler, leaving the full-length validation
    # pipeline (scaler_std / sigw1metric_val) untouched.
    assert capped._scaler_std_train is not capped.scaler_std
    assert capped.train_seq_cap == CAP


@pytest.mark.parametrize("use_teacher_forcing", [False, True])
def test_training_step_with_cap_returns_finite_loss(use_teacher_forcing):
    model = _build_model(train_seq_cap=CAP)
    model.use_teacher_forcing = use_teacher_forcing  # True => teacher-forced path (epoch 0 ratio = 1.0)
    model.train()

    data_train = _make_cumulative_times(N, EVENTS, seed=3)
    lens = torch.full((N,), EVENTS + 1, dtype=torch.long)
    marks = torch.zeros(N, EVENTS + 1, dtype=torch.long)
    batch = (data_train, lens, marks)

    loss = model.training_step(batch, 0)

    assert torch.is_tensor(loss)
    assert loss.shape == ()
    assert torch.isfinite(loss).item()


def test_cap_training_batch_truncates_all_three_fields():
    model = _build_model(train_seq_cap=CAP)
    data_train = _make_cumulative_times(N, EVENTS, seed=4)
    lens = torch.full((N,), EVENTS + 1, dtype=torch.long)
    marks = torch.arange(N * (EVENTS + 1), dtype=torch.long).reshape(N, EVENTS + 1)

    data_c, lens_c, marks_c = model._cap_training_batch((data_train, lens, marks))

    assert data_c.shape == (N, CAP + 1, 1)
    assert marks_c.shape == (N, CAP + 1)
    assert torch.all(lens_c == CAP + 1)
    assert torch.equal(marks_c, marks[:, : CAP + 1])


def test_capped_residual_anchor_constructs_and_caps_rollout():
    # RESIDUAL requires detach_cum_channel=True; exercises the capped separate-pipeline
    # branch under a non-FREE_ENDPOINT training anchor.
    model = _build_model(
        train_seq_cap=CAP,
        anchor=TerminalAnchorMode.RESIDUAL,
        detach_cum_channel=True,
    )
    assert model._scaler_std_train is not model.scaler_std

    model.train()
    with torch.no_grad():
        samples, _, _ = model.sample(num_seq=8)
    assert samples.shape[1] == CAP
