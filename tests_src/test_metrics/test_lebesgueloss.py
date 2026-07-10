import torch

from src.metrics.lebesgue_loss import (
    get_L1loss,
    get_perc_L1loss,
    get_L2loss_weighted_by_targets,
    get_L1loss_weighted_by_targets,
    get_L1loss_conditional_weighted_by_targets,
    get_perc_L1loss_weighted_by_targets,
)


class TestGetL1Loss:
    def test_identical_tensors_zero(self):
        a = torch.randn(4, 5, 2)
        loss = get_L1loss(a, a)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_known_value(self):
        a = torch.tensor([[[1.0, 2.0]]])
        b = torch.tensor([[[3.0, 5.0]]])
        # |1-3| + |2-5| = 2 + 3 = 5, mean = 2.5
        loss = get_L1loss(a, b)
        assert torch.isclose(loss, torch.tensor(2.5), atol=1e-6)

    def test_nan_excluded(self):
        a = torch.tensor([[[1.0], [float('nan')]]])
        b = torch.tensor([[[3.0], [100.0]]])
        # NaN in (a - b) at position [0,1,0], nanmean should ignore it
        loss = get_L1loss(a, b)
        assert torch.isfinite(loss)
        assert torch.isclose(loss, torch.tensor(2.0), atol=1e-6)

    def test_symmetry(self):
        a = torch.tensor([[[1.0, 3.0]]])
        b = torch.tensor([[[4.0, 1.0]]])
        assert torch.isclose(get_L1loss(a, b), get_L1loss(b, a), atol=1e-6)


class TestGetPercL1Loss:
    def test_identical_tensors_zero(self):
        a = torch.randn(4, 5, 2) + 1.0
        loss = get_perc_L1loss(a, a)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_known_value(self):
        a = torch.tensor([[[2.0]]])
        b = torch.tensor([[[4.0]]])
        # |2-4| / (|4| + 1e-6) = 2 / 4.000001 ≈ 0.5
        loss = get_perc_L1loss(a, b)
        assert torch.isclose(loss, torch.tensor(0.5), atol=1e-4)

    def test_double_the_value_gives_100_percent(self):
        a = torch.tensor([[[10.0]]])
        b = torch.tensor([[[5.0]]])
        # |10-5| / (|5| + 1e-6) = 5/5 ≈ 1.0
        loss = get_perc_L1loss(a, b)
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-4)


class TestWeightedLosses:
    def test_l2_weighted_identical_zero(self):
        a = torch.randn(4, 5, 2)
        loss = get_L2loss_weighted_by_targets(a, a)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_l1_weighted_identical_zero(self):
        a = torch.randn(4, 5, 2)
        loss = get_L1loss_weighted_by_targets(a, a)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_perc_l1_weighted_identical_zero(self):
        a = torch.randn(4, 5, 2) + 1.0
        loss = get_perc_L1loss_weighted_by_targets(a, a)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_l1_weighted_nan_excluded_with_value_check(self):
        """NaN in targets should be excluded, and loss computed on valid entries only."""
        a = torch.tensor([[[1.0], [2.0], [3.0]]])
        b = torch.tensor([[[1.0], [float('nan')], [5.0]]])
        loss = get_L1loss_weighted_by_targets(a, b)
        assert torch.isfinite(loss)
        # Valid entries: (1,1) -> |0|=0 and (3,5) -> |2|=2
        # Weighted mean of these should be positive
        assert loss.item() > 0.0

    def test_l2_weighted_known_value(self):
        """Known exact value: all entries valid, uniform weighting."""
        a = torch.tensor([[[1.0], [2.0]]])
        b = torch.tensor([[[2.0], [4.0]]])
        # elem = (1-2)^2=1, (2-4)^2=4
        loss = get_L2loss_weighted_by_targets(a, b)
        assert loss.item() > 0.0
        # Mean of [1, 4] = 2.5
        assert torch.isclose(loss, torch.tensor(2.5), atol=1e-5)

    def test_all_nan_targets_returns_inf(self):
        """An undefined loss must not look like a perfect zero-valued score."""
        a = torch.tensor([[[1.0], [2.0]]])
        b = torch.full((1, 2, 1), float('nan'))
        loss = get_L1loss_weighted_by_targets(a, b)
        assert loss.shape == torch.Size([])
        assert loss.dtype == a.dtype
        assert torch.isposinf(loss)


class TestL1ConditionalWeighted:
    def test_perfect_samples_zero(self):
        """When all S samples equal the target, MAE should be 0."""
        torch.manual_seed(0)
        targets = torch.rand(4, 6)  # (N, L)
        samples = targets.unsqueeze(1).expand(4, 5, 6)  # (N, S, L) all equal to target
        loss = get_L1loss_conditional_weighted_by_targets(samples, targets)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_single_sample_matches_l1_weighted(self):
        """When S=1, conditional MAE should equal standard L1 weighted loss."""
        torch.manual_seed(0)
        pred = torch.rand(4, 6)  # (N, L)
        targets = torch.rand(4, 6)  # (N, L)
        samples = pred.unsqueeze(1)  # (N, 1, L)
        mae_cond = get_L1loss_conditional_weighted_by_targets(samples, targets)
        mae_std = get_L1loss_weighted_by_targets(pred.unsqueeze(-1), targets.unsqueeze(-1))
        assert torch.isclose(mae_cond, mae_std, atol=1e-5)

    def test_nan_excluded(self):
        """NaN-padded positions in targets should be excluded."""
        targets = torch.tensor([[1.0, 2.0, float('nan')]])  # (1, 3)
        samples = torch.tensor([[[3.0, 4.0, 999.0]]])  # (1, 1, 3)
        loss = get_L1loss_conditional_weighted_by_targets(samples, targets)
        assert torch.isfinite(loss)
        # Valid entries: |3-1|=2, |4-2|=2 → mean=2.0
        assert torch.isclose(loss, torch.tensor(2.0), atol=1e-5)

    def test_known_value_multiple_samples(self):
        """Known value with S=2 samples."""
        targets = torch.tensor([[1.0, 2.0]])  # (1, 2)
        samples = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])  # (1, 2, 2): S=2
        # sample 0: |3-1|=2, |4-2|=2
        # sample 1: |5-1|=4, |6-2|=4
        # mean over S: (2+4)/2=3, (2+4)/2=3 → weighted mean = 3.0
        loss = get_L1loss_conditional_weighted_by_targets(samples, targets)
        assert torch.isclose(loss, torch.tensor(3.0), atol=1e-5)
