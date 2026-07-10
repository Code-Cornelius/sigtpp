import torch

from src.metrics.histogram_loss import HistogramLoss


class TestHistogramLoss:
    def test_identical_distributions_near_zero(self):
        torch.manual_seed(42)
        N, L, D = 1000, 3, 2
        x = torch.randn(N, L, D)
        loss_fn = HistogramLoss(x, n_bins=20)
        loss = loss_fn(x)
        assert loss.item() < 0.05

    def test_non_overlapping_distributions_near_one(self):
        torch.manual_seed(42)
        N, L, D = 1000, 3, 1
        x_real = torch.randn(N, L, D)
        x_fake = torch.randn(N, L, D) + 100.0
        loss_fn = HistogramLoss(x_real, n_bins=20)
        loss = loss_fn(x_fake)
        assert loss.item() > 0.9

    def test_nan_in_fake_no_crash_and_bounded(self):
        torch.manual_seed(0)
        N, L, D = 200, 3, 2
        x_real = torch.randn(N, L, D)
        x_fake = x_real.clone()
        x_fake[0:50, 1, 0] = float('nan')
        loss_fn = HistogramLoss(x_real, n_bins=10)
        loss = loss_fn(x_fake)
        assert torch.isfinite(loss)
        assert 0.0 <= loss.item() <= 1.0

    def test_loss_bounded_between_0_and_1(self):
        torch.manual_seed(1)
        N, L, D = 500, 2, 2
        x_real = torch.randn(N, L, D)
        x_fake = torch.randn(N, L, D) * 2.0
        loss_fn = HistogramLoss(x_real, n_bins=20)
        loss = loss_fn(x_fake)
        assert 0.0 <= loss.item() <= 1.0

    def test_freedman_diaconis_rule(self):
        assert HistogramLoss.num_bins_freedman_diaconis_rule(1000) == 20

    def test_all_nan_feature_in_real_no_crash(self):
        torch.manual_seed(0)
        N, L, D = 200, 3, 2
        x_real = torch.randn(N, L, D)
        x_real[:, 2, 1] = float('nan')  # entire (t=2, feature=1) slot is NaN
        x_fake = torch.randn(N, L, D)
        loss_fn = HistogramLoss(x_real, n_bins=10)
        loss = loss_fn(x_fake)
        assert torch.isfinite(loss)

    def test_constant_real_data_no_crash(self):
        # All values identical: should not produce division-by-zero
        N, L, D = 100, 2, 1
        x_real = torch.ones(N, L, D)
        x_fake = torch.ones(N, L, D) * 1.5
        loss_fn = HistogramLoss(x_real, n_bins=10)
        loss = loss_fn(x_fake)
        assert torch.isfinite(loss)
