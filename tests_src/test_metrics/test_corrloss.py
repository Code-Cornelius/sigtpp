import torch
import pytest
from src.metrics.corrloss import CorrLoss, corr_torch


class TestCorrTorch:
    def test_diagonal_is_one(self):
        torch.manual_seed(0)
        x = torch.randn(100, 3)
        corr = corr_torch(x)
        # corr_torch returns float64 (via numpy)
        assert torch.allclose(torch.diag(corr), torch.ones(3, dtype=corr.dtype), atol=1e-5)

    def test_output_shape_2d(self):
        x = torch.randn(50, 4)
        corr = corr_torch(x)
        assert corr.shape == (4, 4)

    def test_output_shape_3d_flattened(self):
        # (50, 4, 2) -> flattened to (50, 8) -> corr (8, 8)
        x = torch.randn(50, 4, 2)
        corr = corr_torch(x)
        assert corr.shape == (8, 8)

    def test_single_feature_crashes(self):
        # np.ma.corrcoef returns a scalar for 1-variable input, so .filled() fails.
        # This is a known limitation in the source code.
        x = torch.randn(100, 1)
        with pytest.raises((AttributeError, RuntimeError)):
            corr_torch(x)

    def test_symmetric(self):
        torch.manual_seed(1)
        x = torch.randn(100, 4)
        corr = corr_torch(x)
        assert torch.allclose(corr, corr.T, atol=1e-6)


class TestCorrLoss:
    def test_identity_loss_near_zero(self):
        torch.manual_seed(0)
        N, L, D = 200, 3, 2
        x = torch.randn(N, L, D)
        loss_fn = CorrLoss(x)
        loss = loss_fn.loss(x)
        assert loss.item() < 0.05

    def test_single_feature_loss_is_zero(self):
        # corr([[1]]) is always [[1]], so corr(x) - corr(x) = 0
        # Note: corr_torch returns float64; compare with matching dtype
        torch.manual_seed(0)
        N, L, D = 200, 2, 1
        x = torch.randn(N, L, D)
        loss_fn = CorrLoss(x)
        loss = loss_fn.loss(x)
        assert torch.isclose(loss, torch.zeros(1, dtype=loss.dtype), atol=1e-6)

    def test_truncation_with_sparse_late_timesteps(self):
        torch.manual_seed(0)
        N, L, D = 200, 5, 2
        x = torch.randn(N, L, D)
        # Make last two time steps have very few valid samples (< 50)
        x[10:, 3, :] = float('nan')
        x[10:, 4, :] = float('nan')
        loss_fn = CorrLoss(x)
        # slice_t should be capped at 3
        assert loss_fn.slice_t.item() <= 3
        # Should still run without error
        result = loss_fn(x)
        assert result is not None

    def test_different_data_gives_nonzero_loss(self):
        torch.manual_seed(42)
        N, L, D = 300, 3, 3
        x_real = torch.randn(N, L, D)
        # Strongly correlated fake: first feature = second feature
        x_fake = torch.randn(N, L, D)
        x_fake[:, :, 1] = x_fake[:, :, 0]
        loss_fn = CorrLoss(x_real)
        loss = loss_fn.loss(x_fake)
        assert loss.item() > 0.0
