import torch

from src.lead_lag import lead_lag_transformation


class TestLeadLagTransformation:
    def test_shape_general(self):
        # (N=2, L=5, D=3) -> (2, 2*5-1, 2*3) = (2, 9, 6)
        x = torch.randn(2, 5, 3)
        assert lead_lag_transformation(x).shape == (2, 9, 6)

    def test_shape_docstring_example(self):
        data = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        assert lead_lag_transformation(data).shape == (1, 5, 2)

    def test_known_values(self):
        # Input [[1], [2], [3]] with N=1, L=3, D=1
        # Duplicated: [[1],[1],[2],[2],[3],[3]]
        # lead = [1:] = [[1],[2],[2],[3],[3]]
        # lag  = [:-1] = [[1],[1],[2],[2],[3]]
        # concat along D: [[1,1],[2,1],[2,2],[3,2],[3,3]]
        data = torch.tensor([[[1.0], [2.0], [3.0]]])
        out = lead_lag_transformation(data)
        expected = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [3.0, 2.0], [3.0, 3.0]]])
        assert torch.allclose(out, expected)

    def test_single_time_step(self):
        # (N, L=1, D) -> (N, 2*1-1, 2*D) = (N, 1, 2D)
        x = torch.randn(3, 1, 4)
        assert lead_lag_transformation(x).shape == (3, 1, 8)

    def test_feature_doubling(self):
        x = torch.randn(5, 3, 2)
        out = lead_lag_transformation(x)
        assert out.shape[2] == 4

    def test_batch_size_preserved(self):
        x = torch.randn(7, 4, 2)
        assert lead_lag_transformation(x).shape[0] == 7

    def test_time_dimension_formula(self):
        for L in [2, 4, 10]:
            x = torch.randn(1, L, 1)
            out = lead_lag_transformation(x)
            assert out.shape[1] == 2 * L - 1
