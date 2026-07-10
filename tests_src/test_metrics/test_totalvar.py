import torch
import pytest
from src.metrics.totalvar import total_var


class TestTotalVar:
    def test_known_value(self):
        # Path [[0,0], [1,1], [3,0]]:
        # diffs: [[1,1], [2,-1]] -> abs sum = 1+1+2+1 = 5
        path = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]]])
        result = total_var(path)
        assert torch.isclose(result, torch.tensor([5.0]))

    def test_constant_path_is_zero(self):
        path = torch.ones(3, 5, 2)
        assert torch.all(total_var(path) == 0.0)

    def test_single_time_step_is_zero(self):
        # No diffs possible for L=1
        path = torch.randn(4, 1, 3)
        assert torch.all(total_var(path) == 0.0)

    def test_batch_each_gets_own_value(self):
        path = torch.tensor([
            [[0.0], [1.0], [3.0]],
            [[0.0], [2.0], [2.0]],
        ])
        result = total_var(path)
        # First: |1| + |2| = 3, Second: |2| + |0| = 2
        assert result[0].item() == pytest.approx(3.0)
        assert result[1].item() == pytest.approx(2.0)

    def test_output_shape(self):
        path = torch.randn(5, 4, 3)
        assert total_var(path).shape == (5,)

    def test_requires_3d_input(self):
        with pytest.raises(AssertionError):
            total_var(torch.randn(3, 4))
