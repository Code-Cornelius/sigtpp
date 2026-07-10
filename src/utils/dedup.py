import unittest

import torch


def dedup_consecutive_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Remove consecutive duplicates along each 1-D sequence in a 3-D tensor while preserving shape.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, 1) and dtype float32 or float64.
            NaN values are treated as missing values and are not considered duplicates.

    Returns:
        torch.Tensor: Tensor of shape (B, L, 1) where for each batch:
            - The first occurrence of each run of equal (non-NaN) values is kept in order.
            - Subsequent duplicates are removed and survivors are left-packed.
            - Remaining tail positions are filled with NaN.

    Raises:
        ValueError: If input is not a 3-D tensor with last dimension equal to 1.
        TypeError: If input tensor dtype is not float32 or float64.
    """
    assert x.ndim == 3 and x.shape[-1] == 1, f"Expected input of shape (B, L, 1), got {tuple(x.shape)}"
    assert x.dtype in (torch.float32, torch.float64), f"Expected float32 or float64 tensor, got {x.dtype}"

    x2 = x.squeeze(-1)  # (B, L)
    valid = ~torch.isnan(x2)
    # Identify positions where both current and previous elements are valid (not NaN)
    # and equal, i.e., consecutive duplicates
    dup = valid[:, 1:] & valid[:, :-1] & (x2[:, 1:] == x2[:, :-1])
    keep = torch.ones_like(x2, dtype=torch.bool)
    keep[:, 1:] &= ~dup
    positions = torch.cumsum(keep, dim=1) - 1

    B, L = x2.shape
    out = torch.full_like(x2, float('nan'))
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, L)
    out[batch_idx[keep], positions[keep]] = x2[keep]
    return out.unsqueeze(-1)


class TestDedupConsecutiveTorch(unittest.TestCase):
    """Unit tests for dedup_consecutive_torch with debug prints for each case."""

    def assertTensorAllCloseWithNan(self, actual: torch.Tensor, expected: torch.Tensor, **kwargs):
        torch.testing.assert_close(actual, expected, equal_nan=True, **kwargs)

    def test_basic_cases(self):
        print("\n=== test_basic_cases ===")
        cases = [
            (
                torch.tensor([[[1.0], [1.0], [2.0], [2.0], [3.0]]], dtype=torch.float64),
                torch.tensor([[[1.0], [2.0], [3.0], [float('nan')], [float('nan')]]], dtype=torch.float64),
            ),
            (
                torch.tensor([[[5.0], [5.0], [6.0], [6.0], [float('nan')]]], dtype=torch.float32),
                torch.tensor([[[5.0], [6.0], [float('nan')], [float('nan')], [float('nan')]]], dtype=torch.float32),
            ),
            (
                torch.tensor([[[1e-12], [1e-12], [2e-12], [3e-12], [3e-12], [4e-12]]], dtype=torch.float32),
                torch.tensor(
                    [[[1e-12], [2e-12], [3e-12], [4e-12], [float('nan')], [float('nan')]]], dtype=torch.float32
                ),
            ),
        ]
        for inp, exp in cases:
            print("\n-- case --")
            print("Input:\n", inp)
            print("Expected:\n", exp)
            out = dedup_consecutive_torch(inp)
            print("Output:\n", out)
            self.assertTensorAllCloseWithNan(out, exp)

    def test_multi_batch(self):
        print("\n=== test_multi_batch ===")
        x = torch.tensor(
            [
                [[1.0], [1.0], [2.0]],
                [[3.0], [3.0], [3.0]],
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [[1.0], [2.0], [float('nan')]],
                [[3.0], [float('nan')], [float('nan')]],
            ],
            dtype=torch.float32,
        )
        print("Input multi-batch:\n", x)
        print("Expected multi-batch:\n", expected)
        out = dedup_consecutive_torch(x)
        print("Output multi-batch:\n", out)
        self.assertTensorAllCloseWithNan(out, expected)

    def test_invalid_shape(self):
        print("\n=== test_invalid_shape ===")
        shapes = [(2, 3), (2, 3, 2)]
        for shape in shapes:
            with self.subTest(shape=shape):
                arr = torch.randn(*shape)
                print(f"Testing invalid shape: {shape}")
                with self.assertRaises(AssertionError):
                    dedup_consecutive_torch(arr)

    def test_invalid_dtype(self):
        print("\n=== test_invalid_dtype ===")
        arr = torch.randint(0, 5, (1, 4, 1), dtype=torch.int32)
        print("Testing invalid dtype input:\n", arr)
        with self.assertRaises(AssertionError):
            dedup_consecutive_torch(arr)


if __name__ == '__main__':
    unittest.main()
