import unittest
import numpy as np
import torch
from src.metrics.wasstpp import w1_between_processes_via_tpp_norm
from src.generators.hp import gen


def _poisson_cumtimes(n_seq: int, lam: float, n_events: int, seed: int) -> torch.Tensor:
    """Generate Poisson inter-arrivals, return cumulative times (n_seq, n_events, 1)."""
    rng = np.random.default_rng(seed)
    iat = gen(n_seq, lam, num_elements_in_ts=n_events, rng=rng)
    return torch.from_numpy(iat).float().cumsum(dim=1)


class TestW1Tpp(unittest.TestCase):

    def test_identical_distributions_near_zero(self):
        """W1(P, P) should be close to zero."""
        torch.manual_seed(0)
        X = torch.rand(30, 8, 1).cumsum(dim=1)
        w1 = w1_between_processes_via_tpp_norm(X, X.clone(), T=20.0)
        self.assertAlmostEqual(w1, 0.0, places=3)

    def test_different_distributions_positive(self):
        """W1(P, Q) > 0 for different distributions."""
        torch.manual_seed(0)
        X = torch.rand(30, 8, 1).cumsum(dim=1)
        Y = (torch.rand(30, 8, 1) * 5).cumsum(dim=1)
        w1 = w1_between_processes_via_tpp_norm(X, Y, T=20.0)
        self.assertGreater(w1, 0.0)

    def test_with_nans(self):
        """Variable-length sequences should work."""
        torch.manual_seed(0)
        X = torch.rand(20, 10, 1).cumsum(dim=1)
        Y = torch.rand(20, 10, 1).cumsum(dim=1)
        X[0, 7:, :] = float("nan")
        Y[5, 4:, :] = float("nan")
        w1 = w1_between_processes_via_tpp_norm(X, Y, T=20.0)
        self.assertTrue(w1 >= 0.0)

    def test_regression_value(self):
        torch.manual_seed(123)
        X = torch.rand(15, 6, 1).cumsum(dim=1)
        Y = torch.rand(12, 6, 1).cumsum(dim=1)
        w1 = w1_between_processes_via_tpp_norm(X, Y, T=10.0)
        # Reference computed with reg=None (exact EMD via ot.emd2)
        self.assertAlmostEqual(w1, 0.016413328220447, places=8)

    def test_sinkhorn_close_to_exact(self):
        """Sinkhorn with small reg should approximate exact EMD."""
        torch.manual_seed(0)
        X = torch.rand(30, 8, 1).cumsum(dim=1)
        Y = torch.rand(30, 8, 1).cumsum(dim=1)
        w1_exact = w1_between_processes_via_tpp_norm(X, Y, T=20.0, reg=None)
        w1_sink = w1_between_processes_via_tpp_norm(X, Y, T=20.0, reg=0.05)
        self.assertAlmostEqual(w1_sink, w1_exact, delta=abs(w1_exact) * 0.2 + 1e-4)

    def test_default_call_returns_finite(self):
        """Default call (no reg) should return finite non-negative float."""
        torch.manual_seed(0)
        X = torch.rand(20, 8, 1).cumsum(dim=1)
        Y = torch.rand(20, 8, 1).cumsum(dim=1)
        w1 = w1_between_processes_via_tpp_norm(X, Y, T=20.0)
        self.assertTrue(w1 >= 0.0)
        self.assertTrue(w1 < float('inf'))


class TestW1TppPoisson(unittest.TestCase):
    """Integration tests using real Poisson process samples.

    Mirrors the old __main__ block (Poisson(1) vs Poisson(3), exact vs Sinkhorn)
    and extends it with monotonicity checks.
    """

    N = 100  # sequences per batch: small enough for exact EMD to be fast
    L = 15  # fixed events per sequence
    T = 20.0  # observation horizon

    def _w1(self, lam1, lam2, seed=0, reg=None):
        """Exact W1 between Poisson(lam1) and Poisson(lam2) samples."""
        X = _poisson_cumtimes(self.N, lam1, self.L, seed=seed)
        Y = _poisson_cumtimes(self.N, lam2, self.L, seed=seed + 1)
        return w1_between_processes_via_tpp_norm(X, Y, T=self.T, reg=reg)

    def test_within_rate_smaller_than_cross_rate(self):
        """Mean within-rate W1 < cross-rate W1 for clearly different λ.

        Note: W1 between finite empirical samples doesn't converge to 0
        (unlike the unbiased energy distance estimator), so we test the
        relative ordering rather than nearness to zero.
        """
        for lam1, lam2 in [(1.0, 3.0), (1.0, 5.0), (0.5, 2.0)]:
            with self.subTest(lam1=lam1, lam2=lam2):
                X = _poisson_cumtimes(self.N, lam1, self.L, seed=0)
                Y = _poisson_cumtimes(self.N, lam2, self.L, seed=1)
                w1_xx = w1_between_processes_via_tpp_norm(X, X.clone(), T=self.T, reg=None)
                w1_yy = w1_between_processes_via_tpp_norm(Y, Y.clone(), T=self.T, reg=None)
                w1_xy = w1_between_processes_via_tpp_norm(X, Y, T=self.T, reg=None)
                self.assertGreater(
                    w1_xy,
                    max(w1_xx, w1_yy),
                    msg=f"Expected W1_cross > W1_within for lam1={lam1}, lam2={lam2}: "
                    f"w1_xy={w1_xy:.4f}, w1_xx={w1_xx:.4f}, w1_yy={w1_yy:.4f}",
                )

    def test_different_rates_positive(self):
        """W1(Poisson(λ1), Poisson(λ2)) > 0 when λ1 ≠ λ2."""
        for lam1, lam2 in [(1.0, 3.0), (0.5, 2.0), (1.0, 5.0)]:
            with self.subTest(lam1=lam1, lam2=lam2):
                w1 = self._w1(lam1, lam2)
                self.assertGreater(w1, 0.0, msg=f"Expected W1>0 for lam1={lam1}, lam2={lam2}, got {w1:.6f}")

    def test_monotone_in_lambda_gap(self):
        """Larger rate gap → larger W1: W1(1,2) < W1(1,5) < W1(1,10)."""
        w1_small = self._w1(1.0, 2.0)
        w1_mid = self._w1(1.0, 5.0)
        w1_large = self._w1(1.0, 10.0)
        self.assertLess(w1_small, w1_mid, msg=f"Expected W1(1,2)<W1(1,5): {w1_small:.6f} vs {w1_mid:.6f}")
        self.assertLess(w1_mid, w1_large, msg=f"Expected W1(1,5)<W1(1,10): {w1_mid:.6f} vs {w1_large:.6f}")

    def test_sinkhorn_close_to_exact_poisson(self):
        """Sinkhorn ≈ exact EMD on Poisson data (mirrors old __main__ printout)."""
        X = _poisson_cumtimes(self.N, 1.0, self.L, seed=0)
        Y = _poisson_cumtimes(self.N, 3.0, self.L, seed=1)
        w1_exact = w1_between_processes_via_tpp_norm(X, Y, T=self.T, reg=None)
        w1_sink = w1_between_processes_via_tpp_norm(X, Y, T=self.T, reg=0.05)
        self.assertAlmostEqual(
            w1_sink,
            w1_exact,
            delta=abs(w1_exact) * 0.25 + 1e-4,
            msg=f"Sinkhorn {w1_sink:.4f} too far from exact {w1_exact:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
