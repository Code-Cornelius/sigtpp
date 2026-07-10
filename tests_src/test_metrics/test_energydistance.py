import unittest
import numpy as np
import torch
from src.metrics.energy_distance_tpp import energy_distance_tpp
from src.generators.hp import gen


def _poisson_cumtimes(n_seq: int, lam: float, n_events: int, seed: int) -> torch.Tensor:
    """Generate Poisson inter-arrivals, return cumulative times (n_seq, n_events, 1)."""
    rng = np.random.default_rng(seed)
    iat = gen(n_seq, lam, num_elements_in_ts=n_events, rng=rng)  # (N, L, 1) numpy
    return torch.from_numpy(iat).float().cumsum(dim=1)


class TestEnergyDistanceTpp(unittest.TestCase):

    def test_identical_distributions_near_zero(self):
        """ED(P, P) should be close to zero."""
        torch.manual_seed(0)
        X = torch.rand(50, 10, 1).cumsum(dim=1)
        ed = energy_distance_tpp(X, X.clone(), T=20.0)
        self.assertAlmostEqual(ed.item(), 0.0, places=2)

    def test_different_distributions_positive(self):
        """ED(P, Q) > 0 for clearly different distributions."""
        torch.manual_seed(0)
        X = torch.rand(50, 10, 1).cumsum(dim=1)
        Y = (torch.rand(50, 10, 1) * 5).cumsum(dim=1)
        ed = energy_distance_tpp(X, Y, T=20.0)
        self.assertGreater(ed.item(), 0.0)

    def test_deterministic(self):
        """Same input should give same output."""
        torch.manual_seed(0)
        X = torch.rand(30, 8, 1).cumsum(dim=1)
        Y = torch.rand(20, 8, 1).cumsum(dim=1)
        ed1 = energy_distance_tpp(X, Y, T=15.0)
        ed2 = energy_distance_tpp(X, Y, T=15.0)
        self.assertEqual(ed1.item(), ed2.item())

    def test_symmetry(self):
        """ED(P, Q) == ED(Q, P)."""
        torch.manual_seed(0)
        X = torch.rand(30, 8, 1).cumsum(dim=1)
        Y = torch.rand(25, 8, 1).cumsum(dim=1)
        ed_xy = energy_distance_tpp(X, Y, T=15.0)
        ed_yx = energy_distance_tpp(Y, X, T=15.0)
        self.assertAlmostEqual(ed_xy.item(), ed_yx.item(), places=6)

    def test_regression_value(self):
        torch.manual_seed(123)
        X = torch.rand(15, 6, 1).cumsum(dim=1)
        Y = torch.rand(12, 6, 1).cumsum(dim=1)
        ed = energy_distance_tpp(X, Y, T=10.0)
        self.assertAlmostEqual(ed.item(), -0.001007664832287, places=8)

    def test_with_nans(self):
        """Variable-length sequences (NaN-padded) should work."""
        torch.manual_seed(0)
        X = torch.rand(20, 10, 1).cumsum(dim=1)
        Y = torch.rand(20, 10, 1).cumsum(dim=1)
        X[0, 7:, :] = float("nan")
        Y[5, 4:, :] = float("nan")
        ed = energy_distance_tpp(X, Y, T=20.0)
        self.assertTrue(torch.isfinite(torch.tensor(ed.item())))


class TestEnergyDistanceTppPoisson(unittest.TestCase):
    """Integration tests using real Poisson process samples (replaces old __main__ block)."""

    N = 500  # sequences per batch: large enough for stable estimates
    L = 20  # fixed events per sequence (avoids shape-padding)
    T = 20.0  # normalisation horizon

    def _ed(self, lam1, lam2, seed=0):
        X = _poisson_cumtimes(self.N, lam1, self.L, seed=seed)
        Y = _poisson_cumtimes(self.N, lam2, self.L, seed=seed + 1)
        return energy_distance_tpp(X, Y, T=self.T).item()

    def test_same_rate_near_zero(self):
        """ED(Poisson(λ), Poisson(λ)) ≈ 0 for several rates."""
        for lam in [0.5, 1.0, 3.0, 5.0]:
            with self.subTest(lam=lam):
                X = _poisson_cumtimes(self.N, lam, self.L, seed=42)
                Y = _poisson_cumtimes(self.N, lam, self.L, seed=99)
                ed = energy_distance_tpp(X, Y, T=self.T).item()
                self.assertAlmostEqual(ed, 0.0, delta=2e-3, msg=f"Expected ED≈0 for lam={lam}, got {ed:.6f}")

    def test_different_rates_positive(self):
        """ED(Poisson(λ1), Poisson(λ2)) > 0 when λ1 ≠ λ2."""
        for lam1, lam2 in [(1.0, 3.0), (0.5, 2.0), (1.0, 5.0)]:
            with self.subTest(lam1=lam1, lam2=lam2):
                ed = self._ed(lam1, lam2)
                self.assertGreater(ed, 0.0, msg=f"Expected ED>0 for lam1={lam1}, lam2={lam2}, got {ed:.6f}")

    def test_monotone_in_lambda_gap(self):
        """Larger rate gap → larger ED: ED(P(1),P(2)) < ED(P(1),P(5)) < ED(P(1),P(10))."""
        ed_small = self._ed(1.0, 2.0)
        ed_mid = self._ed(1.0, 5.0)
        ed_large = self._ed(1.0, 10.0)
        self.assertLess(ed_small, ed_mid, msg=f"Expected ED(1,2)<ED(1,5), got {ed_small:.6f} vs {ed_mid:.6f}")
        self.assertLess(ed_mid, ed_large, msg=f"Expected ED(1,5)<ED(1,10), got {ed_mid:.6f} vs {ed_large:.6f}")


if __name__ == "__main__":
    unittest.main()
