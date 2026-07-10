import unittest
import numpy as np
import torch
from src.metrics.sliced_energy_distance import sliced_energy_distance_tpp
from src.generators.hp import gen


def _poisson_cumtimes(n_seq: int, lam: float, n_events: int, seed: int) -> torch.Tensor:
    """Generate Poisson inter-arrivals, return cumulative times (n_seq, n_events, 1)."""
    rng = np.random.default_rng(seed)
    iat = gen(n_seq, lam, num_elements_in_ts=n_events, rng=rng)
    return torch.from_numpy(iat).float().cumsum(dim=1)


class TestSlicedEnergyDistance(unittest.TestCase):

    def test_identical_near_zero(self):
        """SED(P, P) ~ 0."""
        torch.manual_seed(0)
        X = torch.rand(100, 10, 1).cumsum(dim=1)
        sed = sliced_energy_distance_tpp(X, X.clone(), T=20.0, num_projections=200)
        self.assertAlmostEqual(sed, 0.0, places=2)

    def test_different_positive(self):
        """SED(P, Q) > 0 for clearly different distributions."""
        torch.manual_seed(0)
        X = torch.rand(100, 10, 1).cumsum(dim=1)
        Y = (torch.rand(100, 10, 1) * 5).cumsum(dim=1)
        sed = sliced_energy_distance_tpp(X, Y, T=20.0, num_projections=200)
        self.assertGreater(sed, 0.0)

    def test_symmetry(self):
        """SED(P, Q) == SED(Q, P) for same random seed."""
        torch.manual_seed(0)
        X = torch.rand(50, 8, 1).cumsum(dim=1)
        Y = torch.rand(40, 8, 1).cumsum(dim=1)
        gen = torch.Generator().manual_seed(99)
        sed_xy = sliced_energy_distance_tpp(X, Y, T=15.0, num_projections=100, generator=gen)
        gen = torch.Generator().manual_seed(99)
        sed_yx = sliced_energy_distance_tpp(Y, X, T=15.0, num_projections=100, generator=gen)
        self.assertAlmostEqual(sed_xy, sed_yx, places=6)

    def test_with_nans(self):
        """Variable-length sequences (NaN-padded) should work."""
        torch.manual_seed(0)
        X = torch.rand(50, 10, 1).cumsum(dim=1)
        Y = torch.rand(50, 10, 1).cumsum(dim=1)
        X[0, 7:, :] = float("nan")
        Y[5, 4:, :] = float("nan")
        sed = sliced_energy_distance_tpp(X, Y, T=20.0, num_projections=100)
        self.assertTrue(sed >= 0.0)

    def test_scales_with_divergence(self):
        """More different distributions should give larger SED."""
        torch.manual_seed(0)
        X = torch.rand(80, 10, 1).cumsum(dim=1)
        Y_close = (torch.rand(80, 10, 1) * 1.5).cumsum(dim=1)
        Y_far = (torch.rand(80, 10, 1) * 10).cumsum(dim=1)
        sed_close = sliced_energy_distance_tpp(X, Y_close, T=20.0, num_projections=200)
        sed_far = sliced_energy_distance_tpp(X, Y_far, T=20.0, num_projections=200)
        self.assertGreater(sed_far, sed_close)


class TestSlicedEnergyDistancePoisson(unittest.TestCase):
    """Integration tests using real Poisson process samples."""

    N = 500  # sequences per batch: large enough for stable projection averages
    L = 20  # fixed events per sequence
    T = 20.0  # observation horizon
    K = 300  # projections: enough for stable SED estimates

    def _sed(self, lam1, lam2, seed=0):
        X = _poisson_cumtimes(self.N, lam1, self.L, seed=seed)
        Y = _poisson_cumtimes(self.N, lam2, self.L, seed=seed + 1)
        return sliced_energy_distance_tpp(X, Y, T=self.T, num_projections=self.K)

    def test_same_rate_near_zero(self):
        """SED(Poisson(λ), Poisson(λ)) ≈ 0 for several rates.

        SED uses the unbiased 1D ED estimator per projection, so the average
        across projections converges to 0 as N → ∞ when the distributions match.
        The tolerance is looser than for ED (2e-2 vs 2e-3) because random projections
        add variance: projecting a high-dimensional process onto a random 1D direction
        mixes dimensions, inflating residual noise at finite N.
        """
        for lam in [0.5, 1.0, 3.0, 5.0]:
            with self.subTest(lam=lam):
                X = _poisson_cumtimes(self.N, lam, self.L, seed=42)
                Y = _poisson_cumtimes(self.N, lam, self.L, seed=99)
                sed = sliced_energy_distance_tpp(X, Y, T=self.T, num_projections=self.K)
                self.assertAlmostEqual(sed, 0.0, delta=2e-2, msg=f"Expected SED≈0 for lam={lam}, got {sed:.6f}")

    def test_different_rates_positive(self):
        """SED(Poisson(λ1), Poisson(λ2)) > 0 when λ1 ≠ λ2."""
        for lam1, lam2 in [(1.0, 3.0), (0.5, 2.0), (1.0, 5.0)]:
            with self.subTest(lam1=lam1, lam2=lam2):
                sed = self._sed(lam1, lam2)
                self.assertGreater(sed, 0.0, msg=f"Expected SED>0 for lam1={lam1}, lam2={lam2}, got {sed:.6f}")

    def test_monotone_in_lambda_gap(self):
        """Larger rate gap → larger SED: SED(1,2) < SED(1,5) < SED(1,10)."""
        sed_small = self._sed(1.0, 2.0)
        sed_mid = self._sed(1.0, 5.0)
        sed_large = self._sed(1.0, 10.0)
        self.assertLess(sed_small, sed_mid, msg=f"Expected SED(1,2)<SED(1,5): {sed_small:.6f} vs {sed_mid:.6f}")
        self.assertLess(sed_mid, sed_large, msg=f"Expected SED(1,5)<SED(1,10): {sed_mid:.6f} vs {sed_large:.6f}")


if __name__ == "__main__":
    unittest.main()
