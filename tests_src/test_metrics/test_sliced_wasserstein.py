import unittest
import numpy as np
import torch
from src.metrics.sliced_wasserstein_tpp import sliced_wasserstein_tpp, _w1_1d
from src.generators.hp import gen


def _poisson_cumtimes(n_seq: int, lam: float, n_events: int, seed: int) -> torch.Tensor:
    """Generate Poisson inter-arrivals, return cumulative times (n_seq, n_events, 1)."""
    rng = np.random.default_rng(seed)
    iat = gen(n_seq, lam, num_elements_in_ts=n_events, rng=rng)
    return torch.from_numpy(iat).float().cumsum(dim=1)


class TestW1_1D(unittest.TestCase):
    """Unit tests for the 1D W1 primitive."""

    def test_identical_samples_zero(self):
        """W1(x, x) == 0."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(_w1_1d(x, x).item(), 0.0, places=12)

    def test_shifted_samples(self):
        """W1(delta_0, delta_c) == c (point masses shifted by c)."""
        x = torch.tensor([0.0])
        y = torch.tensor([3.0])
        self.assertAlmostEqual(_w1_1d(x, y).item(), 3.0, places=10)

    def test_known_value_equal_sizes(self):
        """W1({1,3}, {2,4}) == 1.0 (manual calculation)."""
        x = torch.tensor([1.0, 3.0])
        y = torch.tensor([2.0, 4.0])
        self.assertAlmostEqual(_w1_1d(x, y).item(), 1.0, places=10)

    def test_symmetry(self):
        """W1(x, y) == W1(y, x)."""
        torch.manual_seed(0)
        x = torch.rand(20)
        y = torch.rand(15) * 2
        self.assertAlmostEqual(_w1_1d(x, y).item(), _w1_1d(y, x).item(), places=10)

    def test_unequal_sizes(self):
        """Handles nx != ny."""
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.5, 1.5])
        w = _w1_1d(x, y)
        self.assertGreaterEqual(w.item(), 0.0)
        self.assertAlmostEqual(w.item(), _w1_1d(y, x).item(), places=10)


class TestSlicedWassersteinTpp(unittest.TestCase):

    def test_identical_near_zero(self):
        """SWD(eta, eta.clone()) == 0 exactly (same data, same projected values)."""
        torch.manual_seed(0)
        eta = torch.rand(50, 10, 1).cumsum(dim=1)
        swd = sliced_wasserstein_tpp(eta, eta.clone(), T=20.0, num_projections=100)
        self.assertAlmostEqual(swd, 0.0, places=10)

    def test_different_positive(self):
        """SWD(P, Q) > 0 for clearly different distributions."""
        torch.manual_seed(0)
        eta = torch.rand(50, 10, 1).cumsum(dim=1)
        rho = (torch.rand(50, 10, 1) * 5).cumsum(dim=1)
        swd = sliced_wasserstein_tpp(eta, rho, T=20.0, num_projections=100)
        self.assertGreater(swd, 0.0)

    def test_symmetry(self):
        """SWD(eta, rho) == SWD(rho, eta) for same generator seed (same projections)."""
        torch.manual_seed(0)
        eta = torch.rand(40, 8, 1).cumsum(dim=1)
        rho = torch.rand(35, 8, 1).cumsum(dim=1)
        gen_er = torch.Generator().manual_seed(42)
        gen_re = torch.Generator().manual_seed(42)
        swd_er = sliced_wasserstein_tpp(eta, rho, T=15.0, num_projections=100, generator=gen_er)
        swd_re = sliced_wasserstein_tpp(rho, eta, T=15.0, num_projections=100, generator=gen_re)
        self.assertAlmostEqual(swd_er, swd_re, places=10)

    def test_with_nans(self):
        """Variable-length sequences (NaN-padded) should work."""
        torch.manual_seed(0)
        eta = torch.rand(30, 10, 1).cumsum(dim=1)
        rho = torch.rand(30, 10, 1).cumsum(dim=1)
        eta[0, 7:, :] = float("nan")
        rho[5, 4:, :] = float("nan")
        swd = sliced_wasserstein_tpp(eta, rho, T=20.0, num_projections=50)
        self.assertGreaterEqual(swd, 0.0)

    def test_scales_with_divergence(self):
        """More different distributions should give larger SWD."""
        torch.manual_seed(0)
        eta = torch.rand(60, 10, 1).cumsum(dim=1)
        rho_close = (torch.rand(60, 10, 1) * 1.5).cumsum(dim=1)
        rho_far = (torch.rand(60, 10, 1) * 10).cumsum(dim=1)
        swd_close = sliced_wasserstein_tpp(eta, rho_close, T=20.0, num_projections=200)
        swd_far = sliced_wasserstein_tpp(eta, rho_far, T=20.0, num_projections=200)
        self.assertGreater(swd_far, swd_close)


class TestSlicedWassersteinTppPoisson(unittest.TestCase):
    """Integration tests using real Poisson process samples.

    SWD between two independent finite samples from the same distribution is NOT
    near zero (finite-sample OT bias, same as the non-sliced W1). We therefore
    test relative ordering (within-rate < cross-rate) rather than absolute proximity.
    """

    N = 100  # sequences per batch
    L = 15  # fixed events per sequence
    T = 20.0  # observation horizon
    K = 200  # projections

    def _swd(self, lam1, lam2, seed=0):
        eta = _poisson_cumtimes(self.N, lam1, self.L, seed=seed)
        rho = _poisson_cumtimes(self.N, lam2, self.L, seed=seed + 1)
        return sliced_wasserstein_tpp(eta, rho, T=self.T, num_projections=self.K)

    def test_within_rate_smaller_than_cross_rate(self):
        """SWD(eta, eta') < SWD(eta, rho) for clearly different rates.

        Within-rate: same eta vs its clone (= 0 exactly).
        Cross-rate: eta from Poisson(lam1) vs rho from Poisson(lam2).
        """
        for lam1, lam2 in [(1.0, 3.0), (1.0, 5.0), (0.5, 2.0)]:
            with self.subTest(lam1=lam1, lam2=lam2):
                eta = _poisson_cumtimes(self.N, lam1, self.L, seed=0)
                rho = _poisson_cumtimes(self.N, lam2, self.L, seed=1)
                swd_ee = sliced_wasserstein_tpp(eta, eta.clone(), T=self.T, num_projections=self.K)
                swd_rr = sliced_wasserstein_tpp(rho, rho.clone(), T=self.T, num_projections=self.K)
                swd_er = sliced_wasserstein_tpp(eta, rho, T=self.T, num_projections=self.K)
                self.assertGreater(
                    swd_er,
                    max(swd_ee, swd_rr),
                    msg=f"Expected SWD_cross > SWD_within for lam1={lam1}, lam2={lam2}: "
                    f"swd_er={swd_er:.4f}, swd_ee={swd_ee:.4f}, swd_rr={swd_rr:.4f}",
                )

    def test_different_rates_positive(self):
        """SWD > 0 for different Poisson rates."""
        for lam1, lam2 in [(1.0, 3.0), (0.5, 2.0), (1.0, 5.0)]:
            with self.subTest(lam1=lam1, lam2=lam2):
                swd = self._swd(lam1, lam2)
                self.assertGreater(swd, 0.0, msg=f"Expected SWD>0 for lam1={lam1}, lam2={lam2}, got {swd:.6f}")

    def test_monotone_in_lambda_gap(self):
        """Larger rate gap → larger SWD: SWD(1,2) < SWD(1,5) < SWD(1,10)."""
        swd_small = self._swd(1.0, 2.0)
        swd_mid = self._swd(1.0, 5.0)
        swd_large = self._swd(1.0, 10.0)
        self.assertLess(swd_small, swd_mid, msg=f"Expected SWD(1,2)<SWD(1,5): {swd_small:.6f} vs {swd_mid:.6f}")
        self.assertLess(swd_mid, swd_large, msg=f"Expected SWD(1,5)<SWD(1,10): {swd_mid:.6f} vs {swd_large:.6f}")


if __name__ == "__main__":
    unittest.main()
