import unittest
import numpy as np
import torch
from src.metrics.tppnorm import tpp_area_distance_mean, pairwise_tpp_area_distance
from src.generators.hp import gen


def _poisson_cumtimes(n_seq: int, lam: float, n_events: int, seed: int) -> torch.Tensor:
    """Generate Poisson inter-arrivals, return cumulative times (n_seq, n_events, 1)."""

    rng = np.random.default_rng(seed)
    iat = gen(n_seq, lam, num_elements_in_ts=n_events, rng=rng)
    return torch.from_numpy(iat).float().cumsum(dim=1)


class TestPairwiseTppAreaDistance(unittest.TestCase):
    """Verify chunked pairwise matches brute-force all-pairs."""

    def _brute_force_pairwise(self, X, Y, T):
        """Compute (Nf, Nt) cost matrix by materializing all pairs (reference impl)."""
        Nf, Nt = X.shape[0], Y.shape[0]
        X_rep = X.repeat_interleave(Nt, dim=0)
        Y_tile = Y.repeat((Nf, 1, 1))
        d_flat = tpp_area_distance_mean(X_rep, Y_tile, T)
        return d_flat.view(Nf, Nt)

    def test_matches_brute_force_small(self):
        """Small example: 5x4 cost matrix, chunk_size larger than Nf."""
        torch.manual_seed(42)
        X = torch.rand(5, 10, 1).cumsum(dim=1)
        Y = torch.rand(4, 10, 1).cumsum(dim=1)
        T = 20.0
        expected = self._brute_force_pairwise(X, Y, T)
        actual = pairwise_tpp_area_distance(X, Y, T, chunk_size=512)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_matches_brute_force_with_nans(self):
        """Sequences padded with NaN (variable-length)."""
        torch.manual_seed(7)
        X = torch.rand(6, 8, 1).cumsum(dim=1)
        Y = torch.rand(4, 8, 1).cumsum(dim=1)
        X[0, 5:, :] = float("nan")
        X[3, 3:, :] = float("nan")
        Y[1, 6:, :] = float("nan")
        T = 15.0
        expected = self._brute_force_pairwise(X, Y, T)
        actual = pairwise_tpp_area_distance(X, Y, T, chunk_size=2)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_chunk_size_smaller_than_nf(self):
        """Chunk size forces multiple iterations over X rows."""
        torch.manual_seed(99)
        X = torch.rand(10, 6, 1).cumsum(dim=1)
        Y = torch.rand(8, 6, 1).cumsum(dim=1)
        T = 10.0
        expected = self._brute_force_pairwise(X, Y, T)
        actual = pairwise_tpp_area_distance(X, Y, T, chunk_size=3)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_multidimensional(self):
        """D > 1."""
        torch.manual_seed(0)
        X = torch.rand(4, 5, 2).cumsum(dim=1)
        Y = torch.rand(3, 5, 2).cumsum(dim=1)
        T = 10.0
        expected = self._brute_force_pairwise(X, Y, T)
        actual = pairwise_tpp_area_distance(X, Y, T, chunk_size=2)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


class TestPairwiseTppAreaDistancePoisson(unittest.TestCase):
    """Integration tests using real Poisson process samples."""

    N = 300  # sequences per batch
    L = 20  # fixed events per sequence
    T = 20.0  # observation horizon

    def _mean_cross_dist(self, lam1, lam2, seed=0):
        """Mean of the (N x N) cross-distance matrix between Poisson(lam1) and Poisson(lam2)."""
        X = _poisson_cumtimes(self.N, lam1, self.L, seed=seed)
        Y = _poisson_cumtimes(self.N, lam2, self.L, seed=seed + 1)
        return pairwise_tpp_area_distance(X, Y, self.T).mean().item()

    def test_within_rate_smaller_than_cross_rate(self):
        """Mean within-rate distance < mean cross-rate distance for clearly different λ."""
        for lam1, lam2 in [(1.0, 3.0), (1.0, 5.0), (0.5, 2.0)]:
            with self.subTest(lam1=lam1, lam2=lam2):
                X = _poisson_cumtimes(self.N, lam1, self.L, seed=0)
                Y = _poisson_cumtimes(self.N, lam2, self.L, seed=1)
                d_xx = pairwise_tpp_area_distance(X, X, self.T).mean().item()
                d_yy = pairwise_tpp_area_distance(Y, Y, self.T).mean().item()
                d_xy = pairwise_tpp_area_distance(X, Y, self.T).mean().item()
                self.assertGreater(
                    d_xy,
                    max(d_xx, d_yy),
                    msg=f"Expected d_cross > d_within for lam1={lam1}, lam2={lam2}: "
                    f"d_xy={d_xy:.4f}, d_xx={d_xx:.4f}, d_yy={d_yy:.4f}",
                )

    def test_monotone_in_lambda_gap(self):
        """Larger rate gap → larger mean cross-distance."""
        d_small = self._mean_cross_dist(1.0, 2.0)
        d_mid = self._mean_cross_dist(1.0, 5.0)
        d_large = self._mean_cross_dist(1.0, 10.0)
        self.assertLess(d_small, d_mid, msg=f"Expected d(1,2)<d(1,5): {d_small:.4f} vs {d_mid:.4f}")
        self.assertLess(d_mid, d_large, msg=f"Expected d(1,5)<d(1,10): {d_mid:.4f} vs {d_large:.4f}")

    def test_within_rate_distances_nonnegative(self):
        """All entries of the within-rate cost matrix should be non-negative."""
        for lam in [0.5, 1.0, 3.0]:
            with self.subTest(lam=lam):
                X = _poisson_cumtimes(self.N, lam, self.L, seed=7)
                C = pairwise_tpp_area_distance(X, X, self.T)
                self.assertTrue((C >= 0).all().item(), msg=f"Negative distance found for lam={lam}")

    def test_diagonal_is_zero(self):
        """d(x, x) == 0: each sequence compared with itself gives zero distance."""
        X = _poisson_cumtimes(50, 2.0, self.L, seed=5)
        # Compare each sequence against itself via tpp_area_distance_mean
        d_self = tpp_area_distance_mean(X, X, torch.tensor(self.T))
        torch.testing.assert_close(d_self, torch.zeros_like(d_self), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
