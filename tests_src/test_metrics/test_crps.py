import unittest

import numpy as np
import scipy.stats as st
import torch

from src.metrics.crps import get_crps_loss


class TestCRPSLoss(unittest.TestCase):

    def test_perfect_forecast(self):
        samples = torch.full((2, 3, 5), 0.8)  # [N, L, S]
        targets = torch.full((2, 3), 0.8)  # [N, L]
        crps_result = get_crps_loss(samples, targets)

        self.assertTrue(
            torch.allclose(crps_result, torch.zeros_like(crps_result), atol=1e-6),
            f"Expected all zeros, but got {crps_result}",
        )

    def test_constant_samples(self):
        # Constant forecast c=0.6 vs target y=0.3: CRPS = |c - y| = 0.3 (E|X-X'|=0)
        samples = torch.full((1, 1, 5), 0.6)  # [N, L, S]
        targets = torch.tensor([[0.3]])  # [N, L]
        expected_crps = torch.tensor(0.3).double()
        crps_result = get_crps_loss(samples, targets)

        self.assertTrue(
            torch.allclose(crps_result, expected_crps, atol=1e-6), f"Expected {expected_crps}, but got {crps_result}"
        )

    def test_monte_carlo(self):
        S = 5_000
        mu, sigma = 0.0, 1.0
        target_value = 0.5
        pass_count = 0
        num_trials = 100
        required_passes = 80

        print(f"\nRunning Monte Carlo CRPS test {num_trials} times...\n")

        # There are numerical instabilities when using float32! Try it out with S = 20_000.
        DTYPE_TENSORS = torch.float64

        for i in range(1, num_trials + 1):
            samples = torch.normal(mu, sigma, size=(3, 1, S), dtype=DTYPE_TENSORS)  # [N, L, S]
            targets = torch.tensor([[target_value]], dtype=DTYPE_TENSORS)  # [N, L]

            crps_mc = get_crps_loss(samples, targets)

            z = ((targets - mu) / sigma).item()
            phi_z = torch.tensor(st.norm.pdf(z))
            Phi_z = torch.tensor(st.norm.cdf(z))
            crps_theory = (sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))).view(-1, 1)

            if torch.allclose(crps_mc.float(), crps_theory.float(), atol=1e-2):
                pass_count += 1
                print(
                    f"Trial {i}: ✅ Passed ({pass_count}/{num_trials}) - CRPS MC: {crps_mc.item()}, CRPS Theory: {crps_theory.item()}."
                )
            else:
                print(
                    f"Trial {i}: ❌ Failed ({pass_count}/{num_trials}) - CRPS MC: {crps_mc.item()}, CRPS Theory: {crps_theory.item()}."
                )

        print(f"\nFinal result: {pass_count}/{num_trials} tests passed.")

        self.assertTrue(
            pass_count >= required_passes,
            f"Monte Carlo CRPS failed: Passed {pass_count} times out of {num_trials} (need at least {required_passes}).",
        )

    def test_nans_in_samples(self):
        """NaNs in samples (variable-length sequences) must not corrupt the aggregate scalar."""
        samples = torch.tensor(
            [
                [[0.2, 0.5, float('nan')], [0.3, 0.6, 0.9]],  # seq 0: one timestep has a NaN sample
                [[0.1, 0.4, 0.7], [float('nan'), float('nan'), float('nan')]],  # seq 1: timestep 1 all NaN
            ]
        )
        targets = torch.tensor([[0.4, 0.6], [0.3, 0.7]])  # [N, L]

        # get_crps_loss returns a scalar aggregate; NaN positions are excluded via nansum.
        #         # We verify that all-NaN timesteps don't propagate NaN into the final scalar.
        crps_result = get_crps_loss(samples, targets)
        print(crps_result)
        self.assertFalse(torch.isnan(crps_result), f"Expected a finite scalar but got NaN")
        self.assertTrue(torch.isfinite(crps_result), f"Expected a finite scalar but got {crps_result}")
