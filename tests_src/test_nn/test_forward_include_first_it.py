"""Tests for forward(include_first_it) parameter."""

import pytest

pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

import unittest
from unittest.mock import patch

import torch


def make_stub_arch():
    """Return a minimal stub that exposes forward() without a full Lightning setup."""
    from src.nn.architectures.tpp_architecture import TPPArchitecture

    # TPPArchitecture is abstract; create a minimal concrete subclass.
    class _ConcreteArch(TPPArchitecture):
        def filter_patho_seqs(self, *a, **kw):
            pass

        def sample(self, *a, **kw):
            pass

        def training_step(self, *a, **kw):
            pass

        def validation_step(self, *a, **kw):
            pass

    arch = object.__new__(_ConcreteArch)
    # Minimal Lightning/nn.Module internal state so attribute lookups don't crash.
    arch.__dict__['_parameters'] = {}
    arch.__dict__['_buffers'] = {}
    arch.__dict__['_modules'] = {}
    arch.__dict__['_backward_hooks'] = {}
    arch.__dict__['_forward_hooks'] = {}
    arch.__dict__['_forward_pre_hooks'] = {}
    arch.__dict__['_state_dict_hooks'] = {}
    arch.__dict__['_load_state_dict_pre_hooks'] = {}
    arch.__dict__['_non_persistent_buffers_set'] = set()
    arch.time_max = 10.0
    arch.use_marks = False

    class IdentityScaler:
        def unscale(self, x):
            return x

    arch.scaler_exp = IdentityScaler()
    return arch


def _make_sample_output(n, l, d, time_max):
    """Produce deterministic inter-arrival times whose cumsum stays below time_max."""
    dt = time_max / (l + 1)
    return torch.full((n, l, d), dt), torch.full((n,), l, dtype=torch.long), None


class TestForwardIncludeFirstIt(unittest.TestCase):

    def test_forward_false_strips_first_value(self):
        print("\nTesting forward(include_first_it=False)...")
        arch = make_stub_arch()
        N, L, D = 4, 6, 1
        sample_output = _make_sample_output(N, L, D, arch.time_max)
        with patch.object(arch, 'sample', return_value=sample_output):
            samples, lens, _ = arch.forward(N, include_first_it=False)

        print(f"Sample shape: {samples.shape}, Max length: {lens.max().item()}")
        self.assertEqual(samples.shape[1], L - 1, f"Expected length {L-1}, got {samples.shape[1]}")
        self.assertLessEqual(lens.max().item(), L - 1, f"Expected max length <= {L-1}, got {lens.max().item()}")
        print("Test forward(include_first_it=False) passed.")

    def test_forward_true_keeps_first_value(self):
        print("\nTesting forward(include_first_it=True)...")
        arch = make_stub_arch()
        N, L, D = 4, 6, 1
        sample_output = _make_sample_output(N, L, D, arch.time_max)
        with patch.object(arch, 'sample', return_value=sample_output):
            samples, lens, _ = arch.forward(N, include_first_it=True)

        print(f"Sample shape: {samples.shape}, Max length: {lens.max().item()}")
        self.assertEqual(samples.shape[1], L, f"Expected length {L}, got {samples.shape[1]}")
        self.assertLessEqual(lens.max().item(), L, f"Expected max length <= {L}, got {lens.max().item()}")
        print("Test forward(include_first_it=True) passed.")

    def test_forward_true_lens_one_more_than_false(self):
        print("\nTesting lens difference between include_first_it=True and False...")
        arch = make_stub_arch()
        N, L, D = 4, 6, 1
        fixed_sample = _make_sample_output(N, L, D, arch.time_max)

        with patch.object(arch, 'sample', return_value=fixed_sample):
            _, lens_false, _ = arch.forward(N, include_first_it=False)
        with patch.object(arch, 'sample', return_value=fixed_sample):
            _, lens_true, _ = arch.forward(N, include_first_it=True)

        print(f"Lens (False): {lens_false}")
        print(f"Lens (True):  {lens_true}")
        self.assertTrue(torch.all(lens_true == lens_false + 1), "lens_true should be lens_false + 1")
        print("Test lens difference passed.")

    def test_forward_no_default_raises_type_error(self):
        print("\nTesting forward() without include_first_it parameter (should raise TypeError)...")
        arch = make_stub_arch()
        with self.assertRaises(TypeError):
            arch.forward(4)
        print("Test TypeError on missing parameter passed.")


if __name__ == "__main__":
    unittest.main()
