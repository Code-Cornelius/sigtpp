"""Lock the paired-testing invariant of the bootstrap resampler.

Two independently trained models must draw the *same* (B, N) bootstrap index
matrix when evaluated on the same test split. Without this, paired statistical
tests (paired t / Wilcoxon / Diebold-Mariano) across models are invalid.

The production loop in ``TPPArchitecture._run_bootstrap_metrics`` delegates
index generation to :func:`generate_bootstrap_indices`. These tests exercise
that helper directly so a regression in the RNG protocol is caught here, not
silently across a paper run.
"""

import torch

from src.data_types.bootstrap_eval import generate_bootstrap_indices


def test_generate_bootstrap_indices_shape_and_range():
    idx = generate_bootstrap_indices(N=50, B=10, seed=42)
    assert idx.shape == (10, 50)
    assert idx.dtype == torch.int64
    assert idx.device.type == "cpu"
    assert int(idx.min()) >= 0
    assert int(idx.max()) < 50


def test_generate_bootstrap_indices_paired_across_calls():
    """The defining invariant: identical (N, B, seed) -> bit-identical matrix."""
    a = generate_bootstrap_indices(N=100, B=20, seed=42)
    b = generate_bootstrap_indices(N=100, B=20, seed=42)
    assert torch.equal(a, b)


def test_generate_bootstrap_indices_isolated_from_global_rng():
    """Perturbing the global torch RNG between calls must not affect the matrix."""
    a = generate_bootstrap_indices(N=64, B=8, seed=42)
    torch.manual_seed(123456)
    _ = torch.randn(1000)  # consume entropy from the global generator
    b = generate_bootstrap_indices(N=64, B=8, seed=42)
    assert torch.equal(a, b)


def test_generate_bootstrap_indices_different_seeds_differ():
    a = generate_bootstrap_indices(N=100, B=20, seed=42)
    b = generate_bootstrap_indices(N=100, B=20, seed=43)
    assert not torch.equal(a, b)


def test_generate_bootstrap_indices_b1_is_arange():
    """B=1 is the no-bootstrap fast path: deterministic single pass, no RNG draw."""
    idx = generate_bootstrap_indices(N=7, B=1, seed=42)
    assert idx.shape == (1, 7)
    assert torch.equal(idx[0], torch.arange(7))


def test_generate_bootstrap_indices_rejects_bad_args():
    import pytest

    with pytest.raises(AssertionError):
        generate_bootstrap_indices(N=0, B=5)
    with pytest.raises(AssertionError):
        generate_bootstrap_indices(N=10, B=0)
