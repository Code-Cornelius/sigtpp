"""Bootstrap aggregation helpers for test-step evaluation.

The bootstrap unit is the test sequence. Generated samples are produced once
upstream and reused across replicates. This module provides two views of the
same per-replicate data:

* ``aggregate_bootstrap_metrics`` — collapses B replicates to mean/std scalars
  for the existing txt output schema.
* ``build_per_replicate_matrix`` — preserves the raw (B,) vectors so that
  downstream code can run paired statistical tests (paired t / Wilcoxon /
  Diebold–Mariano) across models without re-running the bootstrap loop.
"""

import logging
from typing import Dict, Iterable, List

import numpy as np
import torch


logger = logging.getLogger(__name__)
from src.utils.result_helpers import coerce_float, summarise_values


def generate_bootstrap_indices(N: int, B: int, seed: int = 42) -> torch.Tensor:
    """Generate the ``(B, N)`` index matrix for paired bootstrap resampling.

    Uses a private CPU ``torch.Generator`` so the matrix is deterministic across
    runs and isolated from global RNG state. Two calls with identical
    ``(N, B, seed)`` return bit-identical tensors — this is the precondition
    for paired statistical tests across independently trained models.

    ``B == 1`` is the no-bootstrap fast path and returns ``torch.arange(N)``
    reshaped to ``(1, N)`` with no RNG draw (degenerate single deterministic pass).

    Args:
        N: number of test sequences (the resample pool size).
        B: number of bootstrap replicates.
        seed: deterministic seed for the local CPU generator.

    Returns:
        Long tensor of shape ``(B, N)`` on CPU; callers move it to the model device.
    """
    assert N >= 1, f"N must be >= 1, got {N}"
    assert B >= 1, f"B must be >= 1, got {B}"
    if B == 1:
        return torch.arange(N).unsqueeze(0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.stack([torch.randint(0, N, (N,), generator=gen) for _ in range(B)])


def aggregate_bootstrap_metrics(per_replicate: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-replicate metric dicts into a flat ``{name_mean, name_std}`` dict.

    The bootstrap output schema records only mean and standard deviation across
    replicates. NaN-only metrics (every replicate failed) emit ``nan`` for both;
    single-replicate metrics emit the value as the mean and ``0.0`` as the std.

    Args:
        per_replicate: One dict of ``{metric_name: value}`` per replicate. Values
            must be coercible to ``float``; missing keys in some replicates are
            treated as NaN.

    Returns:
        Dict mapping ``"<name>_mean"`` and ``"<name>_std"`` to floats.
    """
    assert per_replicate, f"aggregate_bootstrap_metrics requires at least one replicate, got {per_replicate!r}."

    keys: Iterable[str] = sorted({k for rep in per_replicate for k in rep.keys()})
    aggregated: Dict[str, float] = {}
    for k in keys:
        mean, std, _n_valid = summarise_values(rep.get(k, float("nan")) for rep in per_replicate)
        aggregated[f"{k}_mean"] = mean
        aggregated[f"{k}_std"] = std
    return aggregated


def build_per_replicate_matrix(per_replicate: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    """Convert list of per-replicate metric dicts to a dict of ``(B,)`` float arrays.

    Missing keys in some replicates are filled with NaN, matching the convention
    in :func:`aggregate_bootstrap_metrics`. Each returned array is asserted to
    have shape ``(len(per_replicate),)`` so the per-replicate contract is enforced
    at the only construction site.
    """
    B = len(per_replicate)
    keys: Iterable[str] = sorted({k for rep in per_replicate for k in rep.keys()})
    out: Dict[str, np.ndarray] = {
        k: np.array(
            [coerce_float(rep.get(k, float("nan"))) for rep in per_replicate],
            dtype=float,
        )
        for k in keys
    }
    for k, arr in out.items():
        assert arr.shape == (B,), f"per-replicate vector for {k!r} has shape {arr.shape}, expected ({B},)"
    return out
