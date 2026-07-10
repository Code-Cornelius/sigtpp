import logging
from typing import List

import signatory
import torch

logger = logging.getLogger(__name__)


class SigW1DegreeDetector(object):
    """
    Infer the effective signature truncation degree from `base_paths`.

    A signature degree d is "dead" when every signature term at depth d has
    std < 1e-8 across `base_paths`. Dead degrees carry no distributional
    signal for the SigW1 loss, so the cost of computing them is wasted.

    The effective degree is `k - 1`, where `k` is the start of the contiguous
    block of dead degrees ending at `max(dead_degrees)`. Equivalently: find the
    smallest k such that {k, k+1, ..., max(dead)} are all dead. We do **not**
    assume the dead set is contiguous as a whole — alive degrees below the
    trailing block (even if some are isolated-dead) are kept. Examples
    (sig_degree = 10):
      - dead = {10}              -> effective = 9   (drop just the top)
      - dead = {9, 10}           -> effective = 8   (drop the dead tail)
      - dead = {8, 9, 10}        -> effective = 7   (drop the whole tail)
      - dead = {4, 7}            -> effective = 6   (trim from 7; keep 1-6
                                                      including isolated
                                                      dead degree 4)
      - dead = {4, 7, 8, 9, 10}  -> effective = 6   (trim from 7; keep 1-6
                                                      including dead 4)
      - dead = {5}               -> effective = 4   (singleton starts a
                                                      one-element trailing
                                                      block)
      - dead = {1, 2, 3}         -> effective = 1   (floored)

    Rationale: alive degrees *above* the trailing dead block are treated as
    noise that escaped the threshold and dropped together with the tail.
    Alive degrees *below* the trailing block are kept even if there are
    isolated dead degrees among them, because truncation is a single
    max-depth cutoff and we'd rather pay for one wasted depth than discard
    real signal. Floored at 1 so it remains a valid signature truncation
    depth.
    """

    @staticmethod
    def _trim_to_effective_degree(dead_degrees: List[int], sig_degree: int) -> int:
        """Return `k - 1` where `k` is the start of the trailing dead block.

        That block is the maximal contiguous run of dead degrees ending at
        `max(dead_degrees)`. Floored at 1; returns `sig_degree` if no dead
        degrees were detected.
        """
        if not dead_degrees:
            return sig_degree
        dead_set = set(dead_degrees)
        k = max(dead_set)
        while k - 1 in dead_set:
            k -= 1
        return max(1, k - 1)

    def __init__(
        self,
        base_paths: torch.Tensor,
        sig_degree: int,
    ):
        assert sig_degree > 0, f"Signature degree must be positive. Instead, received {sig_degree}."
        assert (
            len(base_paths.shape) == 3
        ), f"`base_paths` must be a tensor of shape (N, L, D_in). Instead, received shape: {list(base_paths.shape)}"

        self.sig_degree = sig_degree
        self.paths_dim = base_paths.shape[2]

        paths_sig = signatory.signature(base_paths, depth=sig_degree)
        raw_stds = paths_sig.std(0)

        # Dtype-aware "dead" threshold. The previous hardcoded 1e-8 sat below
        # float32 eps (1.19e-7), so mathematically-zero coordinates with f32
        # accumulation noise (~5e-8) were not flagged as dead, while the same
        # coordinates in f64 (round-off ~1e-15) were. Using 10 * dtype_eps
        # tracks the actual noise floor; the 1e-12 floor stops the f64 path
        # from getting so tight that round-off-perturbed deterministic terms
        # (~1e-14) escape detection.
        dtype_eps = torch.finfo(paths_sig.dtype).eps
        dead_threshold = max(10.0 * dtype_eps, 1e-12)

        dead: List[int] = []
        ptr = 0
        for _deg in range(1, sig_degree + 1):
            n = self.paths_dim**_deg
            if (raw_stds[ptr : ptr + n] < dead_threshold).all():
                dead.append(_deg)
            ptr += n
        self.dead_degrees = dead
        self.effective_sig_degree = self._trim_to_effective_degree(dead, sig_degree)

        if dead:
            if self.effective_sig_degree == 1 and 1 in dead:
                logger.warning(
                    "SigW1DegreeDetector: trailing dead block reaches degree 1 — " "flooring effective_sig_degree at 1."
                )
            logger.info(
                "SigW1DegreeDetector: dead degrees %s detected (sig_degree=%d, D=%d, "
                "dtype=%s, threshold=%.2e). Effective signature degree "
                "(trim from start of trailing dead block): %d.",
                dead,
                sig_degree,
                self.paths_dim,
                paths_sig.dtype,
                dead_threshold,
                self.effective_sig_degree,
            )
