import logging
import math
from typing import Optional

import signatory
import torch
from torch import nn

logger = logging.getLogger(__name__)

from src.data_transformations.standardscaler import StandardScaler


class SigW1MetricExp(nn.Module):
    """Expected-signature W1 metric.

    nn.Module because it has a parameter to register.
    scale_high_degrees is not useful, and may hinder convergence; the values are standardised regardless.
    Standardise should always be true as well.

    Precision mode (`use_float64_signature`)
    ----------------------------------------
    The boundary is always float32 (inputs arrive float32, the loss is returned
    float32). The flag only changes the *internal* dtype:

    - **False (default)** — everything in float32. Bit-identical to before the
      flag existed.
    - **True** — float32 `paths` are upcast in `_signature_of`, so the
      signature, aggregation, scaler and `diff.pow(2).sum()` *and the backward
      through all of them* run in float64; the scalar loss is cast back to
      float32 on the way out. All data-derived buffers are built in float64.

    Why it matters: the loss squares `base - sample` signature coordinates.
    At convergence the generator matches the data, so that difference → 0 — a
    cancellation of two O(1) numbers. float32 bottoms out at its ~1e-7 floor,
    and once the true mismatch falls below it the loss and its **gradient**
    (which depends on the same difference) become noise — the end-of-training
    instability. float64 lowers that floor to ~1e-16. The optimiser acts on the
    gradient, so the *backward* must be float64 too, not just the loss value;
    hence the whole block is upcast. The gradient is rounded back to float32 at
    the boundary, which is fine — the cancellation is resolved in float64 first,
    so the direction is correct.

    Note: `signatory.signature` preserves dtype but is a compiled extension —
    verify the float64 path on your wheel (older CUDA builds had float64 gaps).
    Covered on CPU by `test_sigw1metric.py::TestFloat64Signature`.
    """

    def __init__(
        self,
        base_paths: torch.Tensor,
        sig_degree: int,
        scale_high_degrees: bool = True,
        standardise: bool = True,
        effective_sig_degree: Optional[int] = None,
        use_float64_signature: bool = False,
    ):
        super().__init__()

        assert sig_degree > 0, f"Signature degree must be positive. Instead, received {sig_degree}."
        assert (
            len(base_paths.shape) == 3
        ), f"`base_paths` must be a tensor of shape (N, L, D_in). Instead, received shape: {list(base_paths.shape)}"
        if effective_sig_degree is not None:
            assert (
                0 < effective_sig_degree <= sig_degree
            ), f"effective_sig_degree must be in (0, sig_degree]. Got {effective_sig_degree} with sig_degree={sig_degree}."

        # Precision mode. `input_dtype` is the boundary dtype (the model's dtype, e.g. float32):
        # inputs must arrive in it and the loss is returned in it. `_compute_dtype` is the dtype
        # of the whole internal block — signature, mean(0), scaler, diff, sum-of-squares, AND the
        # backward through all of them. When use_float64_signature is True we upcast to float64 for
        # that block so the `base - sample` cancellation at convergence is resolved to ~1e-16 rather
        # than the ~1e-7 float32 floor, then cast the scalar loss back to `input_dtype`.
        self.use_float64_signature = use_float64_signature
        self.input_dtype = base_paths.dtype
        self._compute_dtype = torch.float64 if use_float64_signature else base_paths.dtype

        self.sig_degree = sig_degree
        self._active_degree = effective_sig_degree if effective_sig_degree is not None else sig_degree
        self.paths_dim = base_paths.shape[2]
        self.sig_len = (
            self.paths_dim * (self.paths_dim**self._active_degree - 1) // (self.paths_dim - 1)
            if self.paths_dim > 1
            else self.paths_dim * self._active_degree
        )
        self.scale_high_degrees = scale_high_degrees
        self.standardise = standardise

        if self.scale_high_degrees:
            self.register_buffer('factorials', self._get_factorial_scaling_factor(), persistent=False)

        # Signature in shape (self.sig_len). Saved as non-persistent buffers so they travel with
        # .to(device) but are excluded from state_dict (data-derived, not learnable weights).
        paths_sig = self._signature_of(base_paths)
        mean_sig = paths_sig.mean(0)
        self.register_buffer('exp_paths_sig_before_scaling', mean_sig.clone(), persistent=False)
        self.register_buffer('base_exp_sig', mean_sig.clone(), persistent=False)

        # self.base_exp_sig = nn.Parameter(self._exp_sig(base_paths, False), requires_grad=False)
        assert not torch.isnan(self.base_exp_sig).any(), "The base signature contains NaNs."

        if self.standardise:
            logger.debug("Standardising the base signature.")
            # Vectors passed, of shape (base_paths.shape[2],)
            self.scaler: StandardScaler = StandardScaler(means=mean_sig, stds=paths_sig.std(0))
            self.base_exp_sig.data = self.scaler(self.base_exp_sig.data)
            # The target signature is scaled but not multiplied by factorial yet if required by scale_high_degrees.
        return

    def __call__(self, paths: torch.Tensor) -> torch.Tensor:
        assert len(paths.shape) == 3 and paths.shape[2] == self.paths_dim, (
            f"`paths` must be a 3D tensor of shape (N, L, {self.paths_dim}) and match the dimension of `base_paths`. "
            f"Instead, received shape: {str(paths.shape)}."
        )
        assert paths.dtype == self.input_dtype, (
            f"`paths.dtype` ({paths.dtype}) must match the metric's boundary dtype "
            f"({self.input_dtype}). In float64 mode the upcast to the compute dtype happens "
            f"internally; feed paths in the model's dtype, do not pre-cast at the call site."
        )

        loss = self.compute_loss(paths)
        return loss

    def compute_loss(self, paths: torch.Tensor) -> torch.Tensor:
        # \E[ ||   \E_{real}[Sig(X)] - \E_{generated}[Sig(X)]   ||^2 ]
        samples_signature_before_scaling = self._signature_of(paths)
        if self.standardise:
            samples_signature = self.scaler(samples_signature_before_scaling)
        else:
            samples_signature = samples_signature_before_scaling

        diff = self.base_exp_sig - samples_signature.mean(0)

        if self.scale_high_degrees:
            diff = diff * self.factorials

        # Normalise by signature length so the metric scale does not grow just because more signature coordinates are included.
        loss = diff.pow(2).sum()  ####### / self.sig_len
        # Cast back to the boundary dtype. In float64 mode the line above ran in float64
        # (forward and, on backward, the gradient); the optimiser sees an `input_dtype` loss.
        return loss.to(self.input_dtype)

    def _signature_of(self, paths: torch.Tensor) -> torch.Tensor:
        # Cast up to the compute dtype before signatory so the signature forward *and* its backward
        # run in `_compute_dtype` (float64 in f64 mode). autograd casts the gradient back to the
        # input dtype at this boundary, so an f32 model still receives an f32 gradient.
        if paths.dtype != self._compute_dtype:
            paths = paths.to(self._compute_dtype)
        return signatory.signature(paths, depth=self._active_degree)

    def _get_factorial_scaling_factor(self) -> torch.Tensor:
        lengths = [self.paths_dim ** (i + 1) for i in range(self._active_degree)]
        factorials = torch.cat(
            [
                torch.full((length,), math.factorial(i + 1), dtype=self._compute_dtype)
                for i, length in enumerate(lengths)
            ]
        )
        return factorials
