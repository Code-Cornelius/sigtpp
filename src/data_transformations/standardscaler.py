import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class StandardScaler(nn.Module):
    def __init__(self, *, means: torch.Tensor, stds: torch.Tensor, **kwargs):
        super().__init__()
        assert means.shape == stds.shape, (
            f"`means` and `stds` should have the same shape but are of shapes "
            f"{list(means.shape)} and {list(stds.shape)}."
        )

        # Check for NaN or infinite values in means and stds
        if torch.isnan(means).any() or torch.isnan(stds).any():
            logger.error(f"NaN values detected in means: {means} or stds: {stds}")
            raise ValueError("NaN values detected in means or stds.")

        if torch.isinf(means).any() or torch.isinf(stds).any():
            logger.error(f"Infinite values detected in means: {means} or stds: {stds}")
            raise ValueError("Infinite values detected in means or stds.")

        is_scaling_scalar = len(means.shape) == 0

        # When it is a scalar, the shape is wrong and cannot be broadcast so we reshape it first.
        if is_scaling_scalar:
            assert len(stds.shape) == 0, f"`stds` should be a scalar but is of shape {list(stds.shape)}."
            means = means.view(1)
            stds = stds.view(1)

        self.register_buffer('mean_paths', means, persistent=False)  # Shape (D,)
        self.register_buffer('std_paths', stds, persistent=False)  # Shape (D,)

        # Prevents division by 0. Happens when the vector is constant.
        # Dtype-aware floor: matches SigW1DegreeDetector's dead-degree threshold so
        # a channel flagged "dead" by the detector also gets neutralised here. The
        # previous hardcoded 1e-8 sat below float32 eps (~1.19e-7), so f32-noise
        # coordinates could slip past and produce gradients amplified by 1/(small).
        constant_threshold = max(10.0 * torch.finfo(self.std_paths.dtype).eps, 1e-12)
        cst_tensor = self.std_paths < constant_threshold
        if cst_tensor.any():
            self.std_paths[cst_tensor] = 1.0
            # Log the dimensions that are constants
            logger.log(
                5,
                "During standardisation, some of the passed data's dimensions were constant "
                "(std < %.2e for dtype %s). We use a std of 1 for these dimensions: %s.",
                constant_threshold,
                self.std_paths.dtype,
                cst_tensor.nonzero(as_tuple=True)[0],
            )

        logger.log(
            5,
            "Standardisation with " + ", ".join("%s (+/- %s)" for _ in means),
            *interleave_tensors(means, stds),
        )
        return

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # Does not transform the input
        return (inputs - self.mean_paths) / self.std_paths

    def inverse_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        # Does not transform the input
        return inputs * self.std_paths + self.mean_paths

    def __repr__(self) -> str:
        stats = ", ".join(
            f"{mean:.4f} (+/- {std:.4f})" for mean, std in zip(self.mean_paths.tolist(), self.std_paths.tolist())
        )
        return f"{self.__class__.__name__}({stats})"


def interleave_tensors(vect1: torch.Tensor, vect2: torch.Tensor):
    if len(vect1) != len(vect2):
        raise ValueError("Tensors must be of the same length")
    return [item for pair in zip(vect1, vect2) for item in pair]
