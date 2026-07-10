import logging
from enum import Enum
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


from src.data_transformations.standardscaler import StandardScaler
from src.data_transformations.statscompute import (
    variable_len_standard_stats,
)


class ScalingStrategy(str, Enum):
    """
    Enum representing the available scaling strategies for the ExpScaler.

    - NO_SCALING: No additional scaling is applied after the logarithmic transformation.
    - NAIVE: The data is scaled using simple mean and standard deviation, computed through
             the `variable_len_standard_stats` function.
    """

    NO_SCALING = "no_scaling"
    """
    No scaling is applied. The data undergoes a logarithmic transformation, but no further
    standardization is performed. Instead we just log and shift.
    This option is suitable when the original scale of the data should be preserved post-transformation. 
    """

    NAIVE = "naive"
    """
    The data is scaled using a simple approach where the mean and standard deviation are 
    computed across the time steps and features. The `variable_len_standard_stats` function 
    is used to handle variable-length sequences, and the data is standardized based on these 
    computed statistics. This is a general-purpose scaling method.
    """


class ExpScaler(nn.Module):
    """
    A scaler that applies a logarithmic transformation and optionally standard scaling to the input tensor.

    The transformation is defined as:

    .. math::
        y = \log(x + \text{{shift_param}})

    where `x` is the original input, and `shift_param` ensures numerical stability.

    The scaling strategy (optional) can be chosen from an enum, and the data is standardized using a `StandardScaler`,
    which is fitted based on the selected scaling strategy:

    - `NO_SCALING`: No scaling is applied.
    - `NAIVE`: A simple mean and variance-based scaling is applied using the `variable_len_standard_stats` function.

    When `lengths` is not provided, it is assumed that all data have the same length, and this is handled internally
    by generating lengths as the full size of the sequence dimension of `data`.
    """

    def __init__(
        self,
        fitted_data: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        concentration_factor: float = 1.0,
        shift_param: float = 0.0,
        scaling_strategy: ScalingStrategy = ScalingStrategy.NO_SCALING,
    ) -> None:
        """
        Initialize the ExpScaler with paths, lengths, and transformation parameters.

        Args:
            fitted_data (torch.Tensor): A 3D tensor of shape `(batch_size, sequence_length, num_features)`
                                  representing the input paths.
            lengths (Optional[torch.Tensor]): A 1D tensor representing the lengths of each path.
                                              If not provided, it defaults to full path lengths.
            concentration_factor (float): A factor used to concentrate the target distribution.
                                          This has proven to be effective in reducing mode collapse. A value between 1 and 10 is recommended.
            shift_param (float): A shift parameter to ensure numerical stability in the log transformation.
            scaling_strategy (ScalingStrategy): The strategy to compute the mean and standard deviation for
                                                scaling. Defaults to `NO_SCALING`.
        """
        super().__init__()

        assert len(fitted_data.shape) == 3, f"Expected `fitted_data` to be 3D, but got {list(fitted_data.shape)}."
        assert (
            fitted_data.shape[1] > 0
        ), f"Expected `fitted_data` to have a sequence length > 0, but got {fitted_data.shape[1]}."
        assert (
            fitted_data.shape[2] > 0
        ), f"Expected `fitted_data` to have a feature dimension > 0, but got {fitted_data.shape[2]}."

        # If lengths are not provided, assume full lengths
        if lengths is None:
            lengths = torch.full((fitted_data.shape[0],), fitted_data.shape[1] - 1, dtype=torch.long)
        else:
            assert len(lengths.shape) == 1, f"Expected `lengths` to be 1D, but got {list(lengths.shape)}."
            assert (
                fitted_data.shape[0] == lengths.shape[0]
            ), f"Mismatched batch sizes: fitted_data have shape {list(fitted_data.shape)}, but lengths have shape {list(lengths.shape)}."

        self.dim_paths = fitted_data.shape[-1]

        self.concentration_factor = concentration_factor
        self.shift_param = shift_param

        # Be careful to ensure it remains positive!
        assert self.shift_param >= 0.0, f"Expected `shift_param` to be non-negative, but got {self.shift_param}."

        ################ Observation:
        # Lots of values in our data are consecutive, example:
        # [ 589824.  348160. 6340608.    4096.       0.  417792.       0.]
        # It would need to be investigated, but it is probably due to poor data collection.
        # A simple fix is to add a shift param, making the data slightly above zero.

        self.scaling_fct = lambda x: torch.log(x + self.shift_param)
        self.unscaling_fct = lambda x: torch.exp(x) - self.shift_param

        # Set up the standard scaler based on the selected scaling strategy
        if scaling_strategy is ScalingStrategy.NO_SCALING:
            logger.debug("No scaling strategy selected, skipping standard scaling inside the exponential scaler.")
            self.standard_scaler: Optional[StandardScaler] = None
        elif scaling_strategy is ScalingStrategy.NAIVE:
            logger.debug("Using naive estimators to set the standard scaler inside the exponential scaler.")
            scaled_data = self.scaling_fct(fitted_data)
            mean_log_intensity, std_log_intensity = variable_len_standard_stats(scaled_data, lengths, True)
            self.standard_scaler: Optional[StandardScaler] = StandardScaler(
                means=mean_log_intensity, stds=std_log_intensity
            )
        else:
            raise ValueError(f"Unknown scaling strategy: {scaling_strategy}.")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the logarithmic transformation followed by optional standard scaling to the input `data`.

        The transformation is:

        .. math::
            y = \frac{{\text{{standard}}\left[\log(x + \text{{shift_param}})\right]}}{{\text{{concentration_factor}}}}

        Args:
            data (torch.Tensor): A 3D tensor of input data with shape `(batch_size, sequence_length, num_features)`.

        Returns:
            torch.Tensor: The transformed tensor with the same shape as the input.
        """
        assert len(data.shape) == 3, f"Expected `data` to be 3D, but got {list(data.shape)}."
        assert data.shape[2] == self.dim_paths, (
            f"Expected the last dimension of the input to be {self.dim_paths}, " f"but got {data.shape[2]} instead."
        )

        # Apply log transformation to the first feature (dim 0)
        values = self.scaling_fct(data)
        if self.standard_scaler:
            values = self.standard_scaler(values)
        values = values / self.concentration_factor
        return values

    def unscale(self, data: torch.Tensor) -> torch.Tensor:
        r"""
        Revert the scaling transformation to recover the original data.

        The inverse transformation is:

        .. math::
            x = \exp\left[\text{{standard}}^{-1}\left(y \times \text{{concentration_factor}}\right)\right] - \text{{shift_param}}

        Args:
            data (torch.Tensor): A 3D tensor of transformed data with shape `(batch_size, sequence_length, num_features)`.

        Returns:
            torch.Tensor: The unscaled tensor.
        """
        assert data.shape[2] == self.dim_paths, (
            f"Expected the last dimension of the input to be {self.dim_paths}, " f"but got {data.shape[2]} instead."
        )
        values = data * self.concentration_factor
        if self.standard_scaler:
            values = self.standard_scaler.inverse_transform(values)
        values = self.unscaling_fct(values)
        return values

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  shift_param={self.shift_param},\n"
            f"  concentration_factor={self.concentration_factor},\n"
            f"  standard_scaler={self.standard_scaler}\n"
            f")"
        )
