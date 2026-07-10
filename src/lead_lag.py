import torch


def lead_lag_transformation(data: torch.Tensor) -> torch.Tensor:
    """
    Applies a 'double-then-lead-lag' transformation to a 3D time-series tensor.

    Given an input of shape (N, L, D):
      1) Duplicate along the time dimension (L) so each point is repeated once.
         This changes the time dimension from L -> 2L.
      2) Apply the lead-lag slicing along the duplicated time dimension:
         - 'lead' row is everything except the first point (shape: 2L - 1)
         - 'lag' row is everything except the last point  (shape: 2L - 1)

    The final output shape is (N, 2L - 1, 2 * D):
      - dimension=0: batch dimension (N)
      - dimension=1: new time dimension (2L - 1)
      - dimension=2: feature dimension (2*D)

    Args:
        data (torch.Tensor):
            A 3D tensor of shape (N, L, D), where:
                - N is the batch size
                - L is the number of time steps
                - D is the number of features per time step

    Returns:
        torch.Tensor:
            A 4D tensor of shape (N, 2L - 1, 2*D).

    Example:
        >>> # Example with N=1 (single sequence), L=3, D=1
        >>> import torch
        >>> data = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        >>> output = lead_lag_transformation(data)
        >>> # output will have shape (1, 5, 2):
        >>> #   The time dimension: 3 -> duplicated to 6 -> lead-lag => 5
        >>> print(output.shape)
        torch.Size([1, 5, 2])
    """

    # Check input dimensions
    assert data.dim() == 3, f"Expected a 3D tensor (N, L, D), but got shape {data.shape}."

    # 1) Duplicate each entry along the time dimension: (N, L, D) -> (N, 2L, D)
    data_duplicated = data.repeat_interleave(2, dim=1)

    # 2) Lead-Lag slicing along the time dimension
    lead = data_duplicated[:, 1:, :]  # (N, 2L - 1, D)
    lag = data_duplicated[:, :-1, :]  # (N, 2L - 1, D)

    # 3) Stack into (N, 2L - 1, 2*D)
    output = torch.cat((lead, lag), dim=2)
    return output
