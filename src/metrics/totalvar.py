import torch


def total_var(paths: torch.Tensor) -> torch.Tensor:
    """
    Compute the total variation norm of a batch of paths.

    Parameters:
        paths (torch.Tensor): An NxLxD tensor representing N paths with L points in D dimensions.

    Returns:
        torch.Tensor: The bounded variation norm of the path.
    """
    assert (
        len(paths.shape) == 3
    ), f"`paths` must be a 3D tensor of shape (N, L, D_in). Instead, received shape: {list(paths.shape)}"
    return torch.sum(torch.abs(torch.diff(paths, dim=1)), dim=(1, 2))


if __name__ == "__main__":
    # Example usage: (2,4,3) tensor.
    path = torch.tensor(
        [[[0, 1, 1], [1, 2, 1], [2, 1, 1], [3, 3, 1]], [[0, 1, 1], [1, 2, 1], [2, 1, 1], [4, 4, 1]]],
        dtype=torch.float32,
    )

    print(path.shape)
    tot_var = total_var(path)
    print("Total Variation Norm:", tot_var)

    print(tot_var.repeat(path.shape[0], path.shape[1], 1))
    tot_var_scale = tot_var.view(-1, 1, 1)
    print(tot_var_scale)
    scaled_tot_var = total_var(path / tot_var_scale)
    print("Scaled Total Variation Norm:", scaled_tot_var)
