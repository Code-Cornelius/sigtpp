# General function to compute parameter combinations
import itertools


def parameters_product(params_grid):
    # Separate keys and values
    keys, values = zip(*params_grid.items())

    # Return a list of dictionaries with all parameter configurations
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


if __name__ == "__main__":
    # Define a grid of parameters
    params_grid = {
        "lr_G": [0.0008, 0.001],
        "lr_D": [0.002, 0.003],
        "D_steps_per_G_step": [1, 2],
        "GPU": [[0], [1]],
    }

    # Iterate over all parameter combinations
    for config in parameters_product(params_grid):
        print(config)
