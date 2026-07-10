import torch
import torch.nn as nn


class UniformRangeSampler(nn.Module):
    def __init__(self, start: float, end: float, step: float) -> None:
        """
        Initializes the discrete uniform sampler with the given range and step size.

        Args:
            start (float): The start of the range.
            end (float): The end of the range.
            step (float): The step size between values in the range.
        """
        super().__init__()
        values: torch.Tensor = torch.arange(start, end, step, dtype=torch.float32)
        self.sampling_state_space = nn.Parameter(values, requires_grad=False)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Samples values uniformly from the range.

        Args:
            num_samples (int): The number of samples to draw.

        Returns:
            torch.Tensor: A tensor of sampled values.
        """
        indices = torch.randint(
            0, len(self.sampling_state_space), (num_samples,), device=self.sampling_state_space.device
        )
        sampled_values = self.sampling_state_space[indices]
        return sampled_values


if __name__ == '__main__':
    # Example usage
    start = 30000
    end = 40000
    step = 15

    # Initialize the sampler
    sampler = UniformRangeSampler(start, end, step)

    # Sample values
    num_samples = 5
    samples = sampler.sample(num_samples)

    print(f"Sampled values: {samples}")
