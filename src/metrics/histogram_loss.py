import logging
import math
import typing

import matplotlib.pyplot as plt
import torch
from torch import nn

logger = logging.getLogger(__name__)


def histogram_torch(x, bins, density=True):
    """
    Computes the histogram of a tensor using provided bins, ignoring NaNs.

    Args:
    - x (torch.Tensor): Input tensor.
    - bins (torch.Tensor): Precomputed bin edges.
    - density (bool): Whether to normalize the histogram.

    Returns:
    - count (torch.Tensor): Counts of the histogram bins on the device of x.
    - bins (torch.Tensor): Bin edges on the device of x.
    """
    # Remove NaNs from the input tensor
    x = x[~torch.isnan(x)]
    # Return zero count if no valid entries remain
    if x.numel() == 0:
        logger.warning("There are only NaNs in the input tensor for histogram computation.")
        return torch.zeros(len(bins) - 1, device=x.device), bins

    n_bins = len(bins) - 1

    # torch.histogram does not exist before 1.10 pytorch.
    count = torch.histc(x, bins=n_bins, min=bins[0].item(), max=bins[-1].item())
    if density:
        # Divide by the total count but multiple by n_bins so the average bin height is 1.
        count = count / x.numel() * n_bins
    return count, bins


class HistogramLoss(nn.Module):
    r"""
        HistogramLoss compares the distributions of real and generated (fake) data
        using histograms for each feature and time step. It quantifies how closely
        the generated data replicates the statistical characteristics of the real data.


    Formulas used for the loss computation:

    1. **Bin Centers**:

    .. math::
        \text{center\_bin\_loc} = \frac{bins[i+1] + bins[i]}{2}

    where `bins[i]` are the edges of the bins.

    2. **Density of Fake Data**:

    .. math::
        \text{density} = \frac{1}{|\Delta|} \cdot \frac{1}{N} \sum_{j=1}^N I\left(\frac{|\text{fake\_sample} - \text{center\_bin\_loc}|}{|\Delta|/2} \leq 1 \right)

    where `\Delta` is the bin width, and the indicator function ensures that the
    fake sample is within the bin centered at `center_bin_loc`.

    3. **Loss**:

    .. math::
        \text{loss} = \frac{1}{|\Delta|} \cdot | \text{density}_{real} - \text{density}_{fake} |

    This computes the absolute difference between the real and fake densities,
    normalized by the bin width `\Delta`.

    ----
    Core Idea:
        For each time-feature pair:
        - It constructs a histogram from the real data (precomputed).
        - It constructs a histogram from the generated data (on-the-fly).
        - The loss is computed as the average bin-wise absolute difference
          between these histograms, plus a penalty for generated samples falling outside real data bounds.

    ----
    Loss Range and Interpretation:
        - The returned loss is normalized and lies between **0.0 and 1.0**.
        - A loss of 0.0 means the real and fake histograms are identical.
        - A loss of 1.0 means the distributions are completely non-overlapping.
        - This can be interpreted as a **percentage error between the two empirical distributions**.
        - The division by 2.0 ensures a proper normalization since the maximal bin-wise difference per pair is 2.

    ----
    Special Considerations:
        - NaNs in data are ignored.
        - If a time-feature pair contains only NaNs, a maximal error is assigned for that entry.
        - Features with more valid samples contribute more to the final loss (weighted by sample size).

    ----
    Suggested Usage:
        ```python
        N, L, D = 1000, 5, 3
        real_data = torch.randn(N, L, D)
        fake_data = torch.randn(N, L, D) * 0.8 + 0.2  # Slightly different distribution

        loss_fn = HistogramLoss(real_data, n_bins=20)
        loss_value = loss_fn(fake_data)
        print("Histogram Loss:", loss_value.item())
        ```

    ----
    Method to Use:
        - Call `loss_fn(fake_tensor)` after constructing with real data.
        - Optionally call `.plot_histograms(fake_tensor)` to visually inspect differences.
    """

    @staticmethod
    def num_bins_freedman_diaconis_rule(num_samples):
        """
        Calculate the number of bins using the Freedman-Diaconis rule.

        The Freedman-Diaconis rule suggests the optimal number of bins to use in a histogram
        by minimizing the variance of the histogram.

        The number of bins is computed as:

        .. math::

            bins = 2 * n^{1/3}

        where:
            - n is the number of samples.

        Args:
            num_samples (int): The number of samples in the dataset.

        Returns:
            int: The computed number of bins based on the Freedman-Diaconis rule.
        """
        return int(round(2.0 * math.pow(num_samples, 1.0 / 3.0), 0))

    @staticmethod
    def precompute_histograms(x: torch.Tensor, n_bins: int):
        densities: typing.List = []
        center_bin_locs: typing.List = []
        bin_widths: typing.List = []
        bin_edges: typing.List = []
        sample_sizes: typing.List = []

        for time_step in range(x.shape[1]):
            per_time_densities = []
            per_time_center_bin_locs = []
            per_time_bin_widths = []
            feature_bins = []
            sample_size = []
            for feature_idx in range(x.shape[2]):
                x_ti = x[:, time_step, feature_idx].reshape(-1)
                x_ti = x_ti[~torch.isnan(x_ti)]  # Remove NaNs

                if x_ti.numel() == 0:
                    # Handle the case where all values are NaNs by appending a tensor of zeros
                    per_time_densities.append(torch.zeros(n_bins, device=x.device))
                    per_time_center_bin_locs.append(torch.zeros(n_bins, device=x.device))
                    per_time_bin_widths.append(torch.tensor(1.0, device=x.device))  # Default bin width
                    feature_bins.append(torch.zeros(n_bins + 1, device=x.device))
                    sample_size.append(torch.tensor(0, device=x.device))
                    continue

                min_val, max_val = x_ti.min().item(), x_ti.max().item()
                # We catch here the case when the values are all the same for a time and feature.
                if len(x_ti) > 1 and abs(max_val - min_val) < 1e-10:
                    max_val = max_val + 1e-5
                    min_val = min_val - 1e-5
                    logger.debug(
                        "All values are the same for a time and feature. "
                        "Adding a small perturbation to the range. "
                        "The loss might not be as representative as desired."
                    )
                elif len(x_ti) == 1:
                    # Case that happens naturally for our application, so no need to log it.
                    max_val = max_val + 1e-5
                    min_val = min_val - 1e-5

                bins = torch.linspace(min_val, max_val, n_bins + 1, device=x.device)
                density, bins = histogram_torch(x_ti, bins, density=True)
                per_time_densities.append(density)
                bin_width = bins[1] - bins[0]
                center_bin_loc = 0.5 * (bins[1:] + bins[:-1])
                per_time_center_bin_locs.append(center_bin_loc)
                per_time_bin_widths.append(bin_width)
                feature_bins.append(bins)
                sample_size.append(torch.tensor(x_ti.numel(), device=x.device))

            densities.append(per_time_densities)
            center_bin_locs.append(per_time_center_bin_locs)
            bin_widths.append(per_time_bin_widths)
            bin_edges.append(feature_bins)
            sample_sizes.append(sample_size)

        # For all time stamps, they should be the same dimensions, hence stackable.
        # Can't do ParamList of ParamList. First nest per feature, second per time and inside per bin.
        densities: typing.List = [torch.stack(d) for d in densities]
        center_bin_locs: typing.List = [torch.stack(l) for l in center_bin_locs]
        bin_widths: typing.List = [torch.stack(d) for d in bin_widths]
        bin_edges: typing.List = [torch.stack(b) for b in bin_edges]
        sample_sizes: typing.List = [torch.stack(s) for s in sample_sizes]

        return densities, center_bin_locs, bin_widths, bin_edges, sample_sizes

    def __init__(self, x_real: torch.Tensor, n_bins: int):
        """
        Initializes the HistogramLoss with the real data distribution.

        Args:
        - x_real (torch.Tensor): Real data tensor of shape (N, L, D).
        - n_bins (int): Number of bins for the histograms.
        """
        super().__init__()
        self.n_bins = n_bins
        self.num_samples, self.num_time_steps, self.num_features = x_real.shape

        # Log the initialization details
        logger.debug(
            f"Initializing HistogramLoss with {self.num_samples} samples, {self.num_time_steps} time steps, and {self.num_features} features."
        )

        (_densities, _center_bin_locs, _bin_widths, _bin_edges, _sample_sizes) = self.precompute_histograms(
            x_real, n_bins
        )

        # Stacked tensors registered as non-persistent buffers: they travel with .to(device)
        # but are excluded from state_dict (data-derived statistics, not learnable weights).
        # Shapes: densities (L, D, n_bins), center_bin_locs (L, D, n_bins),
        #         bin_widths (L, D), bin_edges (L, D, n_bins+1), sample_sizes (L, D).
        self.register_buffer('densities', torch.stack(_densities, dim=0), persistent=False)
        self.register_buffer('center_bin_locs', torch.stack(_center_bin_locs, dim=0), persistent=False)
        self.register_buffer('bin_widths', torch.stack(_bin_widths, dim=0), persistent=False)
        self.register_buffer('bin_edges', torch.stack(_bin_edges, dim=0), persistent=False)
        self.register_buffer('sample_sizes', torch.stack(_sample_sizes, dim=0), persistent=False)
        return

    def compute(self, x_fake):
        """
        Computes the histogram loss between real and fake data distributions.
        We noticed issues in the case of the comparison of densities ala Dirac measure. Use with caution in that case.


        Args:
        - x_fake (torch.Tensor): Fake data tensor of shape (N, L, D).

        Returns:
        - all_losses (torch.Tensor): Component-wise loss. Shape (L, D), representing loss per time per feature.
        """
        assert (
            x_fake.shape[2] == self.num_features
        ), f"Expected {self.num_features} features in x_fake, but got {x_fake.shape[2]}."
        assert (
            x_fake.shape[1] == self.num_time_steps
        ), f"Expected {self.num_time_steps} time steps in x_fake, but got {x_fake.shape[1]}."

        all_losses: typing.List = []
        # To store time steps with NaNs
        nan_features_per_time_step: typing.List[typing.Tuple[int, typing.List[int]]] = []

        for time_step in range(self.num_time_steps):
            per_time_losses = []
            nan_features: typing.List[int] = []  # Collect indices with NaNs for this time step
            for feature_idx in range(self.num_features):
                # Localisation of the bins
                loc: torch.Tensor = self.center_bin_locs[time_step, feature_idx]
                if (loc < 1e-16).all():
                    # Means it is a case when the distribution computed had no values for that feature so the error has to be maximal.
                    per_time_losses.append(torch.tensor(2.0, device=x_fake.device))
                    continue

                # Fake samples at time step t for feature i
                x_ti = x_fake[:, time_step, feature_idx].reshape(-1)
                x_ti = x_ti[~torch.isnan(x_ti)].reshape(-1)  # Remove NaNs
                if x_ti.numel() == 0:
                    # Record feature index with all NaNs
                    nan_features.append(feature_idx)
                    # Maximal loss for this time-feature because the discrepancy is total.
                    per_time_losses.append(torch.tensor(2.0, device=x_fake.device))
                    nan_features_per_time_step.append(
                        (time_step, nan_features)
                    )  # Store if all features are NaNs for this step
                    continue  # Skip if no valid entries after removing NaNs

                # Compute histogram using torch.histc
                # Shape (num_bins)
                density_fake = torch.histc(
                    x_ti,
                    bins=self.n_bins,
                    min=self.bin_edges[time_step, feature_idx][0].item(),
                    max=self.bin_edges[time_step, feature_idx][-1].item(),
                )

                # Normalize fake densities. Divide by total number of elements, not just the ones in the range.
                density_fake = density_fake / x_ti.numel() * self.n_bins

                # Compute absolute difference between real and fake densities
                abs_metric = torch.abs(density_fake - self.densities[time_step, feature_idx]).mean()

                # Very fast. Is ok.
                num_samples_oob = torch.sum(
                    (x_ti < self.bin_edges[time_step, feature_idx][0])
                    | (x_ti > self.bin_edges[time_step, feature_idx][-1])
                )

                out_of_bound_error = num_samples_oob / x_ti.numel()

                # Compute final loss by averaging across bins
                per_time_losses.append(abs_metric + out_of_bound_error)
            all_losses.append(torch.stack(per_time_losses))

        # Commented out to avoid unnecessary spamming, due to the fact that we now have variable length sequences.
        # # Log NaNs at the end of processing
        # if nan_features_per_time_step:
        #     nan_warnings = [
        #         f"Time step {time_step} has only NaNs for features {features}"
        #         for time_step, features in nan_features_per_time_step
        #     ]
        #     nan_warnings = nan_warnings[:10] + (["..."] if len(nan_warnings) > 5 else [])
        #     logger.warning(", ".join(nan_warnings))

        # Raise error if no valid data was found for any time step and feature
        if not all_losses:
            logger.error("All time steps and features contain NaNs or empty data, yielding no valid losses.")

        all_losses: torch.Tensor = torch.stack(all_losses)
        return all_losses / 2.0  # dividing by 2 to represent the difference between both histograms.

    def forward(self, x_fake, ignore_features: list = None, verbose: bool = False):
        # Note: We use _weigh_by_sample_size instead of reduce_weighted_per_num_samples because:
        # - HistogramLoss compares distributions (unconditional metric), not pointwise predictions
        # - The loss is per (time, feature), not per sample
        # - We must weight by reference data statistics (self.sample_sizes), not by current batch validity
        # - reduce_weighted_per_num_samples is for conditional metrics (MAE/MSE/CRPS) that weight by batch
        try:
            if ignore_features is None or (hasattr(ignore_features, '__len__') and len(ignore_features) == 0):
                return self._weigh_by_sample_size(self.compute(x_fake), verbose=verbose)

            ignore_indices = torch.tensor(ignore_features, dtype=torch.long)
            mask = torch.ones(x_fake.shape[2], dtype=torch.bool)
            mask[ignore_indices] = False
            return self._weigh_by_sample_size(self.compute(x_fake), verbose=verbose)[:, mask].mean()
        except Exception as e:
            logger.error(f"Error in the computation of the HistogramLoss. Return maximal error. Details: {e}")
            return torch.tensor(1.0, device=x_fake.device)

    def _weigh_by_sample_size(self, loss, verbose: bool = False):
        assert loss.shape == self.sample_sizes.shape, (
            "Loss and sample sizes should have the same shape, " f"but got {loss.shape} and {self.sample_sizes.shape}."
        )
        if verbose:
            logger.info(f"Sample sizes: {self.sample_sizes}")
            logger.info(f"Loss before weighting: {loss}")
        return (loss * self.sample_sizes / self.sample_sizes.sum()).sum()

    def plot_histograms(self, x_fake):
        """
        Plots histograms for real and fake data for each feature at each time step.
        For debugging, call this method after computing the loss to visualize the distributions.

        Examples:
            call self.histogram_loss.plot_histograms(FAKE_VALUES)


        Args:
        - x_fake (torch.Tensor): Fake data tensor of shape (N, L, D).
        """
        for time_step in range(self.num_time_steps):
            for feature_idx in range(self.num_features):

                plt.figure(figsize=(10, 5))

                # Compute histogram for fake data using the same bins as real data
                fake_density, _ = histogram_torch(
                    x_fake[:, time_step, feature_idx], self.bin_edges[time_step, feature_idx], density=True
                )

                bin_centers = self.center_bin_locs[time_step, feature_idx].cpu().numpy()
                real_data_density = self.densities[time_step, feature_idx].cpu().detach().numpy()
                fake_data_density = fake_density.cpu().detach().numpy()
                # Plot real data histogram
                plt.bar(
                    bin_centers,
                    real_data_density,
                    width=(bin_centers[1] - bin_centers[0]),
                    alpha=0.5,
                    label='Real Data',
                )

                # Plot fake data histogram
                plt.bar(
                    bin_centers,
                    fake_data_density,
                    width=(bin_centers[1] - bin_centers[0]),
                    alpha=0.5,
                    label='Fake Data',
                )

                plt.title(f'Feature {feature_idx + 1}, Time Step {time_step + 1}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.grid(True)
                plt.legend()
                plt.pause(0.01)
