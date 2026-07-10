from collections import defaultdict

import torch
import torch.nn as nn


class CategoricalSampler(nn.Module):

    @staticmethod
    def _float_to_str(key: float) -> str:
        """Convert float key to string, replacing dots to avoid conflicts."""
        return f"key_{str(key).replace('.', '_')}"

    def __init__(self, dataset: torch.Tensor, no_categories: bool) -> None:
        """
        Initializes the sampler with the given dataset. Only works for datasets of the form N,L,D.
        The first element of the sequence in the first dimension is the key.
        The values are the whole time series associated to that first value.

        Args:
            dataset (torch.Tensor): A tensor where each row is a time series containing tuples of category and value.
            no_categories (bool): If True, ignore the categories passed but return samples from the passed sequences.
        """
        super().__init__()

        self.no_categories = no_categories
        self.category_data = self._prepare_data(dataset)
        self.length_ts, self.dim_ts = dataset.shape[1:]

        # Mock parameter:
        self.mock_parameter = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        # Cache to avoid moving tensors again everytime.
        self._category_data_cache = {}

    def _apply(self, fn):
        # Invalidate cached per-device tensors whenever the module is moved/cast.
        self._category_data_cache.clear()
        return super()._apply(fn)

    def _prepare_data(self, dataset: torch.Tensor) -> dict:
        """
        Prepares the data.

        Args:
            dataset (torch.Tensor): The input dataset.

        Returns:
            category_data (dict): A dictionary mapping categories to their values.
        """
        map_sampling = defaultdict(list)

        for time_series in dataset:
            if self.no_categories:
                key = 0.0
            else:
                key = time_series[0, 0].item()
            map_sampling[key].append(time_series)
        ans = dict(map_sampling)

        for category in ans:
            ans[category] = torch.stack(ans[category])
        return ans

    def sample(self, categories: torch.Tensor, return_indices: bool = False):
        """
        Samples values based on the given categories.

        Args:
            categories (torch.Tensor): A tensor of categories to sample from. 1D, length corresponds to the number of samples.
            If no categories setting, we sample uniformly.
            return_indices: When True, also return the indices into the original dataset
                that were sampled. Only meaningful with no_categories=True, where index i
                corresponds to training sequence i.

        Returns:
            torch.Tensor: A tensor of sampled values, one for each category. The shape is (len(categories), L, D).
            Each row of the first axis represents one sample of a time series. Slice if one wants only the first value.
            If return_indices=True, returns (sampled_values, indices) where indices is (len(categories),) long tensor.
        """
        # This is fast. 0.1 sec for 1E6 samples. Got slightly slower with the to device.
        assert len(categories.shape) == 1, f"Categories should be a 1D tensor. Got shape {categories.shape}"

        if self.no_categories:
            categories = torch.zeros_like(categories)

        categories = categories.to(self.device)
        sampled_values = torch.empty((len(categories), self.length_ts, self.dim_ts), device=self.device)
        sampled_indices = torch.empty(len(categories), dtype=torch.long, device=self.device) if return_indices else None

        unique_categories = categories.unique()

        # Explosive pattern to save computational time
        try:
            for category in unique_categories:
                category = category.item()
                mask = categories == category
                count4category = mask.sum().item()
                possible_samples = self._get_category_tensor(category)
                indices_values = torch.randint(
                    0,
                    len(possible_samples),
                    (count4category,),
                    device=self.device,
                )
                sampled_values[mask] = possible_samples[indices_values]
                if sampled_indices is not None:
                    sampled_indices[mask] = indices_values
        except KeyError:
            missing_categories = [
                category.item()
                for category in unique_categories
                if category.item() not in set(self.category_data.keys())
            ]
            available_categories = sorted(self.category_data.keys())
            error_message = "Some categories are not represented in the dataset.\n"
            error_message += f"Missing categories: {missing_categories}\n"
            error_message += "Closest available categories:\n"

            for missing_category in missing_categories:
                closest = self._find_closest_categories(missing_category, available_categories)
                error_message += f"Category {missing_category}: closest available categories are {closest}\n"

            raise ValueError(error_message)

        if return_indices:
            return sampled_values, sampled_indices
        return sampled_values

    def _get_category_tensor(self, category):
        device_key = str(self.device)
        device_cache = self._category_data_cache.setdefault(device_key, {})
        if category not in device_cache:
            device_cache[category] = self.category_data[category].to(self.device)
        return device_cache[category]

    def _find_closest_categories(self, category, available_categories):
        """Find the two closest categories to the given category."""
        closest_below = None
        closest_above = None
        min_diff_below = float('inf')
        min_diff_above = float('inf')

        for avail_category in available_categories:
            diff = abs(avail_category - category)
            if avail_category <= category and diff < min_diff_below:
                min_diff_below = diff
                closest_below = avail_category
            elif avail_category >= category and diff < min_diff_above:
                min_diff_above = diff
                closest_above = avail_category

        return closest_below, closest_above

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == '__main__':
    # Create a toy dataset with categories and values
    dataset = torch.tensor(
        [
            [[0, 1.1], [0, 1.2], [0, 1.3]],
            [[0, 1.2], [0, 1.3], [0, 1.4]],
            [[1, 2.0], [1, 2.4], [1, 2.0]],
            [[1, 2.0], [1, 2.4], [1, 2.0]],
            [[1, 2.8], [1, 2.8], [1, 2.8]],
        ]
    )

    # Initialize the sampler
    sampler = CategoricalSampler(dataset, no_categories=False)

    # Sample from the dataset based on the categories tensor
    categories = torch.tensor([0, 1, 0, 1])
    samples = sampler.sample(categories)
    print(f"Sampled values: {samples}")

    # Benchmarking with 1E5 samples
    import time

    categories_large = torch.tensor([0, 1] * int(1e6))
    start_time = time.time()
    samples_large = sampler.sample(categories_large)
    end_time = time.time()
    print(f"Time taken for sampling 1E5 samples: {end_time - start_time} seconds")
