import unittest.mock

import torch
import pytest

from test.paper_experiments.data.real.easytpp.easytpp_dataset import EasyTPPDataModule


class TestConvertSplitNoPadNaN:
    """_convert_split must produce no NaN values in `inputs`."""

    def _make_split_data(self):
        """Two sequences with different lengths to force padding.
        time_since_start already includes 0.0 as its first element (the anchor),
        matching the EasyTPP HuggingFace dataset format.
        """
        return [
            {"seq_len": 4, "time_since_start": [0.0, 0.5, 1.0, 2.0]},
            {"seq_len": 2, "time_since_start": [0.0, 0.7]},
        ]

    def test_inputs_contains_no_nan(self):
        data = self._make_split_data()
        inputs, inputs_len, inputs_marks, num_marks = EasyTPPDataModule._convert_split(data)
        assert not torch.any(torch.isnan(inputs)), (
            "inputs tensor must not contain NaN after _convert_split; "
            "padding positions should be constant-filled with the last valid cumulative time."
        )

    def test_padding_equals_last_valid_time(self):
        """Padding positions must repeat the last valid cumulative time, not be NaN or 0."""
        data = self._make_split_data()
        inputs, inputs_len, _, _ = EasyTPPDataModule._convert_split(data)
        # Sequence 1 (index 1): seq_len=1, so position 1 holds 0.7, positions 2+ should be 0.7
        last_valid_time = 0.7
        padded_length = inputs.shape[1]
        for pos in range(2, padded_length):
            assert inputs[1, pos, 0].item() == pytest.approx(
                last_valid_time
            ), f"Position {pos} should be constant-padded with {last_valid_time}, got {inputs[1, pos, 0].item()}"

    def test_valid_positions_unchanged(self):
        """Valid positions (0..seq_len) must still hold the original cumulative times."""
        data = self._make_split_data()
        inputs, _, _, _ = EasyTPPDataModule._convert_split(data)
        # Sequence 0: seq_len=4, time_since_start=[0.0, 0.5, 1.0, 2.0]
        assert inputs[0, 0, 0].item() == pytest.approx(0.0)
        assert inputs[0, 1, 0].item() == pytest.approx(0.5)
        assert inputs[0, 2, 0].item() == pytest.approx(1.0)
        assert inputs[0, 3, 0].item() == pytest.approx(2.0)

    def test_zero_length_sequence_no_nan(self):
        """seq_len=1 (anchor only, no events) must produce no NaN.
        In the EasyTPP dataset format time_since_start already includes 0.0 as anchor,
        so the minimum realistic seq_len is 1.
        """
        data = [
            {"seq_len": 1, "time_since_start": [0.0]},
            {"seq_len": 3, "time_since_start": [0.0, 0.3, 0.9]},
        ]
        inputs, inputs_len, _, _ = EasyTPPDataModule._convert_split(data)
        assert not torch.any(torch.isnan(inputs)), "Anchor-only sequence must not produce NaN"
        # Anchor-only sequence: position 0 holds 0.0; padding positions also 0.0
        assert inputs[0, 0, 0].item() == pytest.approx(0.0)
        assert inputs[0, 1, 0].item() == pytest.approx(0.0)
        assert inputs[0, 2, 0].item() == pytest.approx(0.0)
        assert inputs_len[0].item() == 1  # anchor only


class TestLoadSplitFromCacheMapLocation:
    def test_load_split_from_cache_uses_map_location_cpu(self, tmp_path, monkeypatch):
        """EasyTPPDataModule._load_split_from_cache must pass map_location='cpu' to
        torch.load so CUDA-saved caches can be loaded on CPU-only machines."""
        cache_file = tmp_path / "train.pt"
        torch.save({"x": torch.zeros(2)}, str(cache_file))

        # Point the module's cache directory to tmp_path.
        dm = object.__new__(EasyTPPDataModule)
        dm.DATASET_NAME = "test"
        monkeypatch.setattr(dm, "_cache_dir", lambda: str(tmp_path))

        with unittest.mock.patch(
            "test.paper_experiments.data.real.easytpp.easytpp_dataset.torch.load",
            wraps=torch.load,
        ) as mock_load:
            dm._load_split_from_cache("train")

        _, kwargs = mock_load.call_args
        assert kwargs.get("map_location") == "cpu", f"torch.load must be called with map_location='cpu', got: {kwargs}"
