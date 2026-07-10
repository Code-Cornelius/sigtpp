"""Tests for Hawkes3x3DataModule."""

import pytest

pytest.importorskip("tick")

import torch

from test.paper_experiments.data.synthetic.hawkes.hawkes_3x3_dataset import Hawkes3x3DataModule
from test.paper_experiments.data.synthetic_tpp_data_module import SyntheticTPPDataModule

SMALL = dict(
    data_size=60,
    seed=42,
    baseline=[0.2, 0.4, 1.0],
    adjacency=[[0.6, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.5, 0.1]],
    decays=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
)


@pytest.fixture()
def patched_cache(tmp_path, monkeypatch):
    """Redirect cache I/O to a temp directory."""
    monkeypatch.setattr(SyntheticTPPDataModule, "_DATA_LINKER", lambda parts: tmp_path / "__".join(parts))
    yield tmp_path


class TestHawkes3x3DatasetContract:
    def test_num_marks_is_3(self, patched_cache):
        dm = Hawkes3x3DataModule(**SMALL)
        assert dm.num_marks == 3

    def test_cumtime_shape(self, patched_cache):
        dm = Hawkes3x3DataModule(**SMALL)
        # (N, L+1, 1)
        assert dm.train_in.ndim == 3
        assert dm.train_in.shape[2] == 1
        assert dm.val_in.ndim == 3
        assert dm.val_in.shape[2] == 1

    def test_marks_shape_matches_cumtimes(self, patched_cache):
        dm = Hawkes3x3DataModule(**SMALL)
        assert dm.train_marks.shape == dm.train_in.shape[:2]
        assert dm.val_marks.shape == dm.val_in.shape[:2]

    def test_marks_dtype_long(self, patched_cache):
        dm = Hawkes3x3DataModule(**SMALL)
        assert dm.train_marks.dtype == torch.long
        assert dm.val_marks.dtype == torch.long

    def test_marks_in_valid_range(self, patched_cache):
        dm = Hawkes3x3DataModule(**SMALL)
        # Collect all valid event marks (skip column 0 = anchor, always 0)
        for lengths, marks in [
            (dm.train_in_len, dm.train_marks),
            (dm.val_in_len, dm.val_marks),
        ]:
            for i in range(len(lengths)):
                valid_len = int(lengths[i].item())
                event_marks = marks[i, 1:valid_len]  # skip anchor
                if event_marks.numel() > 0:
                    assert event_marks.min().item() >= 0
                    assert event_marks.max().item() <= 2

    def test_lengths_include_anchor(self, patched_cache):
        dm = Hawkes3x3DataModule(**SMALL)
        # Every length must be >= 1 (at least the anchor)
        assert (dm.train_in_len >= 1).all()
        assert (dm.val_in_len >= 1).all()

    def test_cumtimes_nondecreasing(self, patched_cache):
        dm = Hawkes3x3DataModule(**SMALL)
        for i in range(min(10, len(dm.train_in_len))):
            L = int(dm.train_in_len[i].item())
            times = dm.train_in[i, :L, 0]
            diffs = times[1:] - times[:-1]
            assert (diffs >= 0).all(), f"sample {i}: non-monotone cumulative times"

    def test_cache_roundtrip(self, patched_cache):
        """Loading from cache yields identical tensors."""
        dm1 = Hawkes3x3DataModule(**SMALL)
        dm2 = Hawkes3x3DataModule(**SMALL)
        assert torch.equal(dm1.train_in, dm2.train_in)
        assert torch.equal(dm1.train_marks, dm2.train_marks)
        assert torch.equal(dm1.train_in_len, dm2.train_in_len)

    def test_split_sizes(self, patched_cache):
        n = 60
        dm = Hawkes3x3DataModule(**SMALL)
        assert len(dm.train_in) == int(0.60 * n)
        assert len(dm.val_in) == int(0.80 * n) - int(0.60 * n)
        assert len(dm.test_in) == n - int(0.80 * n)
