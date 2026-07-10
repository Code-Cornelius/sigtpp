"""Tests for PoissonDataModule changes A-D from the marked-poisson feature plan."""

import unittest.mock

import numpy as np
import pytest
import torch

from test.paper_experiments.data.synthetic.poisson.poisson_dataset import PoissonDataModule
from test.paper_experiments.data.synthetic_tpp_data_module import SyntheticTPPDataModule

SMALL = dict(data_size=50, seed=0)


@pytest.fixture()
def patched_cache(tmp_path, monkeypatch):
    """Redirect all cache I/O to a temp directory so tests don't write to data/."""
    monkeypatch.setattr(SyntheticTPPDataModule, "_DATA_LINKER", lambda parts: tmp_path / "__".join(parts))
    yield tmp_path


# ── Change A: base_intensity parameter ───────────────────────────────────────


class TestBaseIntensityParam:
    def test_default_is_one(self, patched_cache):
        dm = PoissonDataModule(**SMALL, use_IHP_or_HP=False)
        assert dm.base_intensity == 1.0

    def test_custom_value_stored(self, patched_cache):
        dm = PoissonDataModule(**SMALL, use_IHP_or_HP=False, base_intensity=3.0)
        assert dm.base_intensity == 3.0

    def test_mean_interarrival_scales_with_intensity(self, patched_cache):
        """HP with λ=3 should produce ~3x shorter inter-arrivals than λ=1."""
        dm1 = PoissonDataModule(data_size=500, seed=1, use_IHP_or_HP=False, base_intensity=1.0)
        dm3 = PoissonDataModule(data_size=500, seed=1, use_IHP_or_HP=False, base_intensity=3.0)
        mean1 = dm1.train_in.diff(dim=1)[dm1.train_in.diff(dim=1) > 0].mean().item()
        mean3 = dm3.train_in.diff(dim=1)[dm3.train_in.diff(dim=1) > 0].mean().item()
        # mean inter-arrival for HP(λ) = 1/λ; ratio should be ~3
        assert 2.0 < mean1 / mean3 < 4.0, f"ratio={mean1/mean3:.2f}, expected ~3"


# ── Change B: base_intensity in cache key ────────────────────────────────────


class TestCacheKey:
    def test_different_intensity_different_filename(self):
        fname1 = PoissonDataModule.format_filename(
            seed=0, use_IHP_or_HP=False, data_size=100, tmax=12.0, base_intensity=1.0
        )
        fname3 = PoissonDataModule.format_filename(
            seed=0, use_IHP_or_HP=False, data_size=100, tmax=12.0, base_intensity=3.0
        )
        assert fname1 != fname3

    def test_filename_contains_intensity_token(self):
        fname = PoissonDataModule.format_filename(
            seed=0, use_IHP_or_HP=False, data_size=100, tmax=12.0, base_intensity=3.0
        )
        assert "lam3.0" in fname or "lam3" in fname

    def test_mark_probs_are_rounded_into_filename(self):
        fname = PoissonDataModule.format_filename(
            seed=0,
            use_IHP_or_HP=False,
            data_size=100,
            tmax=12.0,
            num_marks=3,
            mark_probs=np.array([0.123456, 0.333333, 0.543211]),
        )
        assert "probs_0p1235_0p3333_0p5432" in fname

    def test_mark_probs_are_canonicalized_for_computation(self, patched_cache):
        dm = PoissonDataModule(
            **SMALL,
            use_IHP_or_HP=False,
            num_marks=3,
            mark_probs=np.array([0.123456, 0.333333, 0.543211]),
        )
        np.testing.assert_allclose(dm.mark_probs, np.array([0.1235, 0.3333, 0.5432]))

    def test_cache_integrity_warning_on_mismatch(self, patched_cache, caplog):
        """Loading a cache written with intensity=1.0 when requesting intensity=3.0 should warn."""
        import logging

        # Write a cache with intensity=1.0
        dm1 = PoissonDataModule(**SMALL, use_IHP_or_HP=False, base_intensity=1.0)
        # Manually save it under the intensity=3.0 filename to simulate a collision.
        # Use the same path construction as the patched_cache fixture: tmp_path / "__".join(parts)
        path3 = patched_cache / "__".join(
            [
                "synthetic",
                PoissonDataModule.format_filename(
                    seed=0, use_IHP_or_HP=False, data_size=50, tmax=12.0, base_intensity=3.0
                ),
            ]
        )
        import torch

        cache_data = torch.load(
            patched_cache
            / "__".join(
                [
                    "synthetic",
                    PoissonDataModule.format_filename(
                        seed=0, use_IHP_or_HP=False, data_size=50, tmax=12.0, base_intensity=1.0
                    ),
                ]
            )
        )
        # Overwrite with wrong base_intensity stored
        cache_data["base_intensity"] = 1.0
        torch.save(cache_data, path3)
        with caplog.at_level(logging.WARNING):
            dm3 = PoissonDataModule(**SMALL, use_IHP_or_HP=False, base_intensity=3.0)
        # Cache loader detects format mismatch and regenerates with a warning.
        assert any(
            "base_intensity" in r.message or "incompatible" in r.message for r in caplog.records
        ), f"Expected a cache-mismatch warning; got: {[r.message for r in caplog.records]}"


# ── Change C: marks shape (N, L+1) not (N, L+1, 1) ─────────────────────────


class TestMarksShape:
    def test_marks_are_2d(self, patched_cache):
        dm = PoissonDataModule(**SMALL, use_IHP_or_HP=False, num_marks=3, mark_probs=np.array([1 / 6, 1 / 3, 1 / 2]))
        assert dm.train_marks.ndim == 2, f"train_marks should be 2D (N, L+1), got shape {dm.train_marks.shape}"

    def test_marks_shape_is_n_lplus1(self, patched_cache):
        dm = PoissonDataModule(**SMALL, use_IHP_or_HP=False, num_marks=3, mark_probs=np.array([1 / 6, 1 / 3, 1 / 2]))
        N, L1 = dm.train_marks.shape
        assert dm.train_in.shape[:2] == (N, L1)

    def test_const_mark_anchor_zero_rest_one(self, patched_cache):
        """Anchor mark (col 0) must be 0; all VALID non-anchor events must be mark 1.
        Padding positions use 0 as sentinel (from hp.gen np.where fallback), so we
        only check positions 1..seq_len-1 for each sequence.
        """
        dm = PoissonDataModule(**SMALL, use_IHP_or_HP=False, num_marks=2, mark_probs=np.array([0.0, 1.0]))
        assert (dm.train_marks[:, 0] == 0).all(), "anchor marks must be 0"
        # Check only valid (non-padding) positions: col j is valid if j < train_in_len[i]
        for i in range(dm.train_marks.shape[0]):
            seq_len = dm.train_in_len[i].item()  # includes anchor, so valid events are cols 1..seq_len-1
            if seq_len > 1:
                assert (
                    dm.train_marks[i, 1:seq_len] == 1
                ).all(), f"seq {i}: marks at valid positions must be 1, got {dm.train_marks[i, 1:seq_len]}"


# ── Change D: compare_validation_to_fresh forwards params ───────────────────


class TestCompareValidationFresh:
    def test_forwards_base_intensity(self, patched_cache, monkeypatch):
        """compare_validation_to_fresh must create the fresh module with base_intensity=3.0."""
        calls = []
        original_init = PoissonDataModule.__init__

        def spy_init(self_inner, **kwargs):
            calls.append(kwargs.get("base_intensity", 1.0))
            original_init(self_inner, **kwargs)

        monkeypatch.setattr(PoissonDataModule, "__init__", spy_init)
        dm = PoissonDataModule(**SMALL, use_IHP_or_HP=False, base_intensity=3.0)
        calls.clear()  # discard the construction above
        try:
            dm.compare_validation_to_fresh(other_seed=99)
        except Exception:
            pass  # we only care about the call, not the result
        assert any(
            v == 3.0 for v in calls
        ), f"compare_validation_to_fresh did not pass base_intensity=3.0; calls={calls}"


# ── Change E: dataloaders include marks in batch when marks are set ──────────


class TestDataloaderIncludesMarks:
    def test_train_dataloader_returns_3tuple_when_marks_set(self, patched_cache):
        """Critical-4 regression: batch must have 3 elements (times, lens, marks)."""
        dm = PoissonDataModule(**SMALL, use_IHP_or_HP=False, num_marks=3, mark_probs=np.array([1 / 3, 1 / 3, 1 / 3]))
        batch = next(iter(dm.train_dataloader()))
        assert len(batch) == 3, f"Expected 3-tuple batch, got len={len(batch)}"
        times, lens, marks = batch
        assert marks.dtype == torch.long, f"marks must be long, got {marks.dtype}"
        assert marks.shape[1] == times.shape[1], f"marks L+1={marks.shape[1]} must match times L+1={times.shape[1]}"

    def test_train_dataloader_returns_3tuple_without_marks(self, patched_cache):
        """Unmarked datasets emit 3-tuples with trivial all-zeros marks (canonical spec)."""
        dm = PoissonDataModule(**SMALL, use_IHP_or_HP=False)
        batch = next(iter(dm.train_dataloader()))
        assert len(batch) == 3, f"Expected 3-tuple batch for unmarked data, got len={len(batch)}"
        _, _, marks = batch
        assert (marks == 0).all(), "unmarked dataset marks must be all zeros"

    def test_val_dataloader_returns_3tuple_when_marks_set(self, patched_cache):
        dm = PoissonDataModule(**SMALL, use_IHP_or_HP=False, num_marks=2, mark_probs=np.array([0.5, 0.5]))
        batch = next(iter(dm.val_dataloader()))
        assert len(batch) == 3, f"Expected 3-tuple val batch, got len={len(batch)}"


# ── map_location="cpu" on cache load ─────────────────────────────────────────


class TestLoadFromCacheMapLocation:
    def test_load_from_cache_uses_map_location_cpu(self, tmp_path):
        """SyntheticTPPDataModule._load_from_cache must pass map_location='cpu' to
        torch.load so that caches saved on CUDA (Linux) can be loaded on CPU-only Windows."""
        filepath = tmp_path / "dummy.pt"
        torch.save({"x": torch.zeros(3)}, str(filepath))

        with unittest.mock.patch(
            "test.paper_experiments.data.synthetic_tpp_data_module.torch.load",
            wraps=torch.load,
        ) as mock_load:
            SyntheticTPPDataModule._load_from_cache(str(filepath))

        _, kwargs = mock_load.call_args
        assert kwargs.get("map_location") == "cpu", f"torch.load must be called with map_location='cpu', got: {kwargs}"
