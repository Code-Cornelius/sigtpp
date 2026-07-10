import numpy as np
import os
import tempfile

import pytest

from src.data_types.bootstrap_eval import aggregate_bootstrap_metrics, build_per_replicate_matrix


def test_build_per_replicate_matrix_shape():
    reps = [{"w1": 0.1, "ed": 0.2}, {"w1": 0.3, "ed": 0.4}, {"w1": 0.5, "ed": 0.6}]
    mat = build_per_replicate_matrix(reps)
    assert set(mat.keys()) == {"w1", "ed"}
    assert mat["w1"].shape == (3,)
    assert mat["ed"].dtype == float


def test_build_per_replicate_matrix_nan_for_missing():
    reps = [{"w1": 1.0}, {"w1": 2.0, "ed": 3.0}]
    mat = build_per_replicate_matrix(reps)
    assert np.isnan(mat["ed"][0])
    assert mat["ed"][1] == pytest.approx(3.0)


def test_build_per_replicate_matrix_values():
    reps = [{"m": 0.5}, {"m": 1.5}]
    mat = build_per_replicate_matrix(reps)
    np.testing.assert_array_almost_equal(mat["m"], [0.5, 1.5])


def test_aggregate_bootstrap_metrics_rejects_empty_replicates():
    with pytest.raises(AssertionError, match="requires at least one replicate"):
        aggregate_bootstrap_metrics([])


# ---------------------------------------------------------------------------
# Per-replicate .npz round-trip tests
# ---------------------------------------------------------------------------


def _make_rows(B: int):
    rng = np.random.default_rng(0)
    return [
        {
            "model_name": "model_A",
            "metrics": {"w1_mean": 0.5, "w1_std": 0.01},
            "per_replicate": {"w1": rng.random(B), "ed": rng.random(B)},
        },
        {
            "model_name": "model_B",
            "metrics": {"w1_mean": float("nan"), "w1_std": float("nan"), "error": "fail"},
            "per_replicate": None,
        },
    ]


def test_write_per_replicate_npz_shape():
    from test.paper_experiments.recompute_bootstrap import _write_per_replicate_npz

    B = 10
    rows = _make_rows(B)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "per_replicate.npz")
        _write_per_replicate_npz(rows, path, B)
        assert os.path.exists(path)
        with np.load(path, allow_pickle=True) as npz:
            assert npz["B"] == B
            assert npz["schema_version"] == 1
            assert "model_A" in list(npz["model_names"])
            assert "model_B" in list(npz["model_names"])
            M, K, Bp = npz["data"].shape
            assert M == 2
            assert Bp == B


def test_write_per_replicate_npz_failed_row_is_nan():
    from test.paper_experiments.recompute_bootstrap import _write_per_replicate_npz

    B = 5
    rows = _make_rows(B)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "per_replicate.npz")
        _write_per_replicate_npz(rows, path, B)
        with np.load(path, allow_pickle=True) as npz:
            i_b = list(npz["model_names"]).index("model_B")
            assert np.all(np.isnan(npz["data"][i_b]))


def test_write_per_replicate_npz_values():
    from test.paper_experiments.recompute_bootstrap import _write_per_replicate_npz

    B = 3
    w1_values = np.array([0.1, 0.2, 0.3])
    rows = [{"model_name": "m", "metrics": {}, "per_replicate": {"w1": w1_values}}]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "per_replicate.npz")
        _write_per_replicate_npz(rows, path, B)
        with np.load(path, allow_pickle=True) as npz:
            j = list(npz["metric_names"]).index("w1")
            np.testing.assert_array_almost_equal(npz["data"][0, j, :], w1_values)
