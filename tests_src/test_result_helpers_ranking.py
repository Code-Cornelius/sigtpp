"""Tests for namespaced ranking in ``apply_competition_ranking``.

Validation ranking and test reporting can store metrics on the same row, so the
ranking helper must be able to write namespaced score/rank keys (e.g.
``val_norm_score`` / ``rank_val_W1``) instead of always clobbering the default
``norm_score`` / ``rank_W1`` keys.
"""

import numpy as np

from src.utils.result_helpers import apply_competition_ranking


def _rows():
    return [
        {"model_name": "a", "metrics": {"val_W1_mean": 0.9, "val_ED_mean": 0.9}},
        {"model_name": "b", "metrics": {"val_W1_mean": 0.1, "val_ED_mean": 0.1}},
    ]


def test_ranking_writes_namespaced_score_and_rank_keys():
    rows = _rows()

    apply_competition_ranking(
        rows,
        ["W1", "ED"],
        value_key_for_metric=lambda m: f"val_{m}_mean",
        score_key="val_norm_score",
        rank_key_for_metric=lambda m: f"rank_val_{m}",
    )

    # Sorted ascending by the namespaced score: row "b" (lower metrics) wins.
    assert rows[0]["model_name"] == "b"
    # Namespaced keys present, default keys absent.
    for row in rows:
        assert "val_norm_score" in row["metrics"]
        assert "rank_val_W1" in row["metrics"]
        assert "norm_score" not in row["metrics"]
        assert "rank_W1" not in row["metrics"]


def test_ranking_defaults_preserve_legacy_keys():
    rows = [
        {"model_name": "a", "metrics": {"W1_mean": 0.9}},
        {"model_name": "b", "metrics": {"W1_mean": 0.1}},
    ]

    apply_competition_ranking(rows, ["W1"], lambda m: f"{m}_mean")

    assert rows[0]["model_name"] == "b"
    assert "norm_score" in rows[0]["metrics"]
    assert "rank_W1" in rows[0]["metrics"]
    assert not np.isnan(rows[0]["metrics"]["norm_score"])
