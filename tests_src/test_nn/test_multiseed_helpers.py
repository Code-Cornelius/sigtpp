import math

import pytest

from test.paper_experiments.experiment_results import ExperimentResults
from test.paper_experiments.multiseed_helpers import (
    aggregate_across_seeds,
    aggregate_test_rows_across_seeds,
    apply_seed_ranking,
    build_seed_config,
    seed_summary_column_names,
    seed_test_summary_column_names,
    seeds_from_config,
    strip_seed_suffix,
    write_multiseed_outputs,
    write_multiseed_test_by_seed_txt,
    write_multiseed_test_summary_txt,
)


def test_strip_seed_suffix_peels_only_the_trailing_tag():
    # Trailing `_seed<digits>` tag (both name forms end in this) is removed.
    assert strip_seed_suffix("hawkes_sigtpp_TX20_lr1e-3_seed42") == "hawkes_sigtpp_TX20_lr1e-3"
    assert strip_seed_suffix("sigtpp_mymodel_seed0") == "sigtpp_mymodel"
    # No tag: returned unchanged.
    assert strip_seed_suffix("hawkes_sigtpp_TX20_lr1e-3") == "hawkes_sigtpp_TX20_lr1e-3"
    # `_seed<N>` that is NOT the trailing token must be preserved (anchored match only).
    assert strip_seed_suffix("model_seed5_lr1e-3") == "model_seed5_lr1e-3"
    # Only the final tag peels, never a mid-string lookalike.
    assert strip_seed_suffix("model_seed5_seed3") == "model_seed5"
    # `_seed` with no digits is not a tag.
    assert strip_seed_suffix("model_seedling") == "model_seedling"


def test_seeds_from_config_reads_canonical_seed_list():
    cfg = {"seeds": [0, 1, 2]}

    assert seeds_from_config(cfg) == [0, 1, 2]


def test_build_seed_config_tags_seed_and_disables_bootstrap():
    cfg = {
        "seeds": [0, 1],
        "output_dir": "test/paper_experiments",
        "n_bootstraps": 200,
    }

    seed_cfg = build_seed_config(cfg, 7)

    # TrainingManager._resolve_single_seed requires a one-element ``seeds`` list.
    assert seed_cfg["seeds"] == [7]
    # Seed isolation now happens via the model-name suffix, not the output_dir.
    assert seed_cfg["output_dir"] == "test/paper_experiments"
    assert seed_cfg["_multiseed_seed_tag"] == 7
    assert seed_cfg["n_bootstraps"] == 1
    # Base cfg must not be mutated.
    assert "_multiseed_seed_tag" not in cfg


def test_build_seed_config_disables_bootstrap_refinement():
    """Multi-seed reports across-seed variance only; per-seed refinement would
    produce a confusing ``<version>_refine_*.txt`` per seed."""
    cfg = {
        "seeds": [0, 1],
        "output_dir": "test/paper_experiments",
        "n_bootstraps": 200,
        "refine_best_n_bootstraps": 100,
    }

    seed_cfg = build_seed_config(cfg, 7)

    assert seed_cfg["refine_best_n_bootstraps"] is None
    # Base cfg must not be mutated.
    assert cfg["refine_best_n_bootstraps"] == 100


def test_aggregate_across_seeds_uses_explicit_val_seed_columns():
    # Per-seed manager rows carry val_-prefixed diagnostics.
    seed_results = [
        (0, ExperimentResults([{"model_name": "model_a", "metrics": {"val_hist_it_mean": 1.0}}], version="sigtpp")),
        (1, ExperimentResults([{"model_name": "model_a", "metrics": {"val_hist_it_mean": 3.0}}], version="sigtpp")),
    ]

    rows = aggregate_across_seeds(seed_results)
    metrics = rows[0]["metrics"]

    assert metrics["val_hist_it_seed_mean"] == pytest.approx(2.0)
    assert metrics["val_hist_it_seed_std"] == pytest.approx(2**0.5)
    assert metrics["val_hist_it_seed_n_valid"] == 2
    assert "val_hist_it_std" not in metrics


def test_aggregate_across_seeds_groups_seed_suffixed_names_into_one_row():
    """The seed is baked into the model name (``..._seed<N>``). Aggregation must
    strip that suffix so the same config across seeds lands in one bucket, not a
    singleton per ``(config, seed)``."""
    seed_results = [
        (
            0,
            ExperimentResults(
                [{"model_name": "model_a_seed0", "metrics": {"val_hist_it_mean": 1.0}}], version="sigtpp"
            ),
        ),
        (
            1,
            ExperimentResults(
                [{"model_name": "model_a_seed1", "metrics": {"val_hist_it_mean": 3.0}}], version="sigtpp"
            ),
        ),
    ]

    rows = aggregate_across_seeds(seed_results)

    assert len(rows) == 1
    assert rows[0]["model_name"] == "model_a"
    metrics = rows[0]["metrics"]
    assert metrics["val_hist_it_seed_n_valid"] == 2
    assert metrics["val_hist_it_seed_mean"] == pytest.approx(2.0)
    assert metrics["val_hist_it_seed_std"] == pytest.approx(2**0.5)
    assert not math.isnan(metrics["val_hist_it_seed_std"])


def test_aggregate_across_seeds_keeps_distinct_configs_separate():
    """Stripping the seed tag must not over-merge: two different configs stay in
    two buckets, each aggregating its own seeds (grid-search × seeds scenario)."""
    seed_results = [
        (
            0,
            ExperimentResults(
                [
                    {"model_name": "cfg_a_seed0", "metrics": {"val_hist_it_mean": 1.0}},
                    {"model_name": "cfg_b_seed0", "metrics": {"val_hist_it_mean": 10.0}},
                ],
                version="sigtpp",
            ),
        ),
        (
            1,
            ExperimentResults(
                [
                    {"model_name": "cfg_a_seed1", "metrics": {"val_hist_it_mean": 3.0}},
                    {"model_name": "cfg_b_seed1", "metrics": {"val_hist_it_mean": 20.0}},
                ],
                version="sigtpp",
            ),
        ),
    ]

    rows = aggregate_across_seeds(seed_results)
    by_name = {row["model_name"]: row["metrics"] for row in rows}

    assert set(by_name) == {"cfg_a", "cfg_b"}
    assert by_name["cfg_a"]["val_hist_it_seed_n_valid"] == 2
    assert by_name["cfg_a"]["val_hist_it_seed_mean"] == pytest.approx(2.0)
    assert by_name["cfg_b"]["val_hist_it_seed_n_valid"] == 2
    assert by_name["cfg_b"]["val_hist_it_seed_mean"] == pytest.approx(15.0)


def test_aggregate_across_seeds_reads_plain_non_bootstrap_metrics():
    seed_results = [
        (0, ExperimentResults([{"model_name": "model_a", "metrics": {"val_mark_ce": 1.0}}], version="sigtpp")),
        (1, ExperimentResults([{"model_name": "model_a", "metrics": {"val_mark_ce": 2.0}}], version="sigtpp")),
    ]

    rows = aggregate_across_seeds(seed_results)
    metrics = rows[0]["metrics"]

    assert metrics["val_mark_ce_seed_mean"] == pytest.approx(1.5)
    assert metrics["val_mark_ce_seed_std"] == pytest.approx(2**-0.5)
    assert metrics["val_mark_ce_seed_n_valid"] == 2


def test_aggregate_across_seeds_train_time_stays_unprefixed():
    seed_results = [
        (0, ExperimentResults([{"model_name": "model_a", "metrics": {"train_time": 10.0}}], version="sigtpp")),
        (1, ExperimentResults([{"model_name": "model_a", "metrics": {"train_time": 20.0}}], version="sigtpp")),
    ]

    rows = aggregate_across_seeds(seed_results)
    metrics = rows[0]["metrics"]

    assert metrics["train_time_seed_mean"] == pytest.approx(15.0)
    assert "val_train_time_seed_mean" not in metrics


def test_apply_seed_ranking_ranks_by_val_seed_means():
    rows = [
        {"model_name": "worse", "metrics": {"val_W1_seed_mean": 2.0, "val_ED_seed_mean": 2.0}},
        {"model_name": "better", "metrics": {"val_W1_seed_mean": 1.0, "val_ED_seed_mean": 1.0}},
    ]

    apply_seed_ranking(rows)

    assert rows[0]["model_name"] == "better"
    assert "val_norm_score" in rows[0]["metrics"]
    assert "rank_val_W1" in rows[0]["metrics"]
    # Test-namespace keys must not be produced by validation ranking.
    assert "norm_score" not in rows[0]["metrics"]


def test_seed_summary_column_names_are_val_prefixed_without_flat():
    names = seed_summary_column_names()

    assert "val_hist_it_seed_mean" in names
    assert "train_time_seed_mean" in names
    assert names[-1] == "val_norm_score"
    assert all("_flat" not in name for name in names)
    assert "norm_score" not in names


def test_write_multiseed_test_by_seed_txt_writes_unprefixed_test_columns(tmp_path):
    test_rows = [
        (0, {"model_name": "model_a", "metrics": {"ED_mean": 0.5, "W1_mean": 0.7}}),
        (1, {"model_name": "model_a", "metrics": {"ED_mean": 0.6, "W1_mean": 0.8}}),
    ]

    path = write_multiseed_test_by_seed_txt(test_rows, "sigtpp", str(tmp_path))

    assert "multiseed_test_by_seed" in path
    with open(path) as handle:
        header = handle.readline()
        body = handle.read()
    assert "SEED" in header
    assert "ED_mean" in header
    assert "val_ED_mean" not in header
    # Both rows are the SAME config (the point of this file, vs. the retired
    # per-seed-winner report which could mix configs across seeds).
    assert body.count("model_a") == 2


def test_aggregate_test_rows_across_seeds_computes_mean_std():
    test_rows = [
        (0, {"model_name": "model_a", "metrics": {"ED_mean": 1.0}}),
        (1, {"model_name": "model_a", "metrics": {"ED_mean": 3.0}}),
    ]

    metrics = aggregate_test_rows_across_seeds(test_rows)

    assert metrics["ED_seed_mean"] == pytest.approx(2.0)
    assert metrics["ED_seed_std"] == pytest.approx(2**0.5)
    assert metrics["ED_seed_n_valid"] == 2
    # Test metrics are unprefixed -- no val_ namespace here.
    assert "val_ED_seed_mean" not in metrics


def test_aggregate_test_rows_across_seeds_nan_seed_shrinks_n_valid():
    test_rows = [
        (0, {"model_name": "model_a", "metrics": {"ED_mean": 1.0}}),
        (1, {"model_name": "model_a", "metrics": {"ED_mean": float("nan")}}),
        (2, {"model_name": "model_a", "metrics": {"ED_mean": 3.0}}),
    ]

    metrics = aggregate_test_rows_across_seeds(test_rows)

    assert metrics["ED_seed_n_valid"] == 2
    assert metrics["ED_seed_mean"] == pytest.approx(2.0)
    assert metrics["ED_seed_std"] == pytest.approx(2**0.5)


def test_aggregate_test_rows_across_seeds_train_time_reads_plain_key():
    test_rows = [
        (0, {"model_name": "model_a", "metrics": {"train_time": 10.0}}),
        (1, {"model_name": "model_a", "metrics": {"train_time": 20.0}}),
    ]

    metrics = aggregate_test_rows_across_seeds(test_rows)

    assert metrics["train_time_seed_mean"] == pytest.approx(15.0)


def test_seed_test_summary_column_names_are_unprefixed_without_flat():
    names = seed_test_summary_column_names()

    assert "ED_seed_mean" in names
    assert "train_time_seed_mean" in names
    assert all("_flat" not in name for name in names)
    assert all(not name.startswith("val_") for name in names)


def test_write_multiseed_test_summary_txt_writes_one_row(tmp_path):
    summary_metrics = {"ED_seed_mean": 0.5, "ED_seed_std": 0.1, "ED_seed_n_valid": 2}

    path = write_multiseed_test_summary_txt("model_a", summary_metrics, "sigtpp", str(tmp_path))

    assert "multiseed_test_summary" in path
    with open(path) as handle:
        header = handle.readline()
        body = handle.read()
    assert "ED_seed_mean" in header
    assert "model_a" in body


def test_aggregate_across_seeds_nan_seed_reduces_n_valid_and_uses_survivors():
    """A failed seed (NaN per-seed mean) must shrink ``_seed_n_valid`` and not poison the std."""
    seed_results = [
        (0, ExperimentResults([{"model_name": "model_a", "metrics": {"val_hist_it_mean": 1.0}}], version="sigtpp")),
        (
            1,
            ExperimentResults(
                [{"model_name": "model_a", "metrics": {"val_hist_it_mean": float("nan")}}], version="sigtpp"
            ),
        ),
        (2, ExperimentResults([{"model_name": "model_a", "metrics": {"val_hist_it_mean": 3.0}}], version="sigtpp")),
    ]

    rows = aggregate_across_seeds(seed_results)
    metrics = rows[0]["metrics"]

    assert metrics["val_hist_it_seed_n_valid"] == 2
    assert metrics["val_hist_it_seed_mean"] == pytest.approx(2.0)
    # std over survivors {1.0, 3.0} with ddof=1 == sqrt(2).
    assert metrics["val_hist_it_seed_std"] == pytest.approx(2**0.5)


def test_aggregate_across_seeds_all_nan_produces_nan_summary_with_zero_n_valid():
    """If every seed failed, summary stays nan and n_valid is 0 — never silently 0.0 std."""
    seed_results = [
        (
            0,
            ExperimentResults(
                [{"model_name": "model_a", "metrics": {"val_hist_it_mean": float("nan")}}], version="sigtpp"
            ),
        ),
        (
            1,
            ExperimentResults(
                [{"model_name": "model_a", "metrics": {"val_hist_it_mean": float("nan")}}], version="sigtpp"
            ),
        ),
    ]

    rows = aggregate_across_seeds(seed_results)
    metrics = rows[0]["metrics"]

    assert metrics["val_hist_it_seed_n_valid"] == 0
    assert math.isnan(metrics["val_hist_it_seed_mean"])
    assert math.isnan(metrics["val_hist_it_seed_std"])


def test_aggregate_across_seeds_warns_when_nonzero_bootstrap_std_present(caplog):
    """If a per-seed row carries a non-zero ``*_std``, the dropped variance must be flagged."""
    seed_results = [
        (
            0,
            ExperimentResults(
                [{"model_name": "model_a", "metrics": {"val_hist_it_mean": 1.0, "val_hist_it_std": 0.42}}],
                version="sigtpp",
            ),
        ),
        (
            1,
            ExperimentResults(
                [{"model_name": "model_a", "metrics": {"val_hist_it_mean": 3.0, "val_hist_it_std": 0.31}}],
                version="sigtpp",
            ),
        ),
    ]

    with caplog.at_level("WARNING", logger="test.paper_experiments.multiseed_helpers"):
        aggregate_across_seeds(seed_results)

    assert any(
        "dropping non-zero bootstrap" in record.message and "model_a" in record.message for record in caplog.records
    ), f"Expected a bootstrap-drop warning, got: {[r.message for r in caplog.records]}"


def test_aggregate_across_seeds_silent_when_all_bootstrap_std_zero(caplog):
    """The B=1 happy path must not log the bootstrap-drop warning."""
    seed_results = [
        (
            0,
            ExperimentResults(
                [{"model_name": "model_a", "metrics": {"val_hist_it_mean": 1.0, "val_hist_it_std": 0.0}}],
                version="sigtpp",
            ),
        ),
        (
            1,
            ExperimentResults(
                [{"model_name": "model_a", "metrics": {"val_hist_it_mean": 3.0, "val_hist_it_std": 0.0}}],
                version="sigtpp",
            ),
        ),
    ]

    with caplog.at_level("WARNING", logger="test.paper_experiments.multiseed_helpers"):
        aggregate_across_seeds(seed_results)

    assert not any("dropping non-zero bootstrap" in record.message for record in caplog.records)


def test_write_multiseed_outputs_names_seed_std_unambiguously(tmp_path):
    seed_results = [
        (0, ExperimentResults([{"model_name": "model_a", "metrics": {"val_hist_it_mean": 1.0}}], version="sigtpp")),
        (1, ExperimentResults([{"model_name": "model_a", "metrics": {"val_hist_it_mean": 3.0}}], version="sigtpp")),
    ]

    by_seed_path, summary_path = write_multiseed_outputs(seed_results, "sigtpp", str(tmp_path))

    with open(by_seed_path) as handle:
        by_seed_header = handle.readline()
    with open(summary_path) as handle:
        summary_header = handle.readline()

    assert "SEED" in by_seed_header
    assert "val_hist_it_mean" in by_seed_header
    assert "multiseed_summary" in summary_path
    assert "val_hist_it_seed_std" in summary_header
    assert "val_hist_it_std" not in summary_header
