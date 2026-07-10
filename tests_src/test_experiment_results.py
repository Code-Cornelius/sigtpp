from pathlib import Path

import numpy as np
import pytest

from test.paper_experiments.experiment_results import ExperimentResults


def _make_rows():
    return [
        {"model_name": "model_a", "metrics": {"W1": 0.5, "ED": 1.2, "corr": 0.9}},
        {"model_name": "model_b", "metrics": {"W1": 0.3, "ED": 1.5, "corr": 0.8}},
    ]


def _make_rows_for_ranking():
    """3 models with known metric values for ranking tests."""
    return [
        {
            "model_name": "worst",
            "metrics": {
                "sigW_loword_notstd_mean": 3.0,
                "hist_it_mean": 3.0,
                "hist_int_mean": 3.0,
                "autocorr_it_mean": 3.0,
                "autocorr_it_short_mean": 3.0,
                "autocorr_mean": 3.0,
                "corr_mean": 3.0,
                "ED_mean": 3.0,
                "W1_mean": 3.0,
                "MAE_proper_mean": 1.0,
                "MSE_proper_mean": 1.0,
                "CRPS_mean": 3.0,
            },
        },
        {
            "model_name": "middle",
            "metrics": {
                "sigW_loword_notstd_mean": 2.0,
                "hist_it_mean": 2.0,
                "hist_int_mean": 2.0,
                "autocorr_it_mean": 2.0,
                "autocorr_it_short_mean": 2.0,
                "autocorr_mean": 2.0,
                "corr_mean": 2.0,
                "ED_mean": 2.0,
                "W1_mean": 2.0,
                "MAE_proper_mean": 1.0,
                "MSE_proper_mean": 1.0,
                "CRPS_mean": 2.0,
            },
        },
        {
            "model_name": "best",
            "metrics": {
                "sigW_loword_notstd_mean": 1.0,
                "hist_it_mean": 1.0,
                "hist_int_mean": 1.0,
                "autocorr_it_mean": 1.0,
                "autocorr_it_short_mean": 1.0,
                "autocorr_mean": 1.0,
                "corr_mean": 1.0,
                "ED_mean": 1.0,
                "W1_mean": 1.0,
                "MAE_proper_mean": 1.0,
                "MSE_proper_mean": 1.0,
                "CRPS_mean": 1.0,
            },
        },
    ]


@pytest.fixture
def saved_results(tmp_path):
    ExperimentResults(_make_rows_for_ranking(), version="sigtpp").save(str(tmp_path))
    return {
        "dir": tmp_path,
        "txt": next(tmp_path.glob("*.txt")),
    }


class TestExperimentResults:
    def test_construction(self):
        er = ExperimentResults(_make_rows(), version="sigtpp")
        assert len(er.rows) == 2
        assert er.version == "sigtpp"
        assert len(er) == 2


def test_ranking_metrics_match_validation_selection_subset():
    assert ExperimentResults.RANKING_METRICS == [
        "sigW_loword_notstd",
        "hist_it",
        "hist_int",
        "ED",
        "W1",
        "autocorr_it_short",
        "corr",
        "CRPS",
    ]


class TestNormalizeAndRank:
    def test_returns_new_instance(self):
        er = ExperimentResults(_make_rows_for_ranking(), version="sigtpp")
        ranked = er.normalize_and_rank()
        assert ranked is not er
        assert "rank_W1" not in er.rows[0]["metrics"]

    def test_borda_ranks_correct(self):
        ranked = ExperimentResults(_make_rows_for_ranking(), version="sigtpp").normalize_and_rank()
        by_name = {r["model_name"]: r["metrics"] for r in ranked.rows}
        assert by_name["best"]["norm_score"] == 8.0
        assert by_name["best"]["rank_W1"] == 1
        assert by_name["middle"]["norm_score"] == 16.0
        assert by_name["middle"]["rank_W1"] == 2
        assert by_name["worst"]["norm_score"] == 24.0
        assert by_name["worst"]["rank_W1"] == 3

    def test_sorted_by_norm_score(self):
        ranked = ExperimentResults(_make_rows_for_ranking(), version="sigtpp").normalize_and_rank()
        scores = [r["metrics"]["norm_score"] for r in ranked.rows]
        assert scores == sorted(scores)
        assert ranked.rows[0]["model_name"] == "best"

    def test_no_mutation_of_original(self):
        er = ExperimentResults(_make_rows_for_ranking(), version="sigtpp")
        ranked1 = er.normalize_and_rank()
        ranked2 = er.normalize_and_rank()
        scores1 = [r["metrics"]["norm_score"] for r in ranked1.rows]
        scores2 = [r["metrics"]["norm_score"] for r in ranked2.rows]
        assert scores1 == scores2

    def test_nan_gets_excluded_from_ranking(self):
        rows = _make_rows_for_ranking()
        rows[2]["metrics"]["W1_mean"] = float("nan")
        ranked = ExperimentResults(rows, version="sigtpp").normalize_and_rank()
        ranked_no_nan = ExperimentResults(_make_rows_for_ranking(), version="sigtpp").normalize_and_rank()
        best = next(r for r in ranked.rows if r["model_name"] == "best")
        best_no_nan = next(r for r in ranked_no_nan.rows if r["model_name"] == "best")
        assert np.isnan(best["metrics"]["rank_W1"])
        # NaN is penalised with n+1 in norm_score, so missing a metric hurts more than ranking last.
        assert best["metrics"]["norm_score"] > best_no_nan["metrics"]["norm_score"]

    def test_all_nan_metric_skipped(self):
        rows = _make_rows_for_ranking()
        for r in rows:
            r["metrics"]["W1_mean"] = float("nan")
        ranked = ExperimentResults(rows, version="sigtpp").normalize_and_rank()
        assert all(np.isnan(r["metrics"]["rank_W1"]) for r in ranked.rows)
        # norm_score still computed from the remaining ranked metrics
        assert not np.isnan(ranked.rows[0]["metrics"]["norm_score"])

    def test_tied_values_get_same_rank(self):
        rows = _make_rows_for_ranking()
        rows[0]["metrics"]["W1_mean"] = 1.0  # same as "best"
        ranked = ExperimentResults(rows, version="sigtpp").normalize_and_rank()
        by_name = {r["model_name"]: r["metrics"] for r in ranked.rows}
        assert by_name["worst"]["rank_W1"] == by_name["best"]["rank_W1"]

    def test_single_valid_value_gets_rank_one(self):
        rows = _make_rows_for_ranking()
        rows[0]["metrics"]["W1_mean"] = float("nan")
        rows[1]["metrics"]["W1_mean"] = float("nan")  # "best" (rows[2]) is the only valid entry
        ranked = ExperimentResults(rows, version="sigtpp").normalize_and_rank()
        best = next(r for r in ranked.rows if r["model_name"] == "best")
        assert best["metrics"]["rank_W1"] == 1

    def test_no_ranked_metrics_gives_nan_score(self):
        ranked = ExperimentResults([{"model_name": "solo", "metrics": {}}], version="sigtpp").normalize_and_rank()
        assert np.isnan(ranked.rows[0]["metrics"]["norm_score"])

    def test_no_renorm_columns_are_added(self):
        ranked = ExperimentResults(_make_rows_for_ranking(), version="sigtpp").normalize_and_rank()
        for row in ranked.rows:
            assert all(not k.startswith("renorm_") for k in row["metrics"])


def _make_rows_val_vs_test():
    """Two configs where the val-best row is the test-worst row.

    Regression for the selection leak: hyperparameter ranking must use the
    validation score, even when the same row carries (worse) test metrics.
    """
    val_best = {
        "model_name": "val_best_test_worst",
        "metrics": {},
    }
    val_worst = {
        "model_name": "val_worst_test_best",
        "metrics": {},
    }
    for metric in ExperimentResults.RANKING_METRICS:
        val_best["metrics"][f"val_{metric}_mean"] = 1.0
        val_best["metrics"][f"{metric}_mean"] = 5.0
        val_worst["metrics"][f"val_{metric}_mean"] = 3.0
        val_worst["metrics"][f"{metric}_mean"] = 0.5
    return [val_worst, val_best]


class TestValPrefixedRanking:
    def test_ranks_by_val_score_even_when_test_metrics_disagree(self):
        ranked = ExperimentResults(_make_rows_val_vs_test(), version="sigtpp").normalize_and_rank(prefix="val_")
        assert ranked.rows[0]["model_name"] == "val_best_test_worst"
        scores = [r["metrics"]["val_norm_score"] for r in ranked.rows]
        assert scores == sorted(scores)

    def test_val_rank_keys_do_not_clobber_test_rank_keys(self):
        er = ExperimentResults(_make_rows_val_vs_test(), version="sigtpp")
        ranked_val = er.normalize_and_rank(prefix="val_")
        for row in ranked_val.rows:
            assert "rank_val_W1" in row["metrics"]
            assert "val_norm_score" in row["metrics"]
            # Unprefixed rank/score keys belong to the test pass only.
            assert "rank_W1" not in row["metrics"]
            assert "norm_score" not in row["metrics"]

    def test_default_prefix_preserves_legacy_behaviour(self):
        ranked = ExperimentResults(_make_rows_for_ranking(), version="sigtpp").normalize_and_rank()
        assert ranked.rows[0]["model_name"] == "best"
        assert "norm_score" in ranked.rows[0]["metrics"]


class TestValPrefixedColumns:
    def test_val_columns_are_prefixed_and_flat_metrics_dropped(self):
        columns = ExperimentResults.build_column_names(prefix="val_")
        assert "val_ED_mean" in columns
        assert "val_ED_std" in columns
        assert "rank_val_W1" in columns
        assert "rank_val_autocorr_it_short" in columns
        assert "rank_val_CRPS" in columns
        assert "rank_val_autocorr_it" not in columns
        assert "rank_val_autocorr" not in columns
        assert columns[-1] == "val_norm_score"
        # _flat histograms are test-only artifacts: never in the val tuning table.
        assert all("_flat" not in c for c in columns)
        # Unprefixed metric columns must not leak into the val table.
        assert "ED_mean" not in columns
        assert "norm_score" not in columns

    def test_train_time_stays_unprefixed_in_val_table(self):
        columns = ExperimentResults.build_column_names(prefix="val_")
        assert "train_time" in columns
        assert "val_train_time" not in columns

    def test_save_with_prefix_writes_val_tuning_file(self, tmp_path):
        rows = _make_rows_val_vs_test()
        ExperimentResults(rows, version="sigtpp").save(str(tmp_path), prefix="val_")
        files = list(tmp_path.glob("*.txt"))
        assert len(files) == 1
        assert "val_tuning" in files[0].name
        header = files[0].read_text().splitlines()[0]
        assert "val_norm_score" in header


class TestSave:

    def test_extra_metrics_are_written_before_rank_columns(self):
        columns = ExperimentResults.build_column_names()

        assert columns.index("hist_it_flat_mean") < columns.index("rank_sigW_loword_notstd")
        assert columns.index("hist_int_flat_mean") < columns.index("rank_sigW_loword_notstd")
        assert "rank_autocorr_it_short" in columns
        assert "rank_CRPS" in columns
        assert "rank_autocorr_it" not in columns
        assert "rank_autocorr" not in columns
        # hist_it_flat/hist_int_flat are now in DISPLAY_METRICS, right after hist_int
        assert columns.index("hist_it_flat_mean") == columns.index("hist_int_mean") + 2
        assert columns.index("hist_int_flat_mean") == columns.index("hist_int_mean") + 4

    def test_non_bootstrap_metrics_are_plain_scalar_columns(self):
        columns = ExperimentResults.build_column_names()

        for metric in ["mark_ce", "top1_mark_acc", "top3_mark_acc", "train_time"]:
            assert metric in columns
            assert f"{metric}_mean" not in columns
            assert f"{metric}_std" not in columns

    def test_save_creates_txt(self, saved_results):
        files = list(saved_results["dir"].iterdir())
        txt_files = [f for f in files if f.suffix == ".txt"]
        assert len(txt_files) == 1
        assert txt_files[0].name.startswith("sigtpp_")

    def test_txt_header_has_no_renorm_columns(self, saved_results):
        header = saved_results["txt"].read_text().splitlines()[0]
        assert "renorm_" not in header

    def test_empty_results_no_crash(self, tmp_path):
        ExperimentResults([], version="sigtpp").save(str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_save_returns_none_when_no_rows(self, tmp_path):
        assert ExperimentResults([], version="sigtpp").save(str(tmp_path)) is None

    def test_save_returns_the_exact_path_written(self, tmp_path):
        # Callers (the sig-degree ablation) must read back this exact file, not
        # re-glob the directory for "the latest" one -- see ExperimentResults.save.
        # A decoy file with a lexicographically-later name already sits in the
        # same directory (simulating a concurrent run's own tuning file); the
        # returned path must still point at the file just written, never the decoy.
        decoy = tmp_path / "sigtpp_2099-01-01_00-00-00.txt"
        decoy.write_text("MODEL x\n", encoding="utf-8")

        written = ExperimentResults(_make_rows_for_ranking(), version="sigtpp").save(str(tmp_path))

        assert written is not None
        assert Path(written) != decoy
        assert Path(written).exists()
        assert Path(written).parent == tmp_path
