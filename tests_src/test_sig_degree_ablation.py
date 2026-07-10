"""Unit tests for the sig-degree ablation helper modules.

The ablation runs in-process from ``TrainingManager`` (opt-in via the
``sig_degree_ablation`` config flag). These tests exercise the torch-free
building blocks it delegates to - parsing, per-degree selection, fixed-width
report writing, and the injectable orchestration - without importing
torch/signatory/tick. The GPU test step is mocked via injected fake helpers.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from test.paper_experiments.sig_degree_ablation import (
    run_sig_degree_ablation_from_val_file,
)
from test.paper_experiments.sig_degree_report import write_report
from test.paper_experiments.sig_degree_selection import (
    find_latest_val_tuning_file,
    parse_results,
    select_winner_rows_by_sig_degree,
    select_winners_by_sig_degree,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_val_tuning(path: Path, rows: List[Tuple[str, Any]]) -> Path:
    """Write a minimal validation tuning file: MODEL + val_norm_score columns."""
    lines = ["MODEL  val_norm_score"]
    lines += [f"{name}  {score}" for name, score in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _line_for(text: str, needle: str) -> str:
    return next(line for line in text.splitlines() if needle in line)


@dataclass
class _FakeSettings:
    n_bootstraps: int
    trainer_seed: int = 42


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_results_blank_first_line_raises_valueerror(tmp_path):
    # A blank/whitespace first line must raise a clear ValueError, not an
    # IndexError from formatting the error message off an empty header.
    path = tmp_path / "blank_header.txt"
    path.write_text("\ntaxi_sigtpp_sig_2_use_gru  0.1  0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MODEL"):
        parse_results(path)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_select_winners_picks_min_val_norm_score(tmp_path):
    # For BOTH degrees the winning (lowest-score) row is NOT first in file order,
    # so this fails if selection keys off file order instead of val_norm_score.
    val = _write_val_tuning(
        tmp_path / "val_tuning.txt",
        [
            ("taxi_sigtpp_TX1000_sig_2_use_lstm", 0.90),  # deg 2, first-in-file, loser
            ("taxi_sigtpp_TX1000_sig_3_use_gru", 0.70),   # deg 3, first-in-file, loser
            ("taxi_sigtpp_TX1000_sig_2_use_gru", 0.40),   # deg 2 winner
            ("taxi_sigtpp_TX1000_sig_3_use_lstm", 0.20),  # deg 3 winner
        ],
    )
    winners = select_winners_by_sig_degree(val)
    assert winners == {
        2: "taxi_sigtpp_TX1000_sig_2_use_gru",
        3: "taxi_sigtpp_TX1000_sig_3_use_lstm",
    }


def test_select_winners_parses_relative_degree_tokens(tmp_path):
    # The shipped sigtpp grid sweeps `relative_sig_degree`, whose name token is
    # `rela<offset>` (offsets may be negative), not `_sig_<d>_`. Selection must
    # group on those real tokens.
    val = _write_val_tuning(
        tmp_path / "val_tuning.txt",
        [
            ("taxi_sigtpp_TX1000_use_F_hid_32_rela-2_lrscT_anchfree_detaF", 0.90),
            ("taxi_sigtpp_TX1000_use_T_hid_32_rela-2_lrscT_anchfree_detaF", 0.40),
            ("taxi_sigtpp_TX1000_use_F_hid_32_rela0_lrscT_anchfree_detaF", 0.20),
        ],
    )
    winners = select_winners_by_sig_degree(val)
    assert winners == {
        -2: "taxi_sigtpp_TX1000_use_T_hid_32_rela-2_lrscT_anchfree_detaF",
        0: "taxi_sigtpp_TX1000_use_F_hid_32_rela0_lrscT_anchfree_detaF",
    }


def test_select_winners_warns_on_mixed_degree_token_modes(tmp_path, caplog):
    # Absolute (`_sig_<d>_`) and relative (`rela<offset>`) degrees are different
    # axes; a file mixing both is out of contract and must be flagged, not
    # silently conflated into one degree column.
    val = _write_val_tuning(
        tmp_path / "val_tuning.txt",
        [
            ("taxi_sigtpp_TX1000_sig_2_use_gru", 0.50),
            ("taxi_sigtpp_TX1000_use_F_hid_32_rela-1_lrscT_anchfree_detaF", 0.30),
        ],
    )
    with caplog.at_level(logging.WARNING, logger="test.paper_experiments.sig_degree_selection"):
        winners = select_winners_by_sig_degree(val)

    assert set(winners) == {2, -1}
    assert "mixed absolute and relative" in caplog.text


def test_select_winners_skips_rows_without_sig_degree(tmp_path):
    # The wgan row has the best score overall but no _sig_<d>_use_ token, so it
    # must be excluded rather than becoming a spurious winner.
    val = _write_val_tuning(
        tmp_path / "val_tuning.txt",
        [
            ("taxi_wgan_TX1000_hidden_64", 0.10),
            ("taxi_sigtpp_TX1000_sig_2_use_lstm", 0.50),
        ],
    )
    winners = select_winners_by_sig_degree(val)
    assert winners == {2: "taxi_sigtpp_TX1000_sig_2_use_lstm"}
    assert "taxi_wgan_TX1000_hidden_64" not in winners.values()


def test_select_winners_excludes_failed_rows(tmp_path):
    # Failed rows carry a numeric penalty val_norm_score (not NaN), so they must
    # be excluded by their ERROR message, mirroring _evaluate_winner_on_test.
    # Degree 2: the failed row has the LOWEST score; it must still lose.
    # Degree 3: every config failed; the degree must elect no winner at all.
    path = tmp_path / "val_tuning.txt"
    lines = [
        "MODEL  val_norm_score  ERROR",
        "taxi_sigtpp_TX1000_sig_2_use_gru  0.50",
        "taxi_sigtpp_TX1000_sig_2_use_lstm  0.10  CUDA out of memory",
        "taxi_sigtpp_TX1000_sig_3_use_gru  0.20  Training failed: nan loss",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    winners = select_winners_by_sig_degree(path)
    assert winners == {2: "taxi_sigtpp_TX1000_sig_2_use_gru"}


def test_select_winners_logs_and_excludes_nan_scores(tmp_path, caplog):
    val = _write_val_tuning(
        tmp_path / "val_tuning.txt",
        [
            ("taxi_sigtpp_TX1000_sig_2_use_lstm", "nan"),
            ("taxi_sigtpp_TX1000_sig_2_use_gru", 0.40),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="test.paper_experiments.sig_degree_selection"):
        winners = select_winners_by_sig_degree(val)

    assert winners == {2: "taxi_sigtpp_TX1000_sig_2_use_gru"}
    assert "non-finite val_norm_score" in caplog.text


def test_find_latest_val_tuning_file_picks_newest(tmp_path):
    (tmp_path / "sigtpp_val_tuning_2026-01-01_00-00-00.txt").write_text("MODEL x\n", encoding="utf-8")
    newest = tmp_path / "sigtpp_val_tuning_2026-01-02_00-00-00.txt"
    newest.write_text("MODEL x\n", encoding="utf-8")
    assert find_latest_val_tuning_file(tmp_path, "sigtpp") == newest


def test_find_latest_val_tuning_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="val_tuning"):
        find_latest_val_tuning_file(tmp_path, "sigtpp")



# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_report_renders_raw_mean_std_columns(tmp_path):
    winners = {
        2: "taxi_sigtpp_TX1000_sig_2_use_gru",
        3: "taxi_sigtpp_TX1000_sig_3_use_lstm",
    }
    bootstrap_rows = [
        {"model_name": "taxi_sigtpp_TX1000_sig_2_use_gru", "metrics": {"W1_mean": 0.1234, "W1_std": 0.0056, "mark_ce": 1.5}},
        {"model_name": "taxi_sigtpp_TX1000_sig_3_use_lstm", "metrics": {"W1_mean": 0.2000, "W1_std": 0.0100, "mark_ce": 2.0}},
    ]
    out = write_report(winners, bootstrap_rows, tmp_path / "report.txt", ["W1", "mark_ce"])
    text = out.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    line = _line_for(text, "sig_2_use_gru")

    assert "W1_mean" in header
    assert "W1_std" in header
    assert "mark_ce" in header
    assert "mark_ce_mean" not in header
    assert "0.1234" in line
    assert "0.0056" in line
    assert "1.5" in line
    assert "(" not in line


def test_report_default_columns_match_final_test_metric_list(tmp_path):
    winners = {2: "taxi_sigtpp_TX1000_sig_2_use_gru"}
    bootstrap_rows = [
        {
            "model_name": "taxi_sigtpp_TX1000_sig_2_use_gru",
            "metrics": {"W1_mean": 0.1234, "W1_std": 0.0056, "top3_mark_acc": 0.9},
        }
    ]
    out = write_report(winners, bootstrap_rows, tmp_path / "report.txt")
    header = out.read_text(encoding="utf-8").splitlines()[0]

    assert "sig_degree" in header
    assert "sigW_loword_notstd_mean" in header
    assert "hist_it_flat_mean" in header
    assert "W1_mean" in header
    assert "W1_std" in header
    assert "top3_mark_acc" in header
    assert "top3_mark_acc_mean" not in header


def test_report_sigW_columns_keep_raw_small_values(tmp_path):
    winners = {2: "taxi_sigtpp_TX1000_sig_2_use_gru"}
    bootstrap_rows = [
        {
            "model_name": "taxi_sigtpp_TX1000_sig_2_use_gru",
            "metrics": {"sigW_loword_notstd_mean": 2.5e-4, "sigW_loword_notstd_std": 1.0e-5},
        }
    ]
    out = write_report(winners, bootstrap_rows, tmp_path / "report.txt", ["sigW_loword_notstd"])
    text = out.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    line = _line_for(text, "sig_2_use_gru")

    assert "sigW_loword_notstd_mean" in header
    assert "sigW_loword_notstd_std" in header
    assert "0.00025" in line
    assert "1e-05" in line
    assert "25.00(1.00)" not in line


def test_report_missing_bootstrap_row_is_nan(tmp_path):
    # Degree 3's winner is entirely absent from the recompute rows.
    winners = {
        2: "taxi_sigtpp_TX1000_sig_2_use_gru",
        3: "taxi_sigtpp_TX1000_sig_3_use_lstm",
    }
    bootstrap_rows = [
        {"model_name": "taxi_sigtpp_TX1000_sig_2_use_gru", "metrics": {"W1_mean": 0.1234, "W1_std": 0.0056}}
    ]
    out = write_report(winners, bootstrap_rows, tmp_path / "report.txt", ["W1"])
    text = out.read_text(encoding="utf-8")

    assert "nan" in _line_for(text, "sig_3_use_lstm")
    assert "nan" not in _line_for(text, "sig_2_use_gru")


def test_report_failed_bootstrap_row_is_nan(tmp_path):
    # Degree 3's winner is present but failed; every metric cell must be NaN.
    winners = {
        2: "taxi_sigtpp_TX1000_sig_2_use_gru",
        3: "taxi_sigtpp_TX1000_sig_3_use_lstm",
    }
    bootstrap_rows = [
        {"model_name": "taxi_sigtpp_TX1000_sig_2_use_gru", "metrics": {"W1_mean": 0.1234, "W1_std": 0.0056}},
        {"model_name": "taxi_sigtpp_TX1000_sig_3_use_lstm", "metrics": {"W1_mean": 0.0, "W1_std": 0.0, "error": "model load failed"}},
    ]
    out = write_report(winners, bootstrap_rows, tmp_path / "report.txt", ["W1"])
    text = out.read_text(encoding="utf-8")

    assert "nan" in _line_for(text, "sig_3_use_lstm")
    assert "nan" not in _line_for(text, "sig_2_use_gru")


# ---------------------------------------------------------------------------
# Orchestration (GPU step injected)
# ---------------------------------------------------------------------------


def test_run_ablation_uses_the_given_val_file_and_injected_bootstrap(tmp_path):
    results_dir = tmp_path / "out" / "taxi" / "results_on_val_txt"
    ablation_dir = tmp_path / "out" / "taxi" / "results_on_ablation"
    target = _write_val_tuning(
        results_dir / "sigtpp_val_tuning_2026-01-01_00-00-00.txt",
        [
            ("taxi_sigtpp_TX1000_sig_2_use_lstm", 0.90),
            ("taxi_sigtpp_TX1000_sig_2_use_gru", 0.40),
            ("taxi_sigtpp_TX1000_sig_3_use_lstm", 0.20),
        ],
    )
    # Written after `target`, sharing the same `{version}_val_tuning_*.txt`
    # prefix and directory -- simulates a second, unrelated run (e.g. a
    # concurrent multiseed sub-run) landing in the same folder. Regression for
    # the bug where the ablation re-globbed "the latest" file and silently
    # picked this one up instead of the file it was actually told to use.
    _write_val_tuning(
        results_dir / "sigtpp_val_tuning_2026-01-02_00-00-00.txt",
        [("taxi_sigtpp_TX1000_sig_8_use_other_run", 0.01)],
    )

    recompute_calls = []

    def recompute_one_row(run_name: str, settings: _FakeSettings, gpu_id: int) -> Dict[str, Any]:
        recompute_calls.append((run_name, settings.n_bootstraps, settings.trainer_seed, gpu_id))
        return {
            "model_name": "ignored_by_runner",
            "metrics": {"W1_mean": 0.10, "W1_std": 0.01},
            "per_replicate": {"W1": [0.09, 0.11]},
        }

    npz_calls = []

    def write_npz(rows: List[Dict[str, Any]], path: str, b: int) -> None:
        npz_calls.append((rows, Path(path), b))
        Path(path).write_text("fake npz", encoding="utf-8")

    outputs = run_sig_degree_ablation_from_val_file(
        target,
        "sigtpp",
        n_bootstraps=25,
        trainer_seed=123,
        gpu_id=7,
        recompute_one_row_fn=recompute_one_row,
        write_npz_fn=write_npz,
        settings_cls=_FakeSettings,
        results_dir=ablation_dir,
    )

    # The given path wins, even though a newer same-prefix file sits in the
    # same directory; the newer (other run's) sig_8 winner never enters selection.
    assert outputs.val_tuning_path == target
    assert outputs.n_bootstraps == 25
    assert outputs.report_path.parent == ablation_dir
    assert outputs.npz_path.parent == ablation_dir
    assert "_sig_degree_ablation_B25_" in outputs.report_path.name

    # One recompute per degree winner, in ascending-degree order, carrying the
    # injected bootstrap count / seed / gpu id verbatim.
    assert recompute_calls == [
        ("taxi_sigtpp_TX1000_sig_2_use_gru", 25, 123, 7),
        ("taxi_sigtpp_TX1000_sig_3_use_lstm", 25, 123, 7),
    ]
    assert len(npz_calls) == 1
    assert npz_calls[0][1] == outputs.npz_path
    assert npz_calls[0][2] == 25

    report_text = outputs.report_path.read_text(encoding="utf-8")
    assert "use_other_run" not in report_text
    assert "taxi_sigtpp_TX1000_sig_3_use_lstm" in report_text


def test_run_ablation_defaults_output_dir_to_val_file_parent(tmp_path):
    val = _write_val_tuning(
        tmp_path / "results" / "sigtpp_val_tuning_2026-01-01_00-00-00.txt",
        [("taxi_sigtpp_TX1000_sig_2_use_x", 0.10)],
    )

    outputs = run_sig_degree_ablation_from_val_file(
        val,
        "sigtpp",
        n_bootstraps=1,
        trainer_seed=1,
        gpu_id=None,
        recompute_one_row_fn=lambda run_name, settings, gpu_id: {"model_name": run_name, "metrics": {}},
        write_npz_fn=lambda rows, path, b: Path(path).write_text("fake npz", encoding="utf-8"),
        settings_cls=_FakeSettings,
    )

    assert outputs.report_path.parent == val.parent
    assert outputs.npz_path.parent == val.parent
