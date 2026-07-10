"""Behavioural locks for ``run_experiment_config``.

These tests focus on the multi-seed dispatch and the contracts that the runner
enforces on behalf of the aggregation layer â€” chiefly that a non-unit
``n_bootstraps`` cannot silently survive into multi-seed aggregation, and that
the cross-seed test finalization evaluates the SAME globally-selected config
for every seed rather than each seed's own local winner.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("signatory")  # training_runner â†’ tppmetrics â†’ signatory

from test.paper_experiments import training_runner
from test.paper_experiments.experiment_results import ExperimentResults


@pytest.fixture
def multiseed_cfg(tmp_path):
    return {
        "seeds": [1, 2],
        "n_bootstraps": 100,
        "experiment_type": "poisson_three_marks",
        "version": "sigtpp",
        "server_training": False,
        "output_dir": str(tmp_path),
    }


def _empty_seed_result(version="sigtpp"):
    """A seed sub-run result with no rankable rows: ``_finalize_multiseed_test``
    should no-op on this rather than crash on ``result.rows``."""
    return ExperimentResults([], version=version)


def test_multiseed_dispatch_logs_override_with_original_b(multiseed_cfg, caplog):
    """build_seed_config must log an error naming the original B when n_bootstraps > 1."""
    with caplog.at_level("ERROR", logger="test.paper_experiments.multiseed_helpers"):
        with patch.object(
            training_runner, "_run_training", return_value=_empty_seed_result()
        ) as mock_train, patch.object(training_runner, "write_multiseed_outputs") as mock_write, patch.object(
            training_runner.TrainingManager, "get_pathlinker", return_value=lambda parts: "/tmp/" + "/".join(parts)
        ):
            training_runner.run_experiment_config(multiseed_cfg)

    assert mock_train.call_count == 2  # one per seed
    mock_write.assert_called_once()
    assert mock_write.call_args.args[2] == "/tmp/out/poisson_three_marks/results_on_multiseed/"

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "forcing n_bootstraps from 100 to 1" in msg for msg in messages
    ), f"Expected error naming B=100, got: {messages}"


def test_multiseed_dispatch_silent_when_b_already_one(tmp_path, caplog):
    """No error fires for the standard ``n_bootstraps=1`` path."""
    cfg = {
        "seeds": [1, 2],
        "n_bootstraps": 1,
        "experiment_type": "poisson_three_marks",
        "version": "sigtpp",
        "server_training": False,
        "output_dir": str(tmp_path),
    }

    with caplog.at_level("ERROR", logger="test.paper_experiments.multiseed_helpers"):
        with patch.object(training_runner, "_run_training", return_value=_empty_seed_result()), patch.object(
            training_runner, "write_multiseed_outputs"
        ), patch.object(training_runner.TrainingManager, "get_pathlinker", return_value=lambda parts: "/tmp/"):
            training_runner.run_experiment_config(cfg)

    assert not any(
        "forcing n_bootstraps" in record.message for record in caplog.records
    ), "Override error fired even though n_bootstraps was already 1."


def test_single_seed_dispatch_does_not_override_bootstrap(tmp_path, caplog):
    """Single-seed runs preserve the bootstrap path untouched (no override, no error)."""
    cfg = {
        "seeds": [42],
        "n_bootstraps": 100,
        "experiment_type": "poisson_three_marks",
        "version": "sigtpp",
        "server_training": False,
        "output_dir": str(tmp_path),
    }

    with caplog.at_level("ERROR", logger="test.paper_experiments.multiseed_helpers"):
        with patch.object(training_runner, "_run_training", return_value=None) as mock_train:
            training_runner.run_experiment_config(cfg)

    assert mock_train.call_count == 1
    forwarded_cfg = mock_train.call_args.args[0]
    assert forwarded_cfg["n_bootstraps"] == 100, "Single-seed run must not override B."
    assert not any("forcing n_bootstraps" in record.message for record in caplog.records)


def _seed_result_for(model_name: str, seed_cfg: dict, val_ed: float) -> ExperimentResults:
    """A seed sub-run result with one rankable config, mirroring what
    ``TrainingManager.run()`` returns: rows carry ``val_``-prefixed diagnostics
    and ``config_by_model_name`` resolves the model name back to its cfg."""
    results = ExperimentResults([{"model_name": model_name, "metrics": {"val_ED_mean": val_ed}}], version="sigtpp")
    results.config_by_model_name = {model_name: seed_cfg}
    return results


def test_finalize_multiseed_test_evaluates_same_config_for_every_seed(multiseed_cfg, tmp_path):
    """The cross-seed winner (best mean val_ED across seeds) must be evaluated
    on test for BOTH seeds, even though nothing here says which seed's local
    grid search would have picked it â€” that per-seed independence is exactly
    the ambiguity this finalization step removes."""
    seed1_cfg = {**multiseed_cfg, "seeds": [1]}
    seed2_cfg = {**multiseed_cfg, "seeds": [2]}
    seed_results = [
        (1, _seed_result_for("cfg_a_seed1", seed1_cfg, val_ed=1.0)),
        (2, _seed_result_for("cfg_a_seed2", seed2_cfg, val_ed=3.0)),
    ]

    fake_manager = MagicMock()
    fake_manager.evaluate_named_model_on_test.side_effect = lambda model_name, *a, **kw: {
        "model_name": model_name,
        "metrics": {"ED_mean": 0.5},
    }

    with patch.object(training_runner, "_build_manager", return_value=fake_manager) as mock_build, patch.object(
        training_runner, "write_multiseed_test_by_seed_txt"
    ) as mock_by_seed, patch.object(training_runner, "write_multiseed_test_summary_txt") as mock_summary:
        training_runner._finalize_multiseed_test(multiseed_cfg, seed_results, str(tmp_path))

    assert mock_build.call_count == 2
    evaluated_names = [call.args[0] for call in fake_manager.evaluate_named_model_on_test.call_args_list]
    assert evaluated_names == ["cfg_a_seed1", "cfg_a_seed2"]
    # write_report=False: the per-seed report is retired, superseded by the two
    # consolidated files below.
    for call in fake_manager.evaluate_named_model_on_test.call_args_list:
        assert call.kwargs.get("write_report") is False or call.args[-1] is False

    assert fake_manager.prune_all_except.call_count == 2
    pruned_keep_sets = [call.args[1] for call in fake_manager.prune_all_except.call_args_list]
    assert {"cfg_a_seed1"} in pruned_keep_sets
    assert {"cfg_a_seed2"} in pruned_keep_sets

    mock_by_seed.assert_called_once()
    test_rows_arg = mock_by_seed.call_args.args[0]
    assert [seed for seed, _row in test_rows_arg] == [1, 2]

    mock_summary.assert_called_once()
    assert mock_summary.call_args.args[0] == "cfg_a"


def test_finalize_multiseed_test_skips_seed_missing_winner_checkpoint(multiseed_cfg, tmp_path, caplog):
    """If a seed's config_by_model_name has no entry for the cross-seed winner
    (defensive: shouldn't happen given deferred pruning), that seed is skipped
    with a warning rather than silently evaluating the wrong config."""
    seed1_cfg = {**multiseed_cfg, "seeds": [1]}
    seed_results = [
        (1, _seed_result_for("cfg_a_seed1", seed1_cfg, val_ed=1.0)),
        (2, ExperimentResults([{"model_name": "cfg_b_seed2", "metrics": {"val_ED_mean": 3.0}}], version="sigtpp")),
    ]
    seed_results[1][1].config_by_model_name = {"cfg_b_seed2": {**multiseed_cfg, "seeds": [2]}}

    fake_manager = MagicMock()
    fake_manager.evaluate_named_model_on_test.return_value = {"model_name": "cfg_a_seed1", "metrics": {"ED_mean": 0.5}}

    with caplog.at_level("WARNING", logger="test.paper_experiments.training_runner"):
        with patch.object(training_runner, "_build_manager", return_value=fake_manager), patch.object(
            training_runner, "write_multiseed_test_by_seed_txt"
        ) as mock_by_seed, patch.object(training_runner, "write_multiseed_test_summary_txt"):
            training_runner._finalize_multiseed_test(multiseed_cfg, seed_results, str(tmp_path))

    assert fake_manager.evaluate_named_model_on_test.call_count == 1
    test_rows_arg = mock_by_seed.call_args.args[0]
    assert [seed for seed, _row in test_rows_arg] == [1]
    assert any("no surviving checkpoint" in record.message for record in caplog.records)


def test_finalize_multiseed_test_no_op_when_no_rankable_config(multiseed_cfg, tmp_path, caplog):
    seed_results = [(1, _empty_seed_result()), (2, _empty_seed_result())]

    with caplog.at_level("WARNING", logger="test.paper_experiments.training_runner"):
        with patch.object(training_runner, "_build_manager") as mock_build, patch.object(
            training_runner, "write_multiseed_test_by_seed_txt"
        ) as mock_by_seed, patch.object(training_runner, "write_multiseed_test_summary_txt") as mock_summary:
            training_runner._finalize_multiseed_test(multiseed_cfg, seed_results, str(tmp_path))

    mock_build.assert_not_called()
    mock_by_seed.assert_not_called()
    mock_summary.assert_not_called()
    assert any("no rankable config" in record.message for record in caplog.records)


def test_finalize_multiseed_test_respects_evaluate_winner_on_test_false(multiseed_cfg, tmp_path):
    """The single-seed path in TrainingManager.run() honors ``evaluate_winner_on_test:
    false``; the multi-seed finalization step must too, instead of unconditionally
    running the (expensive) test pass regardless of that config flag."""
    cfg = {**multiseed_cfg, "evaluate_winner_on_test": False}
    seed1_cfg = {**cfg, "seeds": [1]}
    seed_results = [(1, _seed_result_for("cfg_a_seed1", seed1_cfg, val_ed=1.0))]

    with patch.object(training_runner, "_build_manager") as mock_build, patch.object(
        training_runner, "write_multiseed_test_by_seed_txt"
    ) as mock_by_seed, patch.object(training_runner, "write_multiseed_test_summary_txt") as mock_summary:
        training_runner._finalize_multiseed_test(cfg, seed_results, str(tmp_path))

    mock_build.assert_not_called()
    mock_by_seed.assert_not_called()
    mock_summary.assert_not_called()


def test_finalize_multiseed_test_skips_winner_with_nan_val_norm_score(multiseed_cfg, tmp_path, caplog):
    """``skip_diagnostics: true`` (or every config failing) leaves every row's val
    metrics empty, so val_norm_score is NaN for every config; aggregate_across_seeds's
    row order is then arbitrary insertion order, not a real ranking. Finalization must
    refuse to pick a "winner" out of it rather than test-evaluating an arbitrary config."""
    seed1_cfg = {**multiseed_cfg, "seeds": [1]}
    seed_results = [(1, _seed_result_for("cfg_a_seed1", seed1_cfg, val_ed=float("nan")))]

    with caplog.at_level("WARNING", logger="test.paper_experiments.training_runner"):
        with patch.object(training_runner, "_build_manager") as mock_build, patch.object(
            training_runner, "write_multiseed_test_by_seed_txt"
        ) as mock_by_seed, patch.object(training_runner, "write_multiseed_test_summary_txt") as mock_summary:
            training_runner._finalize_multiseed_test(multiseed_cfg, seed_results, str(tmp_path))

    mock_build.assert_not_called()
    mock_by_seed.assert_not_called()
    mock_summary.assert_not_called()
    assert any("no valid validation ranking" in record.message for record in caplog.records)
