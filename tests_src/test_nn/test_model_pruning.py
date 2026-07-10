"""Unit tests for TrainingManager._prune_non_top_models.

No Trainer is constructed: the manager is built with dummy factories and a cfg
whose output_dir points into a tmp dir, so path_link resolves model dirs there.
"""

import os

import pytest

pytest.importorskip("signatory")

from test.paper_experiments.experiment_results import ExperimentResults
from test.paper_experiments.trainingmanager import TrainingManager


def _manager(tmp_path, keep_k):
    cfg = {
        "output_dir": str(tmp_path),
        "experiment_type": "dummy_exp",
        "version": "sigtpp",
        "keep_top_k_models": keep_k,
    }
    return TrainingManager(
        data_factory=lambda c: None,
        model_factory=lambda *a, **k: None,
        model_namer=lambda *a, **k: "x",
        loss_metrics_fn=lambda *a, **k: [],
        config=cfg,
    )


def _make_model_dir(tmp_path, name):
    # Mirrors path_link([OUT_FILE_NAME, experiment_type, "models", name, ""]).
    directory = os.path.join(str(tmp_path), "out", "dummy_exp", "models", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "model.ckpt"), "wb") as handle:
        handle.write(b"0" * 100)
    return directory


def _row(name, ed):
    # Only val_ED_mean varies, so ranking order is deterministic (lower = better).
    return {"model_name": name, "metrics": {"val_ED_mean": ed}}


def test_prune_keeps_top_k_and_deletes_lowrank_and_failed(tmp_path):
    mgr = _manager(tmp_path, keep_k=2)
    rows = [_row("m1", 1.0), _row("m2", 2.0), _row("m3", 3.0), _row("m4", 4.0)]
    failed = {"model_name": "mfail", "metrics": {"error": "boom"}}
    results = ExperimentResults(rows + [failed], version="sigtpp")
    results.final_test_row = {"model_name": "m1"}
    dirs = {n: _make_model_dir(tmp_path, n) for n in ["m1", "m2", "m3", "m4", "mfail"]}

    mgr._prune_non_top_models(results)

    assert os.path.isdir(dirs["m1"])  # winner, kept
    assert os.path.isdir(dirs["m2"])  # 2nd best, kept
    assert not os.path.isdir(dirs["m3"])  # outside top-2, deleted
    assert not os.path.isdir(dirs["m4"])  # outside top-2, deleted
    assert not os.path.isdir(dirs["mfail"])  # failed config, deleted


@pytest.mark.parametrize("keep_k", [0, -1])
def test_prune_disabled_keeps_all_dirs(tmp_path, keep_k):
    mgr = _manager(tmp_path, keep_k=keep_k)
    rows = [_row("m1", 1.0), _row("m2", 2.0), _row("m3", 3.0)]
    results = ExperimentResults(rows, version="sigtpp")
    dirs = {n: _make_model_dir(tmp_path, n) for n in ["m1", "m2", "m3"]}

    mgr._prune_non_top_models(results)

    assert all(os.path.isdir(d) for d in dirs.values())


def test_prune_default_keep_k_is_ten(tmp_path):
    """``keep_top_k_models`` absent/None defaults to 10, not "disabled"."""
    mgr = _manager(tmp_path, keep_k=None)
    rows = [_row(f"m{i}", float(i)) for i in range(1, 13)]  # m1..m12, m1 best
    results = ExperimentResults(rows, version="sigtpp")
    results.final_test_row = {"model_name": "m1"}
    dirs = {row["model_name"]: _make_model_dir(tmp_path, row["model_name"]) for row in rows}

    mgr._prune_non_top_models(results)

    for i in range(1, 11):
        assert os.path.isdir(dirs[f"m{i}"])  # top-10, kept
    for i in range(11, 13):
        assert not os.path.isdir(dirs[f"m{i}"])  # outside top-10, deleted


def test_prune_skipped_when_interrupted(tmp_path):
    mgr = _manager(tmp_path, keep_k=1)
    mgr._stop_requested = True
    rows = [_row("m1", 1.0), _row("m2", 2.0)]
    results = ExperimentResults(rows, version="sigtpp")
    dirs = {n: _make_model_dir(tmp_path, n) for n in ["m1", "m2"]}

    mgr._prune_non_top_models(results)

    assert all(os.path.isdir(d) for d in dirs.values())


def test_prune_noop_when_no_rankable_rows(tmp_path):
    mgr = _manager(tmp_path, keep_k=1)
    rows = [
        {"model_name": "m1", "metrics": {"error": "x"}},
        {"model_name": "m2", "metrics": {"error": "y"}},
    ]
    results = ExperimentResults(rows, version="sigtpp")
    dirs = {n: _make_model_dir(tmp_path, n) for n in ["m1", "m2"]}

    mgr._prune_non_top_models(results)

    # No val ranking possible -> safe no-op, nothing deleted.
    assert all(os.path.isdir(d) for d in dirs.values())
