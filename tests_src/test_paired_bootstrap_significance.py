import os
import tempfile

import numpy as np
import pytest

from test.paper_extra_experiments.paired_bootstrap_significance import (
    adjust_pvalues,
    build_outcome_summary,
    load_npz_records,
    run_paired_tests,
)


def _write_npz(path: str) -> None:
    model_names = np.array(
        [
            "hawkes_sigtpp_TX20_good",
            "hawkes_wgan_TX20_base",
        ],
        dtype=object,
    )
    metric_names = np.array(["ED", "W1"], dtype=object)
    data = np.array(
        [
            [[0.10, 0.11, 0.10, 0.09, 0.10], [0.40, 0.41, 0.42, 0.40, 0.39]],
            [[0.20, 0.21, 0.19, 0.22, 0.20], [0.35, 0.36, 0.35, 0.34, 0.35]],
        ],
        dtype=float,
    )
    np.savez_compressed(
        path,
        model_names=model_names,
        metric_names=metric_names,
        data=data,
        schema_version=np.array(1),
        B=np.array(5),
    )


def test_load_npz_records_reads_schema_v1_vectors():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "paired.npz")
        _write_npz(path)

        records = load_npz_records([path])

    assert len(records) == 4
    ed = [r for r in records if r.model == "sigtpp" and r.metric == "ED"][0]
    assert ed.dataset == "hawkes"
    assert ed.values.shape == (5,)
    np.testing.assert_allclose(ed.values, [0.10, 0.11, 0.10, 0.09, 0.10])


def test_run_paired_tests_emits_pvalues_and_replicate_counts():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "paired.npz")
        _write_npz(path)
        records = load_npz_records([path])

    df = run_paired_tests(
        records,
        compared_model="sigtpp",
        baseline_models=["wgan"],
        metrics=["ED", "W1"],
        alternative="less",
        correction="holm",
    )

    assert set(df["metric"]) == {"ED", "W1"}
    ed = df[df["metric"] == "ED"].iloc[0]
    assert ed["replicate_wins"] == 5
    assert ed["replicate_ties"] == 0
    assert ed["replicate_losses"] == 0
    assert ed["paired_t_p"] < 0.001
    assert ed["paired_t_p_adj"] <= 0.01
    assert bool(ed["paired_t_reject"])

    w1 = df[df["metric"] == "W1"].iloc[0]
    assert w1["replicate_wins"] == 0
    assert w1["replicate_losses"] == 5


def test_build_outcome_summary_counts_corrected_wins_ties_losses():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "paired.npz")
        _write_npz(path)
        records = load_npz_records([path])

    df = run_paired_tests(
        records,
        compared_model="sigtpp",
        baseline_models=["wgan"],
        metrics=["ED", "W1"],
        alternative="less",
        correction="holm",
    )
    summary = build_outcome_summary(df)

    row = summary[
        (summary["test_family"] == "paired_t") & (summary["metric"] == "__all_metrics__") & (summary["split"] == "all")
    ].iloc[0]
    assert row["significant_wins"] == 1
    assert row["ties_or_not_significant"] == 1
    assert row["significant_losses"] == 0
    assert row["total"] == 2


def test_adjust_pvalues_holm_is_monotone_and_rejects_expected_values():
    adjusted, reject = adjust_pvalues([0.001, 0.02, 0.5], method="holm", alpha=0.05)

    np.testing.assert_allclose(adjusted, [0.003, 0.04, 0.5])
    assert reject.tolist() == [True, True, False]


def test_load_npz_records_rejects_bad_schema():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.npz")
        np.savez_compressed(
            path,
            model_names=np.array(["hawkes_sigtpp_TX20_good"], dtype=object),
            metric_names=np.array(["ED"], dtype=object),
            data=np.zeros((1, 1, 2)),
            schema_version=np.array(999),
            B=np.array(2),
        )

        with pytest.raises(ValueError, match="unsupported schema_version"):
            load_npz_records([path])
