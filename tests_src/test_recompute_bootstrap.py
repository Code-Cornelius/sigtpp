"""Unit tests for recompute_bootstrap helpers that don't need torch."""


# ---------------------------------------------------------------------------
# Run-name round-trip contract (relied on by recompute_bootstrap consumers)
# ---------------------------------------------------------------------------


def test_split_run_name_roundtrip_matches_input():
    # recompute_bootstrap writes each row's MODEL column as split_run_name(run_name)[2],
    # and downstream consumers (results tables, the per-replicate NPZ, per-degree winner
    # lookups) join rows back to their input run name on that value. If the third element
    # ever stops equalling the input run name, those joins silently miss. Pin the contract
    # here so a regression in split_run_name is caught.
    from test.paper_experiments.recompute_bootstrap import split_run_name

    for run_name in [
        "taxi_sigtpp_TX1000_sig_2_use_gru",
        "stackoverflow_sigwgan_TX500_sig_3_use_lstm",
    ]:
        _data_name, _version, model_dir = split_run_name(run_name)
        assert model_dir == run_name
