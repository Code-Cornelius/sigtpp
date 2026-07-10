import pytest

from test.paper_experiments.training_helpers import (
    get_dir_name_from_params,
    parse_model_dir_to_cfg,
)


@pytest.fixture
def hawkes_cfg_root(tmp_path):
    """Write a minimal reference YAML and return the configs root."""
    (tmp_path / "hawkes").mkdir()
    (tmp_path / "hawkes" / "sigtpp.yaml").write_text(
        "experiment_type: hawkes\n"
        "version: sigtpp\n"
        "epochs: 50\n"
        "period_log: 5\n"
        "gpu_id: 0\n"
        "seeds: [0]\n"
        "output_dir: out\n"
        "parameter_sets:\n"
        "  learning_rate: 0.0015\n"
        "  batch_size: 32\n"
        "  hidden_size: 64\n"
    )
    return tmp_path


def test_round_trip_dir_name(hawkes_cfg_root):
    parameter_sets = {"learning_rate": 0.0015, "batch_size": 32, "hidden_size": 64}
    dir_name = get_dir_name_from_params(
        data_name="hawkes",
        version="sigtpp",
        config=parameter_sets,
        time_max=10.0,
    )
    recovered = parse_model_dir_to_cfg(dir_name, "hawkes", str(hawkes_cfg_root))
    assert recovered["parameter_sets"]["learning_rate"] == pytest.approx(0.0015, rel=1e-3)
    assert recovered["parameter_sets"]["batch_size"] == 32
    assert recovered["parameter_sets"]["hidden_size"] == 64


def test_collision_raises(tmp_path):
    (tmp_path / "hawkes").mkdir()
    (tmp_path / "hawkes" / "sigtpp.yaml").write_text("parameter_sets:\n  learning_rate: 0.001\n  learner_type: foo\n")
    with pytest.raises(ValueError, match="abbreviation collision"):
        parse_model_dir_to_cfg("hawkes_sigtpp_TX10_lear0,001_learfoo", "hawkes", str(tmp_path))


def test_missing_reference_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_model_dir_to_cfg("hawkes_sigtpp_TX10", "hawkes", str(tmp_path))


def test_round_trip_dir_name_with_seed_tag(hawkes_cfg_root):
    """Multi-seed runs append _seed<N>; the reverse parser must peel it off."""
    parameter_sets = {"learning_rate": 0.0015, "batch_size": 32, "hidden_size": 64}
    dir_name = get_dir_name_from_params(
        data_name="hawkes",
        version="sigtpp",
        config=parameter_sets,
        time_max=10.0,
        seed=42,
    )
    assert dir_name.endswith("_seed42"), dir_name
    recovered = parse_model_dir_to_cfg(dir_name, "hawkes", str(hawkes_cfg_root))
    assert recovered["parameter_sets"]["learning_rate"] == pytest.approx(0.0015, rel=1e-3)
    assert recovered["parameter_sets"]["batch_size"] == 32
    assert recovered["parameter_sets"]["hidden_size"] == 64
    assert recovered["seeds"] == [42]


def test_round_trip_dir_name_without_seed_is_backwards_compatible(hawkes_cfg_root):
    """Single-seed runs do not emit a seed token; parser must still work and leave seeds untouched."""
    parameter_sets = {"learning_rate": 0.0015, "batch_size": 32, "hidden_size": 64}
    dir_name = get_dir_name_from_params(
        data_name="hawkes",
        version="sigtpp",
        config=parameter_sets,
        time_max=10.0,
    )
    assert "_seed" not in dir_name
    recovered = parse_model_dir_to_cfg(dir_name, "hawkes", str(hawkes_cfg_root))
    # seeds comes from the reference YAML, not from a parsed token.
    assert recovered["seeds"] == [0]
