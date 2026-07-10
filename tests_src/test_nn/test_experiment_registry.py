"""Verify that all experiment types are registered after trainingmanager imports."""

import pytest
pytest.importorskip("signatory")  # skips entire module in CI where signatory cannot be built

# Side-effect imports happen when trainingmanager is imported
import test.paper_experiments.trainingmanager  # noqa: F401
from test.paper_experiments.experiment_registry import EXPERIMENT_REGISTRY


@pytest.mark.parametrize(
    "name",
    [
        "poisson_three_marks",
        "inh_poisson_three_marks",
        "hawkes",
        "hawkes_3x3",
        "taxi",
        "stackoverflow",
        "taobao",
        "earthquake",
    ],
)
def test_experiment_registered(name):
    assert name in EXPERIMENT_REGISTRY, f"'{name}' not found in EXPERIMENT_REGISTRY"
