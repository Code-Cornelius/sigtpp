"""Lightweight dataset-name metadata shared by experiment scripts.

This module must remain free of training-stack imports so reporting and
analysis tools can use the run-name mapping without optional ML dependencies.
"""

import typing


# Run names embed the namer prefix rather than the registered experiment type.
# This public-release table intentionally contains only the nine paper datasets.
DATA_NAME_TO_EXPERIMENT: typing.Dict[str, str] = {
    "hp_three_marks": "poisson_three_marks",
    "ihp_three_marks": "inh_poisson_three_marks",
    "hawkes": "hawkes",
    "hawkes_3x3": "hawkes_3x3",
    "earthquake": "earthquake",
    "stackoverflow": "stackoverflow",
    "taobao": "taobao",
    "taxi": "taxi",
    "yelp_mississauga": "yelp_mississauga",
}

ALL_DATASET_NAMES: typing.Tuple[str, ...] = tuple(DATA_NAME_TO_EXPERIMENT)
