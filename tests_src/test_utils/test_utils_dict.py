import pytest

from src.utils.utils_dict import verbose_get


def test_verbose_get_missing_key_reports_compact_dict_summary():
    registry = {
        "earthquake": {"data_factory": object()},
        "poisson_three_marks": {"data_factory": object()},
        "yelp_mississauga": {"data_factory": object()},
    }

    with pytest.raises(KeyError) as exc_info:
        verbose_get(registry, "yelp")

    message = str(exc_info.value)
    assert "dict(len=3, keys=['earthquake', 'poisson_three_marks', 'yelp_mississauga'])" in message
    assert "data_factory" not in message
