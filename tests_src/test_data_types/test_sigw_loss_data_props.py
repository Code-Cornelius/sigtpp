import pytest

from src.data_types.sigw_loss_data_props import (
    DEFAULT_MAX_CONSIDERED_SIG_DEGREE,
    SigWLossDataProps,
    SkipSigDegreeConfig,
    sigw_loss_data_props_from_config,
)


def test_absolute_sig_degree_config_preserves_existing_behavior():
    props = sigw_loss_data_props_from_config(
        {"sig_degree": 6},
        scale_high_degrees=False,
        standardise_sig=True,
    )

    assert props.sig_degree == 6
    assert props.relative_sig_degree is None
    assert props.use_degree_detector is False


def test_relative_sig_degree_config_uses_default_detector_limit():
    props = sigw_loss_data_props_from_config(
        {"relative_sig_degree": -2},
        scale_high_degrees=False,
        standardise_sig=True,
    )

    assert props.sig_degree == DEFAULT_MAX_CONSIDERED_SIG_DEGREE
    assert props.relative_sig_degree == -2
    assert props.use_degree_detector is True


def test_relative_sig_degree_resolves_around_largest_ok_degree():
    resolved = [
        sigw_loss_data_props_from_config(
            {"relative_sig_degree": offset},
            scale_high_degrees=False,
            standardise_sig=True,
        ).resolve_detected_sig_degree(8)
        for offset in [-2, -1, 0, 1]
    ]

    assert resolved == [6, 7, 8, 9]


def test_use_degree_detector_is_derived_from_relative_sig_degree():
    absolute = SigWLossDataProps(
        sig_degree=10,
        scale_high_degrees=False,
        standardise_sig=True,
    )
    relative = SigWLossDataProps(
        sig_degree=10,
        scale_high_degrees=False,
        standardise_sig=True,
        relative_sig_degree=0,
    )

    assert absolute.use_degree_detector is False
    assert relative.use_degree_detector is True


def test_config_rejects_both_absolute_and_relative_sig_degree():
    with pytest.raises(AssertionError, match="exactly one"):
        sigw_loss_data_props_from_config(
            {"sig_degree": 8, "relative_sig_degree": 0},
            scale_high_degrees=False,
            standardise_sig=True,
        )


def test_relative_sig_degree_skips_above_detector_limit():
    props = sigw_loss_data_props_from_config(
        {"relative_sig_degree": 1},
        scale_high_degrees=False,
        standardise_sig=True,
    )

    with pytest.raises(SkipSigDegreeConfig, match="exceeds sig_degree"):
        props.resolve_detected_sig_degree(DEFAULT_MAX_CONSIDERED_SIG_DEGREE)


def test_relative_sig_degree_skips_below_three():
    props = sigw_loss_data_props_from_config(
        {"relative_sig_degree": -2},
        scale_high_degrees=False,
        standardise_sig=True,
    )

    with pytest.raises(SkipSigDegreeConfig, match="below 3"):
        props.resolve_detected_sig_degree(4)


def test_use_float64_signature_defaults_false():
    props = SigWLossDataProps(sig_degree=4, scale_high_degrees=False, standardise_sig=True)
    assert props.use_float64_signature is False


def test_use_float64_signature_read_from_config():
    props = sigw_loss_data_props_from_config(
        {"sig_degree": 6, "use_float64_signature": True},
        scale_high_degrees=False,
        standardise_sig=True,
    )
    assert props.use_float64_signature is True


def test_use_float64_signature_absent_from_config_is_false():
    props = sigw_loss_data_props_from_config(
        {"sig_degree": 6},
        scale_high_degrees=False,
        standardise_sig=True,
    )
    assert props.use_float64_signature is False


def test_use_float64_signature_read_from_relative_config():
    props = sigw_loss_data_props_from_config(
        {"relative_sig_degree": 0, "use_float64_signature": True},
        scale_high_degrees=False,
        standardise_sig=True,
    )
    assert props.use_float64_signature is True
