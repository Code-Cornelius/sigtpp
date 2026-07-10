from dataclasses import dataclass
from typing import Optional

from src.data_types.exceptions import SkipConfig

DEFAULT_MAX_CONSIDERED_SIG_DEGREE = 10
MIN_USEFUL_RELATIVE_SIG_DEGREE = 3


class SkipSigDegreeConfig(SkipConfig):
    """Raised when a relative signature degree resolves to a config worth skipping."""


@dataclass
class SigWLossDataProps:
    """Properties of the data used in the SigW loss.

    Exactly two mutually-exclusive degree-selection modes are supported:
      * absolute: pass ``sig_degree=K`` with ``relative_sig_degree=None`` — the
        loss uses ``K`` directly and the auto-detector is **off**.
      * relative: pass ``sig_degree=K_max`` together with
        ``relative_sig_degree=Δ`` — the detector trims dead degrees, then the
        active loss degree is ``detector_largest_ok + Δ`` (clamped by ``K_max``).
        ``Δ = 0`` keeps the detector's answer as-is; negative values are more
        conservative.

    There is no third "absolute + detector" mode: the detector only runs when
    ``relative_sig_degree`` is set. ``use_degree_detector`` is exposed as a
    derived property so callers can keep their conditional code unchanged.
    """

    sig_degree: int
    scale_high_degrees: bool
    standardise_sig: bool
    relative_sig_degree: Optional[int] = None
    # When True, SigW1MetricExp runs its signature/aggregation/backward in float64
    # (boundary stays float32). Off by default; see SigW1MetricExp for the rationale.
    use_float64_signature: bool = False

    def __post_init__(self):
        assert self.sig_degree > 0, f"The signature degree must be positive, got {self.sig_degree}."

    @property
    def use_degree_detector(self) -> bool:
        return self.relative_sig_degree is not None

    def resolve_detected_sig_degree(self, largest_ok_degree: int) -> int:
        """Return the training signature degree after optional relative offset."""
        if self.relative_sig_degree is None:
            return largest_ok_degree

        resolved = largest_ok_degree + self.relative_sig_degree
        if resolved < MIN_USEFUL_RELATIVE_SIG_DEGREE:
            raise SkipSigDegreeConfig(
                "Skipping relative signature degree config because the resolved degree is below "
                f"{MIN_USEFUL_RELATIVE_SIG_DEGREE}. Got {resolved} from "
                f"largest_ok_degree={largest_ok_degree}, relative_sig_degree={self.relative_sig_degree}."
            )
        if resolved > self.sig_degree:
            raise SkipSigDegreeConfig(
                "Skipping relative signature degree config because the resolved degree exceeds sig_degree. "
                f"Got {resolved} from largest_ok_degree={largest_ok_degree}, "
                f"relative_sig_degree={self.relative_sig_degree}, sig_degree={self.sig_degree}."
            )
        return resolved


def sigw_loss_data_props_from_config(
    config: dict,
    scale_high_degrees: bool,
    standardise_sig: bool,
) -> SigWLossDataProps:
    """Build SigW loss properties from either absolute or relative degree config."""
    has_sig_degree = "sig_degree" in config and config["sig_degree"] is not None
    has_relative_sig_degree = "relative_sig_degree" in config and config["relative_sig_degree"] is not None

    assert has_sig_degree != has_relative_sig_degree, (
        "Specify exactly one of sig_degree or relative_sig_degree. "
        f"Got sig_degree={config.get('sig_degree')}, relative_sig_degree={config.get('relative_sig_degree')}."
    )

    use_float64_signature = bool(config.get("use_float64_signature", False))

    if has_relative_sig_degree:
        max_considered_sig_degree = int(config.get("max_considered_sig_degree", DEFAULT_MAX_CONSIDERED_SIG_DEGREE))
        return SigWLossDataProps(
            max_considered_sig_degree,
            scale_high_degrees,
            standardise_sig,
            relative_sig_degree=config["relative_sig_degree"],
            use_float64_signature=use_float64_signature,
        )

    return SigWLossDataProps(
        config["sig_degree"],
        scale_high_degrees,
        standardise_sig,
        use_float64_signature=use_float64_signature,
    )
