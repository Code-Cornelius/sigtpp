from enum import Enum


# List of anchors:
# - "free_endpoint"
# - "residual"
class TerminalAnchorMode(str, Enum):
    """Strategies for encoding the observation window boundary T_max into signature paths.

    All examples below use the same sequence:
        Events at absolute times: 2, 5, 9
          τ₁ = 2  (time from window start to first event, IGNORED in path construction)
          τ₂ = 3,  τ₃ = 4  (inter-event gaps, these form the path)
        T_max = 12,  D = 2  (interarrival axis = log-scaled inter-arrival gap,  times axis = cumulative time)

    NOTE: interarrival axis is always in log-space (output of scaler_exp), times axis is always in raw time units.
    In table headers below: ``ia`` = interarrival axis, ``t`` = times axis.

    After ``insert_zero_beg`` and batch-padding to L=4, seq_len = 2:

        idx | ia        | t (cum)
          0 |   0       |   0    ← zero anchor prepended
          1 |  log(3)   |   3    ← τ₂
          2 |  log(4)   |   7    ← τ₃  (NOTE: true last event = 9, but cum = 7 because τ₁ is missing)
          3 |  log(4)   |   7    ← constant (interarrival axis = log(τ₃) forward-filled, times axis forward-filled)

    Each mode describes what happens **after** this point, specifically how the constant
    padding row is handled.
    """

    # ------------------------------------------------------------------ #
    # FREE_ENDPOINT                                                         #
    # ------------------------------------------------------------------ #
    # Path ends at the last event, no reference to Tmax encoded.
    # performance: works
    #
    # Example: path unchanged; constant stays at end:
    #   idx | ia       | t
    #     0 |  0       |   0
    #     1 |  log(3)  |   3
    #     2 |  log(4)  |   7   ← last event; cum = 7 (τ₁ ignored), T_max not encoded
    #     3 |  log(4)  |   7   ← constant (unchanged)
    FREE_ENDPOINT = "free_endpoint"

    # ------------------------------------------------------------------ #
    # RESIDUAL                                                              #
    # ------------------------------------------------------------------ #
    # Residual time to Tmax placed in-place at the correct last position.
    # gap = T_max - last_cum; anchor_tau = log(gap); written at index seq_len.
    # performance: works the best.
    #
    # Example: gap = T_max - paths[:,-1,-1] = 12 - 7 = 5 (biased: τ₁=2 missing from cum),
    #           anchor_tau = log(5); written in-place at seq_len slot (slot 3):
    #   idx | ia       | t
    #     0 |  0       |   0
    #     1 |  log(3)  |   3
    #     2 |  log(4)  |   7
    #     3 |  log(5)  |  12   ← constant REPLACED; ia=log(gap)=log(5), t=T_max
    RESIDUAL = "residual"
