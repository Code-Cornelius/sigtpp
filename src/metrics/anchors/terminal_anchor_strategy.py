"""Strategy pattern for terminal anchor computation.

Each strategy encapsulates:
- apply: how to compute the anchor and write it into paths
- extra_len: how many extra timesteps the anchor adds (default 1; FREE_ENDPOINT overrides to 0)

Strategies: FreeEndpointStrategy ("free_endpoint"), ResidualStrategy ("residual").

scaler_exp is injected at construction for ResidualStrategy.

Gradient note:
    After insert_zero_beg, the path layout is:
        position 0:              zero anchor [0, 0]
        positions 1..seq_lens:   real events (gradient-connected to the generator)
        positions seq_lens+1..L: forward-filled (detached by set_seq_to_cst_val_from_index)

    All gap/anchor computations must index position seq_lens (last real event)
    rather than position -1 (forward-filled, detached) to preserve gradient flow
    from the terminal anchor through the signature loss back to the generator.
"""

import logging
import typing
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)

from src.metrics.anchors.terminal_anchor_mode import TerminalAnchorMode
from src.utils.fix_seq_ends import _replace_from_index_with_value_torch

# When True, _last_real_event reads from position seq_lens[n] (gradient-connected).
# When False, it reads from paths[:, -1:, ...] (forward-filled tail, detached).
_LAST_REAL_EVENT_WITH_GRAD: bool = True


def _last_real_event(
    paths: torch.Tensor,
    seq_lens: torch.Tensor,
    channel_slice: slice,
    with_grad: bool = True,
) -> torch.Tensor:
    """Index the last real event for each sequence.

    Returns shape (N, 1, len(channel_slice)) to match the (N, 1, 1) convention
    used by gap computations.

    with_grad=True  (default): reads from position seq_lens[n], which is still
        gradient-connected to the generator. Use this for training.
    with_grad=False: reads from paths[:, -1:, ...] (the forward-filled tail).
        Those positions are detached by set_seq_to_cst_val_from_index's
        torch.no_grad(), so gradients do NOT flow back. Use only when gradient
        flow is not required (e.g. evaluation / debugging).
    """
    if with_grad:
        N = paths.shape[0]
        # paths[n, seq_lens[n], channel_slice] → (N, channels)
        return paths[torch.arange(N, device=paths.device), seq_lens, channel_slice].unsqueeze(1)
    return paths[:, -1:, channel_slice]


class TerminalAnchorStrategy(ABC):
    """ABC for terminal anchor strategies.

    All strategies encode the observation window boundary T_max into the path by writing
    a terminal anchor row. The anchor row has two channels:

        interarrival axis (ia, ch0):  anchor_tau = scaler_exp(gap)
        times axis        (t,  ch1):  T_max

    where ``gap`` is the residual time from the last cumulative event to the effective
    window end, read from position ``seq_lens[n]`` (the last real event, which preserves
    gradient flow):

        RESIDUAL:
            gap = T_max - paths[n, seq_lens[n], -1]

    The anchor is written at positions strictly after ``seq_lens[n]``
    (the forward-filled padding region), replacing the constant tail.
    """

    @abstractmethod
    def apply(
        self,
        paths: torch.Tensor,
        time_max: float,
        seq_lens: typing.Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the anchor and write it into the path tensor.

        Args:
            paths: (N, L, D) paths with zero anchor already prepended.
            time_max: Maximum observation time T_max.
            seq_lens: Per-sequence lengths *before* insert_zero_beg, shape (N,).
                Position seq_lens[n] is where the anchor is written.

        Returns:
            Tensor of shape (N_valid, L, D) where N_valid <= N. Sequences whose
            cumulative time exceeds T_max are excluded to prevent NaN propagation.
            FreeEndpointStrategy always returns N_valid == N.
        """

    @property
    def extra_len(self) -> int:
        """Number of extra timesteps consumed by this strategy's anchor row.

        The batch tensor is pre-padded to length L = max_seq_len + 1 to reserve
        one slot for the anchor. This property tells callers how many of those slots
        are actually used as meaningful anchor steps (as opposed to pure padding):
            0: FREE_ENDPOINT, no anchor written; the padding slot is unused.
            1: RESIDUAL, one padding slot is repurposed as the anchor row.
        Used by upstream code to compute the effective sequence length for signature
        computation.
        """
        return 1

    def terminal_anchor_extra_len(self) -> int:
        """Number of extra timesteps added by this strategy (for effective length tracking)."""
        return self.extra_len

    def append(
        self,
        paths: torch.Tensor,
        time_max: float,
        seq_lens: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Wrapper around apply, kept for call-site compatibility."""
        return self.apply(paths, time_max, seq_lens)


class FreeEndpointStrategy(TerminalAnchorStrategy):
    """No terminal anchor, path ends at the last event."""

    def apply(self, paths, time_max, seq_lens):
        return paths

    @property
    def extra_len(self) -> int:
        return 0


class ResidualStrategy(TerminalAnchorStrategy):
    """Residual time to T_max placed in-place at the correct last position."""

    def __init__(self, scaler_exp: typing.Callable, last_real_event_with_grad: bool = True):
        self._scaler_exp = scaler_exp
        self._last_real_event_with_grad = last_real_event_with_grad

    def apply(self, paths, time_max, seq_lens):
        assert seq_lens is not None, (
            "ResidualStrategy requires seq_lens (pre-insert_zero_beg lengths) "
            "to know where each sequence ends. Pass seq_lens to scale_paths_pre_sig."
        )
        last_cum = _last_real_event(
            paths, seq_lens, slice(-1, None), with_grad=self._last_real_event_with_grad
        )  # (N, 1, 1)
        gap = time_max - last_cum
        valid = gap[:, 0, 0] >= 0  # (N,)
        n_bad = int((~valid).sum().item())
        if n_bad > 0:
            logger.warning(
                "ResidualStrategy: excluding %d/%d sequences with cumulative time > T_max.",
                n_bad,
                paths.shape[0],
            )
            paths = paths[valid]
            seq_lens = seq_lens[valid]
            gap = gap[valid]
        anchor_tau = self._scaler_exp(gap)
        N, L, D = paths.shape
        anchor = torch.zeros(N, 1, D, device=paths.device, dtype=paths.dtype)
        anchor[:, :, :1] = anchor_tau
        anchor[:, 0, -1] = time_max
        return _replace_from_index_with_value_torch(paths, seq_lens, anchor)


def make_anchor_strategy(
    mode: TerminalAnchorMode,
    scaler_exp: typing.Optional[typing.Callable] = None,
    last_real_event_with_grad: bool = True,
) -> TerminalAnchorStrategy:
    """Factory: create a strategy from an enum mode.

    Args:
        mode: The terminal anchor mode.
        scaler_exp: Required for RESIDUAL; ignored by FREE_ENDPOINT.
        last_real_event_with_grad: When True (default), anchor reads the last real event with
            gradient connection to the generator. When False, reads the detached forward-filled tail.
    """
    if mode is TerminalAnchorMode.FREE_ENDPOINT:
        return FreeEndpointStrategy()

    if scaler_exp is None:
        raise ValueError(f"{mode} requires scaler_exp at construction")

    if mode is TerminalAnchorMode.RESIDUAL:
        return ResidualStrategy(scaler_exp, last_real_event_with_grad=last_real_event_with_grad)
    raise ValueError(f"Unknown terminal anchor mode: {mode}")
