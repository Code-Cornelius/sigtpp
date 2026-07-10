import logging
import typing
from abc import abstractmethod

import torch

logger = logging.getLogger(__name__)

from test.paper_experiments.data.tpp_data_module import TPPDataModule


class RealWorldDataModule(TPPDataModule):
    """
    Base class for all real-world TPP datasets.

    Adds jitter_zero_interarrival_times() for handling zero inter-arrivals
    that arise from simultaneous events or float32 precision artifacts.
    Subclasses must implement _load_data() and declare class-level stubs
    for the 11 abstract properties defined in TPPDataModule.
    """

    @abstractmethod
    def _load_data(self) -> None:
        """Load and prepare the dataset. Must set all required attributes."""
        pass

    @staticmethod
    def jitter_zero_interarrival_times(
        inputs: torch.Tensor,
        inputs_len: torch.Tensor,
        inputs_marks: typing.Optional[torch.Tensor] = None,
        jitter_scale: float = 0.5,
        jitter_min: float = 1e-6,
        seed: typing.Optional[int] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Replace zero inter-arrival times with random jitter, optionally
        removing same-mark duplicates instead of jittering them.

        Zero inter-arrivals arise from two sources:
        - True simultaneous events (integer-timestamp datasets where multiple
          events fall within the same second).
        - float32 precision artifacts (e.g. stackoverflow, earthquake: two events
          so close in time that they round to the same float32 value).

        Mark-aware logic per consecutive pair (t_k, t_{k+1}):
        - If inter >= jitter_min: keep as-is.
        - If marks available AND mark[k] == mark[k+1]: remove event k+1
          (duplicate same-mark event at same time).
        - Otherwise: replace inter with Uniform(jitter_min, jitter_scale * resolution)
          where resolution = min positive inter-arrival (or 1.0 if none).

        Args:
            inputs: (N, L+1, 1) cumulative times, constant-padded.
            inputs_len: (N,) sequence lengths (includes anchor at position 0).
            inputs_marks: (N, L+1) long event types, or None.
            jitter_scale: max jitter as fraction of resolution (default 0.5).
            jitter_min: absolute lower bound on the jitter (default 1e-6).
            seed: optional RNG seed for reproducibility.

        Returns:
            (inputs_fixed, inputs_len_fixed): tuple of fixed tensors. Sequence
            lengths may decrease if same-mark duplicates were removed.
        """
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        else:
            gen = None

        inputs = inputs.clone()
        inputs_len = inputs_len.clone()
        N, padded_length, _ = inputs.shape

        # --- Vectorized jitter path (no marks or different-mark zeros) ---
        # Build inter-arrival times: (N, L) where L = padded_length - 1
        cum = inputs[:, :, 0]  # (N, padded_length)
        inter = cum[:, 1:] - cum[:, :-1]  # (N, L)

        if inter.shape[1] == 0:
            return inputs, inputs_len

        # Valid position mask: position j is valid if j < seq_len - 1
        # (inter[i, j] = cum[i, j+1] - cum[i, j], both must be valid)
        pos_idx = torch.arange(inter.shape[1], device=inputs.device).unsqueeze(0)  # (1, L)
        valid_mask = pos_idx < (inputs_len.unsqueeze(1) - 1)  # (N, L)

        zero_mask = (inter < jitter_min) & valid_mask  # (N, L)

        # Compute per-sequence resolution: smallest positive inter-arrival
        positive_inter = inter.clone()
        positive_inter[~valid_mask | (inter < jitter_min)] = float('inf')
        resolution = positive_inter.min(dim=1).values  # (N,)
        resolution[resolution == float('inf')] = 1.0  # fallback if all zeros

        # --- Mark-aware: split zero_mask into remove_mask and jitter_mask ---
        if inputs_marks is not None:
            # marks shape: (N, L+1)
            marks = inputs_marks
            if marks.dim() == 3:
                marks = marks.squeeze(-1)
            # same_mark[i, j] = True if mark at position j+1 equals mark at position j
            same_mark = marks[:, 1:] == marks[:, :-1]  # (N, L)
            remove_mask = zero_mask & same_mark
            jitter_mask = zero_mask & ~same_mark
        else:
            remove_mask = torch.zeros_like(zero_mask)
            jitter_mask = zero_mask

        # --- Apply jitter to jitter_mask positions ---
        if jitter_mask.any():
            jitter_upper = jitter_scale * resolution  # (N,)
            jitter_upper = jitter_upper.unsqueeze(1).expand_as(inter)  # (N, L)
            # Uniform(jitter_min, jitter_scale * resolution) per position
            noise = torch.rand(inter.shape, generator=gen, device=inputs.device, dtype=inputs.dtype)
            jitter_vals = jitter_min + noise * (jitter_upper - jitter_min)
            jitter_vals = jitter_vals.clamp(min=jitter_min)

            inter = inter.clone()
            inter[jitter_mask] = jitter_vals[jitter_mask]

            # Recompute cumulative times from fixed inter-arrivals
            cum_fixed = torch.cat([cum[:, :1], cum[:, :1] + inter.cumsum(dim=1)], dim=1)  # (N, padded_length)
            # Only apply fix where needed: any sequence with at least one jittered position
            seq_has_jitter = jitter_mask.any(dim=1)  # (N,)
            for i in torch.where(seq_has_jitter)[0]:
                sl = int(inputs_len[i].item())
                inputs[i, :sl, 0] = cum_fixed[i, :sl]
                if sl < padded_length:
                    inputs[i, sl:, 0] = cum_fixed[i, sl - 1]

        # --- Remove same-mark duplicates (requires per-sequence processing) ---
        if remove_mask.any():
            seqs_with_removals = torch.where(remove_mask.any(dim=1))[0]
            for i in seqs_with_removals:
                seq_len = int(inputs_len[i].item())
                if seq_len <= 1:
                    continue

                # Iteratively remove duplicates until none remain
                while True:
                    cum_i = inputs[i, :seq_len, 0]
                    inter_i = cum_i[1:] - cum_i[:-1]
                    marks_i = inputs_marks[i, :seq_len]

                    zero_pos = inter_i < jitter_min
                    same_mark_pos = marks_i[1:] == marks_i[:-1]
                    dup_mask = zero_pos & same_mark_pos

                    if not dup_mask.any():
                        break

                    # Keep positions where dup_mask is False (remove the k+1 event)
                    # dup_mask[j] means event at position j+1 should be removed
                    keep = torch.ones(seq_len, dtype=torch.bool, device=inputs.device)
                    # Find first duplicate to remove per pass (to handle chains)
                    first_dup = torch.where(dup_mask)[0][0].item()
                    keep[first_dup + 1] = False

                    new_len = int(keep.sum().item())
                    inputs[i, :new_len, 0] = inputs[i, :seq_len, 0][keep]
                    inputs_marks[i, :new_len] = inputs_marks[i, :seq_len][keep]
                    if new_len < padded_length:
                        inputs[i, new_len:, 0] = inputs[i, new_len - 1, 0]
                        inputs_marks[i, new_len:] = 0
                    inputs_len[i] = new_len
                    seq_len = new_len

        return inputs, inputs_len
