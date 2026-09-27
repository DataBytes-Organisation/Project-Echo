import random

import torch
import torch.nn as nn
import torch.nn.functional as F_nn


class TimeWarp(nn.Module):
    """
    Time warping - the original SpecAugment paper's "W" operation
    (https://arxiv.org/abs/1904.08779), as a standalone alternative to
    augment.py's time/frequency masking (that paper's other two
    operations, "F" and "T"). Deliberately never modifies augment.py -
    a separate, additive module, so masking-based presets and this one
    stay independently comparable through the same benchmark harness.

    Lives here (augmentation/), not reproducible_training_pipeline/,
    since it's an experimental comparison arm, not a change to Nolan's
    pipeline files - importable during an actual training run via
    PYTHONPATH, not by copying it into that folder.

    Picks one anchor point along the time axis and stretches the segment
    on one side of it while compressing the segment on the other side by
    the same number of frames, via 1D linear interpolation - total clip
    length is unchanged, and no content is deleted (unlike masking, which
    zeroes a strip outright). That is the whole point of comparing it
    against masking: it should perturb the embedding less at
    initialisation while still forcing timing/tempo invariance, which
    matters given the benchmark's own finding that heavier masking delays
    CircleLoss's embedding-separation phase transition.
    """

    def __init__(self, p=0.5, max_warp=20):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"TimeWarp: p must be in [0.0, 1.0], got {p}.")
        if max_warp < 1:
            raise ValueError(f"TimeWarp: max_warp must be >= 1, got {max_warp}.")
        self.p = p
        self.max_warp = max_warp

    def forward(self, x):
        if random.random() > self.p:
            return x

        C, F, T = x.shape

        # Need room on both sides of the anchor for a warp of up to
        # max_warp frames in either direction - if the clip is too short
        # for that, skip (same "no-op when it doesn't apply" convention
        # augment.py's masking uses).
        w = min(self.max_warp, T // 4)
        if w < 1:
            return x

        anchor = random.randint(w, T - w)
        shift = random.randint(-w, w)
        # Clamp away from the exact edges (0 or T) - either would mean
        # interpolating one side down to zero-length, which isn't a
        # valid resize.
        warped_anchor = max(1, min(T - 1, anchor + shift))

        left, right = x[..., :anchor], x[..., anchor:]
        # (C, F, L) -> (1, C*F, L) so interpolate's 1D linear mode
        # resamples purely along time, treating every (channel, freq bin)
        # pair as an independent 1D signal.
        left_r = F_nn.interpolate(
            left.reshape(1, C * F, anchor), size=warped_anchor, mode="linear", align_corners=False
        ).reshape(C, F, warped_anchor)
        right_r = F_nn.interpolate(
            right.reshape(1, C * F, T - anchor), size=T - warped_anchor, mode="linear", align_corners=False
        ).reshape(C, F, T - warped_anchor)

        return torch.cat([left_r, right_r], dim=-1)
