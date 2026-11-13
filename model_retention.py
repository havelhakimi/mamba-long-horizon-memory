"""Retention-gated variant of the original Mamba-1 implementation.

The learned gate modifies state decay while leaving the input-write
term unchanged.

Ablation modes:
    MAMBA_DISABLE_RETENTION=1  -> gate fixed at 1
    MAMBA_FIXED_RETENTION=0.5  -> gate fixed at 0.5
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Mamba as OriginalMamba
from model import MambaBlock as OriginalMambaBlock
from model import ModelArgs
from einops import einsum


class RetentionMambaBlock(OriginalMambaBlock):
    def __init__(self, args: ModelArgs):
        super().__init__(args)

        self.retention_proj = nn.Linear(args.d_inner, 1, bias=True)

        # Initialize close to the original recurrence.
        nn.init.zeros_(self.retention_proj.weight)
        nn.init.constant_(self.retention_proj.bias, 3.0)

        self.disable_retention = (
            os.getenv("MAMBA_DISABLE_RETENTION", "0") == "1"
        )

        fixed = os.getenv("MAMBA_FIXED_RETENTION")
        self.fixed_retention = float(fixed) if fixed is not None else None

        if self.fixed_retention is not None and not (
            0.0 < self.fixed_retention <= 1.0
        ):
            raise ValueError(
                "MAMBA_FIXED_RETENTION must be in (0, 1]"
            )

        if self.disable_retention and self.fixed_retention is not None:
            raise ValueError(
                "Do not set DISABLE_RETENTION and FIXED_RETENTION together"
            )

    def selective_scan(self, u, delta, A, B, C, D):
        batch, length, inner = u.shape
        state_dim = A.shape[1]

        if self.disable_retention:
            r = torch.ones_like(delta[..., :1])
        elif self.fixed_retention is not None:
            r = torch.full_like(delta[..., :1], self.fixed_retention)
        else:
            r = torch.sigmoid(self.retention_proj(u))

        # Only modification: gate the state-decay term.
        deltaA = torch.exp(
            einsum(delta * r, A, 'b l d_in, d_in n -> b l d_in n')
        )

        # Original Mamba input-write computation.
        deltaB_u = einsum(
            delta, B, u,
            'b l d_in, b l n, b l d_in -> b l d_in n'
        )

        state = torch.zeros((batch, inner, state_dim), device=deltaA.device)
        ys = []

        for i in range(length):
            state = deltaA[:, i] * state + deltaB_u[:, i]
            y_i = einsum(state, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y_i)

        y = torch.stack(ys, dim=1)
        return y + u * D

class Mamba(OriginalMamba):
    def __init__(self, args: ModelArgs):
        super().__init__(args)

        # Preserve original weights and main RNG sequence.
        with torch.random.fork_rng(devices=[]):
            for layer in self.layers:
                old_mixer = layer.mixer

                new_mixer = RetentionMambaBlock(args)

                new_mixer.load_state_dict(
                    old_mixer.state_dict(),
                    strict=False
                )

                layer.mixer = new_mixer
