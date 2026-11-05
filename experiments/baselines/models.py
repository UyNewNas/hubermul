"""Shared architecture and initialization; only the interaction/training rule differs."""

import torch
from torch import nn

from hubermul import huber_mul

METHODS = ("mul", "hubermul", "clipped_mul", "normalized_mul", "grad_clip", "huber_loss")


class InteractionModel(nn.Module):
    def __init__(self, method, parameter, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 5000)
        self.weight = nn.Parameter(torch.randn(2, 4, generator=g) * 0.3)
        self.bias = nn.Parameter(torch.zeros(()))
        self.method = method
        self.parameter = parameter

    def project(self, x):
        return (x @ self.weight.T).unbind(-1)

    def forward(self, x):
        u, v = self.project(x)
        p = self.parameter
        if self.method == "hubermul":
            interaction = huber_mul(u, v, p)
        elif self.method == "clipped_mul":
            interaction = u.clamp(-p, p) * v.clamp(-p, p)
        elif self.method == "normalized_mul":
            # Identity inside a radius-p RMS disk; bounded radial normalization outside.
            scale = torch.sqrt(((u.square() + v.square()) * 0.5).clamp_min(p * p))
            interaction = (u / scale) * (v / scale) * (p * p)
        else:
            interaction = u * v
        return interaction + self.bias

    @torch.no_grad()
    def constrain(self, bound):
        # Same constraint for EVERY method; no learnable output gain to undo it.
        norms = self.weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.weight.mul_((bound / norms).clamp_max(1.0))
