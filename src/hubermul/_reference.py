"""Literal mathematical reference; NOT a numerically reliable extreme-scale oracle."""

import torch
from torch import Tensor

from .functional import _validate_inputs


def huber_mul_reference(x: Tensor, y: Tensor, radius: float = 2.0) -> Tensor:
    """Readable definition for moderate inputs; can suffer catastrophic cancellation."""
    radius = _validate_inputs(x, y, radius)

    def huber(t: Tensor) -> Tensor:
        return torch.where(
            t.abs() <= radius,
            0.5 * t * t,
            radius * (t.abs() - 0.5 * radius),
        )

    return 0.5 * (huber(x + y) - huber(x - y))
