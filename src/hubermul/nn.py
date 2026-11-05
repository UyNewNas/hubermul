"""Stateless-in-training module wrapper; radius is serialized as extra state."""

from torch import Tensor, nn

from .functional import _validate_radius, huber_mul


class HuberMul(nn.Module):
    """Two-input fixed-radius module with no trainable parameters."""

    def __init__(self, radius: float = 2.0) -> None:
        super().__init__()
        self.radius = _validate_radius(radius)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return huber_mul(x, y, self.radius)

    def extra_repr(self) -> str:
        return f"radius={self.radius}"

    def get_extra_state(self) -> dict[str, float]:
        return {"radius": self.radius}

    def set_extra_state(self, state: dict[str, float]) -> None:
        self.radius = _validate_radius(state["radius"])
