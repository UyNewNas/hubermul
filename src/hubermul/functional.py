"""Fixed-radius Huber-polarized multiplication using ordinary torch autograd."""

import math
from numbers import Real

import torch
from torch import Tensor


def _validate_radius(radius: float) -> float:
    if isinstance(radius, bool) or not isinstance(radius, Real):
        raise TypeError("radius must be a Python real scalar, not a Tensor or bool")
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and strictly positive")
    return radius


def _validate_inputs(x: Tensor, y: Tensor, radius: float) -> float:
    radius = _validate_radius(radius)
    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise TypeError("x and y must be torch Tensors")
    if x.layout != torch.strided or y.layout != torch.strided:
        raise TypeError("only strided tensors are supported")
    if x.dtype not in (torch.float32, torch.float64) or x.dtype != y.dtype:
        raise TypeError("x and y must have the same dtype: float32 or float64")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    limits = torch.finfo(x.dtype)
    if radius < limits.tiny or radius > limits.max:
        raise ValueError("radius must be a normal, finite value in the input dtype")
    return radius


def huber_mul(x: Tensor, y: Tensor, radius: float = 2.0) -> Tensor:
    """Compute (H_R(x+y)-H_R(x-y))/2 without subtracting large Huber values.

    Real arithmetic: equals x*y when |x|+|y| <= R; the scalar input-gradient
    Euclidean norm is at most R. These are not whole-network stability claims.

    Supports same-device/dtype float32/float64 strided tensors with broadcasting.
    Radius is a fixed positive Python scalar (not trainable), normal in dtype.
    Nonfinite input pairs return NaN. Gradients on those pairs have no mathematical
    interpretation. Finite lanes are isolated before arithmetic. Results outside
    the dtype range can overflow; the contract is not arbitrary-precision math.
    """
    radius = _validate_inputs(x, y, radius)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = torch.where(finite, x, 0.0)
    y = torch.where(finite, y, 0.0)

    swap = x.abs() < y.abs()
    u = torch.where(swap, y, x)
    v = torch.where(swap, x, y)
    a, b = u.abs(), v.abs()
    gap = a - b
    # Avoid a+b: it may overflow even when the final result is representable.
    center = (a <= radius) & (b <= radius - a)
    # Near a=R, forming a-b first can erase b and choose the wrong derivative.
    # Far from R, a-b is preferable when a and b nearly cancel.
    near_radius = a * 0.5 <= radius
    outer_test = torch.where(near_radius, (a >= radius) & (b <= a - radius), gap >= radius)
    outer = (~center) & outer_test
    middle = ~(center | outer)

    # Mask operands BEFORE arithmetic. torch.where is not lazy branch execution.
    xc = torch.where(center, x, 0.0)
    yc = torch.where(center, y, 0.0)
    central_value = xc * yc

    # Keep v signed: sign(v)*abs(v) would lose the axis derivative at v=0.
    vo = torch.where(outer, v, 0.0)
    outer_value = (u.sign() * radius) * vo

    t_near = torch.where(a <= radius, (radius - a) + b, b - (a - radius))
    t = torch.where(middle, torch.where(near_radius, t_near, radius - gap), 0.0)
    vm = torch.where(middle, v, 0.0)
    # Let h=t/2. R*b - h**2 = (R-h)*b + h*(b-h).
    # Both magnitude terms are nonnegative: no large subtraction, no division,
    # and neither term can overflow when their sum is representable.
    h = 0.5 * t
    middle_value = u.sign() * ((radius - h) * vm + h * (vm - v.sign() * h))

    value = central_value + outer_value + middle_value
    return torch.where(finite, value, torch.full_like(value, float("nan")))
