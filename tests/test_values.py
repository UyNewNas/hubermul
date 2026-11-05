import pytest
import torch

from hubermul import huber_mul
from hubermul._reference import huber_mul_reference
from tests._oracle import oracle


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("radius", [0.125, 2.0, 7.5])
def test_exact_domain(dtype, radius):
    g = torch.Generator().manual_seed(2718)
    x = (torch.rand(257, generator=g, dtype=dtype) - 0.5) * radius
    y = (torch.rand(257, generator=g, dtype=dtype) - 0.5) * radius
    torch.testing.assert_close(huber_mul(x, y, radius), x * y, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_high_precision_grid(dtype):
    values = [-10.0, -2.5, -2.0, -1.0, -0.125, 0.0, 0.125, 1.0, 2.0, 2.5, 10.0]
    x = torch.tensor(values, dtype=dtype)[:, None]
    y = torch.tensor(values, dtype=dtype)[None, :]
    expected = torch.tensor([[oracle(a, b, 2.0) for b in values] for a in values], dtype=dtype)
    torch.testing.assert_close(huber_mul(x, y), expected)
    torch.testing.assert_close(huber_mul_reference(x, y), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("x,y,radius", [(1e20, 1e-4, 2.0), (-1e20, 1e-4, 2.0),
                                       (1e20, -1e-4, 2.0), (1e20, 1e20, 2.0)])
def test_extreme_scale_cancellation(dtype, x, y, radius):
    a, b = torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype)
    expected = torch.tensor(oracle(a.item(), b.item(), radius), dtype=dtype)
    torch.testing.assert_close(huber_mul(a, b, radius), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_symmetry_sign_and_scale(dtype):
    g = torch.Generator().manual_seed(12)
    x, y = torch.randn(2, 256, generator=g, dtype=dtype) * 3
    z = huber_mul(x, y)
    torch.testing.assert_close(z, huber_mul(y, x))
    torch.testing.assert_close(-z, huber_mul(-x, y))
    torch.testing.assert_close(z, huber_mul(-x, -y))
    torch.testing.assert_close(z, huber_mul(2*x, 2*y, 4.0) / 4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_max_finite_does_not_poison_unselected_branches(dtype):
    maxval = torch.finfo(dtype).max
    x = torch.tensor([maxval, 0.0, 1.0], dtype=dtype, requires_grad=True)
    y = torch.tensor([0.125, maxval, 0.5], dtype=dtype, requires_grad=True)
    z = huber_mul(x, y)
    torch.testing.assert_close(z, torch.tensor([0.25, 0.0, 0.5], dtype=dtype))
    dx, dy = torch.autograd.grad(z.sum(), (x, y))
    assert torch.isfinite(dx).all() and torch.isfinite(dy).all()
    torch.testing.assert_close(dx, torch.tensor([0., 2., 0.5], dtype=dtype))
    torch.testing.assert_close(dy, torch.tensor([2., 0., 1.], dtype=dtype))
