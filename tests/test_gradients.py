import pytest
import torch

from hubermul import huber_mul


def test_gradcheck_all_regions():
    x = torch.tensor([0.2, 3., -3., 1.6, -1.6, 2.7], dtype=torch.double, requires_grad=True)
    y = torch.tensor([0.3, 0.2, -0.2, 1., 1., -1.1], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(huber_mul, (x, y))
    assert torch.autograd.gradgradcheck(huber_mul, (x, y))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_axes(dtype):
    x = torch.zeros(7, dtype=dtype, requires_grad=True)
    y = torch.tensor([-10., -2., -0.5, 0., 0.5, 2., 10.], dtype=dtype, requires_grad=True)
    z = huber_mul(x, y)
    dx, dy = torch.autograd.grad(z.sum(), (x, y))
    torch.testing.assert_close(z, torch.zeros_like(z))
    torch.testing.assert_close(dx, y.clamp(-2, 2))
    torch.testing.assert_close(dy, torch.zeros_like(y))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytic_gradient_including_seams(dtype):
    x = torch.tensor([1., -1., 1.5, 3., 2., 0., -2., 3.], dtype=dtype, requires_grad=True)
    y = torch.tensor([1., 1., 1.5, 1., 0., 2., 0., -3.], dtype=dtype, requires_grad=True)
    dx, dy = torch.autograd.grad(huber_mul(x, y).sum(), (x, y))
    p, m = (x+y).clamp(-2, 2), (x-y).clamp(-2, 2)
    torch.testing.assert_close(dx, (p-m)/2)
    torch.testing.assert_close(dy, (p+m)/2)


def test_gradient_bound_and_broadcast_sum():
    g = torch.Generator().manual_seed(314)
    x = (torch.randn(1000, generator=g, dtype=torch.double)*5).requires_grad_()
    y = (torch.randn(1000, generator=g, dtype=torch.double)*5).requires_grad_()
    dx, dy = torch.autograd.grad(huber_mul(x, y).sum(), (x, y))
    assert (dx.square()+dy.square()).max().item() <= 4+1e-12
    scalar = torch.tensor(0., dtype=torch.double, requires_grad=True)
    values = torch.tensor([-3., 1., 4.], dtype=torch.double)
    huber_mul(scalar, values).sum().backward()
    torch.testing.assert_close(scalar.grad, values.clamp(-2, 2).sum())


def test_gradgrad_at_origin():
    x = torch.tensor(0., dtype=torch.double, requires_grad=True)
    y = torch.tensor(0., dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradgradcheck(huber_mul, (x, y))


@pytest.mark.parametrize("radius", [1e-300, 1e-200, 1e-150, 1.5e154])
def test_tiny_or_large_radius_keeps_representable_gradient(radius):
    x = torch.tensor(radius, dtype=torch.double, requires_grad=True)
    y = torch.tensor(radius, dtype=torch.double, requires_grad=True)
    z = huber_mul(x, y, radius)
    dx, dy = torch.autograd.grad(z, (x, y))
    assert torch.isfinite(z)
    # Value may underflow, but the exact derivatives R/2 are representable.
    torch.testing.assert_close(dx / radius, torch.tensor(0.5, dtype=torch.double))
    torch.testing.assert_close(dy / radius, torch.tensor(0.5, dtype=torch.double))


@pytest.mark.parametrize("dtype,radius", [(torch.float32, 1e10), (torch.float64, 1e20)])
def test_near_radius_branch_selection_does_not_round_away_small_input(dtype, radius):
    x = torch.tensor(radius, dtype=dtype, requires_grad=True)
    y = torch.tensor(1e-4, dtype=dtype, requires_grad=True)
    dx, dy = torch.autograd.grad(huber_mul(x, y, radius), (x, y))
    torch.testing.assert_close(dx, y / 2, rtol=1e-5, atol=0)
    torch.testing.assert_close(dy, x - y / 2)
