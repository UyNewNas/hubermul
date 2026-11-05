import io

import pytest
import torch

from hubermul import HuberMul, huber_mul


@pytest.mark.parametrize("radius", [0., -1., float("nan"), float("inf"), -float("inf")])
def test_bad_radius_value(radius):
    with pytest.raises(ValueError):
        huber_mul(torch.tensor(1.), torch.tensor(1.), radius)
    with pytest.raises(ValueError):
        HuberMul(radius)


@pytest.mark.parametrize("radius", [True, [2.], torch.tensor(2.), "2"])
def test_bad_radius_type(radius):
    with pytest.raises(TypeError):
        huber_mul(torch.tensor(1.), torch.tensor(1.), radius)


@pytest.mark.parametrize("radius", [1e-300, 1e300])
def test_radius_not_normal_representable_in_dtype(radius):
    with pytest.raises(ValueError):
        huber_mul(torch.tensor(1., dtype=torch.float32), torch.tensor(1.), radius)


def test_reject_unsupported_types_and_devices():
    with pytest.raises(TypeError):
        huber_mul(1., torch.tensor(1.))
    for dtype in [torch.float16, torch.bfloat16, torch.int64, torch.complex64]:
        with pytest.raises(TypeError):
            huber_mul(torch.ones(2, dtype=dtype), torch.ones(2, dtype=dtype))
    with pytest.raises(TypeError):
        huber_mul(torch.ones(2), torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        huber_mul(torch.ones(2), torch.ones(2, device="meta"))
    with pytest.raises(TypeError):
        huber_mul(torch.eye(2).to_sparse(), torch.eye(2))


def test_scalar_empty_broadcast_noncontiguous():
    assert huber_mul(torch.tensor(1.), torch.tensor(1.)).shape == ()
    assert huber_mul(torch.empty(0, 3), torch.ones(1, 3)).shape == (0, 3)
    x = torch.arange(6., dtype=torch.double).reshape(2, 3).T
    assert not x.is_contiguous()
    z = huber_mul(x, torch.tensor(0.25, dtype=torch.double))
    assert z.shape == (3, 2)
    torch.testing.assert_close(z, huber_mul(x.contiguous(), torch.full_like(x, 0.25)))
    with pytest.raises(RuntimeError):
        huber_mul(torch.ones(2), torch.ones(3))


def test_nonfinite_sentinel_does_not_poison_valid_lanes():
    x = torch.tensor([float("inf"), float("nan"), 1., -float("inf")], requires_grad=True)
    y = torch.tensor([0., 1., 0.5, 1.], requires_grad=True)
    z = huber_mul(x, y)
    assert torch.isnan(z[[0, 1, 3]]).all()
    assert z[2].item() == 0.5
    dx, dy = torch.autograd.grad(z[2], (x, y))
    assert torch.isfinite(dx).all() and torch.isfinite(dy).all()
    assert dx[2].item() == 0.5 and dy[2].item() == 1.


def test_module_state_and_no_parameters():
    op = HuberMul(3.)
    assert list(op.parameters()) == []
    data = io.BytesIO()
    torch.save(op.state_dict(), data)
    data.seek(0)
    restored = HuberMul(2.)
    restored.load_state_dict(torch.load(data, weights_only=True))
    assert restored.radius == 3.
    assert "3" in repr(restored)
    x = torch.tensor([4.], dtype=torch.double)
    torch.testing.assert_close(restored.to(dtype=torch.double)(x, x), huber_mul(x, x, 3.))


def test_compile_fullgraph_forward_backward():
    compiled = torch.compile(huber_mul, backend="aot_eager", fullgraph=True)
    x = torch.tensor([[0.], [1.], [4.]], dtype=torch.double, requires_grad=True)
    y = torch.tensor([1., 2., -3.], dtype=torch.double, requires_grad=True)
    eager = huber_mul(x, y)
    out = compiled(x, y)
    torch.testing.assert_close(out, eager)
    dg = torch.autograd.grad(out.sum(), (x, y))
    eg = torch.autograd.grad(eager.sum(), (x, y))
    for a, b in zip(dg, eg, strict=True):
        torch.testing.assert_close(a, b)
