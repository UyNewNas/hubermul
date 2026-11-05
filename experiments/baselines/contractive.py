"""Fixed task controls, not additions to the installed HuberMul API."""
import math

import torch
from torch import Tensor

from hubermul import huber_mul
from hubermul.functional import _validate_inputs

METHODS = ('mul', 'hubermul', 'projection')


def project_diamond(z: Tensor, radius: float) -> Tensor:
    """Euclidean projection onto {|x|+|y|<=radius}; reference for moderate inputs.

    This is soft-thresholding with a shared threshold, NOT radial normalization.
    No differentiability is claimed on the polytope's active-set boundaries.
    """
    if z.shape[-1] != 2:
        raise ValueError('last dimension must be 2')
    x, y = z.unbind(-1)
    radius = _validate_inputs(x, y, radius)
    a = z.abs()
    inside = a.sum(-1) <= radius
    high = a.max(-1).values
    low = a.min(-1).values
    theta = torch.where(high - low >= radius, high - radius, (high + low - radius)*.5)
    p = z.sign() * (a - theta.unsqueeze(-1)).clamp_min(0.)
    return torch.where(inside.unsqueeze(-1), z, p)


def projection_mul(x: Tensor, y: Tensor, radius: float = 2.) -> Tensor:
    """Factored product of the two coordinates of the Euclidean projection.

    Inside: x*y. Outside with both projected coordinates active:
    sign(x*y)*(R-d)*(R+d)/4, d=abs(abs(x)-abs(y)). Otherwise zero.
    Shared input and nonfinite checks make the core timing comparison fairer.
    """
    radius = _validate_inputs(x, y, radius)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = torch.where(finite,x,0.), torch.where(finite,y,0.)
    ax, ay = x.abs(), y.abs()
    a, b = torch.maximum(ax,ay), torch.minimum(ax,ay)
    inside = (a <= radius) & (b <= radius-a)
    core = torch.where(inside,x,0.) * torch.where(inside,y,0.)
    # d is capped before multiplication. In this experiment R and inputs are moderate.
    d = (a-b).clamp(max=radius)
    edge = ((radius-d)*.5) * (radius*.5+d*.5)
    edge = torch.where(inside,0.,edge) * x.sign() * y.sign()
    out = core + edge
    return torch.where(finite,out,torch.full_like(out,float('nan')))


def interaction(z: Tensor, method: str, radius: float) -> Tensor:
    x,y = z.unbind(-1)
    if method == 'mul':
        return x*y
    if method == 'hubermul':
        return huber_mul(x,y,radius)
    if method == 'projection':
        return projection_mul(x,y,radius)
    raise ValueError(f'unknown method: {method}')


def residual_map(z: Tensor, method: str, radius: float, q: float, direction: Tensor) -> Tensor:
    """F(z)=z+(q/R)*v*op(z); v must be unit length for the claimed bound."""
    return z + ((q/radius)*interaction(z,method,radius)).unsqueeze(-1)*direction


def _validate_solver(target, initial, method, radius, q, direction, error_tol, max_iterations, divergence):
    if target.ndim != 2 or target.shape[-1] != 2 or target.shape != initial.shape:
        raise ValueError('target and initial must have identical (N,2) shapes')
    _validate_inputs(target, initial, radius)
    if target.dtype != torch.float64 or target.device.type != 'cpu':
        raise ValueError('this experiment is CPU float64 only')
    if not torch.isfinite(target).all() or not torch.isfinite(initial).all():
        raise ValueError('target and initial must be finite')
    if method not in METHODS:
        raise ValueError('unknown method')
    if not math.isfinite(q) or not 0 < q < 1:
        raise ValueError('q must be strictly between 0 and 1')
    if direction.shape != (2,) or direction.dtype != target.dtype or direction.device != target.device:
        raise ValueError('direction must be a same-dtype/device two-vector')
    if not torch.isfinite(direction).all() or abs(torch.linalg.vector_norm(direction).item()-1.) > 1e-12:
        raise ValueError('direction must have unit Euclidean norm')
    if not math.isfinite(error_tol) or error_tol <= 0:
        raise ValueError('error_tol must be positive and finite')
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError('max_iterations must be a nonnegative integer')
    if not math.isfinite(divergence) or divergence <= 0:
        raise ValueError('divergence must be positive and finite')


@torch.no_grad()
def inverse_fixed_point(target: Tensor, initial: Tensor, method: str, radius: float,
                        q: float, direction: Tensor, error_tol: float = 1e-10,
                        max_iterations: int = 2000, divergence: float = 1e100) -> dict:
    """Paired Picard iteration z_next=target-g(z), retaining every failure.

    Same residual threshold (1-q)*error_tol*R for ALL methods. Only HuberMul and
    projection have the real-arithmetic bound error <= residual/(1-q). The caller
    separately checks recovery of a known legal preimage; residual convergence of
    ordinary multiplication can instead be convergence to a wrong root.
    """
    _validate_solver(target,initial,method,radius,q,direction,error_tol,max_iterations,divergence)
    n = len(target)
    z = initial.clone()
    code = torch.zeros(n,dtype=torch.int64)
    iterations = torch.zeros(n,dtype=torch.int64)
    residual = torch.full((n,),float('nan'),dtype=target.dtype)
    excursion = initial.abs().sum(-1)/radius
    threshold = (1-q)*error_tol*radius
    for step in range(max_iterations+1):
        idx = (code == 0).nonzero().flatten()
        if len(idx) == 0:
            break
        current = z[idx]
        iterations[idx] = step
        finite = torch.isfinite(current).all(-1)
        code[idx[~finite]] = 4
        large = finite & (current.abs().max(-1).values > divergence*radius)
        code[idx[large]] = 2
        keep = finite & ~large
        idx, current = idx[keep], current[keep]
        if len(idx) == 0:
            continue
        excursion[idx] = torch.maximum(excursion[idx],current.abs().sum(-1)/radius)
        g = ((q/radius)*interaction(current,method,radius)).unsqueeze(-1)*direction
        r = current+g-target[idx]
        rn = torch.hypot(r[:,0],r[:,1])
        residual[idx] = rn
        finite_r = torch.isfinite(rn) & torch.isfinite(g).all(-1)
        code[idx[~finite_r]] = 4
        converged = finite_r & (rn <= threshold)
        code[idx[converged]] = 1
        remaining = finite_r & ~converged
        if step == max_iterations:
            code[idx[remaining]] = 3
        else:
            z[idx[remaining]] = target[idx[remaining]] - g[remaining]
    labels = {1:'converged',2:'diverged',3:'max_iterations',4:'nonfinite'}
    return {'z':z, 'status':[labels[int(c)] for c in code], 'iterations':iterations,
            'residual_norm':residual, 'max_excursion':excursion,
            'certified_method':torch.full((n,),method != 'mul',dtype=torch.bool)}
