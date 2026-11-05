"""Task-matching validation, independent of the original regression experiment."""
import math

import pytest
import torch

from experiments.baselines.contractive import (
    METHODS, inverse_fixed_point, project_diamond, projection_mul, residual_map,
)
from hubermul import huber_mul


def test_euclidean_projection_values_and_kkt():
    z = torch.tensor([[2., 2.], [3., .5], [-2., 2.], [.2, -.3], [0., 2.]], dtype=torch.double)
    p = project_diamond(z, 1.)
    expected = torch.tensor([[.5,.5], [1.,0.], [-.5,.5], [.2,-.3], [0.,1.]], dtype=torch.double)
    torch.testing.assert_close(p, expected)
    g = torch.Generator().manual_seed(321)
    z = torch.randn(1000, 2, generator=g, dtype=torch.double) * 12
    p = project_diamond(z, 1.)
    assert (p.abs().sum(-1) <= 1. + 1e-14).all()
    # Projection optimality against the vertices suffices for this convex polytope.
    vertices = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]], dtype=torch.double)
    dots = ((z-p)[:,None,:] * (vertices[None,:,:]-p[:,None,:])).sum(-1)
    assert dots.max() < 1e-12


@pytest.mark.parametrize('r', [.25, 1., 2.])
def test_projection_product_matches_independent_vector_projection(r):
    g = torch.Generator().manual_seed(765)
    z = torch.randn(1000, 2, generator=g, dtype=torch.double) * r * 4
    p = project_diamond(z, r)
    torch.testing.assert_close(projection_mul(z[:,0], z[:,1], r), p.prod(-1), atol=2e-14, rtol=2e-13)
    inside = torch.rand(100, 2, generator=g, dtype=torch.double) * (.49*r)
    torch.testing.assert_close(projection_mul(inside[:,0], inside[:,1], r), inside.prod(-1), atol=0, rtol=0)


def test_projection_gradient_and_bound_off_kinks():
    x = torch.tensor([.2, 2., 3., -.8, 0.], dtype=torch.double, requires_grad=True)
    y = torch.tensor([-.3, 1.6, .1, .5, .4], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(projection_mul, (x,y,1.))
    dx, dy = torch.autograd.grad(projection_mul(x,y,1.).sum(), (x,y))
    assert torch.sqrt(dx.square()+dy.square()).max() <= 1. + 1e-12
    assert dx[-1] == .4


def test_projection_axes_and_nonfinite_contract():
    x = torch.tensor([0., 2., float('inf'), .2], dtype=torch.double, requires_grad=True)
    y = torch.tensor([.4, 0., 1., -.3], dtype=torch.double, requires_grad=True)
    out = projection_mul(x,y,1.)
    assert torch.isnan(out[2])
    torch.testing.assert_close(out[[0,1,3]], torch.tensor([0.,0.,-.06], dtype=torch.double))
    dx, dy = torch.autograd.grad(out[[0,1,3]].sum(), (x,y))
    assert dx[0] == .4 and dy[1] == 0.
    assert torch.isfinite(dx).all() and torch.isfinite(dy).all()


@pytest.mark.parametrize('method', METHODS)
def test_legal_outputs_are_identical(method):
    z = torch.tensor([[.45,.45], [.5,-.5], [0.,1.], [-.8,.1]], dtype=torch.double)
    v = torch.tensor([1.,1.], dtype=torch.double) / math.sqrt(2)
    expected = z + .8*z.prod(-1,keepdim=True)*v
    torch.testing.assert_close(residual_map(z,method,1.,.8,v), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('method', ['hubermul','projection'])
def test_inverse_arbitrary_start_recovers_legal_preimage(method):
    z = torch.tensor([[.45,.45], [-.4,.3], [0.,0.], [.99,.01]], dtype=torch.double)
    v = torch.tensor([1.,1.], dtype=torch.double) / math.sqrt(2)
    target = residual_map(z,'mul',1.,.8,v)
    initial = torch.full_like(z,10.)
    keep = initial.clone()
    result = inverse_fixed_point(target, initial, method, 1., .8, v, 1e-10, 2000, 1e100)
    assert result['status'] == ['converged'] * len(z)
    error = torch.linalg.vector_norm(result['z']-z, dim=-1)
    assert error.max() <= 1e-10
    assert (error <= result['residual_norm']/(1-.8) + 1e-14).all()
    torch.testing.assert_close(initial, keep, rtol=0,atol=0)
    assert torch.equal(result['certified_method'], torch.ones(len(z),dtype=torch.bool))


def test_plain_multiplication_wrong_root_is_not_a_certificate():
    v = torch.tensor([1.,1.], dtype=torch.double) / math.sqrt(2)
    other = torch.full((1,2), -math.sqrt(2)/.8, dtype=torch.double)
    target = torch.zeros_like(other)
    torch.testing.assert_close(residual_map(other,'mul',1.,.8,v),target,atol=1e-14,rtol=0)
    result = inverse_fixed_point(target,other,'mul',1.,.8,v,1e-10,2000,1e100)
    assert result['status'] == ['converged']
    assert torch.linalg.vector_norm(result['z']) > 1.
    assert not result['certified_method'].any()


def test_divergence_and_timeout_are_retained():
    v = torch.tensor([1.,1.], dtype=torch.double) / math.sqrt(2)
    z = torch.tensor([[.45,.45]],dtype=torch.double)
    target = residual_map(z,'mul',1.,.8,v)
    result = inverse_fixed_point(target,torch.full_like(z,10.),'mul',1.,.8,v,1e-10,2000,1e100)
    assert result['status'] == ['diverged']
    short = inverse_fixed_point(target,torch.full_like(z,10.),'hubermul',1.,.8,v,1e-10,1,1e100)
    assert short['status'] == ['max_iterations']


@pytest.mark.parametrize('q', [0., 1., float('nan')])
def test_invalid_contraction_rejected(q):
    z = torch.zeros(1,2,dtype=torch.double)
    with pytest.raises(ValueError):
        inverse_fixed_point(z,z,'hubermul',1.,q,torch.tensor([1.,0.],dtype=torch.double),1e-10,2,1e100)


def test_one_sided_seam_derivative_is_continuous_only_for_huber():
    eps=1e-6
    def derivative(op, t):
        t=torch.tensor(t,dtype=torch.double,requires_grad=True)
        f=op(.5+t,.5+t,1.)
        return torch.autograd.grad(f,t)[0].item()
    assert abs(derivative(huber_mul,eps)-derivative(huber_mul,-eps)) < 3e-6
    assert abs(derivative(projection_mul,eps)-derivative(projection_mul,-eps)) > .99


def small_config():
    return dict(radius=1., contractions=[.8], directions=[[1.,1.]], seeds=[11],
                random_points=3, starts=['output','zero','far_pp'],start_scale=10.,
                error_tolerance=1e-10, max_iterations=2000,divergence=1e100,
                boundary_offsets=[1e-4,1e-6], benchmark=False,
                benchmark_sizes=[16],benchmark_min_time=.01,benchmark_repeats=1)


def test_runner_has_paired_cases_and_no_silent_failures(tmp_path):
    import csv
    import json
    from experiments.contractive import run_contractive
    cfg=small_config()
    before=json.dumps(cfg,sort_keys=True)
    a=run_contractive(cfg,tmp_path/'a')
    b=run_contractive(cfg,tmp_path/'b')
    assert a == b
    assert json.dumps(cfg,sort_keys=True) == before
    assert a['methods']['projection']['total'] == 39
    assert a['methods']['hubermul']['recovered'] == 39
    assert a['methods']['projection']['recovered'] == 39
    assert a['methods']['mul']['failed'] > 0
    rows=list(csv.DictReader((tmp_path/'a'/'trials.csv').open()))
    assert len(rows) == 117
    for case_id in {r['case_id'] for r in rows}:
        case=[r for r in rows if r['case_id']==case_id]
        assert len(case)==3
        assert len({(r['true_x'],r['true_y'],r['target_x'],r['target_y'],r['initial_x'],r['initial_y']) for r in case})==1
    for file in ['config.json','environment.json','provenance.json','summary.json',
                 'stdout.log','trials.csv','boundary.csv']:
        assert (tmp_path/'a'/file).stat().st_size>0
    provenance=json.loads((tmp_path/'a'/'provenance.json').read_text())
    assert provenance['training_or_selection'] == 'none'
    assert len(provenance['source_manifest_sha256'])==64
    assert 'experiments/baselines/contractive.py' in provenance['source_files']
    with pytest.raises(FileExistsError):
        run_contractive(cfg,tmp_path/'a')


def test_runner_counts_timeouts_in_denominator(tmp_path):
    from experiments.contractive import run_contractive
    cfg=small_config()
    cfg.update(max_iterations=0, starts=['far_pp'])
    out=run_contractive(cfg,tmp_path/'short')
    for method in METHODS:
        assert out['methods'][method]['total']==13
        assert out['methods'][method]['failed']==13
        assert out['methods'][method]['outcomes']['max_iterations']==13


def test_bad_sweep_config_rejected_before_directory_creation(tmp_path):
    from experiments.contractive import run_contractive
    cfg=small_config()
    cfg['directions']=[[0.,0.]]
    with pytest.raises(ValueError):
        run_contractive(cfg,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
