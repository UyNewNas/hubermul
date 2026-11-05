"""Fixed, paired, no-training verification of a contractive residual task.

Run: python -m experiments.contractive --config experiments/configs/contractive.toml
     --output runs/contractive-001
Numerical diagnostics are not interval-certified proofs or real-application results.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tomllib

import torch
from torch.utils.benchmark import Timer

from experiments.baselines.contractive import METHODS, inverse_fixed_point, projection_mul
from hubermul import huber_mul

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = [[0.,0.], [1.,0.], [0.,1.], [.5,.5], [-.5,-.5], [.5,-.5],
           [.45,.45], [.99,.01], [-1.,0.], [0.,-1.]]
STARTS = ('output','zero','far_pp','far_mm','far_pm')


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def _number(value):
    v=float(value)
    return v if math.isfinite(v) else None


def _git(*args):
    try:
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return None


def _provenance():
    files={}
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        if any(part.startswith('.') or part in ('runs','build','dist','__pycache__')
               for part in relative.parts):
            continue
        if p.is_file() and p.suffix in ('.py','.toml','.md','.yml'):
            files[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return {'local_commit':_git('rev-parse','HEAD'), 'local_tree':_git('rev-parse','HEAD^{tree}'),
            'dirty':bool(_git('status','--porcelain')), 'source_files':files,
            'source_manifest_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'remote_commit':os.environ.get('HUBERMUL_REMOTE_COMMIT'),
            'upstream_base':'7e1121eb86e2e5b69f465ae6920d64e192043a79',
            'training_or_selection':'none', 'generator':'area-uniform diamond + ten fixed anchors',
            'certificate':'real-arithmetic inequality; not interval-verified floating point'}


def _validate(cfg):
    for key in ('radius','start_scale','error_tolerance','divergence','benchmark_min_time'):
        if not math.isfinite(cfg[key]) or cfg[key]<=0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('random_points','benchmark_repeats'):
        if type(cfg[key]) is not int or cfg[key]<=0:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['max_iterations']) is not int or cfg['max_iterations']<0:
        raise ValueError('max_iterations must be nonnegative integer')
    if not cfg['contractions'] or any(not math.isfinite(q) or not 0<q<1 for q in cfg['contractions']):
        raise ValueError('contractions must be strictly between 0 and 1')
    if len(set(cfg['contractions']))!=len(cfg['contractions']):
        raise ValueError('duplicate contraction values')
    if not cfg['directions']:
        raise ValueError('empty directions')
    for v in cfg['directions']:
        if len(v)!=2 or not all(math.isfinite(t) for t in v) or math.hypot(*v)==0:
            raise ValueError('direction must be a finite nonzero two-vector')
    if not cfg['seeds'] or any(type(s) is not int or s<0 for s in cfg['seeds']):
        raise ValueError('seeds must be nonnegative integers')
    if len(set(cfg['seeds']))!=len(cfg['seeds']):
        raise ValueError('duplicate seeds')
    if not cfg['starts'] or any(s not in STARTS for s in cfg['starts']):
        raise ValueError('invalid starts')
    if len(set(cfg['starts']))!=len(cfg['starts']):
        raise ValueError('duplicate starts')
    if not cfg['boundary_offsets'] or any(not 0<e<.25 for e in cfg['boundary_offsets']):
        raise ValueError('boundary offsets must be in (0,.25)')
    if not cfg['benchmark_sizes'] or any(type(n) is not int or n<=0 for n in cfg['benchmark_sizes']):
        raise ValueError('invalid benchmark sizes')
    if type(cfg['benchmark']) is not bool:
        raise ValueError('benchmark must be a bool')


def legal_points(seed, n, radius):
    g=torch.Generator().manual_seed(seed)
    rho=torch.rand(n,generator=g,dtype=torch.double).sqrt()*radius
    t=torch.rand(n,generator=g,dtype=torch.double)
    signs=torch.randint(0,2,(n,2),generator=g)*2-1
    points=torch.stack((rho*t,rho*(1-t)),dim=-1)*signs
    return torch.cat((points,torch.tensor(ANCHORS,dtype=torch.double)*radius))


def _boundary_rows(cfg):
    rows=[]
    for e in cfg['boundary_offsets']:
        for method,op in (('hubermul',huber_mul),('projection',projection_mul)):
            values=[]
            for s in (-1,1):
                t=torch.tensor(s*e*cfg['radius'],dtype=torch.double,requires_grad=True)
                value=op(cfg['radius']/2+t,cfg['radius']/2+t,cfg['radius'])
                values.append(torch.autograd.grad(value,t)[0].item())
            rows.append(dict(method=method,relative_offset=e,left_derivative=values[0],
                             right_derivative=values[1],jump=abs(values[1]-values[0])))
    return rows


def _csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plain(x,y,radius):
    return x*y


def _cost_rows(cfg):
    rows=[]
    r=cfg['radius']
    g=torch.Generator().manual_seed(9441)
    order=random.Random(9441)
    ops={'mul':_plain,'hubermul':huber_mul,'projection':projection_mul}
    for n in cfg['benchmark_sizes']:
        for workload,scale in (('legal',.49),('off_domain',4.)):
            x=((torch.rand(n,generator=g,dtype=torch.double)*2-1)*scale*r).requires_grad_()
            y=((torch.rand(n,generator=g,dtype=torch.double)*2-1)*scale*r).requires_grad_()
            for phase in ('forward','forward_backward'):
                stmt='op(x,y,r)' if phase=='forward' else 'torch.autograd.grad(op(x,y,r).sum(), (x,y))'
                # Forward-only uses no autograd graph, matching inference operation cost.
                xx,yy=(x.detach(),y.detach()) if phase=='forward' else (x,y)
                for repeat in range(cfg['benchmark_repeats']):
                    names=list(METHODS)
                    order.shuffle(names)
                    for name in names:
                        timer=Timer(stmt,globals=dict(op=ops[name],x=xx,y=yy,r=r,torch=torch),num_threads=1)
                        measured=timer.blocked_autorange(min_run_time=cfg['benchmark_min_time'])
                        rows.append(dict(method=name,n=n,workload=workload,phase=phase,repeat=repeat,
                                         median_us=measured.median*1e6,iqr_us=measured.iqr*1e6,
                                         blocks=len(measured.raw_times),number_per_run=measured.number_per_run,
                                         block_seconds=json.dumps(measured.raw_times)))
    return rows


def _summary(rows):
    out={'purpose':'task_matching_not_superiority_trial','methods':{}}
    for method in METHODS:
        subset=[r for r in rows if r['method']==method]
        ok=[r for r in subset if r['outcome']=='recovered']
        iterations=[r['iterations'] for r in ok]
        out['methods'][method]={'total':len(subset),'recovered':len(ok),'failed':len(subset)-len(ok),
            'outcomes':dict(Counter(r['outcome'] for r in subset)),
            'iterations_median_recovered':statistics.median(iterations) if iterations else None,
            'iterations_max_recovered':max(iterations) if iterations else None,
            'max_relative_error_recovered':max((r['error_over_radius'] for r in ok),default=None),
            'certificate_violations_with_roundoff_slack':sum(r['certificate_violation'] for r in subset)}
    lookup={(r['case_id'],r['method']):r for r in rows}
    pairs=[]
    for row in rows:
        if row['method']=='hubermul':
            other=lookup[(row['case_id'],'projection')]
            if row['outcome']==other['outcome']=='recovered':
                pairs.append((row['iterations'],other['iterations']))
    out['paired_protected_iterations']={'both_recovered':len(pairs),
        'excluded_from_comparison':out['methods']['hubermul']['total']-len(pairs),
        'huber_fewer':sum(a<b for a,b in pairs),'tie':sum(a==b for a,b in pairs),
        'projection_fewer':sum(a>b for a,b in pairs),
        'huber_total_updates':sum(a for a,b in pairs),'projection_total_updates':sum(b for a,b in pairs)}
    out['judgment']='No automatic superiority decision; inspect all outcomes and equally certified control.'
    return out


def run_contractive(config: dict, output: Path) -> dict:
    cfg=json.loads(json.dumps(config))
    _validate(cfg)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    _json(output/'config.json',cfg)
    _json(output/'environment.json',dict(python=sys.version,torch=torch.__version__,
          platform=platform.platform(),processor=platform.processor(),threads=1,dtype='float64',
          device='cpu',cuda_executed=False,benchmark_backend='eager'))
    _json(output/'provenance.json',_provenance())
    rows=[]
    r=cfg['radius']
    try:
        with (output/'stdout.log').open('w') as log, (output/'trials.csv').open('w',newline='') as raw:
            writer=None
            for seed in cfg['seeds']:
                points=legal_points(seed,cfg['random_points'],r)
                for iq,q in enumerate(cfg['contractions']):
                    for iv,vraw in enumerate(cfg['directions']):
                        v=torch.tensor(vraw,dtype=torch.double)
                        v=v/torch.linalg.vector_norm(v)
                        true=points.repeat(len(cfg['starts']),1)
                        target=true+(q/r)*true.prod(-1,keepdim=True)*v
                        initial=[]
                        for name in cfg['starts']:
                            if name=='output':
                                initial.append(target[:len(points)].clone())
                            elif name=='zero':
                                initial.append(torch.zeros_like(points))
                            else:
                                signs={'far_pp':[1.,1.],'far_mm':[-1.,-1.],'far_pm':[1.,-1.]}[name]
                                initial.append(torch.tensor(signs,dtype=torch.double).expand_as(points)*cfg['start_scale']*r)
                        initial=torch.cat(initial)
                        for method in METHODS:
                            result=inverse_fixed_point(target,initial,method,r,q,v,
                                      cfg['error_tolerance'],cfg['max_iterations'],cfg['divergence'])
                            errors=torch.linalg.vector_norm(result['z']-true,dim=-1)
                            group=[]
                            for i,status in enumerate(result['status']):
                                start=cfg['starts'][i//len(points)]
                                error=_number(errors[i])
                                residual=_number(result['residual_norm'][i])
                                slack=64*torch.finfo(torch.double).eps*(1+torch.linalg.vector_norm(true[i]).item()+
                                      torch.linalg.vector_norm(target[i]).item())/(1-q)
                                recovered=status=='converged' and error is not None and error<=cfg['error_tolerance']*r+slack
                                outcome='recovered' if recovered else ('wrong_root' if status=='converged' else status)
                                bound=residual/(1-q) if method!='mul' and residual is not None else None
                                row=dict(case_id=f'{seed}-q{iq}-v{iv}-{start}-p{i%len(points)}',seed=seed,
                                    q=q,direction_x=v[0].item(),direction_y=v[1].item(),start=start,
                                    point_kind='random' if i%len(points)<cfg['random_points'] else 'anchor',
                                    method=method,true_x=true[i,0].item(),true_y=true[i,1].item(),
                                    target_x=target[i,0].item(),target_y=target[i,1].item(),
                                    initial_x=initial[i,0].item(),initial_y=initial[i,1].item(),
                                    found_x=_number(result['z'][i,0]),found_y=_number(result['z'][i,1]),
                                    solver_status=status,outcome=outcome,iterations=int(result['iterations'][i]),
                                    residual_norm=residual,error_over_radius=error/r if error is not None else None,
                                    mathematical_error_bound=bound,roundoff_slack=slack,
                                    certificate_violation=(bound is not None and error is not None and error>bound+slack),
                                    max_excursion_over_radius=_number(result['max_excursion'][i]))
                                group.append(row)
                            if writer is None:
                                writer=csv.DictWriter(raw,fieldnames=list(group[0]))
                                writer.writeheader()
                            writer.writerows(group)
                            raw.flush()
                            rows.extend(group)
                        message=f'seed={seed} q={q} direction={vraw}: {len(true)} paired cases complete'
                        print(message)
                        log.write(message+'\n')
                        log.flush()
            _csv(output/'boundary.csv',_boundary_rows(cfg))
            if cfg['benchmark']:
                _csv(output/'costs.csv',_cost_rows(cfg))
            summary=_summary(rows)
            _json(output/'summary.json',summary)
            log.write(json.dumps(summary,sort_keys=True)+'\n')
            return summary
    except Exception as exc:
        _json(output/'failure.json',{'status':'exception','type':type(exc).__name__,
                                     'message':str(exc),'completed_rows':len(rows)})
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true',help='Recorded reduced configuration; no performance claim')
    args=p.parse_args()
    cfg=tomllib.loads(args.config.read_text())
    if args.smoke:
        cfg.update(seeds=cfg['seeds'][:1],contractions=cfg['contractions'][:1],
                   directions=cfg['directions'][:1],random_points=8,
                   starts=['output','zero','far_pp'],benchmark=False,mode='smoke')
    summary=run_contractive(cfg,args.output)
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
