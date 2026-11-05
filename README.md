# HuberMul

Experimental Huber-polarized multiplication for PyTorch. A **small verified core
and reproducible screening harness**, not a new universal arithmetic framework.

基于 Huber 极化构造的实验性乘法模块：先验证数值与梯度，再判断是否有实际收益。
首创性、训练优势与 GPU 性能均未确认。

## Definition

For R>0, let H_R be the Huber function with quadratic part t²/2 and linear tails.

**M_R(x,y) = [H_R(x+y) - H_R(x-y)] / 2.**

In real arithmetic it equals xy on |x|+|y|<=R and has scalar input-gradient norm
at most R. Its global Lipschitz constant is R. These are NOT whole-network
stability claims or floating-point exactness guarantees.

## Install and use

Python >=3.11 and PyTorch >=2.10 are required. From this checkout:

```bash
python -m pip install -e '.[test]'
```

```python
import torch
from hubermul import HuberMul, huber_mul

x = torch.tensor([1.0, 10.0], requires_grad=True)
y = torch.tensor([1.0, 10.0], requires_grad=True)
z = huber_mul(x, y, radius=2.0)  # tensor([1., 19.]), not [1., 100.]
z.sum().backward()
module = HuberMul(radius=2.0)    # no trainable parameters
```

## Verify and reproduce

```bash
python -m pytest -q
python -m experiments.run --config experiments/configs/smoke.toml --output runs/smoke-001
python -m experiments.run --config experiments/configs/screen.toml --output runs/screen-001
```

Run experiment commands from the repository root. Output directories must not
already exist. Experiments are deliberately not installed into the library package.

Six baselines, three separately defined tasks, matched initial weights and candidate
budgets. Validation selects configurations; test data does not. Each run saves
configuration, environment, source provenance, failures, per-step metrics and results.
The three-seed screen is exploratory: **it does not establish superiority**.

## Evidence and limitations

- Paper-derived: exact multiplication on the diamond domain; global Lip=R;
  eight-ReQU representation. No formal proof assistant claim.
- Locally exercised: CPU float32/float64, numerical/gradient/tensor contracts,
  state serialization, fullgraph aot_eager forward/backward, experiment recording.
  Local environment: Python 3.13.5, torch 2.10.0+cpu.
- Unverified: CUDA, low precision/AMP, Inductor speed, training advantages and
  novelty/priority. No Triton backend. Unsupported low-precision inputs are rejected.
- Finite outputs are not guaranteed for unrepresentable mathematical results.
  Nonfinite input pairs return NaN; other finite lanes are isolated.

The stable implementation uses analytic branches and ordinary autograd. The literal
`_reference.py` is readable but can lose precision at extreme scales; tests use an
independent high-precision oracle. CPU CI is defined separately from local results;
check the actual workflow status rather than inferring it from this README.

Read [operator contract and derivations](docs/operator_spec.md) and
[experimental hypotheses and stopping rules](docs/experiment_hypothesis.md).

## Narrow follow-up: contractive inversion

No-training, paired comparison against an equally certified **Euclidean projection
product**, not just unconstrained multiplication. Both protected controls retain
exact multiplication on the declared legal domain. See the frozen protocol in
`docs/experiment_hypothesis.md`.

```bash
python -m experiments.contractive --config experiments/configs/contractive.toml --output runs/contractive-001
# Reduced pipeline check, with adjusted settings saved to config.json:
python -m experiments.contractive --config experiments/configs/contractive.toml --output runs/contractive-smoke --smoke
```

`trials.csv` preserves paired inputs, targets, starts, recovered states, iterations
and failures (including convergence to a wrong preimage). `boundary.csv` records
one-sided derivative diagnostics. `costs.csv` includes repeated, one-thread CPU
forward and forward+backward measurements and raw timing blocks. None of these
constitute a real-application benchmark, floating-point proof, or novelty claim.
