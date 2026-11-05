# Operator specification — v0.1.0a1

## Definition and claims
For a fixed real R>0, H_R(t)=t²/2 if |t|<=R, and R|t|-R²/2 otherwise.
M_R(x,y)=(H_R(x+y)-H_R(x-y))/2.

Paper derivations (not machine-checked proofs):
- M_R(x,y)=xy on |x|+|y|<=R, by the difference-of-squares identity.
- Let s(t)=clip(t,-R,R). The input derivatives are
  M_x=(s(x+y)-s(x-y))/2 and M_y=(s(x+y)+s(x-y))/2.
  Their squared Euclidean norm is (s(x+y)²+s(x-y)²)/2 <= R².
- The global Lipschitz constant is exactly R: on the exact domain the quotient
  between (r,h) and (r,0) equals r, for every r<R and sufficiently small h.
  This proves the lower bound as r approaches R.
- M is C1, not generally C2 on |x+y|=R or |x-y|=R.
- With q(t)=max(t,0)², H_R(t)=[q(t)+q(-t)-q(t-R)-q(-t-R)]/2.
  Consequently M has an exact eight-ReQU representation in real arithmetic.

These statements do not establish novelty, numerical reliability, better learning,
or a bound on a whole network's parameter gradient. Broadcasting and reductions
can accumulate gradients. Floating-point arithmetic is not exact real arithmetic.

## Stable branches
Reorder the *signed* inputs to u,v with a=|u|>=b=|v|. Let d=a-b.
- Center: a<=R and b<=R-a. Return u*v, implemented as x*y.
- Outer: d>=R, after selecting the center. Return sign(u)*R*v.
- Middle: h=(R-d)/2. Return
  sign(u)*[(R-h)*v+h*(v-sign(v)*h)].

In the middle, 0<h<=b and h<=R/2. Magnitudes of the two terms are
(R-h)*b and h*(b-h), both nonnegative. Their sum is Rb-h².
This avoids large Huber differences, squaring a large number, and a t/b division
whose backward can underflow even when its final derivative is representable.

Classification and t=R-(a-b) also use cancellation-aware orderings: near a=R,
compare b with a-R and form t from b and a-R; far from R, use a-b first.
This preserves the small input's contribution to the derivative at a=R.

All inactive branch operands are masked *before* potentially unsafe arithmetic.
The outer branch retains signed v: at x=0 the x-derivative must be clip(y,-R,R).
The sort seam |x|=|y| has matching first derivatives; it is covered by tests.

## API and input contract
`huber_mul(x: Tensor, y: Tensor, radius: float = 2.0) -> Tensor`
and `HuberMul(radius=2.0)(x,y)`.

- x,y: strided, same-device, same-dtype tensors, float32 or float64.
- PyTorch broadcasting, zero-dimensional and empty tensors, and noncontiguous
  tensors are accepted. Incompatible shapes raise PyTorch's RuntimeError.
- Mixed dtypes, integers, complex numbers, sparse tensors, FP16 and BF16 raise
  TypeError. Cross-device inputs raise ValueError. Low precision/AMP is unverified.
- Radius: a Python real scalar, not bool or Tensor. Nonpositive/nonfinite values
  raise ValueError; non-scalar or non-real values raise TypeError. It must lie in
  [finfo(dtype).tiny, finfo(dtype).max]. It is rounded as a scalar in the input dtype.
  Radius is fixed, not differentiated or trainable.
- Any pair containing NaN or +/-Inf returns NaN, including (Inf,0). Such elements
  are disconnected before arithmetic; no mathematical gradient is claimed for them.
  Backpropagating a finite lane must not contaminate other finite lanes.
- No in-place mutation. Returned shape/dtype/device follow broadcasting and inputs.
- HuberMul has no trainable parameters. Radius is preserved through extra state in
  state_dict; device/dtype moves do not turn radius into a trainable Tensor.

## Numerical and support limits
The center uses ordinary multiplication; floating rounding can affect branch
classification near boundaries. Exactness means real arithmetic, not a universal
bitwise or correct-rounding guarantee. Unrepresentable outputs may overflow;
underflow and platform-dependent denormal handling remain possible. A finite
mathematical Lipschitz bound is not a promise that floating-point evaluation is
uniformly Lipschitz at all scales.

`_reference.py` is a literal readable formula for moderate inputs. It is NOT the
oracle for extreme inputs. Tests use an independent mpmath evaluation at 800 digits,
closed-form derivatives, fixed regressions, and finite differences away from seams.

Verified locally: CPU float32/float64; Python 3.13.5, torch 2.10.0+cpu;
fullgraph torch.compile with backend=aot_eager forward and backward.
Not yet verified: CUDA, AMP, FP16/BF16, Inductor performance, torch.export, vmap,
and arbitrary-version compatibility. aot_eager success is not an acceleration claim.

## Sources and contribution boundary
Huber loss, polarization identities, ReQU networks, and Lipschitz extension theory
are existing tools. This repository is an experimental implementation; exact-form
priority and independent theoretical value are not established.
- Huber definition: https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html
- Autograd masking pitfalls: https://docs.pytorch.org/docs/stable/notes/autograd.html
- Numerical limits: https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- RePU/PowerNet: https://arxiv.org/abs/1909.05136
- McShane-Whitney extensions: https://arxiv.org/abs/1804.06757
