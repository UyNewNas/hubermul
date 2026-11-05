# Minimal experimental hypothesis and stopping rules

## Scope
E1/E2 correctness gate first. E4 observation corruption is the primary *question*;
E3 arithmetic extrapolation and legitimate large inputs reveal the cost of limiting
multiplication. This initial runner is a pipeline check, NOT a confirmatory study.
No Triton, learnable radius, expression trees, or symbolic export in this batch.

## Hypotheses and controls
H1: under corrupted observations, HuberMul might reduce challenge prediction error
at acceptable clean-error and failure-rate cost, compared with simple protections.
H2: fixed-radius HuberMul is expected to introduce bias for legitimate large values
and far-range multiplication. Negative results are retained.

Target: (x1+x2)*(x3-x4). Six methods share two trainable linear projections and an
output bias, the actual same initial weights, dataset splits, optimizer, step
budget, and row-norm cap. There is no learnable output gain. A fixed no-gain model
limits rescaling ambiguity but does NOT prove whole-network Lipschitz control.
Record actual projection norms and HuberMul exact-domain occupancy.

Methods: ordinary multiplication; HuberMul; separately clipped inputs; RMS-radial
normalized multiplication; ordinary multiplication with parameter-gradient clipping;
ordinary multiplication with Huber training loss. The normalization baseline is a
specific radial rule, NOT a claim to cover all normalization methods.

Three distinct tasks:
1. corrupted_observation: clean latent input z uniform in [-1,1]^4, row-wise
   Bernoulli contamination adds Gaussian observation noise; labels remain f(z).
2. legitimate_large: rare rows are multiplied by a fixed scale; labels are f(x)
   of the actual large input. No noise is mislabeled as genuine arithmetic.
3. arithmetic: train on [-1,1]^4; challenge validation/test at a specified larger
   scale. This is not unseen-distribution tuning: both use the declared range.
Label contamination is deferred, not silently combined with task 1.

## Budget and selection (frozen for the checked-in configs)
smoke.toml: one seed, one candidate, 16 full-batch steps; infrastructure only.
screen.toml: three seeds, two candidates, 100 steps per candidate and method.
Each method gets the same two joint candidates (lr=.01, parameter=1) and
(lr=.03, parameter=2), not an exhaustive grid. Ordinary multiplication uses each
learning rate and ignores the irrelevant parameter. Others use parameter as radius,
clip threshold, normalization scale, gradient clip, or Huber delta respectively.
This is equal finite candidate count, NOT proof of equal tuning difficulty.
A later confirmatory protocol must expand and freeze a meaningful common budget.

Select the final-step model only by challenge validation NMSE; deterministic ties
favor the earlier candidate. Independent test data is generated/evaluated only after
selection. No test-set choice of seed, radius, candidate, checkpoint, or reporting rule.
NMSE denominator is the mean squared CLEAN training target, shared by all methods
for a task/seed. It is not a variance estimated from the test set.

## Outcomes and records
Primary outcomes: clean and challenge test NMSE, failed candidate/run counts,
first evaluated validation step below the fixed NMSE threshold, and exact-domain
occupancy. Gradient norms and occupancy are diagnostics, not success criteria.
All candidates (including failures) are recorded in trials.jsonl. Failed runs stay
in summary.json with null errors and explicit status, never vanish from denominators.
metrics.csv preserves evaluated steps, losses, validation scores and diagnostics.
A seed pair with any failure cannot be silently removed from aggregate comparisons.

Every invocation refuses to overwrite its output directory. Save config,
environment, provenance (local commit/tree/dirty flag, optional remote commit,
per-file hashes), stdout log, candidate records, metrics and summary. No mandatory
checkpoint, plot, or tracking service. Reproducibility is within the recorded
software/hardware setting, not a cross-platform bitwise promise.

## Gates and stop/continue/undetermined
A: mathematical/API contract and all relevant E1/E2 tests pass before interpreting
training outputs. Tests must include axes, sort/region seams, extreme cancellation,
invalid inputs, scalar gradients, broadcasting, and compile agreement.
B: run smoke and three-seed screen. Check that baselines can learn and the pipeline
captures known failures. Do not keep changing tasks until HuberMul wins.
C: only if there is a plausible useful effect, freeze a new protocol and independent
seeds before a confirmatory run. This batch has no automatic superiority decision.

Provisional engineering target for that future protocol: >=10% relative reduction
in corrupted-observation error versus the best validation-selected simple protection,
<=5% relative increase in clean error, and no increase in failure rate. These are
project thresholds, not mathematical/statistical standards. Define uncertainty,
near-zero-error handling, and the total compute budget BEFORE that next run.

Three possible conclusions: useful effect supported; target benefit excluded or
cost unacceptable; evidence insufficient. Budget exhaustion can stop work without
claiming that the method has been proved useless. No E6/E7 expansion without evidence.

## Frozen follow-up: domain-exact contractive residual map (2026-11-05)

The original regression screen is retained, not retuned. This separate **no-training**
experiment tests an algorithmic task, not real-world demand or a learned-model gain.

For R=1, unit v and q in {0.5,0.8,0.95}, define
F(z)=z+(q/R)*v*op(z). Legal z lies in the closed diamond D_R.
Controls: ordinary multiplication (no global certificate), HuberMul, and the product
of the coordinates of the Euclidean projection P_D(z). Projection is NOT radial
normalization. Both protected maps equal the same bilinear map on D_R and their
residual branches have global Lipschitz bound q. Projection is nonexpansive; the
product on D_R has constant R. Fixed-point inversion is z_next=w-g(z).
The real-arithmetic certificate is ||z-z*|| <= ||F(z)-w||/(1-q).
This floating-point experiment is NOT an interval-verified error certificate.

Generate targets independently using ordinary multiplication on known legal inputs.
Use seeds 101,202,303, 128 area-uniform diamond samples per seed plus ten fixed
anchors (including axes, boundary corners, midpoint seams and (0.45,0.45)).
Use normalized directions (1,0),(1,1),(1,-1), all q values, and starts w,0,
(10R,10R),(-10R,-10R),(10R,-10R). All methods receive identical cases.
CPU float64, one thread. Stop at residual <= (1-q)*1e-10*R, max 2000 updates,
or state component > 1e100*R / nonfinite. Separate solver convergence from recovering
known z*: ordinary multiplication can converge to a different preimage.
All cases, failures, residuals, recovered states and iteration counts remain in CSV.
Recovery/certificate diagnostic comparisons allow documented float64 rounding slack;
raw errors and raw mathematical bounds are saved without hiding violations.

Primary feasibility: protected methods recover every legal preimage in this fixed
suite within budget; numerical checks support, but do not prove, the analytic theorem.
Comparative usefulness: paired iteration counts and forward / forward+backward CPU
cost against **equally certified projection multiplication**. Timing is local,
thread-fixed, randomized method order, warmup/repeated Timer measurements, three
replicates for batch sizes 1,1024,65536 in legal and out-of-domain workloads.
Timed statements isolate operator work, NOT data generation or fixed-point-loop Python.
No assertion of GPU performance, optimal implementation, or training advantage.
Boundary diagnostic: directional derivative on z(t)=(R/2+t,R/2+t), offsets
1e-2,1e-4,1e-6. HuberMul's C1 property is predicted; projection has a kink.
It is NOT retroactively made a task requirement or a superiority outcome.

Decision: no further development if HuberMul offers neither a supported iteration/
cost advantage nor an independently established need for continuous Jacobians.
A numerical advantage would justify only independent validation, not a new general
claim. Equal feasibility alone does not establish an advantage. Keep the reference
library and results if these checks find no such reason; do not archive or delete
without owner instruction.

Theory reference: Behrmann et al., Invertible Residual Networks, ICML 2019,
https://arxiv.org/abs/1811.00995 . The theorem motivates the task; the paper does not
study HuberMul. Euclidean projection's nonexpansiveness follows from its variational
inequality: ||Pz-Pw||^2 <= <Pz-Pw,z-w>. No full-network bound is inferred from a
scalar bound; here the complete residual branch and its scaling are specified.
