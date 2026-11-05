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
