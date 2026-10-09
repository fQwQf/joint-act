# Action-aware supervision study

Status: implemented experimental objectives; effectiveness is not established.
For the 120 GB whole-host limit, follow [the revised resource plan](compute-plan.md)
and [bounded serial launch instructions](memory-safe-running.md). Calibrate RAM
before increasing concurrency; do not launch five independent jobs together.
This study follows the 1,000-step Spatial pilot. Its full validation L1 was
0.177229 for JointAct and 0.179323 for the same-backbone regression control.
The larger difference on the first 100 windows was not representative of all
4,380 validation windows. Existing evidence and checkpoints remain unchanged.

## Mechanism

For demonstration chunk `a`, valid-time mask `v`, fixed prototype `c_k`, and
probabilities `p_k`, define `e_k = masked_mean(abs(a - c_k))` in normalized action
units. Let `y` be the original nearest squared-L2 prototype and `j = argmax p`.
Neither labels nor the codebook are changed in this experiment.

The original CE + Brier + demonstrated-mode residual L1 objective is retained.
Two independently switchable terms are added:

1. **Selected-mode correction:** compute `c_j + r(h,j)` using the same backbone
   features. Supervise this additional action only when `j != y`,
   `e_j <= min_k(e_k) + alignment_margin`, and every valid required residual
   magnitude is strictly below `residual_scale`. This is a demonstration-based
   geometric eligibility test, not proof that the alternate mode is safe or
   successful in simulation. The true-mode residual always keeps its supervision.
   The extra masked error is divided by all valid action elements in the batch,
   so its contribution decreases when coverage is low. The argmax is discrete;
   this term updates the residual and shared features, not the classifier output
   weights directly.
2. **Action cost:** `mean_batch(sum_k p_k * stop_gradient(e_k))`. Fixed prototypes
   prevent the residual from changing the cost targets. This gives the classifier
   a gradient that distinguishes small and large action errors, alongside CE and
   Brier. These costs are distance to a demonstration, not task failure costs or
   success probabilities. A frequent low-error prototype can still dominate;
   monitor mode occupancy and task performance.

Initial, untuned settings are `alignment_weight=0.5`,
`action_cost_weight=1.0`, and `alignment_margin=0.05`. Defaults in the general
configuration are zero weights, preserving old checkpoints and training behavior.
Training adds one small residual-head evaluation and prototype distances;
inference architecture, parameters and action horizon are unchanged.

Implementation precedent and exact inspected commits are in `THIRD_PARTY.md`
and `upstream.lock.json`. In particular, VQ-BeT already supervises correction of
sampled decoded actions. The selected-mode idea alone is not a novelty claim.

## Controlled matrix

| Variant | Extra correction | Expected prototype cost | Purpose |
| --- | ---: | ---: | --- |
| joint | 0 | 0 | Original objective control |
| aligned | 0.5 | 0 | Does deployment-aligned supervision help? |
| cost | 0 | 1.0 | Does accounting for action geometry help selection? |
| aligned_cost | 0.5 | 1.0 | Are the two mechanisms complementary? |
| regression | — | — | Same-backbone continuous-head control |

All variants share observations, train/val/test episodes, codebook, learning-rate
schedule, accumulation, optimizer budget and execution horizon. This matrix is
an internal ablation; official OpenVLA and OpenVLA-OFT comparisons remain separate
upstream reproductions. Multi-seed experiments should initialize independent runs;
forks of a single parent are paired continuations, not independent training seeds.

```bash
jointact prepare-study --config configs/libero_joint.yaml \
  --output runs/alignment-study --seeds 42 43 44 \
  --variants joint aligned cost aligned_cost regression
```

To conserve the completed pilot training, explicitly supply both matched parents:

```bash
jointact prepare-study --config /path/to/original-joint-config.yaml \
  --output /path/to/new-study --seeds 42 \
  --variants joint aligned cost aligned_cost regression \
  --joint-parent /path/to/joint/checkpoints/step-00001000 \
  --regression-parent /path/to/regression/checkpoints/step-00001000
```

Use actual checkpoint paths returned by training. A fork keeps parent weights,
optimizer, scheduler, scaler, sampler position and per-rank RNG; it preserves
seed, model, data, batch topology and optimization schedule. Only the new objective
settings may change. It writes a fresh run and `lineage.json` with parent hashes,
step and changed objectives. Ordinary `--resume` rejects objective changes.
A fork's optimization-time counters cover the new segment; inherited parent cost
must be included when reporting total training compute.

`train --stop-after N` uses an **absolute optimizer step** without changing the
20,000-step schedule. For example, 1,100 stops after 100 additional steps from a
1,000-step parent. A short engineering run is not a completed method comparison.
`run-study` instead executes the full configured budget and full offline val/test
plus export; it does not silently run LIBERO rollouts.

## Measurement and decisions

Periodic study validation uses a deterministic set of evenly spaced windows
across the whole validation index range, with budget `batch_size * eval_batches`.
It records `sampling`, `evaluated_windows`, and `total_validation_windows`.
This is window coverage, not episode- or task-balanced sampling. Full held-out
validation remains the selection metric; test results are reserved for a frozen
recipe. Historical prefix evaluations retain their original meaning.

Training records eligibility coverage, neighborhood/reachability fractions,
alignment contribution, eligible L1 and expected prototype cost. Evaluation
separates correct and wrong selected modes, reports both action and prototype L1,
and their residual gain, with the valid-element counts for each group. Empty
groups have null errors rather than an invented zero error.

Before scaling the study, check:

- The alignment gate has nontrivial coverage. Near-zero coverage means the term
  is inactive; do not interpret its unchanged result as evidence against correction.
- Wrong-mode action error decreases without harming correctly selected modes.
  Oracle-mode error alone is not evidence that deployment improved.
- Cost supervision reduces actual selected action error without severe occupancy
  collapse; lower expected cost alone can be achieved by changing probabilities.
- Full validation supports any improvement seen on the periodic subset.

For policy effectiveness, run the existing paired LIBERO evaluator on all 10
Spatial tasks with the same initial states, 50 trials per task, seed and execution
horizon (8), then compare complete trial ledgers. Keep an identical stopping budget
for all five branches. A seed-42 continuation is exploratory; expand a promising
recipe to independent seeds 42/43/44 and other LIBERO suites before making broad
claims. Report taskwise success, variation across training seeds, action error,
training GPU-hours, and measured batch-one latency. The unchanged inference graph
suggests no extra deployment overhead, but a speed advantage over OpenVLA/OFT
still requires matched end-to-end measurement.
