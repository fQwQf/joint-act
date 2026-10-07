# Research experiments supported by the engineering implementation

The primary question is whether direct **joint** action decisions improve the success/latency or data-efficiency tradeoff over strong direct-action policies. The implementation supplies mechanisms and measurement tools; it contains no assumed positive results.

## Required comparisons

| Comparison | Purpose | Controlled conditions |
| --- | --- | --- |
| JointAct vs same-adapter continuous L1 head | Isolate joint mode plus residual representation | Same base weights, inputs, data, update count and execution horizon |
| JointAct vs official OpenVLA-OFT | Compare with a strong fast action policy | Report any remaining architecture/training/input differences |
| JointAct vs AR OpenVLA | Measure conversion benefit | Same visual observations and explicit action horizon |
| Modes with/without residual | Measure quantization versus precision | Same codebook and training budget |
| Demonstrations vs demonstrations + teacher | Measure supervision value | Same demo split, teacher cost recorded separately |

The regression config is runnable. To disable residual learning for an ablation, use `train.residual_weight: 0` and retain zero initialization; verify its parameters remain at zero or explicitly freeze them before interpreting the result. A loss-weight change alone is not automatically the intended ablation if another loss sends gradient through that component.

Original AR OpenVLA uses one camera and no explicit proprioceptive token. For an
input-matched comparison, convert `--camera-view primary`, set `model.num_images: 1`
and `model.proprio_dim: 0`, and use artifacts for that converted dataset. The
canonical episodes retain their original state; the policy omits the state token
and ignores the state values. Use horizon one for a per-action decoder comparison.
For chunked closed-loop comparisons, report predicted/executed horizons and compare
with the strong OFT baseline under matched chunking and observations.

## Measurements

Closed loop: task success and per-task results across multiple seeds; failures and timeouts remain visible. Efficiency: end-to-end inference median/P95 on identical hardware, batch size one, fixed input resolution and camera count, plus executed horizon. Offline: masked action error, mode NLL/Brier/ECE, codebook occupancy and quantization distortion. Resource cost: trainable parameters, actual elapsed GPU hours and teacher labeling cost.

Do not infer closed-loop success from offline action error. Do not use synthetic fixtures as evidence of robot performance. Do not compare a one-camera model with a two-camera model without identifying the input advantage. Report latency per full replan and executed commands per replan separately.

## Data-efficiency study

Create subsets of whole training episodes, then refit train-only normalization and prototypes for each condition. Retain the same held-out episodes and simulator initial states. Independent per-subset artifacts avoid using full-training action geometry in a nominally low-data experiment.

## Teacher diagnostic

The implemented AR likelihood scorer supports horizon 1 and one camera. It tests whether the existing model can rank fixed low-level actions without sampling them. It is an offline data-generation/diagnostic path; the final deployed policy does not run a candidate scorer. Candidate count and scoring cost must be recorded. A teacher score over a restricted candidate bank is a behavior prior, not outcome supervision.

## Decision points

If joint-head success does not improve at comparable latency, inspect codebook coverage, rare-mode error, residual size and errors after switching between modes. If prototype granularity needs very large K, compare the cost against continuous regression. Keep the simpler model if the added representation has no measured benefit; engineering completeness does not establish research novelty or efficacy.
