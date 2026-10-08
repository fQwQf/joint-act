# Controlled studies and resumable evaluation

The original controlled study compares four output-head settings:
`joint`, `regression`, `prototype` (no residual), and `no_brier`. It preserves the
base checkpoint, observations, episode splits, codebook, optimizer budget and
seed across each group. This is an internal representation study. The regression
variant is not an official OpenVLA-OFT reproduction.

## Prepare and execute a study

First configure `configs/libero_joint.yaml` with the actual dataset, artifacts,
pretrained checkpoint and a training budget. The default 20,000 steps is a
starting recipe; no convergence or benchmark claim has been established for it.
Preparation requires nonempty train, validation and test episode partitions.

```bash
jointact prepare-study --config configs/libero_joint.yaml \
  --output runs/study-spatial --seeds 42 43 44 \
  --variants joint regression prototype no_brier

# Run only on GPU(s) assigned to this project.
CUDA_VISIBLE_DEVICES=0 jointact run-study \
  --plan runs/study-spatial/study.json --device cuda:0

# For an allocated multi-GPU job, run-study launches torchrun itself.
CUDA_VISIBLE_DEVICES=0,1,2,3 jointact run-study \
  --plan runs/study-spatial/study.json --device cuda:0 --world-size 4
```

The runner executes train → full held-out offline evaluation → export for each
configuration. It checks the source, configuration, dataset and artifact hashes.
Rerunning the same command resumes an incomplete training checkpoint and skips
training already at its configured final step. Offline evaluation is rerun, and
an existing export must contain the same weights. `--max-runs 1` limits execution
to the first listed configuration, useful for a bounded resource check. It does
not select the next unfinished configuration automatically.

Each run retains `lifecycle.log`, `lifecycle.json`, `run.json`, `metrics.jsonl`,
`val.json`, `test.json`, checkpoints and the deployment bundle. A lifecycle marked
complete means these stages completed; closed-loop LIBERO evaluation is a
separate command. The runner records `closed_loop_evaluated: false` explicitly.

Do not change world size, accumulation, learning-rate schedule or data order
while resuming a training run. Start another study to change these controls.
The original bitwise continuation guarantee remains restricted to the tested
CPU setup with zero data-loader workers.

## Diagnose the action representation

Offline evaluation keeps `action_l1` based on the model's predicted mode. Its
separate `diagnostics` object includes:

| Field | Interpretation |
| --- | --- |
| `quantization_l1` | Error of the closest prototype before correction |
| `predicted_prototype_l1` | Error after selecting the model's predicted prototype |
| `oracle_mode_action_l1` | Diagnostic correction error given the demonstrated mode; not a deployable policy score |
| `residual_abs_mean` | Magnitude of the continuous correction |
| `residual_saturation_fraction` | Valid action elements near the configured residual bound |
| `required_residual_out_of_bounds_fraction` | Elements beyond the correction bound even for the assigned prototype |
| `predicted_mode_counts`, `target_mode_counts` | Mode occupancy and imbalance |

All errors use normalized action units and valid timesteps only. These metrics
help distinguish quantization, classification and correction failures. They do
not measure execution success. `model.residual_enabled: false` freezes and
bypasses the residual branch for the prototype ablation.

## Resume LIBERO trials

```bash
jointact evaluate-libero --checkpoint runs/study-spatial/runs/joint-seed42/bundle \
  --device cuda:0 --precision bf16 --suite libero_spatial --trials 50 \
  --execute-horizon 8 --output runs/eval-joint-seed42 --max-new-trials 10

jointact evaluate-libero --checkpoint runs/study-spatial/runs/joint-seed42/bundle \
  --device cuda:0 --precision bf16 --suite libero_spatial --trials 50 \
  --execute-horizon 8 --output runs/eval-joint-seed42 --resume
```

The versioned protocol binds weights, model configuration, source digest,
dependency versions, input specification, task files and initial-state assets.
Each trial gets a deterministic seed independent of the preceding trials. An
atomic JSON file per completed trial is the resume authority. The JSONL index
and summary are rebuilt from those files; an interrupted partial summary cannot
hide missing trials. Concurrent writers are rejected. Failed trials that raised
an exception remain pending, and the evaluator exits with the error.

`--max-new-trials` controls how much additional work to perform without changing
the trial set. `--max-steps` changes the evaluation protocol and is only suitable
for a separately labelled bounded integration check. A five-step check is not a
standard benchmark. Videos are optional; frames are not retained when disabled.

```bash
jointact compare-libero --run joint=runs/eval-joint-seed42 \
  --run regression=runs/eval-regression-seed42 --output runs/comparison-seed42.json
```

Comparison rejects incomplete evaluations and mismatched trial sets, initial
states, task definitions, input conditions, precision or execution horizons.
It reports paired successes, disagreements and success-rate differences, and
preserves training configurations for assessing the remaining differences.
Wilson intervals describe trial proportions; they do not estimate variation
over independently trained seeds. Upstream OFT/AR reports require a matching
evaluation adapter and are not automatically imported as equivalent trials.

## Measure deployment cost

```bash
jointact benchmark-dataset --checkpoint runs/study-spatial/runs/joint-seed42/bundle \
  --device cuda:0 --precision bf16 --split test --observations 16 \
  --warmup 10 --repeats 100 --execute-horizon 8 --output runs/latency-joint-seed42.json
```

Observations cycle across held-out episodes at batch size one. The report keeps
all durations, observation identifiers, a workload digest, policy identity,
hardware/precision, median/P95 latency and CUDA peak allocated/reserved memory.
The measurement covers preprocessing, prediction and action unnormalization.
Throughput uses the executed horizon, not unused predicted actions. Compare
policies on the same observation identifiers and hardware; changes in model
inputs or normalization can change the workload digest and must be inspected.

Training logs also record optimization time, consumed samples, throughput and
rank-zero CUDA peaks. These optimization measurements exclude periodic
validation and checkpoint I/O; wall elapsed time remains available separately.
Time/sample counters and CUDA peaks restart at each training invocation, so a
resumed run's cumulative cost requires combining its invocation segments.
Rank-zero validation/checkpoint errors and nonfinite losses are propagated
across ranks. The GPU guard rejects foreign compute processes even when their
memory use is below 1 GiB. Model sharding and quantized training remain outside
the current DDP implementation.

The [action-aware supervision study](alignment-study.md) adds `aligned`, `cost`,
and `aligned_cost` variants, matched parent-checkpoint forks, distributed validation
window coverage, and correct/wrong-mode residual diagnostics. These are implemented
experimental objectives; performance gains remain to be measured.
