# Validation record

Validation date: 2026-10-08. Runtime checks ran through SSH on the authorized
host `WHU140piROOT`. The earlier checks below used CPU with CUDA hidden;
the later RTX 3090 check is recorded separately. Local work consisted of source edits,
downloads relayed to the remote disk, and artifact transfers. Existing GPU jobs
were left running. Environments, models, data and outputs were placed on `/tmp`
because the home filesystem had less than 1 GiB free.

| Scope | Observed result | Retained evidence |
| --- | --- | --- |
| Unit and integration tests | 24 passed, including actual HF Llama/PEFT gradients for LoRA and full decoder finetuning, padding, normalization, teacher contracts, disabling state input, and benchmark initial-state compatibility | [pytest log](evidence/pytest.log) |
| Static checks and packaging | Ruff passed; wheel and source distribution built | [lint](evidence/ruff.log), [build](evidence/build.log) |
| Installed wheel | Imported modules from a fresh wheel-install directory; full fixture CLI lifecycle passed | [wheel hash and lifecycle](evidence/wheel-lifecycle.json) |
| Fixture lifecycle through public CLI | Prepare → train → continue → offline evaluation → export → predict → latency → calibrate passed | [report](evidence/fixture-lifecycle.json) |
| Two-process CPU DDP continuation | Uninterrupted and resumed runs produced 47 bitwise-identical tensors | [report](evidence/ddp-continuation.json) |
| HTTP deployment | Actual loopback HTTP server request passed | [report](evidence/http.json) |
| Actual upstream OpenVLA implementation | Small random fused-vision/Llama checkpoint passed the full CLI lifecycle | [report](evidence/upstream-random-lifecycle.json) |
| Public robot demonstrations | Pinned raw TFRecord converted to 10 episodes / 1,238 frames; hashes verified; small upstream model trained for two steps, continued, exported and reloaded | [report](evidence/real-data-random-lifecycle.json) |
| Original AR teacher interface | Two real-demo samples scored; partial-coverage KL batch backpropagated with coverage 2/3 using the small upstream model | [report](evidence/teacher-integration.json) |
| BF16 calibration | Upstream small-model validation calibration passed with BF16 model loading and autocast | [report](evidence/calibration-bf16.json) |
| Actual LIBERO control | Small model trained on real demos completed task 0, one trial, five policy-control steps with OSMesa; two replans; no task success | [summary](evidence/real-data-simulator.json), [trial](evidence/real-data-simulator-trials.json) |
| Full pretrained OpenVLA-7B adaptation | Verified all three weight shards; BF16 training on real demos, checkpoint continuation, nonzero LoRA updates, export and reloaded prediction passed | [lifecycle](evidence/pretrained7b-lifecycle.json), [source hashes](evidence/pretrained7b-source.json) |
| Full research benchmarks | Not run | No comparative success-rate, speedup, data-efficiency or GPU-memory claim |

## Experiment-lifecycle extension

The next implementation revision adds controlled head studies, diagnostic action
errors, atomic resumable trials, paired comparisons, and cost instrumentation.
Its [validation report](evidence/maturity-validation.json) records the package
source digest separately from the earlier integration records above.

| Scope | Observed result | Retained evidence |
| --- | --- | --- |
| Expanded unit/integration suite | 31 passed in 33.03 seconds; seven new tests cover trial recovery, matched comparison controls, prototype-only training, checkpoint corruption, varied-observation benchmarking, occupied-GPU detection, study execution and diagnostic semantics | [log](evidence/maturity-pytest.log) |
| Final study identity check | The affected study train/evaluate/export/rerun test passed again after the saved-configuration/source validation was added | [log](evidence/maturity-study-test.log) |
| Two-process CPU failure handling | Exact continuation still matches all 47 tensors; injected rank-zero output failure and rank-one nonfinite loss both terminate the worker group within the smoke script's timeout | [report](evidence/maturity-validation.json) |
| Actual LIBERO restart | Task 0, two trials capped at five steps each: stop after one trial, restart and execute only the second; the first record's hash remains unchanged | [report](evidence/maturity-validation.json) |
| Installation and CLI lifecycle | New wheel imported from a fresh target directory; fixture prepare/train/continue/evaluate/export/predict/benchmark/calibrate passed | [wheel hash and report](evidence/maturity-validation.json) |
| Static checks and build | Ruff passed and wheel/source distribution built | [build log](evidence/maturity-build.log) |
| Complete Spatial preparation | All 16 TFRecord shards and two metadata files verified; 432 episodes / 52,970 frames converted and rehashed; 351 train, 35 validation and 46 test episodes; train-only `H=8`, `K=64` artifacts prepared | [preparation](evidence/spatial-preparation.json), [pinned source hashes](evidence/spatial-source.json) |
| Full-data study preparation | Four head variants × three seeds generated with matched controls; training has not started | [preparation](evidence/spatial-preparation.json) |

The 31-test run preceded two final changes: the bounded-trial/device protocol
fields and saved-study identity validation. The real simulator restart and
affected study test above exercised those final changes. The simulator used the
small random upstream model previously adapted for two updates on real demos;
both trials timed out without task success. This is restart evidence, not a
standard success-rate estimate. No GPU was used for these checks.

The software versions are in [environment.json](evidence/environment.json).
The model environment used Python 3.10, PyTorch 2.9.1, Transformers 4.40.1,
PEFT 0.11.1, timm 0.9.10, robosuite 1.4.1 and MuJoCo 2.3.7. The separate
data environment used TensorFlow CPU 2.15.1, TFDS 4.9.3 and
tensorflow-metadata 1.15.0. CI defines Python 3.10/3.11 checks; the retained
remote evidence is for Python 3.10.

## Evidence limits

The downloaded dataset source and checksums are in
[robot-data-source.json](evidence/robot-data-source.json). The ten real episodes
all landed in the training partition under the fixed
episode-hash rule; this bounded prefix has no validation/test episodes. The
check did not invent held-out data. The later complete Spatial conversion uses
the same split rule and has nonempty held-out partitions. All 432 episodes come
from the pinned source release's training split; our validation/test partitions
are internal episode holdouts, separate from LIBERO simulator initial states.
Only the Spatial suite has been prepared, not all modified-LIBERO suites.

Random-checkpoint tests verify upstream model interfaces and serialization,
not pretrained motor capability. A five-step rollout verifies the real
simulator path; its timeout is not a standard LIBERO evaluation result.
Fixture latency, mode probabilities and imitation losses do not establish
robot task success. All retained reports preserve these distinctions.

Full-duration training, full-suite task performance, cross-seed variation and
improvements over official OpenVLA/OFT remain to be measured. The later GPU
probe below establishes the memory fit of one concrete configuration, not all
batch sizes or model settings.

The full-model integration used two 224-pixel cameras, `H=4`, `K=8`, LoRA rank 2,
batch size one and two optimizer steps. It loaded 7,560,446,308 total parameters,
of which 19,209,124 were trainable. The exact [configuration](evidence/pretrained7b-config.json),
[run metadata](evidence/pretrained7b-run.json), [finite training metrics](evidence/pretrained7b-training.json)
and [CPU process timing](evidence/pretrained7b-time.log) are retained. These settings
differ from the initial full-training recipe (`H=8`, `K=64`, rank 16). Two updates
are an adaptation/serialization check, not evidence of a trained task policy.

## RTX 3090 feasibility

A subsequently free RTX 3090 (24 GiB) completed 32 optimizer steps: two initial
steps and 30 after checkpoint restoration. The configuration uses the full
pretrained OpenVLA-7B backbone, two 224-pixel cameras, `H=8`, `K=64`, LoRA rank 16,
BF16, batch size one and accumulation 16. Only the first two validation batches
were evaluated in this bounded probe; the 12-run research study has not started.

Peak PyTorch allocated memory was 14.87 GiB and peak reserved memory was
15.07 GiB. The restored segment averaged 8.68 seconds per optimizer step.
Extrapolating its optimization time gives approximately 48.2 GPU hours per
20,000-step run, excluding startup, validation, checkpoint I/O and simulation.
This is a planning estimate from a short run. It does not demonstrate
convergence, task success or a speedup against another policy.

The exported bundle reloaded on GPU and completed 32 timed predictions over
16 held-out observations after five warmups. End-to-end prediction for an
eight-action chunk measured 200.9 ms median / 206.3 ms P95 on this configuration.
These are absolute latency observations from the short-trained model.

The [GPU report](evidence/gpu3090-validation.json) retains resolved configuration,
training/validation rows, checkpoint identity and deployment benchmark. The
bounded run uses its own output directory and does not replace any study run.
The [server application](proposal/server-application.pdf) uses these observations
for its resource request; an [editable Word copy](proposal/server-application.docx)
is available alongside it.

## First controlled 1,000-step pilot

Both seed-42 jobs completed step 1,000, checkpoint saving and full validation.
The [retained pilot records](evidence/pilot1000-results.json) include both resolved
configurations, source identities, complete logged metrics and checkpoint
manifests. Data, source, backbone, inputs and optimizer schedule match; the head
type and its applicable Brier term differ. Neither job performed closed-loop
evaluation. These are one-seed early-training results.

| Measurement | JointAct | Same-backbone regression |
| --- | --- | --- |
| Full-validation normalized action L1 (lower is better) | 0.1772293 | 0.1793227 |
| Valid evaluated action elements | 238,420 | 238,420 |
| Training windows consumed | 16,000 | 16,000 |
| Optimization GPU hours, summing both invocations | 2.4478 | 2.4401 |
| Peak PyTorch reserved memory, GiB | 15.0742 | 15.0723 |

The full validation covers 35 episodes / 4,380 observation windows, with padded
action elements masked out. JointAct's relative L1 reduction is 1.17%. A single
seed and this small difference do not establish a reliable advantage. The
overlapping action elements are not independent statistical trials. Each model
has consumed about 0.37 training epochs; 500 of the 1,000 steps were warmup.

The periodic validator uses the first 100 validation windows. At steps
100/500/1,000 its L1 values were 0.3141/0.1640/0.1414 for JointAct and
0.2384/0.1744/0.1562 for regression. Those trajectories describe a fixed narrow
subset, not full-validation learning curves; the final full-validation values
in the table are the comparison result.

JointAct's full-validation mode accuracy is 46.55%, with 62 of 64 modes predicted.
Predicted prototypes alone give L1 0.18933; applying their residual corrections
reduces it to 0.17723. Supplying the demonstrated mode to the residual head gives
diagnostic L1 0.13207, which is not deployable and is not a task-success bound.
This gap motivates investigating mode assignment/selection and the discrepancy
between demonstrated-mode training and predicted-mode inference. Residual
saturation is zero at this checkpoint; the observed error does not point to
the residual bound as the immediate bottleneck.

The next useful evidence is a paired LIBERO closed-loop check, followed by a
longer matched training budget if warranted. No further training or evaluation
was automatically launched while collecting these completed pilot results.

## Prototype geometry follow-up

An action-only CPU diagnostic over all 4,380 validation windows checks a mismatch
in the current implementation: mode assignment minimizes masked squared distance,
whereas action supervision and reporting use masked L1. Their nearest-prototype
labels differ on 10.57% of windows. Choosing the nearest prototype by L1 instead
reduces oracle prototype L1 from 0.144594 to 0.143363 (0.85%); this is not a trained
policy gain. In 11.35% of windows, the second-smallest squared distance is within
10% of the smallest. That threshold is descriptive, not a selected training
hyperparameter. The [raw diagnostic](evidence/prototype-geometry-diagnostic.json)
retains the exact definitions.

The measured metric mismatch is secondary to the larger demonstrated-mode versus
predicted-mode residual gap. The original pilot conditioned the residual only on
the demonstrated mode. A candidate intervention is to additionally supervise
nearby predicted modes, retaining demonstrated-mode supervision and excluding
incompatible distant modes. Geometry-aware classification supervision is a
separate candidate ablation. The subsequent implementation is documented in [the alignment study](alignment-study.md).
This diagnostic itself does not establish an improvement from either intervention.

## Action-aware objectives and controlled continuation (2026-10-08)

The selected-mode residual and fixed-prototype expected-cost objectives are now
implemented as separate switches. The [study protocol](alignment-study.md) gives
the equations, eligibility gate, initial untuned weights, inspected upstream code,
and five-way ablation. Runtime checks were performed on `WHU140piROOT`.

- Final source: **38 tests passed**, Ruff passed, and two-rank CPU DDP preserved
  **47 trainable tensors bitwise** across interruption and continuation. Rank-zero
  failures and nonfinite losses propagated to all ranks. The new tests cover
  geometric gating, padding, classifier cost gradients, a single backbone call,
  exact resume versus an explicit objective fork, and conditional-error aggregation.
- Real pretrained **OpenVLA 7B**, BF16, RTX 3090: both objectives enabled,
  checkpoint **1,000 → 1,008**, **128** new training windows, **72.19 seconds**
  measured optimization time and **15.074 GiB** peak framework-reserved memory.
  Periodic validation covered 100 uniformly spaced windows across all 4,380
  validation windows; a checkpoint and parent lineage were saved successfully.
  This is a bounded engineering check, not a matched effectiveness comparison.
- Inspection of that GPU log exposed a conditional-metric aggregation issue:
  empty eligible microbatches diluted `alignment_eligible_l1`. Its original value
  is retained and annotated in the raw probe report. The final implementation
  aggregates error sums and eligible element counts before division, using null
  when there are no eligible elements. This changes logging, not gradients.
  The final CPU suite and DDP check were run after that fix. The GPU probe's earlier
  source is retained as `study-alignment/source-probe.tar.gz` on the server.
- The final five-run continuation plan is prepared at
  `/tmp/jointact-1019/study-alignment-v2/study.json`, bound to the final source,
  configurations, data, codebook and matched 1,000-step parent checkpoints.
  **No formal continuation runs or closed-loop efficacy evaluations have been
  executed for this final plan.** The original pilot checkout and results remain.

Evidence: [validation summary](evidence/alignment-validation.json),
[test log](evidence/alignment-pytest.log), [DDP report](evidence/alignment-ddp.json),
[GPU probe](evidence/alignment-gpu-probe.json),
[final source manifest](evidence/alignment-source.json), and
[prepared study snapshot](evidence/alignment-study.json).
The snapshot's five configuration files are stored under `evidence/alignment-configs/`;
the executable server plan retains its original `configs/` relative directory.

## Reproduction and provenance

Run `pytest`, `ruff check src tests scripts`, `python -m build`, and the
scripts `smoke.py`, `ddp_smoke.py` and `server_smoke.py` on the validation host.
`create_random_openvla.py` constructs the explicitly random upstream checkpoint.
`robot_data_smoke.py` runs bounded real-data adaptation through public CLI commands.
See [installation](installation.md), [data](data.md) and [LIBERO](libero.md)
for external dependencies and commands.
The [remote workspace map](remote-workspace.md) lists the actual interpreters,
model/data directories and renderer settings for continuing on the authorized host.

[source-manifest.json](evidence/source-manifest.json) records file hashes at
evidence collection. Earlier checks can precede later documentation and
integration-script edits; the manifest is an inventory, not a claim that every
historical run executed that exact full tree. Evidence replaces the remote
working-directory and home prefixes with `${REMOTE_WORKDIR}` and `${REMOTE_HOME}`;
reported outcomes, counts, metrics and hashes are preserved.
