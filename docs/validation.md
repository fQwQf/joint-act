# Validation record

Validation date: 2026-10-08. All runtime checks ran through SSH on the authorized
host `WHU140piROOT`, with CUDA hidden. Local work consisted of source edits,
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
check did not invent held-out data. Full-dataset training uses the same split
rule across all source episodes.

Random-checkpoint tests verify upstream model interfaces and serialization,
not pretrained motor capability. A five-step rollout verifies the real
simulator path; its timeout is not a standard LIBERO evaluation result.
Fixture latency, mode probabilities and imitation losses do not establish
robot task success. All retained reports preserve these distinctions.

GPU training, GPU latency, the 24 GiB memory fit, full-suite task performance,
cross-seed variation and improvements over official OpenVLA/OFT remain to be
measured. The default LoRA/BF16 recipe is not a measured VRAM guarantee.

The full-model integration used two 224-pixel cameras, `H=4`, `K=8`, LoRA rank 2,
batch size one and two optimizer steps. It loaded 7,560,446,308 total parameters,
of which 19,209,124 were trainable. The exact [configuration](evidence/pretrained7b-config.json),
[run metadata](evidence/pretrained7b-run.json), [finite training metrics](evidence/pretrained7b-training.json)
and [CPU process timing](evidence/pretrained7b-time.log) are retained. These settings
differ from the initial full-training recipe (`H=8`, `K=64`, rank 16). Two updates
are an adaptation/serialization check, not evidence of a trained task policy.

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
