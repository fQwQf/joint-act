# Remote validation workspace

The authorized host is `WHU140piROOT`. Runtime checks used this workspace:

| Item | Remote path |
| --- | --- |
| Repository | `/tmp/jointact-1019/repo` |
| Model/runtime interpreter | `/tmp/jointact-1019/venv/bin/python` |
| TensorFlow conversion interpreter | `/tmp/jointact-1019/data-venv/bin/python` |
| HF cache | `/tmp/jointact-1019/hf` |
| Verified full pretrained OpenVLA-7B | `/tmp/jointact-1019/pretrained-openvla7b` |
| Small random upstream checkpoint | `/tmp/jointact-1019/random-openvla` |
| Downloaded raw Spatial suite | `/tmp/jointact-1019/raw-libero` |
| Ten converted real episodes | `/tmp/jointact-1019/real-libero` |
| Complete converted Spatial suite and train-only artifacts | `/tmp/jointact-1019/libero-spatial-full` |
| Prepared 12-run head study (two seed-42 pilots completed) | `/tmp/jointact-1019/study-spatial/study.json` |
| Full 7B integration outputs | `/tmp/jointact-1019/pretrained7b-smoke` |
| Small upstream lifecycle outputs | `/tmp/jointact-1019/upstream-smoke` |
| Real-data/small-model lifecycle | `/tmp/jointact-1019/real-data-upstream-smoke` |
| Real LIBERO short-rollout records | `/tmp/jointact-1019/real-data-libero-smoke` |
| Experiment lifecycle checks and resumable LIBERO trials | `/tmp/jointact-1019/maturity-v2` |
| RTX 3090 bounded 32-step training and deployment check | `/tmp/jointact-1019/gpu3090-probe` |
| Pinned LIBERO checkout | `/tmp/jointact-1019/LIBERO` |
| LIBERO path configuration | `/tmp/jointact-1019/libero-config` |

These environments are overlays over an existing conda interpreter. Package
changes stayed inside the overlays. For a fresh machine use the dedicated
environment recipes in [installation](installation.md); exact observed versions
are retained in [validation](validation.md).

The raw dataset directory now contains all sixteen Spatial shards and the two
metadata files from revision `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551` of
`openvla/modified_libero_rlds`. All files were checked against the pinned official
file metadata. The other modified-LIBERO suites have not been downloaded.
The earlier ten-episode integration dataset remains separate.
The complete conversion contains 432 episodes / 52,970 frames, with
351/35/46 train/validation/test episodes. Its normalization and 64-mode,
eight-step codebook use the training partition only.

The prepared study uses the verified OpenVLA-7B base and four head variants with
seeds 42, 43 and 44. Joint and regression seed-42 pilot jobs were launched on
2026-10-08 on physical GPUs 0 and 4, respectively. Each starts from the pretrained
base, saves at step 100, resumes to step 1,000, then evaluates the full validation
partition and stops. The original 20,000-step schedule is retained. The jobs use
`pilot.log`, `pilot-status.json` and `pilot-launch.json` in their respective run
directories; the supervising processes survive the launching SSH connection.
Inspect these records before starting another process for the same run. Both pilots subsequently completed step 1,000 and full validation, with status
`completed`. Their `pilot-val.json` files hold the full results. No subsequent
training or closed-loop evaluation has been launched. The other ten study
configurations have not started. After assigning GPU resources,
run the following from the remote checkout, substituting the allocated GPU IDs:

```bash
CUDA_VISIBLE_DEVICES=0 /tmp/jointact-1019/venv/bin/python -m jointact.cli run-study \
  --plan /tmp/jointact-1019/study-spatial/study.json --device cuda:0
```

For DDP, set the allocated IDs in `CUDA_VISIBLE_DEVICES` and add the corresponding
`--world-size`. Keep this topology unchanged on continuation. The plan is bound
to its source/configuration/data hashes; regenerate it if those inputs change.

Inspect resources and the validated model inventory:

```bash
ssh WHU140piROOT 'df -h /tmp /data1; nvidia-smi'
ssh WHU140piROOT 'cat /tmp/jointact-1019/pretrained-openvla7b/verified.json'
ssh WHU140piROOT 'cat /tmp/jointact-1019/pretrained7b-smoke/report.json'
```

The earlier checks used `CUDA_VISIBLE_DEVICES=` and two CPU threads. Later,
physical GPU 0 became free and completed the bounded RTX 3090 probe. GPU
availability must be rechecked for each new job. Full training should select an allocated GPU and
a disk with adequate space; the host's home filesystem was nearly full during
validation. `/tmp` holds temporary validation artifacts and can be cleared by
host maintenance.

The software-rendered simulator check used these settings on the remote host:

```bash
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_HOME=/tmp/jointact-1019/hf
export HF_HUB_OFFLINE=1
export LIBERO_CONFIG_PATH=/tmp/jointact-1019/libero-config
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/tmp/jointact-1019/osmesa/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
```

OSMesa libraries were extracted into the workspace without changing system
libraries. Start a new evaluation in a fresh directory, or use `--resume` with
the same versioned protocol to finish its pending trials. Retain completed
reports, model/data pins and configuration when moving a run to durable storage.

## Action-aware supervision continuation

The original `/tmp/jointact-1019/repo` and `study-spatial` pilot are retained.
The new implementation is `/tmp/jointact-1019/repo-alignment`. Use
`PYTHONPATH=/tmp/jointact-1019/repo-alignment/src` explicitly with the existing
venv, since its editable install still points to the original source tree.
The five-run continuation plan is `/tmp/jointact-1019/study-alignment-v2/study.json`.
It starts from the matched 1,000-step JointAct and regression checkpoints.

```bash
# Example for an allocated, currently free GPU; absolute stop step is 1,100.
cd /tmp/jointact-1019/repo-alignment
CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/jointact-1019/hf \
PYTHONPATH=/tmp/jointact-1019/repo-alignment/src \
/tmp/jointact-1019/venv/bin/python -m jointact.cli train \
  --config /tmp/jointact-1019/study-alignment-v2/configs/aligned_cost-seed42.yaml \
  --stop-after 1100
```

The generated configuration explicitly names its parent in `train.fork_from`.
For subsequent calls, add `--resume` with that variant's new checkpoint directory. Do not sync new source into the original
pilot checkout. Keep source/data/config hashes attached to each study.

The original eight-step GPU integration check remains in `study-alignment`, with
`source-probe.tar.gz` preserving its source before the conditional-error logging
fix. The finalized `study-alignment-v2` plan is prepared and has no training runs
yet. The command above starts its first continuation segment.
