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
| Downloaded raw Spatial prefix | `/tmp/jointact-1019/raw-libero` |
| Ten converted real episodes | `/tmp/jointact-1019/real-libero` |
| Full 7B integration outputs | `/tmp/jointact-1019/pretrained7b-smoke` |
| Small upstream lifecycle outputs | `/tmp/jointact-1019/upstream-smoke` |
| Real-data/small-model lifecycle | `/tmp/jointact-1019/real-data-upstream-smoke` |
| Real LIBERO short-rollout records | `/tmp/jointact-1019/real-data-libero-smoke` |
| Pinned LIBERO checkout | `/tmp/jointact-1019/LIBERO` |
| LIBERO path configuration | `/tmp/jointact-1019/libero-config` |

These environments are overlays over an existing conda interpreter. Package
changes stayed inside the overlays. For a fresh machine use the dedicated
environment recipes in [installation](installation.md); exact observed versions
are retained in [validation](validation.md).

The raw dataset directory contains the first of sixteen Spatial shards plus
the original full-suite metadata. Use `--max-episodes 10` for the reproduced
prefix conversion, or download the complete suite before an unrestricted
conversion. It is not a full local copy of modified LIBERO.

Inspect resources and the validated model inventory:

```bash
ssh WHU140piROOT 'df -h /tmp /data1; nvidia-smi'
ssh WHU140piROOT 'cat /tmp/jointact-1019/pretrained-openvla7b/verified.json'
ssh WHU140piROOT 'cat /tmp/jointact-1019/pretrained7b-smoke/report.json'
```

All checks used `CUDA_VISIBLE_DEVICES=` and two CPU threads. The GPU allocations
belonged to existing jobs. Full training should select an actually free GPU and
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
libraries. Each rerun needs a fresh output directory. Retain completed reports,
model/data pins and configuration when moving a run to durable storage.
