# Running under the 120 GB whole-host ceiling

All limits below are decimal bytes. Use a host with cgroup v2 and a systemd
user session with delegated memory control, or have the administrator create
the equivalent system scope. Do not run an unbounded fallback on failure.

Put **all project work** in one scope: training, validation, rollout evaluation,
data conversion and checkpoint loading. No separate parallel launcher. The new
queue is serial; dual-job concurrency remains disabled until full-lifecycle RAM
measurements justify a separately reviewed scheduler. One GPU per training run.

```bash
systemd-run --user --scope \
  -p MemoryHigh=88000000000 -p MemoryMax=96000000000 -p MemorySwapMax=0 \
  python scripts/memory_safe_queue.py \
  --queue /absolute/path/queue.json --log /absolute/path/memory.jsonl
```

If the host has no user manager, an administrator can run the same command
without `--user`. The program verifies the current scope's `memory.max` and
`memory.swap.max`; inherited ancestor limits alone are not accepted. A fresh,
dedicated scope provides meaningful `memory.peak` where the kernel supports it.
Choose a lower MemoryMax/MemoryHigh if outside usage is above 14 GB. The entry
check conservatively requires current whole-host estimated usage plus MemoryMax
<=110 GB, without subtracting project cache from MemAvailable-based usage. Keep the
remaining 10 GB below the user's 120 GB ceiling as additional headroom.

Example queue (replace paths with prepared study configs and actual checkpoints):

```json
[
  ["jointact", "train", "--config", "/study/configs/joint-seed42.yaml", "--stop-after", "2000"],
  ["jointact", "train", "--config", "/study/configs/aligned-seed42.yaml", "--stop-after", "2000"]
]
```

Set `CUDA_VISIBLE_DEVICES` before launching the scope, and use the correct venv
on PATH. Extend the queue to all five matched variants. For the next stage pass
`--resume` with each run's actual saved checkpoint and `--stop-after 5000`.
Do not edit max_steps (20,000), world size, or batch topology to shorten a stage.
Keep workers=0 and batch=1/accumulation=16 for the paired continuations.
Use the existing CLI's full validation evaluator between stages; do not use
`run-study` for model selection because it also evaluates the held-out test set.

Before the scientific comparison, use a separate calibration study, stop after
32 additional optimizer steps, and exercise save, resume and validation. Keep
its results separate from the five matched branches. Log loading, training,
checkpoint and evaluation peaks; `memory.current` includes charged file cache,
which matters when loading HDF5 and checkpoints. The queue samples host
MemTotal−MemAvailable and cgroup usage every second and logs memory.events.
Its SIGTERM/SIGKILL cleanup targets only its own child process group. Near memory
pressure it does not request a new checkpoint; resume from the latest completed
atomic checkpoint. Nonzero exit stops the queue.

Linux `memory.max` is the primary project boundary, including child processes;
it can transiently overshoot and trigger a local cgroup OOM. The 110 GB host
watchdog is supplementary and cannot bound unrelated processes. Reserve host
capacity with the server owner; no userspace poll can guarantee a whole-host
ceiling when unrelated jobs allocate memory concurrently. See the
[kernel cgroup v2 memory documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).

The wrapper lives outside `src/jointact`, so it does not invalidate frozen study
source fingerprints. The original migration release remains unchanged. CPU
safety tests cover rejection and queue behavior; the target PRO6000D host must
still validate actual kernel enforcement and model peak memory before training.

Validation on 2026-10-09: five focused CPU tests passed on the existing remote
server, including host-pressure child termination and fail-fast queue behavior;
Ruff passed. An actual unbounded invocation exited with code 2 before launching
its child. Creating a bounded systemd scope required unavailable authorization
on that host, so kernel OOM enforcement was not exercised there. These checks
do not constitute a PRO6000D memory benchmark.
