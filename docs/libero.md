# LIBERO closed-loop evaluation

## Install

Install LIBERO in a dedicated environment compatible with the model runtime:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install -e third_party/LIBERO
pip install robosuite==1.4.1 mujoco==2.3.7 bddl==1.0.1 easydict cloudpickle gym==0.26.2 \
  'numpy<2' 'opencv-python<4.12' 'imageio[ffmpeg]'
```

Record the LIBERO Git commit in evaluation evidence. MuJoCo offscreen rendering needs a working EGL or OSMesa backend. `MUJOCO_GL=osmesa` supports CPU software rendering when the OS provides OSMesa; `MUJOCO_GL=egl` uses the selected rendering device. Avoid rendering on occupied GPUs during validation.

Initialize LIBERO's paths before importing it in an unattended job:

```bash
python scripts/setup_libero.py --source third_party/LIBERO \
  --config-dir /path/to/libero-config --datasets /path/to/libero-datasets
export LIBERO_CONFIG_PATH=/path/to/libero-config
```

The evaluator resolves BDDL and initial-state paths using the benchmark's configuration. Initial-state `.pruned_init` assets are NumPy pickles: the evaluator explicitly uses `weights_only=False` for these trusted installed benchmark files to support PyTorch 2.6+. It records their SHA256 in the evaluation configuration. Install benchmark assets from the pinned upstream checkout; this exception does not apply to model checkpoints, which load with `weights_only=True`.

## Evaluate

```bash
MUJOCO_GL=egl jointact evaluate-libero --checkpoint checkpoints/experiment \
  --device cuda:0 --precision bf16 --suite libero_spatial --trials 50 \
  --seed 42 --execute-horizon 8 --output runs/experiment/eval-spatial-seed42
```

For an integration check, `--task-ids 0 --trials 1 --max-steps 5` exercises an actual simulator with a trained or untrained policy. A short rollout is not a benchmark result. Full default time limits are Spatial 220, Object 280, Goal 300, Long 520 steps. The optional LIBERO-90 limit is explicitly 400 in our implementation and is not a reproduction claim for another protocol.

Each trial starts from a distinct indexed benchmark initial state, performs 10 settling no-op steps, resets the chunk controller, and runs until success or timeout. Camera RGBs rotate 180 degrees to match the OpenVLA convention. State is EEF position, axis-angle and gripper positions. Gripper commands convert from canonical `open01` to LIBERO `-1=open,+1=close`.

## Records

- `config.json`: seed, task IDs, suite, limits, executed horizon and full policy config.
- `episodes.jsonl`: one completed trial per row, success, steps, number of policy
  decisions, timing per decision and per control step, and optional video path.
- `summary.json`: completed flag, successes/trials, overall and per-task success rates, Wilson intervals.
- Optional task/trial MP4s when `--video` is enabled.

Errors are raised; they do not become fabricated failures or successful trials. If a run stops, the completed trial log remains and no completed summary is emitted. Choose a new output directory for another run. Multiple-seed reports should preserve every trial record and distinguish between-seed variation from the binomial interval.

Decision timing includes preprocessing and the full policy call; control-step timing
also includes inexpensive queued actions between replans. Compare decision timing
with the same execution horizon. The standalone `benchmark` command additionally
performs explicit CUDA synchronization and repeated warmup for latency studies.

## Baselines

Use the pinned OpenVLA-OFT reference checkout for its released checkpoints and official evaluation path. Our internal continuous regression head controls for output representation with our backbone adapter; its results must not be labeled as OpenVLA-OFT. Keep camera/proprio inputs, data transformations, horizon, evaluation initial states and precision explicit for all methods.
