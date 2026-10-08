# JointAct

JointAct converts pretrained vision-language-action features into a direct distribution over **complete low-level action chunks**, followed by a mode-conditioned continuous correction. The deployed policy uses one backbone forward pass. Modes represent coordinated robot commands, not named high-level skills.

The repository includes OpenVLA adaptation, episode-preserving LIBERO import, train-only normalization and codebook fitting, single-device and DDP training, exact continuation, offline teacher scoring, held-out mode calibration, inference bundles, HTTP serving, latency measurement, and closed-loop LIBERO evaluation. A small image-conditioned backbone exercises the same lifecycle without downloading foundation-model weights.

Controlled multi-seed head studies, residual diagnostics, failure propagation,
resumable LIBERO trials and paired result comparisons are described in the
[research workflow](docs/research-workflow.md). These facilities prepare and
measure experiments; completed training and task-performance evidence are
recorded separately in the validation log.

This is research software. The synthetic fixture checks engineering behavior; it does not establish robot success or a performance improvement over OpenVLA-OFT. See [validation evidence](docs/validation.md) for what has actually run.

## Install

Python 3.10 or 3.11 is recommended. Use a dedicated environment.

```bash
pip install -e '.[dev,serve]'
jointact doctor --path /path/to/workspace
```

For pretrained OpenVLA, follow the pinned environment in [installation](docs/installation.md), then install `.[openvla]`. TensorFlow data conversion uses a separate environment. Tests and examples can run on CPU; GPU commands refuse an occupied GPU unless sharing is explicitly enabled.

## Run the full engineering example

```bash
jointact fixture --output data/fixture
jointact prepare --root data/fixture --output data/fixture/artifacts \
  --horizon 4 --num-modes 8 --max-samples 1000 --iterations 10
OMP_NUM_THREADS=2 jointact train --config configs/tiny.yaml
jointact evaluate-offline --checkpoint runs/tiny/checkpoints \
  --output runs/tiny/test.json
jointact export --checkpoint runs/tiny/checkpoints --output runs/tiny/bundle
```

The default example has 40 optimizer steps. It writes a resolved configuration, dataset audit, trainable tensors, optimizer/scheduler/AMP and sampler state, per-step metrics, and validation metrics. It retains the last three checkpoints. For an intentional early stop that preserves the planned LR schedule, use `--stop-after 20`; continue with `--resume runs/tiny/checkpoints`.

## Train with public robot demonstrations

The first real-data route uses [modified LIBERO RLDS](https://huggingface.co/datasets/openvla/modified_libero_rlds), also used by OpenVLA and OpenVLA-OFT. Existing actions provide all first-stage labels. Human annotation and simulator branch outcomes are not prerequisites.

```bash
# In the data-conversion environment:
jointact convert-rlds --source /path/to/modified_libero_rlds --output data/libero
# Or import the original LIBERO HDF5 demonstrations:
jointact convert-hdf5 --source /path/to/libero/datasets --output data/libero

jointact validate-data --root data/libero
jointact prepare --root data/libero --output data/libero/artifacts \
  --horizon 8 --num-modes 64 --max-samples 100000

# In the OpenVLA training environment; select an actually free GPU:
CUDA_VISIBLE_DEVICES=0 jointact train --config configs/libero_joint.yaml
```

Edit paths and the pinned pretrained revision in the YAML before training. Train/validation/test are assigned to **whole parent episodes before chunking**. Normalizers and prototypes use training episodes only. The converter does not overwrite datasets, and preparation does not overwrite action artifacts. See [data contracts](docs/data.md) for image orientation, action units, gripper polarity, splits and normalization.

For distributed training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m jointact.cli train --config configs/libero_joint.yaml
```

Effective full batches equal `batch_size × grad_accumulation × world_size`. The last partial batch in an epoch can be smaller. Keep training budget and sampler topology fixed for exact continuation. The CPU fixture verifies exact resume with zero data-loader workers; worker prefetch prevents a bitwise guarantee when `workers > 0`.

## Predict and serve

```python
import numpy as np
from jointact.inference import PolicyRuntime

policy = PolicyRuntime("runs/tiny/bundle", device="cpu")
result = policy.predict({
    "images": ["agentview.png", "wrist.png"],
    "proprio": np.zeros(8, dtype=np.float32),
    "instruction": "pick up the cup",
})
actions = result["actions"]  # [horizon, action_dim], physical commands
```

`mode_probabilities` is a behavior-mode distribution. It is not the probability that an action will successfully finish a task. The gripper convention is `1=open, 0=close`; the LIBERO adapter performs the conversion to its environment commands.

```bash
jointact serve --checkpoint runs/tiny/bundle --device cpu --port 8000
jointact evaluate-libero --checkpoint runs/libero_joint/checkpoints \
  --device cuda:0 --precision bf16 --suite libero_spatial \
  --trials 50 --output runs/libero_joint/eval-spatial
```

See [deployment](docs/deployment.md) for JSON examples, batch requests, chunk execution, calibration and benchmark scope; [LIBERO evaluation](docs/libero.md) covers simulator setup, trial records and videos.

## Method and comparisons

For context `x`, prototypes `C_k ∈ R^(H×D)` and a learned mode-conditioned residual:

```text
h = pretrained_backbone(images, instruction, proprio, joint_readout_token)
p = softmax(mode_head(h))
k = argmax(p)
action_chunk = C[k] + bounded_residual(h, k)
```

Training assigns each demonstrated chunk to its closest prototype using only valid timesteps. The objective is mode cross-entropy + weighted Brier loss + masked residual L1, with optional teacher KL. Inference uses the predicted mode, including during offline validation. Continuous regression with the same backbone is provided in `configs/libero_regression.yaml` for an output-head ablation. It is not labeled as a reproduction of OpenVLA-OFT.

The OpenVLA adapter retains its visual encoders, projector and Llama decoder, and adds a joint readout token and proprioceptive input. The action head is newly trained: replacing autoregression does not automatically preserve motor control. Details and limits are in [architecture](docs/architecture.md) and [experiments](docs/experiments.md).

## Development

```bash
pytest
ruff check src tests
python -m build
```

CI exercises the core CPU lifecycle. External-model and simulator checks are separate integration evidence. No benchmark result is populated from synthetic tests.

For a bounded check on converted robot demonstrations and a downloaded OpenVLA
checkpoint, run this on the validation machine:

```bash
OMP_NUM_THREADS=2 python scripts/robot_data_smoke.py \
  --root /path/to/canonical-libero --pretrained /path/to/openvla-7b \
  --base-role pretrained --precision bf16 --output /path/to/fresh-check
```

It forces CPU execution, performs two optimizer steps with continuation, checks
LoRA updates, exports a bundle and reloads it for prediction. This checks the model
and real-data interfaces; task success requires a separate full closed-loop run.

## Attribution

OpenVLA and OpenVLA-OFT informed preprocessing conventions and integration. Jev-style open implementations informed the decision readout interface. See [third-party provenance](THIRD_PARTY.md) and [upstream pins](upstream.lock.json). Original code is MIT licensed. Pretrained models, datasets and optional simulators retain their respective licenses.

The [action-aware supervision study](docs/alignment-study.md) implements gated
predicted-mode residual supervision and expected prototype-cost supervision as
independent ablations. It supports controlled continuation from existing checkpoints
and does not change the inference architecture.

## Migration package

The private GitHub release `migration-2026-10-08` contains the complete prepared
Spatial dataset and saved training states as checksummed, chunked attachments.
See the [migration guide](docs/migration/README.md) for restoration, pinned base-model
download, environment setup and continuation on another GPU server.
