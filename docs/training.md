# Training, continuation and checkpoints

Configurations are strict YAML. Unknown keys raise errors. `configs/tiny.yaml` exercises CPU training, `configs/libero_joint.yaml` adapts OpenVLA, and `configs/libero_regression.yaml` provides the same-backbone regression control.

For OpenVLA, `lora_rank: 0` enables full decoder finetuning and has a much larger
memory/optimizer cost. The unused vocabulary projection stays frozen when its
weights are untied, since the policy reads decoder features directly. Tied
input/output embeddings remain trainable through their input-embedding use.

## Lifecycle

```bash
jointact validate-data --root data/libero
jointact prepare --root data/libero --output data/libero/artifacts --horizon 8 --num-modes 64
jointact train --config configs/libero_joint.yaml
jointact train --config configs/libero_joint.yaml --resume runs/libero_joint/checkpoints
```

Trainable parameters use AdamW, warmup and cosine decay to 10% of the configured learning rate. Loss is divided by accumulation count; DDP averages gradients. Gradient norms are clipped after AMP unscaling. A nonfinite loss or gradient raises an error rather than silently advancing a scheduler over an invalid update. FP32, BF16 and CUDA FP16 are supported; CUDA FP16 uses a gradient scaler.

Validation runs without augmentation, uses the predicted mode, and reports masked physical-representation-normalized action L1 plus mode accuracy/NLL/Brier/ECE. These metrics measure demonstration imitation, not task completion. Closed-loop task results come from the simulator evaluator.

## Checkpoint content

```text
run/
  config.yaml
  run.json
  data-audit.json
  metrics.jsonl
  checkpoints/
    latest.json
    step-00001000/
      manifest.json
      config.yaml
      artifacts.json
      weights.pt
      training.pt
      processor/          # for pretrained OpenVLA
```

`weights.pt` contains trainable parameters and persistent buffers; frozen base weights remain upstream. Loading it uses PyTorch's restricted `weights_only=True` mode and verifies SHA256. `training.pt` includes optimizer, scheduler, scaler, consumed-sample cursor and per-rank RNG states; it is pickle-based and must come from a trusted local training run.

Checkpoint directories are written atomically before `latest.json` changes. Automatic retention deletes only recognized checkpoint directories inside the current run, retaining `keep_checkpoints`. The save estimates state size and preserves the configured free-disk reserve. A failed disk check leaves earlier checkpoints intact.

## Exact continuation

`--stop-after N` intentionally saves at N without changing the original LR schedule or `max_steps`. Resume checks the model, artifacts, seed, optimizer schedule, loss weights, data/augmentation config, batch size, accumulation and world size. It permits output-path and logging-frequency changes.

The sampler recreates the same per-epoch permutation and stores consumed samples, rather than a data-loader prefetch cursor. RNG state is restored separately on each rank. `workers: 0` is required for the verified bitwise CPU guarantee; asynchronous worker augmentation is not exactly reproducible from the stored main-process RNG alone. CUDA kernel determinism depends on the stack; numerical closeness must be checked separately.

Distributed data is padded to equal per-rank lengths. At most `world_size - 1` samples repeat at an epoch boundary. There is no cross-episode chunking. A partial final minibatch has its own per-element mean; it is not silently dropped.

## Debugging

- Artifact mismatch: prepare a new artifact directory after changing the manifest, horizon or mode count.
- No validation episodes: use more source episodes or a deliberate group assignment, and retain the split manifest.
- OOM: reduce camera count/input length/batch size, freeze large components and use gradient checkpointing. Increase accumulation to preserve effective batch size.
- Disk refusal: choose a path with adequate free space; do not lower the reserve as a substitute for checking expected model/cache sizes.
- Probability quality: calibration is fitted only on held-out validation mode labels. It does not add execution-success supervision.
