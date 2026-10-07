# Dataset and action contracts

## Canonical episode format

```text
dataset/
  manifest.jsonl
  provenance.json
  episodes/<episode_id>.h5
  artifacts/
    artifacts.json
    codebook.npz
```

Each HDF5 file holds `images: uint8[T,V,H,W,3]`, `actions: float32[T,D]`, `proprio: float32[T,P]`, plus the instruction and action-semantic attributes. The manifest records split, camera order, dimensions, source identity, length and SHA256. Each frame's observation precedes the corresponding action. Camera order is stable, typically `[agentview, wrist]`.

Current importers support LIBERO's 7D end-effector delta command:

```text
dx, dy, dz, d_axisangle_x, d_axisangle_y, d_axisangle_z, gripper_open01
```

The first six values preserve source controller units. Gripper commands are converted using the OpenVLA LIBERO convention: `1 - clip(raw_gripper, 0, 1)`, where raw `-1` is open and `+1` is close. The canonical result is `1=open, 0=close`. Do not apply the conversion again to already standardized OXE data. The RLDS importer expects the **raw modified-LIBERO TFDS data**, before OpenVLA's OXE transform.

Proprioception is 8D: end-effector position (3), axis-angle orientation (3), and the two gripper positions (2). Joint angles are not substituted for this state. The HDF5 importer reads `ee_states` or `ee_pos`/`ee_ori`, and `gripper_states`. The TFDS importer reads an explicit state key and rejects a state with the wrong dimension.

## Download and conversion

The source-code comparison, verified public episode counts, and rationale for adapting a pretrained VLA are recorded in [pretrained decision models and data](pretrained-decision-data.md).

```bash
python scripts/download_libero.py --output /path/to/raw-libero --min-free-gb 20
jointact convert-rlds --source /path/to/raw-libero --output data/libero
```

The downloader uses the dataset commit in `upstream.lock.json`. Use `--suite spatial`
for the first suite, or `--revision` to explicitly select another source version.
`convert-rlds --max-episodes 10` bounds a conversion check; it reads a prefix of the
original train split and preserves parent indices. A bounded check is not a full
benchmark training set.

`--primary-key image --wrist-key wrist_image --state-key state` match the public raw LIBERO builder. They can be overridden for a verified source variant. Conversion uses only TFDS's original train split, then partitions parent episodes internally. Source-provided held-out splits are never silently turned into training data.

For HDF5:

```bash
jointact convert-hdf5 --source /path/to/libero/datasets --output data/libero
```

Stored official LIBERO demonstration RGBs are expected in policy orientation. `--rotate-images` performs a 180-degree rotation only when a verified source stores raw simulator render orientation. The evaluator always rotates fresh simulator images by 180 degrees, as OpenVLA's evaluator does. Check one converted episode visually before a long run. No-op filtering is not applied by our HDF5 converter; use the public modified dataset or upstream regeneration when that comparison condition matters.

## Splits and leakage

A SHA256-based assignment of `(seed, source parent episode)` creates stable train/val/test splits before any frame or chunk sampling. Defaults are 80/10/10 in expectation. Small datasets may lack validation episodes; training records that fact and omits validation. No synthetic samples are inserted to fill a split.

Split boundaries are by episode, not by task. Test episodes may share task instructions with training episodes. This measures held-out demonstration prediction; task generalization needs separately held-out tasks. Official LIBERO closed-loop evaluation uses the benchmark's initial-state protocol, independently of our offline split.

For a comparison that trains on all public demonstrations, pass
`--val-fraction 0 --test-fraction 0` during conversion and use the benchmark's
separate initial states for closed-loop evaluation. This disables offline
held-out metrics and temperature fitting. If using the default 80/10/10 split,
give each baseline exactly the same training episodes; do not compare it with
an official checkpoint trained on all demonstrations without recording that
data-budget difference. The reported source total of 1,693 episodes is before
our internal split.

Action normalizers, proprio normalizers and codebook use training episodes only. Action/state statistics are bounded-memory reservoir estimates of the 1%/99% quantiles, with seed and sample count retained. Gripper is excluded from action normalization. Constant normalized dimensions map to zero and inverse-map to their physical constant. Continuous predictions are clipped to normalized bounds at deployment.

## Chunk construction

A frame `t` supervises `[a_t, ..., a_(t+H-1)]`. Chunks never cross episode boundaries. Short tails repeat the last action for storage and carry a validity mask; padded values do not affect assignment or loss. Prototype fitting uses complete training chunks only. Increase episode length or reduce horizon if the training dataset has no complete chunks.

K-means uses all `H×D` values jointly. A prototype therefore captures correlations across coordinates and time. `artifacts.json` records quantization MSE and mode occupancy. These are training diagnostics, not closed-loop quality metrics. Artifacts and teacher files are tied to the exact manifest/codebook digests.

## Optional teacher labels

`score-teacher` produces a metadata line followed by `(episode_id, timestep, probabilities)` rows. It scores fixed candidates using the original AR OpenVLA action-token likelihood. Only **horizon 1, one-camera artifacts** are supported, because original OpenVLA is not trained for multi-step chunk likelihoods. A robot-domain `--unnorm-key` is required when the teacher has multiple action-statistic domains.

Teacher distributions describe behavior likelihood over the restricted candidates; they do not establish task success. They are generated on training episodes only. Partial coverage is masked during KL training. Do not relabel random alternative actions as failures. True action success probabilities require executions and outcome evidence.

For the one-camera diagnostic, import with `--camera-view primary`, prepare `--horizon 1`, and use matching model dimensions. Set `data.teacher` to the generated JSONL and a positive `train.distill_weight` to enable distillation. The default two-camera, horizon-eight recipe uses demonstration supervision without the AR teacher.
