# Installation and resource planning

## Core CPU environment

```bash
conda create -n jointact-core python=3.10 -y
conda activate jointact-core
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev,serve,video]'
jointact doctor --path /path/to/workspace
OMP_NUM_THREADS=2 pytest
```

The core package does not import TensorFlow, LIBERO, Transformers or PEFT unless the corresponding command needs them. A newer PyTorch can run the core pipeline; actual tested versions are listed in `validation.md`.

## OpenVLA environment

The compatibility target follows OpenVLA's original HF implementation. JointAct implements its own direct readout and uses ordinary causal attention, so the OFT bidirectional-attention fork is not required.

```bash
conda create -n jointact-vla python=3.10 -y
conda activate jointact-vla
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[openvla,dev,serve,video]'
```

`transformers==4.40.1`, `tokenizers==0.19.1`, `timm==0.9.10`, and `peft==0.11.1` are pinned in the extra. Set `HF_HOME` to a location with enough free space **before** model loading. A 7B BF16 checkpoint is roughly 15 GB on disk; downloads, conversion, data and checkpoints need additional space. Do not use a nearly full home filesystem for the cache.

```bash
export HF_HOME=/path/to/large-disk/huggingface
export TMPDIR=/path/to/large-disk/tmp
jointact doctor --path "$HF_HOME"
```

Model loading uses the explicitly configured repository's remote Python implementation. Pin a verified commit in `model.revision` to make it reproducible. The resolved commit is captured in the training configuration if HF supplies it. LoRA bundles require that same frozen base checkpoint. Offline use requires the base and processor already cached; set `HF_HUB_OFFLINE=1`.

Batch size 1, frozen visual encoders/projector, LoRA rank 16, BF16 and gradient
checkpointing fit on a tested RTX 3090 24 GiB with two 224-pixel cameras, `H=8`
and `K=64`: a 32-step probe peaked at 14.87 GiB allocated / 15.07 GiB reserved.
See [the measured configuration and limits](validation.md#rtx-3090-feasibility).
Fit still depends on prompt length, camera count, software, image resolution and
batch size. The whole base model is placed on each DDP rank; DDP does not shard
a 7B model. Multi-card training increases throughput, while model sharding
would require a separate implementation.

## Data conversion environment

TensorFlow is isolated to the converter. Conversion intentionally disables CUDA.

The data extra pins `tensorflow-metadata==1.15.0`: newer generated protobuf
modules are incompatible with TensorFlow 2.15's supported protobuf runtime.

```bash
conda create -n jointact-data python=3.10 -y
conda activate jointact-data
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[data]'
jointact convert-rlds --source /path/to/modified_libero_rlds --output /path/to/canonical-libero
```

Raw HDF5 conversion requires no TensorFlow. `scripts/download_libero.py` downloads public modified LIBERO data after checking free space, into an explicitly selected destination. It does not download models or start training.

## Remote validation

`scripts/remote_sync.sh HOST REMOTE_DIR` transfers source only. Set `JOINTACT_PYTHON` on the remote machine to an environment interpreter. Run checks through SSH, for example:

```bash
scripts/remote_sync.sh HOST /path/to/remote/jointact
ssh HOST 'cd /path/to/remote/jointact && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /path/to/python -m pytest -q'
```

Inspect `nvidia-smi` and `df -h` immediately before GPU jobs. The built-in guard
refuses GPUs with another compute process, more than 1 GiB allocated or utilization
above 10%, unless sharing is explicitly enabled. Low memory usage alone is not
evidence that a GPU is free. It never terminates another process or deletes
another user's cache.
