"""Atomic resumable checkpoints and portable inference bundles."""

import json
import os
import platform
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from jointact.config import Config
from jointact.models import JointActionPolicy
from jointact.utils import atomic_json, check_disk, sha256


def trainable_state(model):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    names.update(name for name, _ in model.named_buffers())
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if key in names}


def load_trainable(model, state):
    expected = set(trainable_state(model))
    if set(state) != expected:
        raise ValueError(f"Checkpoint trainable/buffer keys mismatch; missing={expected-set(state)}, extra={set(state)-expected}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint tensors: {result.unexpected_keys}")


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(directory, model, optimizer, scheduler, scaler, sampler, step, config,
                    metadata, rank_states, keep=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = trainable_state(model)
    # Adam moment tensors + model tensors + margin. Keep the configured free-space reserve after writing.
    estimate = sum(t.numel() * t.element_size() for t in state.values()) * 4 / 1024**3
    check_disk(directory, config.train.min_free_disk_gb + estimate)
    path = directory / f"step-{step:08d}"
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    temporary = Path(tempfile.mkdtemp(dir=directory, prefix=".checkpoint-"))
    try:
        torch.save(state, temporary / "weights.pt")
        torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict(), step=step, sampler=sampler.state_dict(), rank_states=rank_states),
                   temporary / "training.pt")
        config.save(temporary / "config.yaml")
        atomic_json(temporary / "artifacts.json", metadata)
        if model.processor is not None:
            model.processor.save_pretrained(temporary / "processor")
        atomic_json(temporary / "manifest.json", dict(format="jointact-checkpoint-v1", step=step,
                    weights_sha256=sha256(temporary / "weights.pt"), torch_version=torch.__version__,
                    training_sha256=sha256(temporary / "training.pt"),
                    config_sha256=sha256(temporary / "config.yaml"), artifacts_sha256=sha256(temporary / "artifacts.json"),
                    python_version=platform.python_version(), base_model=config.model.pretrained,
                    base_revision=config.model.revision))
        os.replace(temporary, path)
        atomic_json(directory / "latest.json", dict(path=path.name, step=step))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    checkpoints = sorted(directory.glob("step-*"))
    for old in checkpoints[:-keep]:
        if (old / "manifest.json").is_file():
            shutil.rmtree(old)
    return path


def resolve_checkpoint(path):
    path = Path(path)
    if (path / "latest.json").is_file():
        with open(path / "latest.json", encoding="utf-8") as stream:
            child = json.load(stream)["path"]
        if Path(child).name != child:
            raise ValueError("Invalid latest checkpoint path")
        path = path / child
    return path


def read_weights(path):
    path = resolve_checkpoint(path)
    with open(path / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest["format"] != "jointact-checkpoint-v1":
        raise ValueError("Unsupported checkpoint format")
    for name in ("config", "artifacts"):
        suffix = "yaml" if name == "config" else "json"
        if f"{name}_sha256" in manifest and manifest[f"{name}_sha256"] != sha256(path / f"{name}.{suffix}"):
            raise ValueError(f"Checkpoint {name} checksum failed")
    if manifest["weights_sha256"] != sha256(path / "weights.pt"):
        raise ValueError("Checkpoint weights checksum failed")
    return path, torch.load(path / "weights.pt", map_location="cpu", weights_only=True)


def load_policy(path, device="cpu", dtype=torch.float32):
    path, weights = read_weights(path)
    config = Config.load(path / "config.yaml")
    with open(path / "artifacts.json", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if config.model.head_type == "joint":
        prototypes = weights["head.prototypes"].numpy()
    else:
        prototypes = np.zeros((config.model.num_modes, config.model.horizon, config.model.action_dim), np.float32)
    model = JointActionPolicy(config.model, prototypes, device, dtype,
                              str(path / "processor") if (path / "processor").is_dir() else None)
    load_trainable(model, weights)
    model.eval()
    return model, config, metadata


def export_bundle(checkpoint, output):
    source, _ = read_weights(checkpoint)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Bundle already exists: {output}")
    check_disk(output, 0.2 + (source / "weights.pt").stat().st_size / 1024**3)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=".export-"))
    try:
        for name in ("weights.pt", "config.yaml", "artifacts.json", "manifest.json"):
            shutil.copy2(source / name, temporary / name)
        if (source / "processor").is_dir():
            shutil.copytree(source / "processor", temporary / "processor")
        calibration = Path(checkpoint) / "calibration.json"
        if not calibration.exists():
            calibration = source / "calibration.json"
        if calibration.exists():
            with open(calibration, encoding="utf-8") as stream:
                calibration_data = json.load(stream)
            if calibration_data["weights_sha256"] != sha256(source / "weights.pt"):
                raise ValueError("Calibration belongs to different model weights")
            shutil.copy2(calibration, temporary / "calibration.json")
        atomic_json(temporary / "bundle.json", dict(format="jointact-inference-v1", base_weights_included=False,
                    note="OpenVLA bundles require the pinned base checkpoint; tiny bundles are self-contained."))
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(output)
