"""Canonical episode format: uint8 RGB, physical actions, explicit semantics."""

import json
import os
from pathlib import Path

import h5py
import numpy as np

from jointact.utils import sha256

SCHEMA_VERSION = 1
ACTION_SEMANTICS = "eef_delta_xyz_axisangle_gripper_open01"


def read_manifest(root: str | Path) -> list[dict]:
    root = Path(root)
    with open(root / "manifest.jsonl", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    seen = set()
    for row in rows:
        if row["schema_version"] != SCHEMA_VERSION or row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid schema/split: {row}")
        if row["episode_id"] in seen:
            raise ValueError(f"Duplicate episode: {row['episode_id']}")
        seen.add(row["episode_id"])
        target = (root / row["path"]).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError(f"Invalid episode path: {row['path']}")
        if row["length"] < 1:
            raise ValueError("Episodes cannot be empty")
    return rows


def episode_split(group_id: str, seed: int = 42, val_fraction: float = 0.1, test_fraction: float = 0.1) -> str:
    """Stable parent-group assignment, before windows are constructed."""
    import hashlib
    if not 0 <= val_fraction + test_fraction < 1 or min(val_fraction, test_fraction) < 0:
        raise ValueError("Invalid split fractions")
    value = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "test" if value < test_fraction else "val" if value < test_fraction + val_fraction else "train"


def write_episode(root: Path, episode_id: str, images: np.ndarray, actions: np.ndarray,
                  proprio: np.ndarray, instruction: str, split: str, source: dict,
                  camera_names: list[str], action_semantics: str = ACTION_SEMANTICS) -> dict:
    actions, proprio = np.asarray(actions, np.float32), np.asarray(proprio, np.float32)
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 5 or images.shape[-1] != 3:
        raise ValueError("images must be uint8 [T,V,H,W,3]")
    if actions.ndim != 2 or proprio.ndim != 2 or len(actions) != len(images) or len(proprio) != len(images):
        raise ValueError("actions/proprio/images must have matching timesteps")
    if not np.isfinite(actions).all() or not np.isfinite(proprio).all() or not instruction.strip():
        raise ValueError("Nonfinite action/state or empty instruction")
    if len(camera_names) != images.shape[1]:
        raise ValueError("Camera name count does not match images")
    if not episode_id or Path(episode_id).name != episode_id or episode_id in {".", ".."}:
        raise ValueError("episode_id must be a safe filename")
    directory = root / "episodes"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{episode_id}.h5"
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".h5.tmp")
    try:
        with h5py.File(temporary, "w") as file:
            file.create_dataset("images", data=images, compression="lzf", chunks=(1, *images.shape[1:]))
            file.create_dataset("actions", data=actions)
            file.create_dataset("proprio", data=proprio)
            file.attrs["instruction"] = instruction
            file.attrs["action_semantics"] = action_semantics
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(schema_version=SCHEMA_VERSION, episode_id=episode_id, path=str(path.relative_to(root)),
                length=len(actions), action_dim=actions.shape[1], proprio_dim=proprio.shape[1],
                num_images=images.shape[1], cameras=camera_names, instruction=instruction, split=split,
                action_semantics=action_semantics, source=source, sha256=sha256(path))


def write_manifest(root: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No episodes were converted")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.jsonl"
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {path}")
    with open(path.with_suffix(".jsonl.tmp"), "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(path.with_suffix(".jsonl.tmp"), path)
