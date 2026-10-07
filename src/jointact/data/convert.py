"""Import LIBERO HDF5 or raw modified-LIBERO TFDS episodes."""

import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from jointact.data.schema import ACTION_SEMANTICS, episode_split, write_episode, write_manifest
from jointact.utils import atomic_json, check_disk


def canonical_libero_actions(actions):
    actions = np.asarray(actions, np.float32).copy()
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("LIBERO importer expects [T,7] delta-pose actions")
    # Same convention as OpenVLA's libero_dataset_transform: -1=open,+1=close -> 1=open,0=close.
    actions[:, -1] = 1 - np.clip(actions[:, -1], 0, 1)
    return actions


def _identifier(source: str):
    return hashlib.sha256(source.encode()).hexdigest()[:24]


def _instruction(file, path):
    for owner in (file, file.get("data")):
        if owner is not None and "language_info" in owner.attrs:
            info = owner.attrs["language_info"]
            if isinstance(info, bytes):
                info = info.decode()
            try:
                info = json.loads(info)
                if isinstance(info, dict):
                    return info.get("language_instruction") or info.get("instruction") or info.get("language")
                if isinstance(info, str):
                    return info
            except json.JSONDecodeError:
                return info
    return path.stem.replace("_demo", "").replace("_", " ")


def convert_hdf5(source: str, output: str, seed=42, val_fraction=0.1, test_fraction=0.1,
                 rotate_images=False, max_episodes=None, min_free_gb=2.0, camera_view="both"):
    source, output = Path(source), Path(output)
    if (output / "manifest.jsonl").exists() or (output / "episodes").exists():
        raise FileExistsError("Choose a fresh output directory for conversion")
    files = sorted(source.rglob("*.hdf5")) + sorted(source.rglob("*.h5")) if source.is_dir() else [source]
    rows = []
    for path in files:
        with h5py.File(path, "r") as file:
            if "data" not in file:
                raise ValueError(f"No LIBERO data group in {path}")
            instruction = _instruction(file, path)
            for demo_name in sorted(file["data"], key=lambda name: int(name.rsplit("_", 1)[-1])):
                if max_episodes is not None and len(rows) >= max_episodes:
                    break
                check_disk(output, min_free_gb)
                demo = file["data"][demo_name]
                obs = demo["obs"]
                primary, wrist = obs["agentview_rgb"][:], obs["eye_in_hand_rgb"][:]
                if rotate_images:
                    primary, wrist = primary[:, ::-1, ::-1], wrist[:, ::-1, ::-1]
                images = np.stack([primary, wrist] if camera_view == "both" else [primary], axis=1)
                if "ee_states" in obs:
                    eef = obs["ee_states"][:]
                else:
                    eef = np.concatenate([obs["ee_pos"][:], obs["ee_ori"][:]], -1)
                proprio = np.concatenate([eef, obs["gripper_states"][:]], -1)
                if proprio.shape[1] != 8:
                    raise ValueError("Expected six EEF pose + two gripper state dimensions")
                group = f"{path.relative_to(source) if source.is_dir() else path.name}/{demo_name}"
                rows.append(write_episode(output, _identifier(group), images, canonical_libero_actions(demo["actions"][:]),
                                          proprio, instruction, episode_split(group, seed, val_fraction, test_fraction),
                                          dict(format="libero_hdf5", file=path.name, demo=demo_name, rotate_images=rotate_images),
                                          ["agentview", "wrist"] if camera_view == "both" else ["agentview"]))
    write_manifest(output, rows)
    atomic_json(output / "provenance.json", dict(format="libero_hdf5", source=str(source), seed=seed,
                                               val_fraction=val_fraction, test_fraction=test_fraction,
                                               action_semantics=ACTION_SEMANTICS, num_episodes=len(rows)))
    return dict(episodes=len(rows), frames=sum(r["length"] for r in rows))


def _decode_image(value):
    if isinstance(value, (bytes, np.bytes_)):
        return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
    return np.asarray(value, dtype=np.uint8)


def convert_rlds(source: str, output: str, seed=42, val_fraction=0.1, test_fraction=0.1,
                 primary_key="image", wrist_key="wrist_image", state_key="state", max_episodes=None,
                 min_free_gb=2.0, camera_view="both"):
    # Keep TensorFlow off GPUs and outside the PyTorch training environment.
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    source, output = Path(source), Path(output)
    if (output / "manifest.jsonl").exists() or (output / "episodes").exists():
        raise FileExistsError("Choose a fresh output directory for conversion")
    dirs = [p.parent for p in sorted(source.rglob("dataset_info.json"))]
    if not dirs:
        raise FileNotFoundError("No TFDS dataset_info.json found; pass the downloaded RLDS root")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    rows = []
    for directory in dirs:
        if max_episodes is not None and len(rows) >= max_episodes:
            break
        builder = tfds.builder_from_directory(str(directory))
        if "train" not in builder.info.splits:
            raise ValueError(f"No train split in {directory}; held-out TFDS splits are not imported as training")
        # A bounded prefix lets integration runs fetch only the required TFRecord
        # shards, while retaining the original parent-episode indices.
        split = "train" if max_episodes is None else f"train[:{max_episodes - len(rows)}]"
        dataset = builder.as_dataset(split=split, shuffle_files=False,
                                     read_config=tfds.ReadConfig(skip_prefetch=True, try_autocache=False))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        dataset = dataset.with_options(options)
        for index, episode in enumerate(dataset):
            if max_episodes is not None and len(rows) >= max_episodes:
                break
            check_disk(output, min_free_gb)
            steps = list(episode["steps"].as_numpy_iterator())
            if not steps:
                continue
            first = steps[0]
            lang = first.get("language_instruction", first["observation"].get("language_instruction"))
            if lang is None:
                raise ValueError("Missing language_instruction in TFDS episode")
            instruction = lang.decode("utf-8") if isinstance(lang, bytes) else str(lang)
            group = f"{directory.relative_to(source)}/{index}"
            image_keys = [primary_key, wrist_key] if camera_view == "both" else [primary_key]
            images = np.stack([np.stack([_decode_image(s["observation"][key]) for key in image_keys]) for s in steps])
            state = np.stack([s["observation"][state_key] for s in steps]).astype(np.float32)
            if state.shape[-1] != 8:
                raise ValueError(f"Expected 8D LIBERO state, got {state.shape}")
            rows.append(write_episode(output, _identifier(group), images,
                                      canonical_libero_actions(np.stack([s["action"] for s in steps])), state,
                                      instruction, episode_split(group, seed, val_fraction, test_fraction),
                                      dict(format="modified_libero_rlds", builder=builder.name, episode_index=index),
                                      ["agentview", "wrist"] if camera_view == "both" else ["agentview"]))
    write_manifest(output, rows)
    atomic_json(output / "provenance.json", dict(format="modified_libero_rlds", source=str(source), seed=seed,
                                               val_fraction=val_fraction, test_fraction=test_fraction,
                                               max_episodes=max_episodes,
                                               action_semantics=ACTION_SEMANTICS, num_episodes=len(rows)))
    return dict(episodes=len(rows), frames=sum(r["length"] for r in rows))
