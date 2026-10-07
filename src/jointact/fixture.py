"""Synthetic image-conditioned episodes solely for engineering verification."""

from pathlib import Path

import numpy as np

from jointact.data.schema import write_episode, write_manifest
from jointact.utils import atomic_json


def create_fixture(output, episodes=12, length=16, image_size=32, seed=42, cameras=2):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Fixture output must be a fresh directory")
    if episodes < 4 or length < 2 or cameras < 1 or image_size < 8:
        raise ValueError("Fixture needs >=4 episodes, >=2 timesteps, >=8 image pixels and >=1 camera")
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(episodes):
        actions = rng.normal(0, 0.02, (length, 7)).astype(np.float32)
        direction = 1 if index % 2 else -1
        actions[:, 0] = direction * 0.1
        actions[:, -1] = float(index % 2)
        proprio = rng.normal(0, 0.1, (length, 8)).astype(np.float32)
        images = np.zeros((length, cameras, image_size, image_size, 3), np.uint8)
        images[..., 0 if direction > 0 else 1] = 220
        images[:, :, image_size // 4:image_size // 2, :, 2] = 120
        split = "val" if index == episodes - 2 else "test" if index == episodes - 1 else "train"
        rows.append(write_episode(output, f"fixture-{index:04d}", images, actions, proprio,
                                  "move right" if direction > 0 else "move left", split,
                                  dict(format="synthetic_engineering_fixture", seed=seed),
                                  [f"camera-{i}" for i in range(cameras)]))
    write_manifest(output, rows)
    atomic_json(output / "provenance.json", dict(purpose="engineering_verification_only", seed=seed))
    return dict(episodes=episodes, frames=episodes * length, research_evidence=False)
