"""Fit train-only normalization/codebook and audit all episode boundaries."""

import json
from pathlib import Path

import h5py
import numpy as np

from jointact.data.codebook import fit_codebook
from jointact.data.normalization import Normalizer
from jointact.data.schema import ACTION_SEMANTICS, read_manifest
from jointact.utils import atomic_json, check_disk, sha256


class Reservoir:
    def __init__(self, capacity, seed):
        self.capacity, self.rng, self.seen = capacity, np.random.default_rng(seed), 0
        self.values = []

    def add(self, batch):
        for value in batch:
            self.seen += 1
            if len(self.values) < self.capacity:
                self.values.append(value.copy())
            else:
                target = self.rng.integers(self.seen)
                if target < self.capacity:
                    self.values[target] = value.copy()

    def array(self):
        if not self.values:
            raise ValueError("No training values available")
        return np.stack(self.values)


def validate_dataset(root, verify_hashes=True):
    rows = read_manifest(root)
    signatures = set()
    split_counts = dict(train=0, val=0, test=0)
    for row in rows:
        path = Path(root) / row["path"]
        if verify_hashes and sha256(path) != row["sha256"]:
            raise ValueError(f"Episode content changed: {row['episode_id']}")
        with h5py.File(path, "r") as file:
            if file["actions"].shape != (row["length"], row["action_dim"]) or file["proprio"].shape != (row["length"], row["proprio_dim"]):
                raise ValueError(f"Episode shape mismatch: {path}")
            if file["images"].shape[:2] != (row["length"], row["num_images"]) or file["images"].dtype != np.uint8:
                raise ValueError(f"Image schema mismatch: {path}")
            if not np.isfinite(file["actions"][:]).all() or not np.isfinite(file["proprio"][:]).all():
                raise ValueError(f"Nonfinite episode: {path}")
        signatures.add((row["action_dim"], row["proprio_dim"], row["num_images"], row["action_semantics"]))
        split_counts[row["split"]] += 1
    if len(signatures) != 1:
        raise ValueError("One dataset must have a consistent action/state/camera schema")
    if not split_counts["train"]:
        raise ValueError("No training episodes")
    return dict(episodes=len(rows), frames=sum(r["length"] for r in rows), splits=split_counts,
                signature=list(signatures.pop()), hashes_verified=verify_hashes,
                manifest_sha256=sha256(Path(root) / "manifest.jsonl"))


def prepare_artifacts(root, output, horizon=8, num_modes=64, max_samples=100000, iterations=40, seed=42):
    output = Path(output)
    if (output / "artifacts.json").exists() or (output / "codebook.npz").exists():
        raise FileExistsError("Artifacts already exist; choose another directory for a new experiment")
    if max_samples < num_modes:
        raise ValueError("max_samples must be at least num_modes")
    check_disk(output, 0.2)
    audit = validate_dataset(root)
    rows = [r for r in read_manifest(root) if r["split"] == "train"]
    action_samples, state_samples = Reservoir(max_samples, seed), Reservoir(max_samples, seed + 1)
    for row in rows:
        with h5py.File(Path(root) / row["path"], "r") as file:
            action_samples.add(file["actions"][:])
            state_samples.add(file["proprio"][:])
    mask = np.ones(rows[0]["action_dim"], bool)
    if rows[0]["action_semantics"] == ACTION_SEMANTICS:
        mask[-1] = False
    action = Normalizer.fit(action_samples.array(), mask)
    state = Normalizer.fit(state_samples.array())
    chunks = Reservoir(max_samples, seed + 2)
    for row in rows:
        with h5py.File(Path(root) / row["path"], "r") as file:
            actions = action.normalize(file["actions"][:])
            # Only complete chunks fit the prototypes. Tail windows remain supervised with a validity mask.
            chunks.add(np.stack([actions[i:i + horizon] for i in range(len(actions) - horizon + 1)])
                       if len(actions) >= horizon else [])
    prototypes, report = fit_codebook(chunks.array(), num_modes, iterations, seed)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "codebook.npz", prototypes=prototypes)
    metadata = dict(schema_version=1, manifest_sha256=audit["manifest_sha256"],
                    codebook_sha256=sha256(output / "codebook.npz"), horizon=horizon, num_modes=num_modes,
                    action_dim=rows[0]["action_dim"], proprio_dim=rows[0]["proprio_dim"],
                    num_images=rows[0]["num_images"], cameras=rows[0]["cameras"],
                    action_semantics=rows[0]["action_semantics"], fit_split="train", seed=seed,
                    action_normalizer=action.to_dict(), proprio_normalizer=state.to_dict(),
                    audit=audit, codebook_report=report, normalization_samples=len(action_samples.values))
    atomic_json(output / "artifacts.json", metadata)
    return metadata


def load_artifacts(path):
    path = Path(path)
    with open(path / "artifacts.json", encoding="utf-8") as stream:
        meta = json.load(stream)
    if sha256(path / "codebook.npz") != meta["codebook_sha256"]:
        raise ValueError("Codebook digest mismatch")
    with np.load(path / "codebook.npz", allow_pickle=False) as file:
        prototypes = file["prototypes"].copy()
    expected = (meta["num_modes"], meta["horizon"], meta["action_dim"])
    if prototypes.shape != expected or not np.isfinite(prototypes).all():
        raise ValueError("Invalid codebook shape/content")
    return meta, prototypes
