import json

import h5py
import numpy as np
import pytest
import torch

from jointact.data.codebook import fit_codebook, nearest_modes
from jointact.data.convert import canonical_libero_actions, convert_hdf5
from jointact.data.dataset import EpisodeDataset, ObservationCollator, ResumableSampler
from jointact.data.normalization import Normalizer
from jointact.data.prepare import load_artifacts, validate_dataset
from jointact.data.schema import episode_split
from jointact.utils import sha256


def test_normalization_roundtrip_constant_and_mask():
    x = np.array([[2, 3, 0], [4, 3, 1]], np.float32)
    norm = Normalizer([2, 3, 0], [4, 3, 1], [True, True, False])
    actual = norm.normalize(x)
    np.testing.assert_allclose(actual, [[-1, 0, 0], [1, 0, 1]])
    np.testing.assert_allclose(norm.unnormalize(actual), x)


def test_joint_codebook_and_tail_mask():
    x = np.stack([np.full((2, 2), -1), np.full((2, 2), 1)] * 8).astype(np.float32)
    codebook, _ = fit_codebook(x, 2, seed=4)
    assert np.max(np.min(np.square(x[:, None] - codebook).sum((2, 3)), axis=1)) < 1e-5
    labels = nearest_modes(torch.tensor([[[1., 1.], [-999., -999.]]]), torch.tensor([[True, False]]), torch.from_numpy(codebook))
    assert codebook[labels[0], 0, 0] == 1


def test_train_only_artifacts_and_boundaries(prepared):
    root, artifacts, config = prepared
    meta, prototypes = load_artifacts(artifacts)
    assert meta["fit_split"] == "train" and prototypes.shape == (4, 3, 7)
    data = EpisodeDataset(root, 3, "train", artifacts)
    last = data[data.rows[0]["length"] - 1]
    np.testing.assert_array_equal(last["valid"], [True, False, False])
    np.testing.assert_array_equal(last["actions"][0], last["actions"][-1])
    assert "fixture-0006" not in {r["episode_id"] for r in data.rows}
    batch = ObservationCollator(config.model, augment=False)([data[0], data[1]])
    assert batch["pixel_values"].shape == (2, 2, 3, 16, 16)
    assert batch["valid"].dtype == torch.bool
    data.close()


def test_episode_hash_detects_mutation(prepared):
    root, _, _ = prepared
    with h5py.File(next((root / "episodes").glob("*.h5")), "r+") as file:
        file["actions"][0, 0] += 1
    with pytest.raises(ValueError, match="content changed"):
        validate_dataset(root)


def test_sampler_resume_and_rank_partition():
    first = ResumableSampler(17, 5, rank=0, world=2)
    other = ResumableSampler(17, 5, rank=1, world=2)
    assert len(list(first)) == len(list(other)) == 9
    full = list(first)
    first.advance(4)
    resumed = ResumableSampler(17, 5, rank=0, world=2)
    resumed.load_state_dict(first.state_dict())
    assert list(resumed) == full[4:]
    resumed.advance(5)
    assert resumed.epoch == 1 and resumed.cursor == 0
    assert list(resumed) != full


def test_stable_group_split():
    assert episode_split("parent", 12) == episode_split("parent", 12)
    with pytest.raises(ValueError):
        episode_split("parent", val_fraction=0.9, test_fraction=0.3)


def test_teacher_contract_and_partial_coverage(prepared, tmp_path):
    root, artifacts, config = prepared
    meta, _ = load_artifacts(artifacts)
    dataset = EpisodeDataset(root, config.model.horizon)
    episode_id = dataset.rows[0]["episode_id"]
    dataset.close()
    header = dict(type="metadata", codebook_sha256=meta["codebook_sha256"],
                  manifest_sha256=sha256(root / "manifest.jsonl"), num_modes=meta["num_modes"],
                  horizon=config.model.horizon)
    path = tmp_path / "teacher.jsonl"
    def write(rows):
        path.write_text("\n".join(json.dumps(r) for r in [header, *rows]) + "\n", encoding="utf-8")
    row = dict(episode_id=episode_id, timestep=0, probabilities=[0.25] * 4)
    write([row])
    labeled = EpisodeDataset(root, config.model.horizon, artifacts=artifacts, teacher=str(path))
    assert labeled[0]["teacher_valid"] and not labeled[1]["teacher_valid"]
    labeled.close()
    for malformed, message in (([], "no labeled samples"), ([dict(row, probabilities=[1.0])], "distribution"),
                                ([dict(row, timestep=-1)], "timestep")):
        write(malformed)
        with pytest.raises(ValueError, match=message):
            EpisodeDataset(root, config.model.horizon, artifacts=artifacts, teacher=str(path))


def test_hdf5_import_gripper_and_proprio(tmp_path):
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as file:
        group = file.create_group("data")
        group.attrs["language_info"] = json.dumps(dict(language_instruction="pick the cup"))
        for index in range(4):
            demo = group.create_group(f"demo_{index}")
            obs = demo.create_group("obs")
            obs["agentview_rgb"] = np.full((3, 16, 16, 3), 123, np.uint8)
            obs["eye_in_hand_rgb"] = np.full((3, 16, 16, 3), 234, np.uint8)
            obs["ee_states"] = np.zeros((3, 6), np.float32)
            obs["gripper_states"] = np.zeros((3, 2), np.float32)
            actions = np.zeros((3, 7), np.float32)
            actions[:, -1] = [-1, 1, -1]
            demo["actions"] = actions
    output = tmp_path / "converted"
    result = convert_hdf5(str(source), str(output), min_free_gb=0)
    assert result == dict(episodes=4, frames=12)
    data = EpisodeDataset(output, 1, split="train")
    assert data[0]["instruction"] == "pick the cup"
    assert data[0]["proprio"].shape == (8,)
    data.close()
    raw = np.zeros((2, 7))
    raw[:, -1] = [-1, 1]
    np.testing.assert_array_equal(canonical_libero_actions(raw)[:, -1], [1, 0])
