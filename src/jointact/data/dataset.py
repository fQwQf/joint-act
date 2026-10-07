"""Map-style episode windows, supporting exact sampler continuation."""

import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from jointact.config import ModelConfig
from jointact.data.normalization import Normalizer
from jointact.data.schema import read_manifest
from jointact.utils import sha256


class EpisodeDataset(Dataset):
    def __init__(self, root: str | Path, horizon: int, split: str = "train", artifacts: str | Path | None = None,
                 teacher: str | None = None):
        self.root, self.horizon = Path(root), horizon
        self.rows = [r for r in read_manifest(root) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} episodes in {root}")
        self.offsets = np.cumsum([0] + [r["length"] for r in self.rows])
        self.files = OrderedDict()
        self.action_normalizer = self.proprio_normalizer = None
        if artifacts is not None:
            with open(Path(artifacts) / "artifacts.json", encoding="utf-8") as stream:
                meta = json.load(stream)
            if meta["manifest_sha256"] != sha256(self.root / "manifest.jsonl"):
                raise ValueError("Artifacts were prepared for a different dataset manifest")
            if meta["horizon"] != horizon:
                raise ValueError("Artifact horizon does not match dataset")
            if meta["codebook_sha256"] != sha256(Path(artifacts) / "codebook.npz"):
                raise ValueError("Codebook digest mismatch")
            self.action_normalizer = Normalizer(**meta["action_normalizer"])
            self.proprio_normalizer = Normalizer(**meta["proprio_normalizer"])
        self.teacher = {}
        if teacher:
            with open(teacher, encoding="utf-8") as stream:
                header = json.loads(next(stream))
                if header.get("type") != "metadata" or artifacts is None:
                    raise ValueError("Teacher file needs a metadata header and matching artifacts")
                if header.get("num_modes") != meta["num_modes"] or header.get("horizon") != horizon:
                    raise ValueError("Teacher action dimensions do not match artifacts")
                if header["codebook_sha256"] != sha256(Path(artifacts) / "codebook.npz") or header["manifest_sha256"] != sha256(self.root / "manifest.jsonl"):
                    raise ValueError("Teacher labels do not match dataset/codebook")
                train_lengths = {r["episode_id"]: r["length"] for r in read_manifest(root) if r["split"] == "train"}
                for line in stream:
                    row = json.loads(line)
                    if row["episode_id"] not in train_lengths:
                        raise ValueError("Teacher labels must contain training episodes only")
                    if type(row["timestep"]) is not int or not 0 <= row["timestep"] < train_lengths[row["episode_id"]]:
                        raise ValueError("Teacher timestep is outside its source episode")
                    key = (row["episode_id"], row["timestep"])
                    p = np.asarray(row["probabilities"], np.float32)
                    if p.shape != (meta["num_modes"],) or key in self.teacher or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-5):
                        raise ValueError(f"Invalid/duplicate teacher distribution: {key}")
                    self.teacher[key] = p
            self.teacher_modes = header["num_modes"]
            if not self.teacher:
                raise ValueError("Teacher file contains no labeled samples")

    def __len__(self):
        return int(self.offsets[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["files"] = OrderedDict()
        return state

    def close(self):
        for file in self.files.values():
            file.close()
        self.files.clear()

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.offsets, index, side="right") - 1)
        row = self.rows[episode]
        start = int(index - self.offsets[episode])
        if episode not in self.files:
            if len(self.files) >= 8:
                self.files.popitem(last=False)[1].close()
            self.files[episode] = h5py.File(self.root / row["path"], "r")
        self.files.move_to_end(episode)
        file = self.files[episode]
        end = min(start + self.horizon, row["length"])
        actions = file["actions"][start:end]
        valid = np.arange(self.horizon) < len(actions)
        actions = np.concatenate([actions, np.repeat(actions[-1:], self.horizon - len(actions), axis=0)])
        proprio = file["proprio"][start]
        if self.action_normalizer:
            actions = self.action_normalizer.normalize(actions)
            proprio = self.proprio_normalizer.normalize(proprio)
        result = dict(images=file["images"][start], actions=actions, valid=valid, proprio=proprio,
                      instruction=row["instruction"], episode_id=row["episode_id"], timestep=start)
        if self.teacher:
            p = self.teacher.get((row["episode_id"], start))
            result["teacher_probs"] = np.zeros(self.teacher_modes, np.float32) if p is None else p
            result["teacher_valid"] = p is not None
        return result


class ResumableSampler(Sampler):
    """Order is deterministic; cursor counts consumed samples, not prefetched ones."""
    def __init__(self, length: int, seed: int, rank: int = 0, world: int = 1):
        if length < world:
            raise ValueError("Dataset must have at least world_size samples")
        self.length, self.seed, self.rank, self.world = length, seed, rank, world
        self.epoch, self.cursor = 0, 0
        self._cached_epoch, self._cached_indices = None, None

    def indices(self):
        if self._cached_epoch == self.epoch:
            return self._cached_indices
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        # Equal rank lengths, with padding made explicit rather than silently dropping data.
        indices = torch.randperm(self.length, generator=generator).tolist()
        size = (self.length + self.world - 1) // self.world * self.world
        indices += (indices * ((size - self.length) // self.length + 1))[:size - self.length]
        self._cached_epoch = self.epoch
        self._cached_indices = indices[self.rank:size:self.world]
        return self._cached_indices

    def __iter__(self):
        yield from self.indices()[self.cursor:]

    def __len__(self):
        return len(self.indices()) - self.cursor

    def advance(self, count):
        self.cursor += count
        if self.cursor >= len(self.indices()):
            self.epoch += 1
            self.cursor = 0

    def state_dict(self):
        return dict(epoch=self.epoch, cursor=self.cursor, length=self.length, seed=self.seed, world=self.world)

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ("length", "seed", "world")):
            raise ValueError("Sampler topology/data changed across resume")
        self.epoch, self.cursor = state["epoch"], state["cursor"]


class ObservationCollator:
    def __init__(self, config: ModelConfig, processor=None, augment=False, crop_scale=0.9):
        self.config, self.processor = config, processor
        self.augment, self.crop_scale = augment, crop_scale

    def __call__(self, records):
        c = self.config
        flat = []
        for record in records:
            if len(record["images"]) != c.num_images:
                raise ValueError(f"Expected {c.num_images} cameras; got {len(record['images'])}")
            for array in record["images"]:
                image = Image.fromarray(array)
                width, height = image.size
                side = np.sqrt(self.crop_scale)
                cw, ch = max(1, round(width * side)), max(1, round(height * side))
                if self.augment:
                    left, top = np.random.randint(width - cw + 1), np.random.randint(height - ch + 1)
                else:
                    left, top = (width - cw) // 2, (height - ch) // 2
                image = image.crop((left, top, left + cw, top + ch))
                flat.append(image.resize((c.image_size, c.image_size), Image.Resampling.LANCZOS))
        instructions = [r["instruction"].strip().lower() for r in records]
        if self.processor is not None:
            images = self.processor.image_processor(images=flat, return_tensors="pt")["pixel_values"]
            text = self.processor.tokenizer(
                [f"In: What action should the robot take to {s}?\nOut:" for s in instructions],
                return_tensors="pt", padding=True, truncation=False,
            )
            if text["input_ids"].shape[1] > c.max_text_length:
                raise ValueError("Instruction exceeds max_text_length; increase the limit explicitly")
            # OpenVLA's action prefix ends with the tokenizer's standalone-space token.
            space = torch.full((len(records), 1), 29871, dtype=text["input_ids"].dtype)
            text["input_ids"] = torch.cat([text["input_ids"], space], dim=1)
            text["attention_mask"] = torch.cat([text["attention_mask"], torch.ones_like(space)], dim=1)
            pixels = images.reshape(len(records), c.num_images, *images.shape[1:])
        else:
            pixels = torch.from_numpy(np.stack([np.asarray(i).copy().transpose(2, 0, 1) for i in flat])).float() / 255
            pixels = pixels.reshape(len(records), c.num_images, *pixels.shape[1:])
            encoded = [list(s.encode("utf-8"))[:c.max_text_length] or [0] for s in instructions]
            ids = torch.zeros(len(records), max(map(len, encoded)), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, values in enumerate(encoded):
                ids[index, :len(values)] = torch.tensor(values)
                mask[index, :len(values)] = 1
            text = dict(input_ids=ids, attention_mask=mask)
        output = dict(pixel_values=pixels, **text, proprio=torch.from_numpy(np.stack([r["proprio"] for r in records])).float())
        for key in ("actions", "valid", "teacher_probs", "teacher_valid"):
            if key in records[0]:
                output[key] = torch.from_numpy(np.asarray([r[key] for r in records]))
        output["episode_id"] = [r.get("episode_id", "inference") for r in records]
        output["timestep"] = [r.get("timestep", 0) for r in records]
        return output
