"""Offline teacher scoring using original autoregressive OpenVLA action logits.

Only horizon=1 is accepted: original OpenVLA is not trained to model action chunks.
Scores are restricted behavior likelihoods, not measured execution success rates.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from jointact.data.dataset import EpisodeDataset
from jointact.data.normalization import Normalizer
from jointact.data.prepare import load_artifacts
from jointact.utils import check_disk, check_gpu, sha256


def action_token_ids(actions, vocab_size, bins=256):
    values = np.digitize(np.clip(actions, -1, 1), np.linspace(-1, 1, bins))
    return vocab_size - values


@torch.inference_mode()
def score_teacher(root, artifacts, output, model_id, revision=None, unnorm_key=None, device="cpu",
                  temperature=1.0, max_samples=None, allow_shared_gpu=False, precision="bf16"):
    if temperature <= 0 or (max_samples is not None and max_samples < 1):
        raise ValueError("temperature and max_samples must be positive")
    meta, prototypes = load_artifacts(artifacts)
    if meta["horizon"] != 1:
        raise ValueError("Autoregressive OpenVLA teacher scoring requires horizon=1 artifacts")
    if meta["num_images"] != 1:
        raise ValueError("Original OpenVLA teacher supports one image; use one-camera data/artifacts")
    check_gpu(device, allow_shared_gpu)
    check_disk(output, 2)
    from transformers import AutoModelForVision2Seq, AutoProcessor
    kwargs = {"revision": revision} if revision else {}
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype,
                                                  attn_implementation="eager", **kwargs).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    if not hasattr(model, "norm_stats"):
        raise ValueError("Teacher checkpoint has no robot action statistics")
    if unnorm_key is None:
        if len(model.norm_stats) != 1:
            raise ValueError("Teacher has multiple action domains; supply --unnorm-key explicitly")
        unnorm_key = next(iter(model.norm_stats))
    stats = model.norm_stats[unnorm_key]["action"]
    low, high = stats.get("q01", stats.get("min")), stats.get("q99", stats.get("max"))
    if low is None or high is None:
        raise ValueError("Unsupported teacher normalization statistics")
    normalizer = Normalizer(low, high, stats.get("mask", np.ones(len(low), bool)))
    student_norm = Normalizer(**meta["action_normalizer"])
    physical = student_norm.unnormalize(prototypes[:, 0])
    tokens = torch.as_tensor(action_token_ids(normalizer.normalize(physical), processor.tokenizer.vocab_size),
                             dtype=torch.long, device=device)
    dataset = EpisodeDataset(root, 1, "train")
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            header = dict(type="metadata", manifest_sha256=sha256(Path(root) / "manifest.jsonl"),
                          codebook_sha256=meta["codebook_sha256"], num_modes=meta["num_modes"],
                          teacher=model_id, revision=revision or getattr(model.config, "_commit_hash", None),
                          unnorm_key=unnorm_key, temperature=temperature, horizon=1,
                          semantics="restricted_autoregressive_behavior_likelihood")
            stream.write(json.dumps(header) + "\n")
            for index in range(count):
                row = dataset[index]
                prompt = f"In: What action should the robot take to {row['instruction'].strip().lower()}?\nOut:"
                inputs = processor(prompt, Image.fromarray(row["images"][0]), return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                ids = inputs["input_ids"]
                if ids[0, -1] != 29871:
                    ids = torch.cat([ids, ids.new_full((1, 1), 29871)], -1)
                scores = []
                # A one-candidate microbatch bounds full-vocabulary logits memory.
                for candidate in tokens:
                    full = torch.cat([ids, candidate[None]], -1)
                    output_logits = model(input_ids=full, attention_mask=torch.ones_like(full),
                                           pixel_values=inputs["pixel_values"], use_cache=False,
                                           return_dict=True).logits
                    offset = output_logits.shape[1] - full.shape[1]
                    positions = torch.arange(len(candidate), device=device) + ids.shape[1] - 1 + offset
                    logp = output_logits[0, positions].float().log_softmax(-1)
                    scores.append(logp.gather(-1, candidate[:, None]).sum())
                probabilities = (torch.stack(scores) / temperature).softmax(-1).cpu().tolist()
                stream.write(json.dumps(dict(episode_id=row["episode_id"], timestep=row["timestep"],
                                             probabilities=probabilities), allow_nan=False) + "\n")
        import os
        os.replace(temporary, path)
    finally:
        dataset.close()
        if temporary.exists():
            temporary.unlink()
    return dict(samples=count, output=str(path), semantics=header["semantics"])
