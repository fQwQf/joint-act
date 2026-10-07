"""Bounded CPU integration on converted robot demos and an explicit base model.

Run this on the validation host. It checks adaptation and serialization, not robot
performance. The public CLI remains the interface for full training/evaluation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from jointact.checkpoint import read_weights
from jointact.config import Config
from jointact.data.dataset import EpisodeDataset
from jointact.data.prepare import load_artifacts
from jointact.utils import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Converted canonical robot episodes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained", required=True, help="Local OpenVLA-compatible checkpoint")
    parser.add_argument("--base-role", choices=["pretrained", "random-integration"], required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Integration output must be fresh")
    output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    commands = []

    def run(*arguments):
        command = [sys.executable, "-m", "jointact.cli", *map(str, arguments)]
        with open(output / f"command-{len(commands):02d}.log", "w", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        commands.append(command[3:])

    artifacts = output / "artifacts"
    run("validate-data", "--root", root)
    run("prepare", "--root", root, "--output", artifacts, "--horizon", 4,
        "--num-modes", 8, "--max-samples", 1000, "--iterations", 6)
    meta, _ = load_artifacts(artifacts)
    config = Config.load(Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml")
    config.model.backbone, config.model.pretrained = "openvla", str(Path(args.pretrained).resolve())
    config.model.lora_rank, config.model.lora_alpha = 2, 4
    config.model.gradient_checkpointing = True
    config.model.num_images, config.model.image_size = meta["num_images"], args.image_size
    config.data.root, config.data.artifacts, config.data.image_aug = str(root), str(artifacts), False
    config.train.output, config.train.max_steps = str(output / "run"), 2
    config.train.precision = args.precision
    config.train.batch_size, config.train.grad_accumulation = 1, 1
    config.train.warmup_steps, config.train.log_every = 0, 1
    config.train.save_every, config.train.eval_every, config.train.eval_batches = 1, 2, 1
    config_path = output / "config.yaml"
    config.save(config_path)
    checkpoints = output / "run" / "checkpoints"
    run("train", "--config", config_path, "--stop-after", 1)
    run("train", "--config", config_path, "--resume", checkpoints)
    _, weights = read_weights(checkpoints)
    assert all(torch.isfinite(value).all() for value in weights.values())
    assert any(value.abs().sum() > 0 for key, value in weights.items() if "lora_B" in key)
    run("export", "--checkpoint", checkpoints, "--output", output / "bundle")
    dataset = EpisodeDataset(root, 1, "train")
    record = dataset[0]
    dataset.close()
    names = []
    for index, rgb in enumerate(record["images"]):
        name = f"camera-{index}.png"
        Image.fromarray(rgb).save(output / name)
        names.append(name)
    atomic_json(output / "observation.json", dict(images=names, instruction=record["instruction"],
                                                 proprio=record["proprio"].tolist()))
    run("predict", "--checkpoint", output / "bundle", "--precision", args.precision,
        "--observation", output / "observation.json", "--output", output / "prediction.json")
    with open(output / "prediction.json", encoding="utf-8") as stream:
        prediction = json.load(stream)
    assert np.asarray(prediction["actions"]).shape == (4, 7)
    assert np.isfinite(prediction["actions"]).all()
    assert np.isclose(sum(prediction["mode_probabilities"]), 1, atol=1e-5)
    report = dict(passed=True, commands=commands, dataset_audit=meta["audit"], base_role=args.base_role,
                  precision=args.precision, optimizer_steps=2, lora_updated=True,
                  prediction_shape=[4, 7], robot_performance_evidence=False)
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
