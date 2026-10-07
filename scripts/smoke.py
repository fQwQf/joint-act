"""Exercise public CLI commands end to end; fixture results are not robot evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from jointact.config import Config
from jointact.data.dataset import EpisodeDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained", help="Exercise an actual OpenVLA-compatible checkpoint on the CPU fixture")
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Smoke output must be a fresh directory")
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
    root, artifacts = output / "data", output / "data" / "artifacts"
    run("fixture", "--output", root)
    run("validate-data", "--root", root)
    run("prepare", "--root", root, "--output", artifacts, "--horizon", 4, "--num-modes", 8, "--iterations", 6)
    config = Config.load(Path(__file__).resolve().parent.parent / "configs" / "tiny.yaml")
    if args.pretrained:
        config.model.backbone, config.model.pretrained = "openvla", args.pretrained
        config.model.lora_rank, config.model.lora_alpha = 2, 4
        config.model.gradient_checkpointing = True
        config.model.image_size = args.image_size
    config.data.root, config.data.artifacts = str(root), str(artifacts)
    config.train.output, config.train.max_steps = str(output / "run"), 12
    config.train.save_every, config.train.eval_every = 6, 6
    config_path = output / "config.yaml"
    config.save(config_path)
    checkpoints = output / "run" / "checkpoints"
    run("train", "--config", config_path, "--stop-after", 6)
    run("train", "--config", config_path, "--resume", checkpoints)
    run("evaluate-offline", "--checkpoint", checkpoints, "--output", output / "test.json")
    run("export", "--checkpoint", checkpoints, "--output", output / "bundle")
    data = EpisodeDataset(root, 1, "test")
    row = data[0]
    data.close()
    images = []
    for index, image in enumerate(row["images"]):
        name = f"camera-{index}.png"
        Image.fromarray(image).save(output / name)
        images.append(name)
    observation = dict(images=images, proprio=row["proprio"].tolist(), instruction=row["instruction"])
    with open(output / "observation.json", "w", encoding="utf-8") as stream:
        json.dump(observation, stream)
    run("predict", "--checkpoint", output / "bundle", "--observation", output / "observation.json", "--output", output / "prediction.json")
    run("benchmark", "--checkpoint", output / "bundle", "--observation", output / "observation.json", "--warmup", 1, "--repeats", 3,
        "--output", output / "latency.json")
    run("calibrate", "--checkpoint", output / "bundle", "--root", root, "--artifacts", artifacts)
    with open(output / "prediction.json", encoding="utf-8") as stream:
        prediction = json.load(stream)
    assert np.asarray(prediction["actions"]).shape == (4, 7)
    report = dict(passed=True, commands=commands, fixture_only=True, robot_performance_evidence=False)
    with open(output / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
