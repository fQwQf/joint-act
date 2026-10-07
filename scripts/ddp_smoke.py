"""Compare uninterrupted and resumed two-rank CPU training at the tensor level."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from jointact.checkpoint import read_weights
from jointact.config import Config
from jointact.data.prepare import prepare_artifacts
from jointact.fixture import create_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh DDP smoke directory")
    output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    create_fixture(output / "data", episodes=8, length=9, image_size=16)
    prepare_artifacts(output / "data", output / "artifacts", horizon=3, num_modes=4, iterations=4)
    config = Config.from_dict(dict(model=dict(hidden_dim=16, head_dim=32, horizon=3, num_modes=4,
                                              image_size=16, gradient_checkpointing=False),
                                  data=dict(root=str(output / "data"), artifacts=str(output / "artifacts"), image_aug=True),
                                  train=dict(output=str(output / "full"), device="cpu", precision="fp32", max_steps=4,
                                             warmup_steps=1, batch_size=5, grad_accumulation=2,
                                             save_every=4, eval_every=4, log_every=1, eval_batches=2, min_free_disk_gb=0)))
    config.save(output / "full.yaml")
    resumed = copy.deepcopy(config)
    resumed.train.output = str(output / "resume")
    resumed.save(output / "resume.yaml")
    def run(name, *extra):
        with open(output / f"{name}.log", "w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                            "-m", "jointact.cli", "train", *map(str, extra)], check=True, stdout=stream, stderr=subprocess.STDOUT)
    run("full", "--config", output / "full.yaml")
    run("partial", "--config", output / "resume.yaml", "--stop-after", 2)
    run("continued", "--config", output / "resume.yaml", "--resume", output / "resume" / "checkpoints")
    _, expected = read_weights(output / "full" / "checkpoints")
    _, actual = read_weights(output / "resume" / "checkpoints")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    report = dict(passed=True, world_size=2, device="cpu", bitwise_continuation=True, tensors=len(expected), fixture_only=True)
    with open(output / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
