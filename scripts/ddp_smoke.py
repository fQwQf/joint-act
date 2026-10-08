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
    parser.add_argument("--alignment", action="store_true", help="Exercise both auxiliary loss branches")
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
    if args.alignment:
        config.train.alignment_weight = 0.5
        config.train.action_cost_weight = 1.0
        config.train.eval_sampling = "uniform"
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
    blocked = copy.deepcopy(config)
    blocked.train.output = str(output / "blocked")
    Path(blocked.train.output).mkdir()
    (Path(blocked.train.output) / "metrics.jsonl").write_text("existing run\n")
    blocked.save(output / "blocked.yaml")
    failure_command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                       "-m", "jointact.cli", "train", "--config", str(output / "blocked.yaml")]
    failure = subprocess.run(failure_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45)
    (output / "rank-zero-failure.log").write_text(failure.stdout)
    if failure.returncode == 0 or "Rank-zero operation failed" not in failure.stdout:
        raise AssertionError("Expected rank-zero failure did not propagate")
    worker = output / "nonfinite_worker.py"
    worker.write_text('''import os, sys
from jointact.config import Config
from jointact.models import JointActionPolicy
from jointact.training import train
original = JointActionPolicy.loss
def injected(self, *args, **kwargs):
    loss, metrics = original(self, *args, **kwargs)
    return (loss * float("nan") if os.environ["RANK"] == "1" else loss), metrics
JointActionPolicy.loss = injected
train(Config.load(sys.argv[1]))
''')
    nonfinite = copy.deepcopy(config)
    nonfinite.train.output = str(output / "nonfinite")
    nonfinite.save(output / "nonfinite.yaml")
    failure = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                              str(worker), str(output / "nonfinite.yaml")], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, timeout=45)
    (output / "nonfinite-failure.log").write_text(failure.stdout)
    if failure.returncode == 0 or "Nonfinite loss on a rank" not in failure.stdout:
        raise AssertionError("Expected nonfinite failure did not propagate")
    report = dict(passed=True, world_size=2, device="cpu", bitwise_continuation=True, tensors=len(expected), fixture_only=True,
                  rank_zero_failure_propagated=True, nonfinite_failure_propagated=True,
                  alignment_objectives=args.alignment)
    with open(output / "report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
