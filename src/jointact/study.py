"""Generate controlled head comparisons and run their resumable training lifecycle."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from jointact.checkpoint import read_weights, resolve_checkpoint
from jointact.config import Config
from jointact.data.prepare import load_artifacts, validate_dataset
from jointact.utils import atomic_json, sha256, source_fingerprint

VARIANTS = ("joint", "regression", "prototype", "no_brier", "aligned", "cost", "aligned_cost")


def prepare_study(config, output, seeds=(42, 43, 44), variants=("joint", "regression", "prototype", "no_brier"),
                  joint_parent=None, regression_parent=None):
    base = Config.load(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh study directory")
    if not seeds or len(set(seeds)) != len(seeds) or not variants or len(set(variants)) != len(variants):
        raise ValueError("Seeds and variants must be nonempty and unique")
    if set(variants) - set(VARIANTS):
        raise ValueError(f"Supported variants: {VARIANTS}")
    if base.train.distill_weight or base.data.teacher:
        raise ValueError("The controlled head study requires demonstration-only training")
    if base.train.fork_from:
        raise ValueError("Supply study parents explicitly instead of base train.fork_from")
    parents = {}
    if regression_parent and not joint_parent:
        raise ValueError("A continuation study requires joint_parent")
    if joint_parent:
        if "regression" in variants and not regression_parent:
            raise ValueError("The regression control requires its own matched regression_parent")
        for head, parent in (("joint", joint_parent), ("regression", regression_parent)):
            if parent:
                resolved, _ = read_weights(parent)
                saved = Config.load(resolved / "config.yaml")
                if list(seeds) != [saved.train.seed] or saved.model.head_type != head:
                    raise ValueError("Parent head/seed differs; independent seeds require independent parents")
                parents[head] = resolved
        parent_steps = {json.loads((parent / "manifest.json").read_text())["step"] for parent in parents.values()}
        if len(parent_steps) != 1:
            raise ValueError("Continuation parents must have the same optimizer step")
    base.data.root = str(Path(base.data.root).resolve())
    base.data.artifacts = str(Path(base.data.artifacts).resolve())
    audit = validate_dataset(base.data.root)
    metadata, _ = load_artifacts(base.data.artifacts)
    if metadata["manifest_sha256"] != audit["manifest_sha256"]:
        raise ValueError("Dataset and artifacts differ")
    if any(audit["splits"][split] == 0 for split in ("train", "val", "test")):
        raise ValueError("A controlled study requires nonempty train, val and test episodes")
    for key in ("horizon", "action_dim", "num_images", "num_modes"):
        if getattr(base.model, key) != metadata[key]:
            raise ValueError(f"model.{key} does not match artifacts")
    runs = []
    for seed in seeds:
        for variant in variants:
            cfg = copy.deepcopy(base)
            cfg.model.head_type = "regression" if variant == "regression" else "joint"
            cfg.model.residual_enabled = variant != "prototype"
            cfg.train.residual_weight = 0.0 if variant == "prototype" else base.train.residual_weight
            cfg.train.brier_weight = 0.0 if variant in {"regression", "no_brier"} else base.train.brier_weight
            cfg.train.alignment_weight = (base.train.alignment_weight or 0.5) if variant in {"aligned", "aligned_cost"} else 0.0
            cfg.train.action_cost_weight = (base.train.action_cost_weight or 1.0) if variant in {"cost", "aligned_cost"} else 0.0
            cfg.train.eval_sampling = "uniform"
            cfg.train.seed, cfg.train.resume = seed, None
            parent = parents.get(cfg.model.head_type)
            cfg.train.fork_from = str(parent) if parent else None
            if parent:
                saved = Config.load(parent / "config.yaml")
                if saved.model != cfg.model or saved.data != cfg.data:
                    raise ValueError("Parent model/data differs from the planned variant")
                for field in ("max_steps", "learning_rate", "weight_decay", "warmup_steps", "batch_size",
                              "grad_accumulation", "precision", "brier_weight", "residual_weight", "distill_weight"):
                    if getattr(saved.train, field) != getattr(cfg.train, field):
                        raise ValueError(f"Parent train.{field} differs from the continuation plan")
            label = f"{variant}-seed{seed}"
            cfg.train.output = str(output / "runs" / label)
            cfg.validate()
            path = output / "configs" / f"{label}.yaml"
            cfg.save(path)
            runs.append(dict(label=label, variant=variant, seed=seed, config=str(path.relative_to(output)),
                             config_sha256=sha256(path),
                             parent_manifest_sha256=sha256(parent / "manifest.json") if parent else None))
    plan = dict(format="jointact-study-v1", manifest_sha256=audit["manifest_sha256"],
                artifacts_sha256=sha256(Path(base.data.artifacts) / "artifacts.json"),
                source_sha256=source_fingerprint()["sha256"], runs=runs,
                comparison="same backbone, inputs, data, optimizer budget and execution horizon; head representation varies",
                external_baselines="Official OpenVLA/OFT require separate upstream reproduction; regression is an internal ablation")
    atomic_json(output / "study.json", plan)
    return dict(plan=str(output / "study.json"), runs=len(runs), episodes=audit["splits"])


def run_study(plan, device=None, world_size=1, max_runs=None):
    path = Path(plan).resolve()
    with open(path, encoding="utf-8") as stream:
        spec = json.load(stream)
    if spec.get("format") != "jointact-study-v1" or world_size < 1 or (max_runs is not None and max_runs < 1):
        raise ValueError("Invalid study or execution limits")
    if spec["source_sha256"] != source_fingerprint()["sha256"]:
        raise ValueError("Study source changed; regenerate a study for the current implementation")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("run-study launches workers itself; invoke it outside torchrun")
    completed = []
    for run in spec["runs"][:max_runs]:
        config_path = (path.parent / run["config"]).resolve()
        if not config_path.is_relative_to(path.parent) or sha256(config_path) != run["config_sha256"]:
            raise ValueError("Study configuration changed or escaped its directory")
        cfg = Config.load(config_path)
        if cfg.train.fork_from and sha256(Path(cfg.train.fork_from) / "manifest.json") != run.get("parent_manifest_sha256"):
            raise ValueError("Study parent checkpoint changed")
        if sha256(Path(cfg.data.root) / "manifest.jsonl") != spec["manifest_sha256"]:
            raise ValueError("Study dataset changed")
        if sha256(Path(cfg.data.artifacts) / "artifacts.json") != spec["artifacts_sha256"]:
            raise ValueError("Study artifacts changed")
        output = Path(cfg.train.output)
        output.mkdir(parents=True, exist_ok=True)
        chosen_device = device or cfg.train.device
        latest = output / "checkpoints"
        stage = "train"
        def execute(arguments):
            with open(output / "lifecycle.log", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(stage=stage, argv=arguments)) + "\n")
                stream.flush()
                subprocess.run(arguments, check=True, stdout=stream, stderr=subprocess.STDOUT)
        try:
            step = 0
            if (latest / "latest.json").is_file():
                resolved, _ = read_weights(latest)
                saved = Config.load(resolved / "config.yaml")
                saved_train, planned_train = saved.to_dict()["train"], cfg.to_dict()["train"]
                for field in ("resume", "device"):
                    saved_train.pop(field)
                    planned_train.pop(field)
                if saved.model != cfg.model or saved.data != cfg.data or saved_train != planned_train:
                    raise ValueError("Existing study checkpoint does not match its planned configuration")
                with open(output / "run.json", encoding="utf-8") as stream:
                    if json.load(stream).get("source", {}).get("sha256") != spec["source_sha256"]:
                        raise ValueError("Existing study checkpoint was trained with different source code")
                with open(resolved / "manifest.json", encoding="utf-8") as stream:
                    step = json.load(stream)["step"]
            if step < cfg.train.max_steps:
                launch = ([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world_size}"]
                          if world_size > 1 else [sys.executable])
                args = launch + ["-m", "jointact.cli", "train", "--config", str(config_path), "--device", chosen_device]
                if step:
                    args += ["--resume", str(latest)]
                execute(args)
            checkpoint = resolve_checkpoint(latest)
            with open(checkpoint / "manifest.json", encoding="utf-8") as stream:
                identity = json.load(stream)["weights_sha256"]
            for split in ("val", "test"):
                stage = f"offline-{split}"
                execute([sys.executable, "-m", "jointact.cli", "evaluate-offline", "--checkpoint", str(checkpoint),
                         "--device", chosen_device, "--precision", cfg.train.precision, "--batch-size", str(cfg.train.batch_size),
                         "--split", split, "--output", str(output / f"{split}.json")])
            stage = "export"
            bundle = output / "bundle"
            if bundle.exists():
                resolved, _ = read_weights(bundle)
                with open(resolved / "manifest.json", encoding="utf-8") as stream:
                    if json.load(stream)["weights_sha256"] != identity:
                        raise ValueError("Existing bundle contains different weights")
            else:
                execute([sys.executable, "-m", "jointact.cli", "export", "--checkpoint", str(checkpoint), "--output", str(bundle)])
            atomic_json(output / "lifecycle.json", dict(completed=True, label=run["label"],
                        weights_sha256=identity, stage="export", closed_loop_evaluated=False))
            completed.append(run["label"])
            print(json.dumps(dict(completed=run["label"])), flush=True)
        except Exception as error:
            atomic_json(output / "lifecycle.json", dict(completed=False, label=run["label"], stage=stage, error=str(error)))
            raise
    return dict(completed=completed, requested_runs=len(spec["runs"]), closed_loop_evaluated=False)
