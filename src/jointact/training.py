"""Single-device/DDP training, AMP, accumulation, evaluation, exact CPU resume."""

import contextlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from jointact.checkpoint import load_trainable, read_weights, restore_rng, rng_state, save_checkpoint
from jointact.config import Config
from jointact.data.dataset import EpisodeDataset, ObservationCollator, ResumableSampler
from jointact.data.prepare import load_artifacts, validate_dataset
from jointact.metrics import classification_metrics
from jointact.models import JointActionPolicy
from jointact.utils import (atomic_json, check_disk, check_gpu, distributed_context, move_batch,
                            rank_zero_call, require_finite, seed_everything, sha256, source_fingerprint)


def autocast_context(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("CPU fp16 is unsupported; use fp32 or bf16")
    return torch.autocast(device_type=device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def validation_indices(length, budget, sampling):
    """Fixed index coverage without consuming training RNG or only seeing early episodes."""
    count = min(length, budget)
    if sampling == "uniform":
        return [(i * length + length // 2) // count for i in range(count)]
    return list(range(count))


@torch.inference_mode()
def evaluate(model, loader, device, precision="fp32", max_batches=None):
    was_training = model.training
    model.eval()
    error, denominator, logits, labels = 0.0, 0.0, [], []
    diagnostic_sums, predictions, grouped = {}, [], {}
    evaluated_windows = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch(batch, device)
        evaluated_windows += len(batch["actions"])
        with autocast_context(device, precision):
            output = model(batch, diagnostics=True)
        mask = batch["valid"].float().unsqueeze(-1)
        error += float(((output["actions"].float() - batch["actions"].float()).abs() * mask).sum())
        denominator += float(mask.sum()) * model.config.action_dim
        if "logits" in output:
            from jointact.data.codebook import nearest_modes
            target = nearest_modes(batch["actions"], batch["valid"], model.head.prototypes)
            labels.append(target.cpu())
            logits.append(output["logits"].cpu())
            predictions.append(output["mode"].cpu())
            truth = batch["actions"].float()
            required = truth - model.head.prototypes[target]
            differences = dict(
                quantization_l1=required.abs(),
                predicted_prototype_l1=(truth - model.head.prototypes[output["mode"]]).abs(),
                oracle_mode_action_l1=(truth - output["oracle_actions"].float()).abs(),
                residual_abs_mean=output["residual"].float().abs(),
                residual_saturation_fraction=(output["residual"].abs() >= .95 * model.head.residual_scale).float(),
                required_residual_out_of_bounds_fraction=(required.abs() > model.head.residual_scale).float(),
            )
            for name, values in differences.items():
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + float((values * mask).sum())
            for name, selected in (("correct_mode", target == output["mode"]),
                                   ("wrong_mode", target != output["mode"])):
                group_mask = mask * selected[:, None, None]
                values = grouped.setdefault(name, dict(elements=0, action_error=0.0, prototype_error=0.0))
                values["elements"] += int(group_mask.sum()) * model.config.action_dim
                values["action_error"] += float(((truth - output["actions"].float()).abs() * group_mask).sum())
                values["prototype_error"] += float((differences["predicted_prototype_l1"] * group_mask).sum())
    model.train(was_training)
    if not denominator:
        raise ValueError("Evaluation consumed no valid action elements")
    result = dict(action_l1=error / denominator, evaluated_action_elements=int(denominator),
                  evaluated_windows=evaluated_windows)
    if logits:
        result.update(classification_metrics(torch.cat(logits), torch.cat(labels)))
        result["diagnostics"] = {name: value / denominator for name, value in diagnostic_sums.items()}
        result["diagnostics"].update(
            oracle_is_deployable=False,
            predicted_mode_counts=torch.bincount(torch.cat(predictions), minlength=model.config.num_modes).tolist(),
            target_mode_counts=torch.bincount(torch.cat(labels), minlength=model.config.num_modes).tolist(),
        )
        result["diagnostics"]["by_mode_correctness"] = {
            key: dict(evaluated_action_elements=value["elements"],
                      action_l1=value["action_error"] / value["elements"] if value["elements"] else None,
                      prototype_l1=value["prototype_error"] / value["elements"] if value["elements"] else None,
                      residual_gain=(value["prototype_error"] - value["action_error"]) / value["elements"]
                      if value["elements"] else None) for key, value in grouped.items()}
    return result


def train(config: Config, stop_after=None):
    config.validate()
    t = config.train
    requested_device = f"cuda:{os.environ['LOCAL_RANK']}" if int(os.environ.get("WORLD_SIZE", "1")) > 1 and t.device.startswith("cuda") else t.device
    check_gpu(requested_device, t.allow_shared_gpu)
    check_disk(t.output, t.min_free_disk_gb)
    rank, world, device = distributed_context(t.device)
    try:
        return _train(config, rank, world, device, stop_after)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(config, rank, world, device, stop_after=None):
    t, m, d = config.train, config.model, config.data
    output = Path(t.output)
    def prepare_output():
        output.mkdir(parents=True, exist_ok=True)
        if (output / "metrics.jsonl").exists() and not t.resume:
            raise FileExistsError("Existing run; resume it or choose a fresh train.output")
    rank_zero_call(prepare_output)
    seed_everything(t.seed)
    metadata, prototypes = load_artifacts(d.artifacts)
    for key in ("action_dim", "num_images", "horizon", "num_modes"):
        if metadata[key] != getattr(m, key):
            raise ValueError(f"model.{key} disagrees with prepared artifacts")
    if m.proprio_dim and metadata["proprio_dim"] != m.proprio_dim:
        raise ValueError("model.proprio_dim disagrees with prepared artifacts")
    audit = rank_zero_call(lambda: validate_dataset(d.root, verify_hashes=True))
    dtype = torch.float32 if t.precision == "fp32" else torch.float16 if t.precision == "fp16" else torch.bfloat16
    model = JointActionPolicy(m, prototypes, device, dtype)
    dataset = EpisodeDataset(d.root, m.horizon, "train", d.artifacts, d.teacher)
    sampler = ResumableSampler(len(dataset), t.seed, rank, world)
    collator = ObservationCollator(m, model.processor, d.image_aug, d.crop_scale)
    loader_generator = torch.Generator().manual_seed(t.seed + rank)
    loader = DataLoader(dataset, batch_size=t.batch_size, sampler=sampler, num_workers=d.workers,
                        collate_fn=collator, pin_memory=device.type == "cuda", generator=loader_generator)
    val_loader = None
    if rank == 0 and metadata["audit"]["splits"]["val"]:
        val_dataset = EpisodeDataset(d.root, m.horizon, "val", d.artifacts)
        val_indices = validation_indices(len(val_dataset), t.batch_size * t.eval_batches, t.eval_sampling)
        val_loader = DataLoader(val_dataset, batch_size=t.batch_size, sampler=val_indices, num_workers=0,
                                collate_fn=ObservationCollator(m, model.processor, False, d.crop_scale),
                                generator=torch.Generator().manual_seed(t.seed))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=t.learning_rate,
                                  weight_decay=t.weight_decay)
    def schedule(step):
        if step < t.warmup_steps:
            return (step + 1) / max(1, t.warmup_steps)
        progress = (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and t.precision == "fp16")
    step = 0
    if t.resume or t.fork_from:
        path, weights = read_weights(t.resume or t.fork_from)
        saved = Config.load(path / "config.yaml")
        is_fork = bool(t.fork_from and not t.resume)
        if is_fork and Path(saved.train.output).resolve() == output.resolve():
            raise ValueError("A training fork requires a fresh output directory")
        with open(path / "artifacts.json", encoding="utf-8") as stream:
            saved_meta = json.load(stream)
        if saved.model != m or saved_meta != metadata:
            raise ValueError("Model or data artifacts changed across resume")
        # Optimizer schedule and batch topology must retain their original meaning.
        objective_keys = ("alignment_weight", "action_cost_weight", "alignment_margin")
        resume_keys = ("seed", "max_steps", "learning_rate", "weight_decay", "warmup_steps", "batch_size",
                       "grad_accumulation", "precision", "brier_weight", "residual_weight", "distill_weight")
        for key in resume_keys + (() if is_fork else objective_keys):
            if getattr(saved.train, key) != getattr(t, key):
                raise ValueError(f"Cannot change train.{key} on exact resume")
        if saved.data != d:
            raise ValueError("Data/augmentation configuration changed across resume")
        load_trainable(model, weights)
        with open(path / "manifest.json", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if "training_sha256" in manifest and sha256(path / "training.pt") != manifest["training_sha256"]:
            raise ValueError("Checkpoint training-state checksum failed")
        # Training state is pickle-based and must come from a trusted local run.
        state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        sampler.load_state_dict(state["sampler"])
        if len(state["rank_states"]) != world:
            raise ValueError("World size changed across resume")
        restore_rng(state["rank_states"][rank])
        step = state["step"]
        if is_fork:
            rank_zero_call(lambda: atomic_json(output / "lineage.json", dict(
                parent_checkpoint=str(path.resolve()), parent_step=step,
                parent_manifest_sha256=sha256(path / "manifest.json"),
                parent_weights_sha256=manifest["weights_sha256"],
                parent_config_sha256=sha256(path / "config.yaml"),
                parent_training_sha256=sha256(path / "training.pt"),
                inherited="weights, optimizer, scheduler, scaler, sampler, per-rank RNG",
                objective_changes={key: dict(before=getattr(saved.train, key), after=getattr(t, key))
                                   for key in objective_keys if getattr(saved.train, key) != getattr(t, key)})))
    def record_run():
        config.save(output / "config.yaml")
        atomic_json(output / "data-audit.json", audit)
        atomic_json(output / "run.json", dict(world_size=world, device=str(device), torch_version=torch.__version__,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    total_parameters=sum(p.numel() for p in model.parameters()), exact_resume_workers_zero=d.workers == 0,
                    source=source_fingerprint(), cuda_version=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    rank_zero_call(record_run)
    wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                       broadcast_buffers=False) if world > 1 else model
    iterator = iter(loader)
    model.train()
    started = time.perf_counter()
    last_checkpoint = None
    compute_seconds, samples_seen = 0.0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    limit = min(t.max_steps, stop_after) if stop_after is not None else t.max_steps
    if limit < step:
        raise ValueError("stop_after precedes the resumed checkpoint")
    while step < limit:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        aggregate = {}
        for micro in range(t.grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            synchronize = wrapped.no_sync() if world > 1 and micro < t.grad_accumulation - 1 else contextlib.nullcontext()
            with synchronize, autocast_context(device, t.precision):
                predictions = wrapped(batch, supervised=True, align_predictions=t.alignment_weight > 0)
                loss, metrics = model.loss(predictions, batch, t.brier_weight, t.residual_weight, t.distill_weight,
                                           t.alignment_weight, t.action_cost_weight, t.alignment_margin)
                require_finite(loss, f"Nonfinite loss on a rank at optimizer step {step}, microbatch {micro}")
                scaler.scale(loss / t.grad_accumulation).backward()
            sampler.advance(len(batch["actions"]))
            samples_seen += len(batch["actions"]) * world
            for key, value in metrics.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value) / t.grad_accumulation
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip, error_if_nonfinite=False)
        require_finite(grad_norm, f"Nonfinite gradient on a rank at optimizer step {step}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_seconds += time.perf_counter() - step_started
        if step % t.log_every == 0 or step == 1 or step == limit:
            if world > 1:
                keys = sorted(aggregate)
                values = torch.tensor([aggregate[k] for k in keys], device=device)
                dist.all_reduce(values)
                aggregate = {k: float(v / world) for k, v in zip(keys, values)}
            if "_alignment_error_sum" in aggregate:
                eligible_error = aggregate.pop("_alignment_error_sum")
                eligible_elements = aggregate.pop("_alignment_element_count")
                aggregate["alignment_eligible_l1"] = eligible_error / eligible_elements if eligible_elements else None
            def record_metrics():
                row = dict(kind="train", step=step, **aggregate, learning_rate=scheduler.get_last_lr()[0],
                           grad_norm=float(grad_norm), elapsed_seconds=time.perf_counter() - started,
                           optimization_seconds=compute_seconds, optimization_samples=samples_seen,
                           optimization_samples_per_second=samples_seen / compute_seconds,
                           rank0_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                           rank0_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None)
                with open(output / "metrics.jsonl", "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
            rank_zero_call(record_metrics)
        if step % t.eval_every == 0 or step == limit:
            def record_validation():
                if val_loader is not None:
                    result = dict(kind="validation", step=step, sampling=t.eval_sampling,
                                  total_validation_windows=len(val_loader.dataset),
                                  **evaluate(model, val_loader, device, t.precision, t.eval_batches))
                    with open(output / "metrics.jsonl", "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, allow_nan=False) + "\n")
                    print(json.dumps(result), flush=True)
            rank_zero_call(record_validation)
        if step % t.save_every == 0 or step == limit:
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng_state())
            else:
                states[0] = rng_state()
            last_checkpoint = rank_zero_call(lambda: str(save_checkpoint(output / "checkpoints", model, optimizer,
                scheduler, scaler, sampler, step, config, metadata, states, t.keep_checkpoints)))
    dataset.close()
    if rank == 0 and val_loader is not None:
        val_loader.dataset.close()
    return dict(step=step, checkpoint=str(last_checkpoint) if last_checkpoint else None)
