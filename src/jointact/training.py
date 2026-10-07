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
from jointact.utils import atomic_json, barrier, check_disk, check_gpu, distributed_context, move_batch, seed_everything


def autocast_context(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("CPU fp16 is unsupported; use fp32 or bf16")
    return torch.autocast(device_type=device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


@torch.inference_mode()
def evaluate(model, loader, device, precision="fp32", max_batches=None):
    was_training = model.training
    model.eval()
    error, denominator, logits, labels = 0.0, 0.0, [], []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch(batch, device)
        with autocast_context(device, precision):
            output = model(batch)
        mask = batch["valid"].float().unsqueeze(-1)
        error += float(((output["actions"].float() - batch["actions"].float()).abs() * mask).sum())
        denominator += float(mask.sum()) * model.config.action_dim
        if "logits" in output:
            from jointact.data.codebook import nearest_modes
            labels.append(nearest_modes(batch["actions"], batch["valid"], model.head.prototypes).cpu())
            logits.append(output["logits"].cpu())
    model.train(was_training)
    if not denominator:
        raise ValueError("Evaluation consumed no valid action elements")
    result = dict(action_l1=error / denominator, evaluated_action_elements=int(denominator))
    if logits:
        result.update(classification_metrics(torch.cat(logits), torch.cat(labels)))
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
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "metrics.jsonl").exists() and not t.resume:
            raise FileExistsError("Existing run; resume it or choose a fresh train.output")
    barrier()
    seed_everything(t.seed)
    metadata, prototypes = load_artifacts(d.artifacts)
    for key in ("action_dim", "num_images", "horizon", "num_modes"):
        if metadata[key] != getattr(m, key):
            raise ValueError(f"model.{key} disagrees with prepared artifacts")
    if m.proprio_dim and metadata["proprio_dim"] != m.proprio_dim:
        raise ValueError("model.proprio_dim disagrees with prepared artifacts")
    audit = validate_dataset(d.root, verify_hashes=True) if rank == 0 else None
    barrier()
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
        val_loader = DataLoader(val_dataset, batch_size=t.batch_size, shuffle=False, num_workers=0,
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
    if t.resume:
        path, weights = read_weights(t.resume)
        saved = Config.load(path / "config.yaml")
        with open(path / "artifacts.json", encoding="utf-8") as stream:
            saved_meta = json.load(stream)
        if saved.model != m or saved_meta != metadata:
            raise ValueError("Model or data artifacts changed across resume")
        # Optimizer schedule and batch topology must retain their original meaning.
        for key in ("seed", "max_steps", "learning_rate", "weight_decay", "warmup_steps", "batch_size", "grad_accumulation", "precision", "brier_weight", "residual_weight", "distill_weight"):
            if getattr(saved.train, key) != getattr(t, key):
                raise ValueError(f"Cannot change train.{key} on exact resume")
        if saved.data != d:
            raise ValueError("Data/augmentation configuration changed across resume")
        load_trainable(model, weights)
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
    if rank == 0:
        config.save(output / "config.yaml")
        atomic_json(output / "data-audit.json", audit)
        atomic_json(output / "run.json", dict(world_size=world, device=str(device), torch_version=torch.__version__,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    total_parameters=sum(p.numel() for p in model.parameters()), exact_resume_workers_zero=d.workers == 0))
    wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                       broadcast_buffers=False) if world > 1 else model
    iterator = iter(loader)
    model.train()
    started = time.perf_counter()
    last_checkpoint = None
    limit = min(t.max_steps, stop_after) if stop_after is not None else t.max_steps
    if limit < step:
        raise ValueError("stop_after precedes the resumed checkpoint")
    while step < limit:
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
                predictions = wrapped(batch, supervised=True)
                loss, metrics = model.loss(predictions, batch, t.brier_weight, t.residual_weight, t.distill_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at optimizer step {step}, microbatch {micro}")
                scaler.scale(loss / t.grad_accumulation).backward()
            sampler.advance(len(batch["actions"]))
            for key, value in metrics.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value) / t.grad_accumulation
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        if step % t.log_every == 0 or step == 1 or step == limit:
            if world > 1:
                keys = sorted(aggregate)
                values = torch.tensor([aggregate[k] for k in keys], device=device)
                dist.all_reduce(values)
                aggregate = {k: float(v / world) for k, v in zip(keys, values)}
            if rank == 0:
                row = dict(kind="train", step=step, **aggregate, learning_rate=scheduler.get_last_lr()[0],
                           grad_norm=float(grad_norm), elapsed_seconds=time.perf_counter() - started)
                with open(output / "metrics.jsonl", "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
        if step % t.eval_every == 0 or step == limit:
            barrier()
            if rank == 0 and val_loader is not None:
                result = dict(kind="validation", step=step, **evaluate(model, val_loader, device, t.precision, t.eval_batches))
                with open(output / "metrics.jsonl", "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                print(json.dumps(result), flush=True)
            barrier()
        if step % t.save_every == 0 or step == limit:
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng_state())
            else:
                states[0] = rng_state()
            if rank == 0:
                last_checkpoint = save_checkpoint(output / "checkpoints", model, optimizer, scheduler, scaler,
                    sampler, step, config, metadata, states, t.keep_checkpoints)
            barrier()
    dataset.close()
    if rank == 0 and val_loader is not None:
        val_loader.dataset.close()
    return dict(step=step, checkpoint=str(last_checkpoint) if last_checkpoint else None)
