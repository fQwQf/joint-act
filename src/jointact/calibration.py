"""Fit a scalar temperature on held-out behavior-mode labels."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jointact.checkpoint import read_weights
from jointact.data.codebook import nearest_modes
from jointact.data.dataset import EpisodeDataset
from jointact.inference import PolicyRuntime
from jointact.metrics import classification_metrics
from jointact.training import autocast_context
from jointact.utils import atomic_json, move_batch, sha256


def fit_temperature(logits, labels):
    logits, labels = logits.detach().float(), labels.detach().long()
    if logits.ndim != 2 or labels.shape != (len(logits),) or len(logits) == 0:
        raise ValueError("Invalid calibration data")
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp().clamp(0.05, 20), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 20).detach())


def calibrate(checkpoint, root=None, artifacts=None, device="cpu", max_samples=10000, batch_size=16,
              precision="fp32", allow_shared_gpu=False):
    if max_samples < 1 or batch_size < 1:
        raise ValueError("max_samples and batch_size must be positive")
    runtime = PolicyRuntime(checkpoint, device, precision, allow_shared_gpu)
    if runtime.config.model.head_type != "joint":
        raise ValueError("Calibration requires a mode-classification head")
    data = EpisodeDataset(root or runtime.config.data.root, runtime.config.model.horizon, "val",
                          artifacts or runtime.config.data.artifacts)
    loader = DataLoader(data, batch_size=batch_size, collate_fn=runtime.collator)
    logits, labels = [], []
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, runtime.device)
            with autocast_context(runtime.device, runtime.precision):
                output = runtime.model(batch)
            target = nearest_modes(batch["actions"], batch["valid"], runtime.model.head.prototypes)
            remain = max_samples - count
            logits.append(output["logits"][:remain].float().cpu())
            labels.append(target[:remain].cpu())
            count += min(remain, len(target))
            if count >= max_samples:
                break
    data.close()
    x, y = torch.cat(logits).clone(), torch.cat(labels).clone()
    temperature = fit_temperature(x, y)
    resolved, _ = read_weights(checkpoint)
    report = dict(temperature=temperature, samples=len(y), split="val", semantics="behavior_mode_distribution",
                  device=str(runtime.device), precision=runtime.precision,
                  weights_sha256=sha256(resolved / "weights.pt"),
                  before=classification_metrics(x, y), after=classification_metrics(x / temperature, y))
    atomic_json(Path(checkpoint) / "calibration.json", report)
    return report
