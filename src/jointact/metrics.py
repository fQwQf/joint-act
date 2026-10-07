"""Offline behavior metrics; mode confidence is not task-success probability."""

import numpy as np
import torch


def classification_metrics(logits, labels, bins=15):
    logits = torch.as_tensor(logits).float()
    labels = torch.as_tensor(labels).long()
    if logits.ndim != 2 or len(logits) != len(labels) or not len(labels):
        raise ValueError("Need nonempty [N,K] logits and [N] labels")
    p = logits.softmax(-1)
    confidence, prediction = p.max(-1)
    correct = prediction == labels
    ece = torch.zeros(())
    for index in range(bins):
        mask = (confidence > index / bins) & (confidence <= (index + 1) / bins)
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
    return dict(mode_accuracy=float(correct.float().mean()), mode_nll=float(-p[torch.arange(len(p)), labels].clamp_min(1e-12).log().mean()),
                mode_brier=float((p - torch.nn.functional.one_hot(labels, p.shape[1])).square().sum(-1).mean()), mode_ece=float(ece))


def wilson_interval(successes, trials, z=1.96):
    if trials < 1:
        raise ValueError("Need at least one completed trial")
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    width = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return [float(max(0, center - width)), float(min(1, center + width))]
