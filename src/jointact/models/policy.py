"""Joint modes represent complete chunks; residuals condition on the selected mode."""

import torch
import torch.nn.functional as F
from torch import nn

from jointact.config import ModelConfig
from jointact.data.codebook import nearest_modes
from jointact.models.backbones import OpenVLABackbone, TinyBackbone


class JointActionHead(nn.Module):
    def __init__(self, hidden_dim, config, prototypes):
        super().__init__()
        self.horizon, self.action_dim = config.horizon, config.action_dim
        self.residual_scale = config.residual_scale
        self.register_buffer("prototypes", torch.as_tensor(prototypes, dtype=torch.float32))
        self.context = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, config.head_dim), nn.GELU())
        self.classifier = nn.Linear(config.head_dim, config.num_modes)
        self.mode_embedding = nn.Embedding(config.num_modes, config.head_dim)
        self.residual = nn.Sequential(nn.Linear(config.head_dim * 2, config.head_dim), nn.GELU(),
                                      nn.Linear(config.head_dim, self.horizon * self.action_dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features, mode=None):
        context = self.context(features.float())
        logits = self.classifier(context)
        mode = logits.argmax(-1) if mode is None else mode
        residual = self.residual(torch.cat([context, self.mode_embedding(mode)], -1))
        residual = self.residual_scale * torch.tanh(residual.reshape(-1, self.horizon, self.action_dim))
        actions = self.prototypes[mode] + residual
        return dict(logits=logits, mode=mode, residual=residual, actions=actions)


class RegressionHead(nn.Module):
    """Same-backbone continuous L1 baseline for a fair output-head ablation."""
    def __init__(self, hidden_dim, config):
        super().__init__()
        self.horizon, self.action_dim = config.horizon, config.action_dim
        self.model = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, config.head_dim), nn.GELU(),
                                   nn.Linear(config.head_dim, config.horizon * config.action_dim))

    def forward(self, features):
        return dict(actions=self.model(features.float()).reshape(-1, self.horizon, self.action_dim))


class JointActionPolicy(nn.Module):
    def __init__(self, config: ModelConfig, prototypes, device="cpu", dtype=torch.float32, processor_path=None):
        super().__init__()
        expected = (config.num_modes, config.horizon, config.action_dim)
        if tuple(prototypes.shape) != expected:
            raise ValueError(f"Codebook shape {prototypes.shape} does not match {expected}")
        self.config = config
        self.backbone = (TinyBackbone(config) if config.backbone == "tiny" else
                         OpenVLABackbone(config, device, dtype, processor_path))
        self.head = (JointActionHead(self.backbone.hidden_dim, config, prototypes) if config.head_type == "joint"
                     else RegressionHead(self.backbone.hidden_dim, config))
        self.to(device)

    @property
    def processor(self):
        return getattr(self.backbone, "processor", None)

    def forward(self, batch, supervised=False):
        features = self.backbone(**{k: batch[k] for k in ("pixel_values", "input_ids", "attention_mask", "proprio")})
        if self.config.head_type == "regression":
            return self.head(features)
        target_mode = nearest_modes(batch["actions"], batch["valid"], self.head.prototypes) if supervised else None
        output = self.head(features, target_mode)
        if supervised:
            output["target_mode"] = target_mode
        return output

    def loss(self, output, batch, brier_weight=0.1, residual_weight=1.0, distill_weight=0.0):
        valid = batch["valid"].float().unsqueeze(-1)
        denominator = (valid.sum() * self.config.action_dim).clamp_min(1)
        l1 = ((output["actions"].float() - batch["actions"].float()).abs() * valid).sum() / denominator
        metrics = dict(action_l1=l1.detach())
        if self.config.head_type == "regression":
            return l1, metrics
        logits, target = output["logits"].float(), output["target_mode"]
        ce = F.cross_entropy(logits, target)
        probs = logits.softmax(-1)
        brier = (probs - F.one_hot(target, self.config.num_modes)).square().sum(-1).mean()
        loss = ce + brier_weight * brier + residual_weight * l1
        metrics.update(ce=ce.detach(), brier=brier.detach(), mode_accuracy=(logits.argmax(-1) == target).float().mean())
        if distill_weight:
            target_probs = batch["teacher_probs"].float()
            if target_probs.shape != logits.shape:
                raise ValueError("Teacher mode count differs from the current codebook")
            teacher_mask = batch["teacher_valid"].float()
            kl = F.kl_div(logits.log_softmax(-1), target_probs, reduction="none").sum(-1)
            kl = (kl * teacher_mask).sum() / teacher_mask.sum().clamp_min(1)
            loss = loss + distill_weight * kl
            metrics["teacher_kl"] = kl.detach()
            metrics["teacher_coverage"] = teacher_mask.mean().detach()
        metrics["loss"] = loss.detach()
        return loss, metrics
