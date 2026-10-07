"""Strict, serializable configuration shared by training and deployment."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    backbone: str = "tiny"
    pretrained: str = "openvla/openvla-7b"
    revision: str | None = None
    hidden_dim: int = 128
    head_dim: int = 256
    action_dim: int = 7
    proprio_dim: int = 8
    horizon: int = 8
    num_modes: int = 64
    num_images: int = 2
    image_size: int = 224
    max_text_length: int = 128
    lora_rank: int = 16
    lora_alpha: int = 32
    freeze_vision: bool = True
    freeze_projector: bool = True
    gradient_checkpointing: bool = True
    residual_scale: float = 1.0
    head_type: str = "joint"


@dataclass
class DataConfig:
    root: str = "data/libero"
    artifacts: str = "data/libero/artifacts"
    teacher: str | None = None
    image_aug: bool = True
    crop_scale: float = 0.9
    workers: int = 0


@dataclass
class TrainConfig:
    output: str = "runs/jointact"
    seed: int = 42
    device: str = "cuda:0"
    precision: str = "bf16"
    batch_size: int = 4
    grad_accumulation: int = 4
    max_steps: int = 20000
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 1000
    eval_every: int = 500
    eval_batches: int = 50
    keep_checkpoints: int = 3
    resume: str | None = None
    brier_weight: float = 0.1
    residual_weight: float = 1.0
    distill_weight: float = 0.0
    min_free_disk_gb: float = 2.0
    allow_shared_gpu: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        m, d, t = self.model, self.data, self.train
        if m.backbone not in {"tiny", "openvla"} or m.head_type not in {"joint", "regression"}:
            raise ValueError("backbone must be tiny/openvla; head_type must be joint/regression")
        for name in ("hidden_dim", "head_dim", "action_dim", "horizon", "num_modes", "num_images", "image_size", "max_text_length"):
            if getattr(m, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if m.proprio_dim < 0 or m.lora_rank < 0 or m.residual_scale <= 0:
            raise ValueError("invalid proprio_dim, lora_rank or residual_scale")
        if m.backbone == "tiny" and m.hidden_dim % 4:
            raise ValueError("tiny hidden_dim must be divisible by four")
        if not 0 < d.crop_scale <= 1 or d.workers < 0:
            raise ValueError("invalid crop_scale or workers")
        if t.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32/fp16/bf16")
        for name in ("batch_size", "grad_accumulation", "max_steps", "log_every", "save_every", "eval_every", "eval_batches", "keep_checkpoints"):
            if getattr(t, name) <= 0:
                raise ValueError(f"train.{name} must be positive")
        if t.warmup_steps < 0 or t.learning_rate <= 0 or t.grad_clip <= 0 or t.min_free_disk_gb < 0:
            raise ValueError("invalid optimizer or storage configuration")
        if min(t.brier_weight, t.residual_weight, t.distill_weight) < 0:
            raise ValueError("loss weights cannot be negative")
        if t.distill_weight > 0 and d.teacher is None:
            raise ValueError("distillation requires data.teacher")
        if m.head_type == "regression" and t.distill_weight:
            raise ValueError("mode distillation requires the joint head")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        unknown = set(raw) - {"model", "data", "train"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        sections = {}
        for key, typ in (("model", ModelConfig), ("data", DataConfig), ("train", TrainConfig)):
            values = raw.get(key, {})
            if not isinstance(values, dict):
                raise ValueError(f"{key} must be a mapping")
            unknown = set(values) - {f.name for f in fields(typ)}
            if unknown:
                raise ValueError(f"Unknown {key} fields: {sorted(unknown)}")
            sections[key] = typ(**values)
        config = cls(**sections)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False, allow_unicode=True)
