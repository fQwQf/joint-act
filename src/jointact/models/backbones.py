"""Tiny verification backbone and an adapter preserving OpenVLA visual features."""

from contextlib import nullcontext

import torch
from torch import nn

from jointact.config import ModelConfig


class TinyBackbone(nn.Module):
    """Small trainable visual-language model for pipeline and numerical tests."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        h = config.hidden_dim
        self.hidden_dim = h
        self.vision = nn.Sequential(nn.Conv2d(3, 32, 5, 4, 2), nn.GELU(), nn.Conv2d(32, h, 3, 2, 1),
                                    nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.tokens = nn.Embedding(256, h)
        self.proprio = nn.Linear(config.proprio_dim, h) if config.proprio_dim else None
        self.camera = nn.Parameter(torch.randn(1, config.num_images, h) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, h * 4, dropout=0,
                                                                           batch_first=True, norm_first=True), 2,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(h)

    def forward(self, pixel_values, input_ids, attention_mask, proprio):
        batch, views, channels, height, width = pixel_values.shape
        visual = self.vision(pixel_values.reshape(batch * views, channels, height, width)).reshape(batch, views, -1)
        text = self.tokens(input_ids)
        weight = attention_mask.unsqueeze(-1).to(text.dtype)
        text = (text * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True).clamp_min(1)
        parts = [visual + self.camera, text]
        if self.proprio is not None:
            parts.append(self.proprio(proprio).unsqueeze(1))
        parts.append(self.query.expand(batch, -1, -1))
        return self.norm(self.transformer(torch.cat(parts, 1))[:, -1])


class OpenVLABackbone(nn.Module):
    """Use pretrained DINOv2/SigLIP, projector, and Llama; add a joint readout token.

    Calls the causal decoder directly, avoiding vocabulary logits and generated tokens.
    Each camera is encoded with the original single-camera vision tower independently.
    """
    def __init__(self, config: ModelConfig, device="cpu", dtype=torch.float32, processor_path=None):
        super().__init__()
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as error:
            raise ImportError("Install joint-act[openvla] in the pinned OpenVLA environment") from error
        # Remote model code is the user's explicitly selected pretrained dependency.
        kwargs = dict(trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True,
                      attn_implementation="eager")
        if config.revision:
            kwargs["revision"] = config.revision
        vla = AutoModelForVision2Seq.from_pretrained(config.pretrained, **kwargs)
        if config.revision is None and getattr(vla.config, "_commit_hash", None):
            config.revision = vla.config._commit_hash
        self.processor = AutoProcessor.from_pretrained(processor_path or config.pretrained,
                                                       trust_remote_code=True,
                                                       **({"revision": config.revision} if config.revision and not processor_path else {}))
        self.processor.tokenizer.padding_side = "right"
        for name in ("vision_backbone", "projector", "language_model"):
            if not hasattr(vla, name):
                raise ValueError(f"Pretrained model is not an OpenVLA/Prismatic checkpoint: missing {name}")
        self.vision, self.projector, self.language = vla.vision_backbone, vla.projector, vla.language_model
        if hasattr(self.vision, "set_num_images_in_input"):
            self.vision.set_num_images_in_input(1)
        self.hidden_dim = self.language.config.hidden_size
        self.freeze_vision = config.freeze_vision
        self.vision.requires_grad_(not config.freeze_vision)
        self.projector.requires_grad_(not config.freeze_projector)
        self.language.requires_grad_(config.lora_rank == 0)
        # The vocabulary projection is bypassed by the direct readout. Freeze an
        # untied output head so full-language DDP has no unused trainable tensor.
        output_embeddings = self.language.get_output_embeddings()
        if output_embeddings.weight is not self.language.get_input_embeddings().weight:
            output_embeddings.requires_grad_(False)
        if config.lora_rank:
            from peft import LoraConfig, get_peft_model
            self.language = get_peft_model(self.language, LoraConfig(r=config.lora_rank,
                lora_alpha=config.lora_alpha, lora_dropout=0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none", task_type="CAUSAL_LM"))
        if config.gradient_checkpointing:
            self.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.language.enable_input_require_grads()
        self.language.config.use_cache = False
        self.proprio = nn.Sequential(nn.Linear(config.proprio_dim, self.hidden_dim), nn.GELU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim)) if config.proprio_dim else None
        self.query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.camera = nn.Parameter(torch.zeros(1, config.num_images, 1, self.hidden_dim))
        del vla
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vision:
            self.vision.eval()
        return self

    def forward(self, pixel_values, input_ids, attention_mask, proprio):
        batch, views, channels, height, width = pixel_values.shape
        dtype = next(self.vision.parameters()).dtype
        with torch.no_grad() if self.freeze_vision else nullcontext():
            patches = self.vision(pixel_values.reshape(batch * views, channels, height, width).to(dtype))
        projected = self.projector(patches).reshape(batch, views, -1, self.hidden_dim)
        projected = (projected + self.camera.to(projected.dtype)).reshape(batch, -1, self.hidden_dim)
        text = self.language.get_input_embeddings()(input_ids)
        parts = [text[:, :1], projected, text[:, 1:]]
        masks = [attention_mask[:, :1], attention_mask.new_ones((batch, projected.shape[1])), attention_mask[:, 1:]]
        if self.proprio is not None:
            state = self.proprio(proprio.float()).to(text.dtype).unsqueeze(1)
            parts.append(state)
            masks.append(attention_mask.new_ones((batch, 1)))
        parts.append(self.query.to(text.dtype).expand(batch, -1, -1))
        masks.append(attention_mask.new_ones((batch, 1)))
        embeddings, mask = torch.cat(parts, 1), torch.cat(masks, 1)
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        base = self.language.get_base_model() if hasattr(self.language, "get_base_model") else self.language
        # LlamaForCausalLM.model returns just decoder states, saving the full vocabulary projection.
        decoder = base.model
        output = decoder(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
                         use_cache=False, return_dict=True)
        return output.last_hidden_state[:, -1].float()
