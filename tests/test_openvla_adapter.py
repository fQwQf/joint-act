"""Real HF Llama/PEFT adapter contracts without downloading foundation weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from jointact.config import ModelConfig
from jointact.models import JointActionPolicy


@pytest.mark.integration
@pytest.mark.parametrize("lora_rank", [0, 2])
def test_openvla_decoder_adapter_with_real_llama(monkeypatch, lora_rank):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 4, 4)

        def forward(self, values):
            return self.conv(values).flatten(2).transpose(1, 2)

    class Prismatic(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_backbone = Vision()
            self.projector = nn.Linear(16, 32)
            self.language_model = LlamaForCausalLM(LlamaConfig(vocab_size=32000, hidden_size=32,
                intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4))
            self.config = SimpleNamespace(_commit_hash="verification-only-random-model")

    monkeypatch.setattr(transformers.AutoModelForVision2Seq, "from_pretrained", lambda *a, **k: Prismatic())
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right")))
    config = ModelConfig(backbone="openvla", head_dim=32, horizon=2, num_modes=4, image_size=16,
                         lora_rank=lora_rank, lora_alpha=4, gradient_checkpointing=True)
    model = JointActionPolicy(config, np.zeros((4, 2, 7), np.float32))
    batch = dict(pixel_values=torch.randn(2, 2, 3, 16, 16), input_ids=torch.tensor([[1, 2, 3, 0], [1, 2, 4, 5]]),
                 attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]), proprio=torch.randn(2, 8),
                 actions=torch.randn(2, 2, 7), valid=torch.ones(2, 2, dtype=torch.bool))
    output = model(batch, supervised=True)
    loss, _ = model.loss(output, batch)
    loss.backward()
    assert model.backbone.query.grad.abs().sum() > 0
    assert model.backbone.vision.conv.weight.grad is None
    assert model.backbone.projector.weight.grad is None
    if lora_rank:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.named_parameters() if "lora_B" in name)
    else:
        assert not model.backbone.language.get_output_embeddings().weight.requires_grad
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        together = model(batch)["logits"]
        single = model({k: v[:1] for k, v in batch.items()})["logits"]
    torch.testing.assert_close(together[:1], single, atol=2e-5, rtol=2e-5)
