"""Build a small random checkpoint using the actual upstream HF implementation.

This exercises serialization, processor, vision/projector and decoder integration.
It does not contain pretrained robot capability and must not be called OpenVLA-7B.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, LlamaConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Local upstream Python/config/tokenizer snapshot")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(args.metadata, trust_remote_code=True)
    config.timm_model_ids = ["vit_tiny_patch16_224", "vit_tiny_patch16_224"]
    config.timm_override_act_layers = [None, None]
    config.image_sizes = [32, 32]
    config.use_fused_vision_backbone = True
    config.text_config = LlamaConfig(vocab_size=32064, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=4, pad_token_id=32000)
    config.norm_stats = {"verification": {"action": {"q01": [-1] * 6 + [0], "q99": [1] * 7,
                                                      "mask": [True] * 6 + [False]}}}
    model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True, attn_implementation="eager")
    model.save_pretrained(output)
    processor = AutoProcessor.from_pretrained(args.metadata, trust_remote_code=True)
    processor.image_processor = type(processor.image_processor)(use_fused_vision_backbone=True,
        image_resize_strategy="resize-naive", input_sizes=[(3, 32, 32), (3, 32, 32)],
        interpolations=["bicubic", "bicubic"], means=[(0.5, 0.5, 0.5)] * 2, stds=[(0.5, 0.5, 0.5)] * 2)
    processor.save_pretrained(output)
    report = dict(purpose="upstream_implementation_integration_only", pretrained=False,
                  parameters=sum(p.numel() for p in model.parameters()), fused_vision=True, image_size=32)
    with open(output / "verification.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
