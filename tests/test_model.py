import numpy as np
import pytest
import torch

from jointact.config import Config
from jointact.data.dataset import EpisodeDataset, ObservationCollator
from jointact.data.prepare import load_artifacts
from jointact.models import JointActionPolicy


def test_configuration_rejects_unknown_and_invalid():
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_dict({"model": {"num_mode": 4}})
    with pytest.raises(ValueError, match="distillation"):
        Config.from_dict({"train": {"distill_weight": 1}})


def test_joint_loss_gradient_and_padding(prepared):
    root, artifacts, config = prepared
    _, prototypes = load_artifacts(artifacts)
    model = JointActionPolicy(config.model, prototypes)
    data = EpisodeDataset(root, 3, "train", artifacts)
    batch = ObservationCollator(config.model)([data[0], data[8]])
    output = model(batch, supervised=True)
    loss, _ = model.loss(output, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.vision[0].weight.grad.abs().sum() > 0
    assert model.head.classifier.weight.grad.abs().sum() > 0
    assert model.head.residual[-1].weight.grad.abs().sum() > 0
    predicted = model(batch)
    assert predicted["actions"].shape == (2, 3, 7)
    assert torch.allclose(predicted["logits"].softmax(-1).sum(-1), torch.ones(2))
    data.close()


def test_image_and_state_affect_predictions(prepared):
    _, artifacts, config = prepared
    _, prototypes = load_artifacts(artifacts)
    model = JointActionPolicy(config.model, prototypes).eval()
    batch = dict(pixel_values=torch.zeros(2, 2, 3, 16, 16), input_ids=torch.ones(2, 3, dtype=torch.long),
                 attention_mask=torch.ones(2, 3, dtype=torch.long), proprio=torch.zeros(2, 8))
    batch["pixel_values"][1] = 1
    output = model(batch)["logits"]
    assert not torch.allclose(output[0], output[1])
    batch["pixel_values"][1] = 0
    batch["proprio"][1] = 1
    output = model(batch)["logits"]
    assert not torch.allclose(output[0], output[1])


def test_continuous_baseline(prepared):
    _, artifacts, config = prepared
    _, prototypes = load_artifacts(artifacts)
    config.model.head_type = "regression"
    model = JointActionPolicy(config.model, prototypes)
    batch = dict(pixel_values=torch.zeros(2, 2, 3, 16, 16), input_ids=torch.ones(2, 3, dtype=torch.long),
                 attention_mask=torch.ones(2, 3, dtype=torch.long), proprio=torch.zeros(2, 8),
                 actions=torch.zeros(2, 3, 7), valid=torch.ones(2, 3, dtype=torch.bool))
    output = model(batch, supervised=True)
    loss, _ = model.loss(output, batch)
    loss.backward()
    assert np.isfinite(float(loss.detach())) and "logits" not in output
