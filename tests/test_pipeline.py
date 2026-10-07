import copy
import json

import numpy as np
import pytest
import torch

from jointact.checkpoint import export_bundle, read_weights
from jointact.data.dataset import EpisodeDataset, ObservationCollator
from jointact.inference import ChunkController, PolicyRuntime
from jointact.training import train


def observation(root):
    dataset = EpisodeDataset(root, 1, "test")
    item = dataset[0]
    dataset.close()
    return dict(images=item["images"], proprio=item["proprio"], instruction=item["instruction"])


def test_exact_resume_export_and_inference(prepared, tmp_path):
    root, _, config = prepared
    uninterrupted = train(config)
    cfg = copy.deepcopy(config)
    cfg.train.output = str(tmp_path / "resumed")
    first = train(cfg, stop_after=2)
    cfg.train.resume = first["checkpoint"]
    second = train(cfg)
    _, expected = read_weights(uninterrupted["checkpoint"])
    _, actual = read_weights(second["checkpoint"])
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    bundle = export_bundle(second["checkpoint"], tmp_path / "bundle")
    assert not (tmp_path / "bundle" / "training.pt").exists()
    runtime = PolicyRuntime(bundle)
    obs = observation(root)
    one = runtime.predict(obs)
    many = runtime.predict_batch([obs, obs])
    np.testing.assert_allclose(one["actions"], many[0]["actions"], atol=1e-6)
    assert np.asarray(one["actions"]).shape == (3, 7)
    assert 0 <= one["mode_confidence"] <= 1
    benchmark = runtime.benchmark(obs, warmup=1, repeats=2)
    assert benchmark["median_ms"] > 0
    controller = ChunkController(runtime, 2)
    np.testing.assert_allclose(controller.action(obs), one["actions"][0])
    np.testing.assert_allclose(controller.action(obs), one["actions"][1])
    controller.reset()
    np.testing.assert_allclose(controller.action(obs), one["actions"][0])
    with open(tmp_path / "run" / "metrics.jsonl") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[-1]["kind"] == "validation"


def test_weight_corruption_detected(prepared):
    _, _, config = prepared
    result = train(config, stop_after=1)
    with open(result["checkpoint"] + "/weights.pt", "ab") as stream:
        stream.write(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        read_weights(result["checkpoint"])


def test_resume_rejects_changed_data_config(prepared):
    _, _, config = prepared
    result = train(config, stop_after=1)
    config.train.resume = result["checkpoint"]
    config.data.crop_scale = 0.8
    with pytest.raises(ValueError, match="augmentation"):
        train(config)


def test_validation_uses_selected_mode(prepared):
    root, artifacts, config = prepared
    from jointact.data.prepare import load_artifacts
    from jointact.models import JointActionPolicy
    from jointact.training import evaluate
    from torch.utils.data import DataLoader
    _, codebook = load_artifacts(artifacts)
    model = JointActionPolicy(config.model, codebook)
    loader = DataLoader(EpisodeDataset(root, 3, "val", artifacts), batch_size=3,
                        collate_fn=ObservationCollator(config.model))
    result = evaluate(model, loader, torch.device("cpu"))
    assert "mode_ece" in result and result["evaluated_action_elements"] > 0
    loader.dataset.close()


def test_disable_proprio_with_canonical_state_and_reload(prepared, tmp_path):
    root, _, config = prepared
    config.model.proprio_dim = 0
    result = train(config, stop_after=1)
    bundle = export_bundle(result["checkpoint"], tmp_path / "no-state-bundle")
    runtime = PolicyRuntime(bundle)
    assert runtime.model.backbone.proprio is None
    obs = observation(root)
    supplied = runtime.predict(obs)
    del obs["proprio"]
    omitted = runtime.predict(obs)
    np.testing.assert_array_equal(supplied["actions"], omitted["actions"])
    np.testing.assert_array_equal(supplied["mode_probabilities"], omitted["mode_probabilities"])
