import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from jointact.calibration import fit_temperature
from jointact.evaluation.libero import action_to_libero, load_initial_states, observation_from_libero, quat_to_axisangle, run_episode
from jointact.metrics import classification_metrics, wilson_interval
from jointact.serve import create_app


def test_calibration_improves_overconfidence():
    logits = torch.tensor([[10., 0], [10., 0], [0, 10.], [0, 10.]])
    target = torch.tensor([0, 1, 1, 1])
    temperature = fit_temperature(logits, target)
    assert temperature > 1
    assert classification_metrics(logits / temperature, target)["mode_nll"] < classification_metrics(logits, target)["mode_nll"]
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high


def test_libero_coordinate_and_gripper_conventions():
    np.testing.assert_allclose(quat_to_axisangle([0, 0, 0, 1]), 0)
    np.testing.assert_allclose(quat_to_axisangle([0, 0, 1, 0]), [0, 0, np.pi], atol=1e-6)
    assert action_to_libero([0, 0, 0, 0, 0, 0, 1])[-1] == -1
    assert action_to_libero([0, 0, 0, 0, 0, 0, 0])[-1] == 1
    raw = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    obs = dict(agentview_image=raw, robot0_eye_in_hand_image=raw, robot0_eef_pos=np.zeros(3),
               robot0_eef_quat=np.array([0, 0, 0, 1]), robot0_gripper_qpos=np.zeros(2))
    converted = observation_from_libero(obs, "test")
    np.testing.assert_array_equal(converted["images"][0], raw[::-1, ::-1])
    assert converted["proprio"].shape == (8,)


def test_numpy_benchmark_initial_states(tmp_path):
    path = tmp_path / "states.pt"
    expected = np.arange(18, dtype=np.float64).reshape(3, 6)
    torch.save(expected, path)
    np.testing.assert_array_equal(load_initial_states(path), expected)
    torch.save(np.array([np.nan]), path)
    with pytest.raises(ValueError, match="finite nonempty matrix"):
        load_initial_states(path)


class FakeRuntime:
    config = SimpleNamespace(model=SimpleNamespace(num_images=2, horizon=3, backbone="tiny"))
    metadata = dict(action_semantics="eef_delta_xyz_axisangle_gripper_open01")

    def predict_batch(self, observations):
        return [self.predict(obs) for obs in observations]

    def predict(self, observation):
        return dict(actions=[[0, 0, 0, 0, 0, 0, 1]] * 3)


def test_http_payload_validation():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    image = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode()
    client = TestClient(create_app(FakeRuntime()))
    assert client.get("/health").status_code == 200
    record = dict(images=[encoded, encoded], proprio=[0] * 8, instruction="pick")
    assert client.post("/predict", json=record).status_code == 200
    assert client.post("/predict", json=dict(images=["bad"], proprio=[], instruction="pick")).status_code == 422


def test_closed_loop_chunk_reset():
    class Environment:
        def reset(self):
            self.steps = 0

        def set_init_state(self, state):
            return self.observation()

        def observation(self):
            image = np.zeros((8, 8, 3), np.uint8)
            return dict(agentview_image=image, robot0_eye_in_hand_image=image, robot0_eef_pos=np.zeros(3),
                        robot0_eef_quat=np.array([0, 0, 0, 1]), robot0_gripper_qpos=np.zeros(2))

        def step(self, action):
            self.steps += 1
            return self.observation(), 0, self.steps >= 12, {}
    result, frames = run_episode(Environment(), None, FakeRuntime(), "pick", max_steps=5, settle_steps=10)
    assert result["success"] and result["steps"] == 2 and len(frames) == 2
    assert result["decisions"] == 1 and result["decision_mean_ms"] >= result["control_mean_ms"]
