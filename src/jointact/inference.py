"""Stateless batched policy inference and explicit chunk execution state."""

from collections import deque
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from jointact.checkpoint import load_policy
from jointact.data.dataset import ObservationCollator
from jointact.data.normalization import Normalizer
from jointact.data.schema import ACTION_SEMANTICS
from jointact.training import autocast_context
from jointact.utils import check_gpu, move_batch


class PolicyRuntime:
    def __init__(self, checkpoint, device="cpu", precision="fp32", allow_shared_gpu=False):
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32/bf16/fp16")
        check_gpu(device, allow_shared_gpu)
        self.device, self.precision = torch.device(device), precision
        dtype = torch.float32 if precision == "fp32" else torch.float16 if precision == "fp16" else torch.bfloat16
        self.model, self.config, self.metadata = load_policy(checkpoint, device, dtype)
        self.action_normalizer = Normalizer(**self.metadata["action_normalizer"])
        self.proprio_normalizer = Normalizer(**self.metadata["proprio_normalizer"])
        self.collator = ObservationCollator(self.config.model, self.model.processor, False, self.config.data.crop_scale)
        self.temperature = 1.0
        self.calibration = None
        calibration_path = Path(checkpoint) / "calibration.json"
        if calibration_path.exists():
            import json
            from jointact.checkpoint import read_weights
            from jointact.utils import sha256
            with open(calibration_path, encoding="utf-8") as stream:
                self.calibration = json.load(stream)
            resolved, _ = read_weights(checkpoint)
            if self.calibration["weights_sha256"] != sha256(resolved / "weights.pt"):
                raise ValueError("Calibration was fitted for different weights")
            self.temperature = self.calibration["temperature"]

    def _record(self, observation):
        images = observation["images"]
        arrays = []
        for image in images:
            if isinstance(image, (str, Path)):
                with Image.open(image) as file:
                    array = np.asarray(file.convert("RGB")).copy()
            elif isinstance(image, Image.Image):
                array = np.asarray(image.convert("RGB")).copy()
            else:
                array = np.asarray(image)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("Inference images must be uint8 HWC RGB")
            arrays.append(array)
        if len(arrays) != self.config.model.num_images:
            raise ValueError("Wrong number of cameras")
        if self.config.model.proprio_dim:
            proprio = np.asarray(observation["proprio"], np.float32)
            if proprio.shape != (self.config.model.proprio_dim,) or not np.isfinite(proprio).all():
                raise ValueError("Wrong proprioceptive state dimensions or nonfinite values")
        else:
            # Keep the canonical dataset schema while omitting the model's state
            # token. This enables input-matched comparisons with original OpenVLA.
            proprio = np.zeros(self.metadata["proprio_dim"], np.float32)
        instruction = observation["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty text")
        return dict(images=arrays, instruction=instruction, proprio=self.proprio_normalizer.normalize(proprio))

    @torch.inference_mode()
    def predict_batch(self, observations):
        if not observations:
            raise ValueError("Observation batch cannot be empty")
        batch = move_batch(self.collator([self._record(obs) for obs in observations]), self.device)
        with autocast_context(self.device, self.precision):
            output = self.model(batch)
        normalized = output["actions"].float().cpu().numpy()
        normalized = np.clip(normalized, -1, 1)
        actions = self.action_normalizer.unnormalize(normalized)
        if self.metadata["action_semantics"] == ACTION_SEMANTICS:
            actions[..., -1] = np.clip(actions[..., -1], 0, 1)
        if not np.isfinite(actions).all():
            raise FloatingPointError("Policy returned nonfinite actions")
        results = []
        p = (output["logits"].float() / self.temperature).softmax(-1).cpu().numpy() if "logits" in output else None
        for index in range(len(observations)):
            result = dict(actions=actions[index].tolist(), action_semantics=self.metadata["action_semantics"],
                          horizon=self.config.model.horizon)
            if p is not None:
                result.update(mode=int(output["mode"][index]), mode_probabilities=p[index].tolist(),
                              mode_confidence=float(p[index].max()), probability_semantics="behavior_mode_distribution")
            results.append(result)
        return results

    def predict(self, observation):
        return self.predict_batch([observation])[0]

    def benchmark(self, observation, warmup=5, repeats=30):
        if warmup < 0 or repeats < 1:
            raise ValueError("Invalid benchmark counts")
        for _ in range(warmup):
            self.predict(observation)
        durations = []
        for _ in range(repeats):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            self.predict(observation)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            durations.append((time.perf_counter() - start) * 1000)
        return dict(scope="preprocessing+model+unnormalization", batch_size=1, warmup=warmup, repeats=repeats,
                    median_ms=float(np.median(durations)), p95_ms=float(np.percentile(durations, 95)),
                    mean_ms=float(np.mean(durations)), horizon=self.config.model.horizon,
                    device=str(self.device), precision=self.precision)


class ChunkController:
    def __init__(self, runtime, execute_horizon=None):
        self.runtime = runtime
        self.execute_horizon = execute_horizon or runtime.config.model.horizon
        if not 1 <= self.execute_horizon <= runtime.config.model.horizon:
            raise ValueError("execute_horizon must be between 1 and the predicted horizon")
        self.queue = deque()

    def reset(self):
        self.queue.clear()

    def action(self, observation):
        if not self.queue:
            actions = self.runtime.predict(observation)["actions"]
            self.queue.extend(actions[:self.execute_horizon])
        return np.asarray(self.queue.popleft(), np.float32)
