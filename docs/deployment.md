# Inference, serving and latency

## Portable bundles

```bash
jointact export --checkpoint runs/experiment/checkpoints --output checkpoints/experiment
```

An inference bundle contains configuration, normalization metadata, trainable weights and buffers, processor files and the weight digest. It excludes optimizer state and frozen base weights. Tiny bundles are self-contained. OpenVLA bundles need the same pinned upstream base checkpoint or its offline cache.

## Stateless prediction

Input is a list of RGB images in training camera order, the physical proprioceptive state and a nonempty instruction. Output actions have shape `[horizon, action_dim]` in physical controller units. The canonical gripper is `1=open,0=close`. `PolicyRuntime.predict_batch` preserves batch order and `predict` handles one observation.

When `model.proprio_dim: 0`, the model omits its state token and inference requests
may omit `proprio`. Canonical training data can retain its measured state.

For CLI prediction, save:

```json
{
  "images": ["agentview.png", "wrist.png"],
  "proprio": [0, 0, 0, 0, 0, 0, 0, 0],
  "instruction": "pick up the cup"
}
```

Image paths resolve relative to this JSON file. CLI commands:

```bash
jointact predict --checkpoint checkpoints/experiment --observation observation.json --output actions.json
jointact benchmark --checkpoint checkpoints/experiment --observation observation.json \
  --device cuda:0 --precision bf16 --warmup 10 --repeats 100 --output latency.json
```

Benchmark scope includes preprocessing, transfer, model and unnormalization. CUDA measurements synchronize before and after each request. Report batch size, precision, device, image count, input size, warmup, repeats, median and P95. A chunk's full inference latency differs from mean control-step latency when cached actions are executed between replans.

## Chunk execution

`ChunkController(runtime, execute_horizon=N)` runs the policy when its queue is empty, then returns the next N commands one at a time. `N` must be between 1 and prediction horizon. Call `reset()` on every episode reset or instruction change. The HTTP endpoint is stateless and leaves chunk scheduling to its client.

## HTTP service

```bash
jointact serve --checkpoint checkpoints/experiment --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

`POST /predict` accepts either one observation or `{"observations": [...]}`. Images are base64 PNG/JPEG payloads, not server-side file paths. Batch size defaults to at most 16. The service bounds image bytes and decoded resolution, checks camera count and serializes model access with a lock. Invalid input returns HTTP 422. Successful output is `{"predictions": [...]}`. The service binds loopback by default; network authentication and transport belong to the deployment infrastructure.

```python
import base64
import requests

def image_payload(path):
    with open(path, "rb") as stream:
        return base64.b64encode(stream.read()).decode("ascii")

response = requests.post("http://127.0.0.1:8000/predict", json={
    "images": [image_payload("agentview.png"), image_payload("wrist.png")],
    "proprio": [0] * 8,
    "instruction": "pick up the cup",
}, timeout=60)
response.raise_for_status()
print(response.json())
```

## Probability meaning and calibration

Mode probabilities classify demonstrated behavior modes. `mode_confidence` is the largest such probability. Neither is an execution-success estimate. The optional scalar temperature fits mode NLL on validation episodes and is tied to the exact weight digest:

```bash
jointact calibrate --checkpoint checkpoints/experiment --root data/libero --artifacts data/libero/artifacts
```

For a 7B model, select the same model precision as deployment and a small batch,
for example `--device cuda:0 --precision bf16 --batch-size 1`. Calibration accepts
the same occupied-GPU guard override as prediction. Default CPU FP32 is intended
for the tiny fixture; it doubles base-weight memory relative to BF16.

The file is `calibration.json` beside the inference bundle. Prediction uses it when present. It changes probability sharpness while preserving the chosen mode for positive temperatures. Evaluate calibration transfer independently when tasks or observations change.
