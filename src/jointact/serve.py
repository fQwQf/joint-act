"""JSON HTTP endpoint with RGB image payloads and batched inference."""

import base64
import binascii
import io
import threading

import numpy as np
from PIL import Image


def create_app(runtime, max_batch=16, max_image_bytes=4 * 1024 * 1024):
    from fastapi import FastAPI, HTTPException
    app = FastAPI(title="JointAct", version="0.1.0")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return dict(status="ready", backbone=runtime.config.model.backbone, horizon=runtime.config.model.horizon,
                    action_semantics=runtime.metadata["action_semantics"])

    @app.post("/predict")
    def predict(payload: dict):
        try:
            records = payload.get("observations", [payload])
            if not isinstance(records, list) or not 1 <= len(records) <= max_batch:
                raise ValueError(f"Supply 1..{max_batch} observations")
            parsed = []
            for record in records:
                images = []
                if len(record["images"]) != runtime.config.model.num_images:
                    raise ValueError("Wrong camera count")
                for encoded in record["images"]:
                    if not isinstance(encoded, str) or len(encoded) > max_image_bytes * 4 // 3 + 8:
                        raise ValueError("Image must be a bounded base64-encoded PNG/JPEG")
                    raw = base64.b64decode(encoded, validate=True)
                    if len(raw) > max_image_bytes:
                        raise ValueError("Image payload too large")
                    with Image.open(io.BytesIO(raw)) as image:
                        if image.width * image.height > 4096 * 4096:
                            raise ValueError("Image resolution too large")
                        images.append(np.asarray(image.convert("RGB")).copy())
                parsed.append(dict(images=images, proprio=record.get("proprio", []), instruction=record["instruction"]))
            with lock:
                results = runtime.predict_batch(parsed)
            return dict(predictions=results)
        except (ValueError, KeyError, TypeError, binascii.Error, OSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
