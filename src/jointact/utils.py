"""Reproducibility, atomic writes, resource checks, and distributed helpers."""

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def check_disk(path: str | Path, min_gb: float) -> None:
    path = Path(path)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free / 1024**3
    if free < min_gb:
        raise RuntimeError(f"Only {free:.2f} GiB free at {path}; require {min_gb:.2f} GiB")


def gpu_inventory() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    records = []
    for line in result.stdout.splitlines():
        idx, uuid, name, total, used, util = [x.strip() for x in line.split(",")]
        records.append(dict(index=int(idx), uuid=uuid, name=name, total_mib=int(total), used_mib=int(used), utilization=int(util)))
    return records


def check_gpu(device: str, allow_shared: bool = False) -> None:
    if not device.startswith("cuda") or allow_shared:
        return
    logical = int(device.split(":")[1]) if ":" in device else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    physical = visible[logical].strip() if visible != [""] else str(logical)
    match = [r for r in gpu_inventory() if str(r["index"]) == physical or r["uuid"].startswith(physical)]
    if len(match) != 1:
        raise RuntimeError(f"Cannot map device {device} to physical GPU {physical!r}")
    gpu = match[0]
    if gpu["used_mib"] > 1024 or gpu["utilization"] > 10:
        raise RuntimeError(f"GPU occupied: {gpu}; select a free GPU or explicitly allow sharing")


def distributed_context(device: str) -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        if device.startswith("cuda"):
            device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    chosen = torch.device(device)
    if chosen.type == "cuda":
        torch.cuda.set_device(chosen)
    return rank, world, chosen


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
