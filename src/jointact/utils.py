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


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = configured.split(",") if configured is not None else None
    if visible is not None and (configured in {"", "-1"} or logical >= len(visible)):
        raise RuntimeError(f"{device} is hidden by CUDA_VISIBLE_DEVICES")
    physical = visible[logical].strip() if visible is not None else str(logical)
    match = [r for r in gpu_inventory() if str(r["index"]) == physical or r["uuid"].startswith(physical)]
    if len(match) != 1:
        raise RuntimeError(f"Cannot map device {device} to physical GPU {physical!r}")
    gpu = match[0]
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    other_pids = []
    for line in processes.stdout.splitlines():
        uuid, pid = [part.strip() for part in line.split(",")]
        if uuid == gpu["uuid"] and int(pid) != os.getpid():
            other_pids.append(int(pid))
    if other_pids:
        raise RuntimeError(f"GPU occupied by compute processes {other_pids}: {gpu}")
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


def rank_zero_call(function):
    """Propagate rank-zero I/O failures instead of leaving other ranks at a barrier."""
    if not dist.is_initialized():
        return function()
    message = [None]
    if dist.get_rank() == 0:
        try:
            message[0] = dict(value=function())
        except Exception as error:
            message[0] = dict(error=f"{type(error).__name__}: {error}")
    dist.broadcast_object_list(message, src=0)
    if "error" in message[0]:
        raise RuntimeError(f"Rank-zero operation failed: {message[0]['error']}")
    return message[0]["value"]


def require_finite(value, message):
    """Every rank must agree before entering the next backward/optimizer collective."""
    finite = torch.isfinite(value).all().to(dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError(message)


def source_fingerprint():
    root = Path(__file__).parent
    files = {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*.py"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=files)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
