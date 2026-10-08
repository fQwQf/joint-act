"""Pack, verify and restore chunked migration archives without importing model code."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def pack(root, output, name, members, chunk_mib=512):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not name.replace("-", "").isalnum() or chunk_mib < 1 or chunk_mib >= 2048:
        raise ValueError("Use an alphanumeric archive name and 1–2047 MiB chunks")
    for item in members:
        path = (root / item).resolve()
        if not path.is_relative_to(root) or path == root or not path.exists():
            raise ValueError(f"Invalid archive member: {item}")
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob(name + "*")):
        raise FileExistsError("Archive already exists; preserve it or select another output/name")
    # argv lists avoid shell interpolation. Existing source files are never changed.
    tar = subprocess.Popen(
        [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-C",
            str(root),
            "-cf",
            "-",
            "--",
            *members,
        ],
        stdout=subprocess.PIPE,
    )
    compressor = subprocess.Popen(["zstd", "-T2", "-3", "-q"], stdin=tar.stdout, stdout=subprocess.PIPE)
    tar.stdout.close()
    parts = []
    try:
        index, remaining, stream = 0, 0, None
        while True:
            block = compressor.stdout.read(min(8 * 1024**2, remaining) if remaining else 8 * 1024**2)
            if not block:
                break
            if stream is None:
                path = output / f"{name}.tar.zst.part-{index:03d}"
                stream = open(path, "xb")
                remaining = chunk_mib * 1024**2
                parts.append(path)
                index += 1
            # Read blocks can exceed a small test chunk size; split them explicitly.
            while block:
                piece, block = block[:remaining], block[remaining:]
                stream.write(piece)
                remaining -= len(piece)
                if not remaining:
                    stream.close()
                    stream = None
                    if block:
                        path = output / f"{name}.tar.zst.part-{index:03d}"
                        stream = open(path, "xb")
                        remaining = chunk_mib * 1024**2
                        parts.append(path)
                        index += 1
        if stream:
            stream.close()
        if compressor.wait() or tar.wait():
            raise RuntimeError("Archive compression failed; incomplete parts retained for inspection")
    finally:
        if tar.poll() is None:
            tar.kill()
        if compressor.poll() is None:
            compressor.kill()
    manifest = dict(
        format="jointact-migration-v1",
        name=name,
        members=members,
        codec="tar+zstd",
        parts=[dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in parts],
    )
    (output / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                manifest=str(output / f"{name}.json"),
                parts=len(parts),
                bytes=sum(p.stat().st_size for p in parts),
            )
        ),
        flush=True,
    )


def verified_parts(manifest):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    if data.get("format") != "jointact-migration-v1" or not data.get("parts"):
        raise ValueError("Invalid archive manifest")
    parts = []
    for part in data["parts"]:
        if Path(part["name"]).name != part["name"]:
            raise ValueError("Unsafe part name")
        path = manifest.parent / part["name"]
        if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
            raise ValueError(f"Checksum mismatch: {path.name}")
        parts.append(path)
    return parts


def restore(manifest, destination):
    parts = verified_parts(manifest)
    destination = Path(destination).resolve()
    # Each archive gets a fresh staging directory; merging into a live run is explicit.
    if destination.exists():
        raise FileExistsError("Restore into a fresh staging directory")
    destination.mkdir(parents=True)
    import threading

    decompressor = subprocess.Popen(["zstd", "-dcq"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    failures = []

    def feed():
        try:
            for part in parts:
                with open(part, "rb") as stream:
                    shutil.copyfileobj(stream, decompressor.stdin, 8 * 1024**2)
        except Exception as error:
            failures.append(error)
        finally:
            decompressor.stdin.close()

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        with tarfile.open(fileobj=decompressor.stdout, mode="r|") as archive:
            for member in archive:
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe archive entry: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, open(target, "xb") as output:
                        shutil.copyfileobj(source, output, 8 * 1024**2)
    except Exception:
        decompressor.kill()
        raise
    finally:
        # Drain trailing tar padding so the producer can close without a broken pipe.
        if decompressor.poll() is None:
            for _ in iter(lambda: decompressor.stdout.read(8 * 1024**2), b""):
                pass
        feeder.join()
        status = decompressor.wait()
    if failures or status:
        raise RuntimeError(f"Restore failed; retain staging for inspection (exit={status})")
    print(json.dumps(dict(restored=str(destination), verified_parts=len(parts))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--members", nargs="+", required=True)
    p.add_argument("--chunk-mib", type=int, default=512)
    p = sub.add_parser("verify")
    p.add_argument("--manifest", required=True)
    p = sub.add_parser("restore")
    p.add_argument("--manifest", required=True)
    p.add_argument("--destination", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "pack":
        pack(**args)
    elif command == "restore":
        restore(**args)
    else:
        print(json.dumps(dict(verified_parts=len(verified_parts(**args)))))


if __name__ == "__main__":
    main()
