"""Fetch or verify the exact public OpenVLA base used by the migration checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "docs/evidence/pretrained7b-source.json").read_text())
    destination = Path(args.output).resolve()
    expected = {name: item["sha256"] for name, item in spec["files"].items()}
    expected.update(spec["metadata_files"])
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=spec["model_id"],
            revision=spec["revision"],
            local_dir=str(destination),
            allow_patterns=list(expected),
        )
    for name, checksum in expected.items():
        digest = hashlib.sha256()
        with open(destination / name, "rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Base model checksum mismatch: {name}")
    print(
        json.dumps(dict(model_id=spec["model_id"], revision=spec["revision"], verified_files=len(expected)))
    )


if __name__ == "__main__":
    main()
