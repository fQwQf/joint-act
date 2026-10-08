"""Upload verified migration assets. Authentication is read once from stdin, never saved."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import requests


def checksum(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--release-id", type=int, required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        raise ValueError("Pass a GitHub credential on stdin")
    session = requests.Session()
    session.headers.update(
        Authorization="Bearer " + token,
        Accept="application/vnd.github+json",
        **{"X-GitHub-Api-Version": "2022-11-28"},
    )
    base = f"https://api.github.com/repos/{args.repository}/releases/{args.release_id}"
    response = session.get(base, timeout=60)
    response.raise_for_status()
    release = response.json()
    upload = release["upload_url"].split("{")[0]
    if not upload.startswith("https://uploads.github.com/"):
        raise ValueError("Unexpected upload endpoint")

    def assets():
        found = {}
        page = 1
        while True:
            response = session.get(base + "/assets", params=dict(per_page=100, page=page), timeout=60)
            response.raise_for_status()
            rows = response.json()
            found.update({item["name"]: item for item in rows})
            if len(rows) < 100:
                return found
            page += 1

    current = assets()
    directory = Path(args.directory)
    manifest_paths = sorted(directory.glob("*.json"))
    part_names = set()
    for manifest in manifest_paths:
        data = json.loads(manifest.read_text())
        if data.get("format") == "jointact-migration-v1":
            part_names.update(part["name"] for part in data["parts"])
    paths = [directory / name for name in sorted(part_names)] + manifest_paths
    if (directory / "SHA256SUMS").is_file():
        paths.append(directory / "SHA256SUMS")
    if not part_names:
        raise ValueError("No migration manifests found")
    uploaded = []
    for path in paths:
        expected = "sha256:" + checksum(path)
        if path.name in current:
            item = current[path.name]
            if item.get("digest") != expected or item["size"] != path.stat().st_size:
                raise ValueError(f"Existing asset does not match: {path.name}")
            print(json.dumps(dict(asset=path.name, status="already_verified")), flush=True)
        else:
            for attempt in range(3):
                try:
                    with open(path, "rb") as stream:
                        response = session.post(
                            upload,
                            params=dict(name=path.name),
                            data=stream,
                            headers={
                                "Content-Type": "application/octet-stream",
                                "Content-Length": str(path.stat().st_size),
                            },
                            timeout=(60, 1800),
                        )
                    response.raise_for_status()
                    item = response.json()
                    break
                except requests.RequestException:
                    # Recover ambiguous success without overwriting any existing asset.
                    existing = assets().get(path.name)
                    if existing and existing.get("digest") == expected:
                        item = existing
                        break
                    if existing or attempt == 2:
                        raise RuntimeError(
                            f"Upload failed for {path.name}; inspect the release before retrying"
                        ) from None
                    time.sleep(5 * (attempt + 1))
            if item.get("digest") != expected or item["size"] != path.stat().st_size:
                raise ValueError(f"GitHub did not confirm the expected digest: {path.name}")
            print(
                json.dumps(dict(asset=path.name, bytes=item["size"], status="uploaded_and_verified")),
                flush=True,
            )
        uploaded.append(dict(name=path.name, bytes=item["size"], digest=item["digest"], id=item["id"]))
    # Keep receipt outside the directory to avoid recursively uploading it on retries.
    receipt = directory.parent / (directory.name + "-upload-receipt.json")
    receipt.write_text(
        json.dumps(dict(repository=args.repository, release_id=args.release_id, assets=uploaded), indent=2)
        + "\n"
    )
    print(json.dumps(dict(completed=True, assets=len(uploaded), receipt=str(receipt))), flush=True)


if __name__ == "__main__":
    main()
