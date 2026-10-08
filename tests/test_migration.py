"""Migration archives must round-trip bytes and reject corruption/traversal."""

import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


@pytest.fixture
def migration():
    if not shutil.which("zstd"):
        pytest.skip("Migration packaging requires the zstd executable")
    path = Path(__file__).resolve().parents[1] / "scripts/migration.py"
    spec = importlib.util.spec_from_file_location("migration_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunked_restore_and_corruption(migration, tmp_path):
    root = tmp_path / "source"
    (root / "episodes").mkdir(parents=True)
    content = os.urandom(2 * 1024**2 + 17)
    (root / "episodes/test.bin").write_bytes(content)
    (root / "episodes/empty").mkdir()
    destination = tmp_path / "parts"
    migration.pack(root, destination, "data", ["episodes"], chunk_mib=1)
    manifest = destination / "data.json"
    assert len(migration.verified_parts(manifest)) == 3
    migration.restore(manifest, tmp_path / "restored")
    assert (tmp_path / "restored/episodes/test.bin").read_bytes() == content
    assert (tmp_path / "restored/episodes/empty").is_dir()
    with pytest.raises(FileExistsError):
        migration.restore(manifest, tmp_path / "restored")
    part = destination / json.loads(manifest.read_text())["parts"][0]["name"]
    with open(part, "r+b") as stream:
        stream.write(b"broken")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        migration.verified_parts(manifest)


def test_restore_rejects_path_traversal(migration, tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    part = tmp_path / "bad.part-000"
    compressed = subprocess.run(["zstd", "-q"], input=buffer.getvalue(), stdout=subprocess.PIPE, check=True)
    part.write_bytes(compressed.stdout)
    manifest = tmp_path / "bad.json"
    manifest.write_text(
        json.dumps(
            dict(
                format="jointact-migration-v1",
                parts=[dict(name=part.name, bytes=part.stat().st_size, sha256=migration.digest(part))],
            )
        )
    )
    with pytest.raises(ValueError, match="Unsafe archive entry"):
        migration.restore(manifest, tmp_path / "restore")
    assert not (tmp_path / "escape").exists()
