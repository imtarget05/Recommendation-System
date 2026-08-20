from __future__ import annotations

import hashlib

from data.scripts.manifest import DatasetManifest, ManifestDataset, sha256_of


def test_manifest_roundtrip(tmp_path) -> None:  # noqa: ANN001
    m = DatasetManifest(
        dataset=ManifestDataset(name="test", source="unit", version="v1", row_count=42),
        splits={"train": 30, "validation": 6, "test": 6},
    )
    path = tmp_path / "manifest.yaml"
    m.save(path)
    loaded = DatasetManifest.load(path)
    assert loaded.dataset.name == "test"
    assert loaded.dataset.row_count == 42
    assert loaded.splits == {"train": 30, "validation": 6, "test": 6}
    assert loaded.generated_at is not None


def test_sha256_of_file(tmp_path) -> None:  # noqa: ANN001
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    digest = sha256_of(f)
    expected = hashlib.sha256(b"hello\n").hexdigest()
    assert digest == expected


def test_sha256_of_directory_returns_none(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "sub").mkdir()
    assert sha256_of(tmp_path) is None
