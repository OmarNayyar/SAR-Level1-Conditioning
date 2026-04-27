from __future__ import annotations

import shutil
from pathlib import Path

from src.datasets.common import deserialize_json_field, read_csv_rows
from src.datasets.custom_loader import build_custom_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_custom_manifest_indexes_future_local_dataset_layout() -> None:
    workspace_root = REPO_ROOT / "outputs" / "test_custom_loader_fixture"
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    dataset_root = workspace_root / "partner_drop"
    _touch(dataset_root / "train" / "images" / "frame_001.png")
    _touch(dataset_root / "train" / "labels" / "frame_001.json", "{}")
    _touch(dataset_root / "test" / "images" / "frame_002.png")
    _touch(dataset_root / "test" / "labels" / "frame_002.json", "{}")

    manifest_path = workspace_root / "partner_manifest.csv"
    try:
        rows = build_custom_manifest(
            dataset_root,
            manifest_path,
            dataset_name="partner_internal_v1",
            pixel_domain="intensity",
            source_name="partner team",
            notes="Restricted handoff",
            extra_metadata={"source_access": "internal"},
        )

        assert len(rows) == 2
        assert {row["sample_id"] for row in rows} == {"train/images/frame_001", "test/images/frame_002"}

        rows_by_split = {row["split"]: row for row in rows}
        assert rows_by_split["train"]["annotation_path"].endswith("frame_001.json")
        assert rows_by_split["test"]["annotation_path"].endswith("frame_002.json")
        assert rows_by_split["train"]["metadata_json"]["pixel_domain"] == "intensity"
        assert rows_by_split["train"]["metadata_json"]["source_name"] == "partner team"

        persisted_rows = read_csv_rows(manifest_path)
        persisted_metadata = deserialize_json_field(persisted_rows[0]["metadata_json"])
        assert persisted_rows[0]["dataset"] == "partner_internal_v1"
        assert persisted_metadata["source_access"] == "internal"
    finally:
        if workspace_root.exists():
            shutil.rmtree(workspace_root)
