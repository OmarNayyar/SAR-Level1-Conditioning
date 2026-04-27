from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image

from src.datasets.common import write_csv
from src.downstream.detection.yolo_dataset import load_prepared_yolo_dataset, prepare_yolo_dataset
from src.reporting.demo_examples import _valid_image_path


def _workspace() -> Path:
    path = Path("outputs") / "test-workspaces" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_prepare_yolo_dataset_converts_voc_boxes() -> None:
    workspace = _workspace()
    image_path = workspace / "000001.jpg"
    Image.new("RGB", (10, 20), color=(128, 128, 128)).save(image_path)
    annotation_path = workspace / "000001.xml"
    annotation_path.write_text(
        """
<annotation>
  <filename>000001.jpg</filename>
  <size><width>10</width><height>20</height></size>
  <object>
    <name>ship</name>
    <bndbox><xmin>2</xmin><ymin>4</ymin><xmax>8</xmax><ymax>14</ymax></bndbox>
  </object>
</annotation>
""".strip(),
        encoding="utf-8",
    )
    manifest_path = workspace / "manifest.csv"
    write_csv(
        manifest_path,
        [
            {
                "record_type": "sample",
                "dataset": "ssdd",
                "sample_id": "000001",
                "split": "train",
                "image_path": image_path.as_posix(),
                "annotation_path": annotation_path.as_posix(),
                "width": "10",
                "height": "20",
                "annotation_count": "1",
            },
            {
                "record_type": "sample",
                "dataset": "ssdd",
                "sample_id": "000002",
                "split": "train",
                "image_path": image_path.as_posix(),
                "annotation_path": annotation_path.as_posix(),
                "width": "10",
                "height": "20",
                "annotation_count": "1",
            },
        ],
    )

    prepared = prepare_yolo_dataset(
        dataset_name="ssdd",
        manifest_path=manifest_path,
        output_root=workspace / "yolo",
        variant="raw",
        limit_per_split=None,
    )

    assert prepared.status == "prepared"
    assert prepared.box_count == 2
    assert (prepared.root / "prepared_summary.json").exists()
    loaded = load_prepared_yolo_dataset(prepared.root)
    assert loaded is not None
    assert loaded.image_count == prepared.image_count
    label_files = sorted((prepared.root / "labels").rglob("*.txt"))
    assert label_files
    assert label_files[0].read_text(encoding="utf-8").strip() == "0 0.50000000 0.45000000 0.60000000 0.50000000"


def test_demo_image_validation_rejects_repo_root() -> None:
    workspace = _workspace()
    image_path = workspace / "panel.png"
    Image.new("L", (4, 4), color=128).save(image_path)

    assert _valid_image_path(workspace.as_posix(), workspace) == ""
    assert _valid_image_path(image_path.as_posix(), workspace).endswith("panel.png")
