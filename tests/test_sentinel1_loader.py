from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from skimage import io as skio

from src.datasets.sentinel1_loader import prepare_sentinel1_record
from src.stage1.pipeline import load_sample


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_sentinel1_fixture(root: Path) -> Path:
    safe_root = root / "S1A_IW_GRDH_1SDV_20240105T180551_20240105T180616_051975_0647CF_FIXT.SAFE"
    measurement_dir = safe_root / "measurement"
    annotation_dir = safe_root / "annotation"
    calibration_dir = annotation_dir / "calibration"
    measurement_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    vv_image = (np.arange(64, dtype=np.uint16).reshape(8, 8) + 1).astype(np.uint16)
    vh_image = np.flipud(vv_image)
    skio.imsave(measurement_dir / "s1a-iw-grd-vv-fixture.tiff", vv_image, check_contrast=False)
    skio.imsave(measurement_dir / "s1a-iw-grd-vh-fixture.tiff", vh_image, check_contrast=False)

    _write_text(
        safe_root / "manifest.safe",
        "<xfdu:XFDU xmlns:xfdu='urn:ccsds:schema:xfdu:1'><metadataSection /></xfdu:XFDU>",
    )
    _write_text(
        annotation_dir / "s1a-iw-grd-vv-fixture.xml",
        """
        <product>
          <adsHeader>
            <polarisation>VV</polarisation>
          </adsHeader>
        </product>
        """.strip(),
    )
    _write_text(
        calibration_dir / "noise-s1a-iw-grd-vv-fixture.xml",
        """
        <noise>
          <noiseRangeVectorList count="2">
            <noiseRangeVector>
              <pixel>0 7</pixel>
              <noiseRangeLut>0.1 0.2</noiseRangeLut>
            </noiseRangeVector>
            <noiseRangeVector>
              <pixel>0 7</pixel>
              <noiseRangeLut>0.2 0.3</noiseRangeLut>
            </noiseRangeVector>
          </noiseRangeVectorList>
        </noise>
        """.strip(),
    )
    _write_text(
        calibration_dir / "calibration-s1a-iw-grd-vv-fixture.xml",
        "<calibration><betaNought>1.0</betaNought></calibration>",
    )

    archive_path = root / f"{safe_root.name}.zip"
    with ZipFile(archive_path, "w") as archive:
        for file_path in safe_root.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(root).as_posix())
    return archive_path


def test_prepare_sentinel1_record_extracts_safe_zip_and_loads_sample(monkeypatch) -> None:
    fixture_root = Path.cwd() / "outputs" / "test_sentinel1_loader_fixture"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    fixture_root.mkdir(parents=True, exist_ok=True)

    archive_path = _build_sentinel1_fixture(fixture_root)
    monkeypatch.setenv("SAR_DATA_LAYOUT_ROOT", str((fixture_root / "layout").resolve()))
    record = {
        "record_type": "product",
        "dataset": "sentinel1",
        "sample_id": "fixture",
        "split": "all",
        "image_path": archive_path.resolve().as_posix(),
        "local_target_path": archive_path.resolve().as_posix(),
        "product_family": "GRD",
        "metadata_json": "",
    }

    prepared = prepare_sentinel1_record(record, repo_root=Path.cwd(), force_reextract=True)
    assert prepared.usable is True
    assert prepared.image_path is not None
    assert "-vv-" in prepared.image_path.name.lower()
    assert prepared.noise_xml_path is not None

    sample = load_sample(record, "sentinel1")
    assert sample.intensity_image.shape == (8, 8)
    assert sample.pixel_domain == "amplitude"
    assert sample.metadata["noise_vector"].shape == (8,)
    assert "Selected" in sample.source_note
    shutil.rmtree(fixture_root)
