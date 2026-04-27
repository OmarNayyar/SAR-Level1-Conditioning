from __future__ import annotations

from src.datasets.sentinel1_catalog import merge_manifest_rows, product_from_manifest_row


def test_merge_manifest_rows_preserves_existing_local_state() -> None:
    existing_rows = [
        {
            "product_id": "scene-1",
            "sample_id": "scene-1",
            "status": "complete",
            "download_status": "complete",
            "prepared_status": "ready",
            "image_path": "C:/prepared/image.tif",
            "prepared_image_path": "C:/prepared/image.tif",
            "notes": "local prep complete",
            "metadata_json": {"geofootprint": {"type": "Polygon"}, "content_type": "application/octet-stream"},
        }
    ]
    incoming_rows = [
        {
            "product_id": "scene-1",
            "sample_id": "scene-1",
            "status": "metadata-only",
            "download_status": "metadata-only",
            "prepared_status": "",
            "image_path": "",
            "prepared_image_path": "",
            "notes": "",
            "metadata_json": {"footprint": "POLYGON(...)", "content_type": "application/octet-stream"},
        }
    ]

    merged = merge_manifest_rows(existing_rows, incoming_rows)
    assert len(merged) == 1
    assert merged[0]["status"] == "complete"
    assert merged[0]["prepared_status"] == "ready"
    assert merged[0]["image_path"] == "C:/prepared/image.tif"
    assert "geofootprint" in merged[0]["metadata_json"]
    assert "footprint" in merged[0]["metadata_json"]


def test_product_from_manifest_row_reconstructs_downloadable_product() -> None:
    product = product_from_manifest_row(
        {
            "product_id": "scene-2",
            "product_name": "S1A_IW_GRDH_1SDV_20240209T172709_20240209T172734_052485_06590B_176C_COG.SAFE",
            "product_type": "GRDH",
            "product_family": "GRD",
            "mode": "IW",
            "sensing_start": "2024-02-09T17:27:09.000000Z",
            "sensing_end": "2024-02-09T17:27:34.000000Z",
            "expected_size_bytes": "1099161241",
            "online": "true",
            "eviction_date": "",
            "s3_path": "/eodata/example",
            "metadata_json": {
                "content_type": "application/octet-stream",
                "geofootprint": {"type": "Polygon"},
                "footprint": "POLYGON((...))",
            },
        }
    )
    assert product.product_id == "scene-2"
    assert product.product_family == "GRD"
    assert product.content_length == 1099161241
    assert product.online is True
