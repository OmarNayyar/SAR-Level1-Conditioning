from __future__ import annotations

from pathlib import Path

from src.datasets.sentinel1_catalog import (
    Sentinel1Query,
    _matches_requested_filters,
    odata_to_product,
    products_to_manifest_rows,
)


def _odata_item(product_id: str, name: str, content_length: int = 1024) -> dict[str, object]:
    return {
        "Id": product_id,
        "Name": name,
        "ContentDate": {
            "Start": "2024-01-05T18:05:51.000000Z",
            "End": "2024-01-05T18:06:16.000000Z",
        },
        "ContentLength": content_length,
        "ContentType": "application/zip",
        "Online": True,
        "EvictionDate": None,
        "S3Path": None,
        "GeoFootprint": None,
        "Footprint": None,
    }


def test_auxiliary_products_are_excluded_by_default() -> None:
    auxiliary = odata_to_product(_odata_item("aux-1", "S1A_AUX_POEORB_20240105T000000_V20240104T225942_20240106T005942.EOF"))
    query = Sentinel1Query(product_type="GRD", mode="IW", max_results=3)
    assert auxiliary.is_auxiliary
    assert not _matches_requested_filters(auxiliary, query)


def test_exact_product_type_filter_does_not_mix_slc_and_grd() -> None:
    slc = odata_to_product(_odata_item("slc-1", "S1A_IW_SLC__1SDV_20240105T180551_20240105T180616_051975_0647CF_1111"))
    grd = odata_to_product(_odata_item("grd-1", "S1A_IW_GRDH_1SDV_20240105T180551_20240105T180616_051975_0647CF_2222"))
    slc_query = Sentinel1Query(product_type="SLC", mode="IW", max_results=3)
    grd_query = Sentinel1Query(product_type="GRD", mode="IW", max_results=3)
    assert _matches_requested_filters(slc, slc_query)
    assert not _matches_requested_filters(grd, slc_query)
    assert _matches_requested_filters(grd, grd_query)
    assert not _matches_requested_filters(slc, grd_query)


def test_manifest_rows_preserve_target_path_and_status() -> None:
    product = odata_to_product(_odata_item("grd-1", "S1A_IW_GRDH_1SDV_20240105T180551_20240105T180616_051975_0647CF_2222", content_length=2048))
    rows = products_to_manifest_rows([product], sentinel1_root=Path("C:/tmp/sentinel1"))
    assert rows[0]["product_id"] == "grd-1"
    assert rows[0]["product_type"] == "GRDH"
    assert rows[0]["download_status"] == "metadata-only"
    assert rows[0]["expected_size_bytes"] == 2048
    assert rows[0]["local_target_path"].endswith("grd/S1A_IW_GRDH_1SDV_20240105T180551_20240105T180616_051975_0647CF_2222.zip")
