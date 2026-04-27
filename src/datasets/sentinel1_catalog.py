from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from .common import (
    bbox_to_polygon_wkt,
    deserialize_json_field,
    human_bytes,
    parse_bbox,
    parse_bool,
    polygon_text_to_wkt,
    read_csv_rows,
    write_csv,
    write_json,
)


CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
AUXILIARY_PRODUCT_KEYWORDS = ("AUX", "PREORB", "RESORB", "POEORB", "OPOD", "EOF")
LOCAL_STATE_KEYS = {
    "image_path",
    "annotation_path",
    "status",
    "download_status",
    "notes",
    "prepared_product_path",
    "prepared_image_path",
    "measurement_paths",
    "measurement_count",
    "primary_polarization",
    "annotation_xml_path",
    "calibration_xml_path",
    "noise_xml_path",
    "manifest_safe_path",
    "noise_vector_path",
    "pixel_domain",
    "preparation_notes",
    "prepared_status",
}


@dataclass(slots=True)
class Sentinel1Query:
    product_type: str | None = None
    mode: str | None = None
    start: str | None = None
    end: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    polygon_wkt: str | None = None
    max_results: int = 10
    metadata_only: bool = True
    order_desc: bool = True
    include_auxiliary: bool = False


@dataclass(slots=True)
class Sentinel1Product:
    product_id: str
    name: str
    product_type: str
    product_family: str
    mode: str
    sensing_start: str
    sensing_end: str
    content_length: int | None
    content_type: str | None
    online: bool | None
    eviction_date: str | None
    s3_path: str | None
    geofootprint: dict[str, Any] | None
    footprint: str | None
    source_json: dict[str, Any]

    @property
    def domain_hint(self) -> str:
        if self.product_family == "SLC":
            return "complex_slc"
        if self.product_family == "GRD":
            return "detected_ground_range"
        return "unknown"

    @property
    def default_subdir(self) -> str:
        return "slc" if self.product_family == "SLC" else "grd"

    @property
    def is_auxiliary(self) -> bool:
        upper_name = self.name.upper()
        upper_type = self.product_type.upper()
        return any(keyword in upper_name or upper_type.startswith(keyword) for keyword in AUXILIARY_PRODUCT_KEYWORDS)


def _to_utc_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _sentinel1_product_family(name: str) -> str:
    tokens = name.split("_")
    if len(tokens) < 3:
        return "UNKNOWN"
    family_token = tokens[2]
    if family_token.startswith("SLC"):
        return "SLC"
    if family_token.startswith("GRD"):
        return "GRD"
    return family_token


def _sentinel1_product_type(name: str) -> str:
    tokens = name.split("_")
    return tokens[2] if len(tokens) > 2 else "UNKNOWN"


def _sentinel1_mode(name: str) -> str:
    tokens = name.split("_")
    return tokens[1] if len(tokens) > 1 else "UNKNOWN"


def normalize_product_type(product_type: str | None) -> str | None:
    if not product_type:
        return None
    normalized = product_type.strip().upper()
    if normalized in {"SLC", "GRD"}:
        return normalized
    return normalized


def build_filter(query: Sentinel1Query) -> str:
    parts = ["Collection/Name eq 'SENTINEL-1'"]
    if query.start:
        parts.append(f"ContentDate/Start ge {_to_utc_timestamp(query.start)}")
    if query.end:
        parts.append(f"ContentDate/Start le {_to_utc_timestamp(query.end)}")

    polygon_wkt = query.polygon_wkt
    if query.bbox:
        polygon_wkt = bbox_to_polygon_wkt(query.bbox)
    if polygon_wkt:
        parts.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon_wkt}')")

    normalized_product_type = normalize_product_type(query.product_type)
    if normalized_product_type and normalized_product_type not in {"SLC", "GRD"}:
        parts.append(
            "Attributes/OData.CSC.StringAttribute/any(att:"
            "att/Name eq 'productType' and "
            f"att/OData.CSC.StringAttribute/Value eq '{normalized_product_type}')"
        )

    return " and ".join(parts)


def odata_to_product(item: dict[str, Any]) -> Sentinel1Product:
    content_date = item.get("ContentDate", {})
    name = str(item["Name"])
    return Sentinel1Product(
        product_id=str(item["Id"]),
        name=name,
        product_type=_sentinel1_product_type(name),
        product_family=_sentinel1_product_family(name),
        mode=_sentinel1_mode(name),
        sensing_start=str(content_date.get("Start", "")),
        sensing_end=str(content_date.get("End", "")),
        content_length=item.get("ContentLength"),
        content_type=item.get("ContentType"),
        online=item.get("Online"),
        eviction_date=item.get("EvictionDate"),
        s3_path=item.get("S3Path"),
        geofootprint=item.get("GeoFootprint"),
        footprint=item.get("Footprint"),
        source_json=item,
    )


def _matches_requested_filters(product: Sentinel1Product, query: Sentinel1Query) -> bool:
    normalized_product_type = normalize_product_type(query.product_type)
    explicit_aux_request = bool(normalized_product_type) and any(
        keyword in normalized_product_type for keyword in AUXILIARY_PRODUCT_KEYWORDS
    )
    if product.is_auxiliary and not (query.include_auxiliary or explicit_aux_request):
        return False
    if normalized_product_type in {"SLC", "GRD"} and product.product_family != normalized_product_type:
        return False
    if normalized_product_type and normalized_product_type not in {"SLC", "GRD"} and product.product_type.upper() != normalized_product_type:
        return False
    if query.mode and product.mode.upper() != query.mode.upper():
        return False
    return True


def search_sentinel1_products(
    query: Sentinel1Query,
    *,
    timeout_seconds: int = 60,
    page_size: int = 100,
) -> list[Sentinel1Product]:
    session = requests.Session()
    session.trust_env = False
    results: list[Sentinel1Product] = []
    skip = 0

    while len(results) < query.max_results:
        params = {
            "$filter": build_filter(query),
            "$orderby": "ContentDate/Start desc" if query.order_desc else "ContentDate/Start asc",
            "$top": page_size,
            "$skip": skip,
        }
        response = session.get(CATALOG_URL, params=params, timeout=timeout_seconds)
        if not response.ok:
            raise requests.HTTPError(
                f"CDSE catalog request failed with {response.status_code}: {response.text[:500]}",
                response=response,
            )
        payload = response.json()
        batch = [odata_to_product(item) for item in payload.get("value", [])]
        if not batch:
            break
        for product in batch:
            if _matches_requested_filters(product, query):
                results.append(product)
                if len(results) >= query.max_results:
                    break
        skip += len(batch)
        if len(batch) < params["$top"]:
            break

    return results[: query.max_results]


def products_to_json(products: list[Sentinel1Product]) -> list[dict[str, Any]]:
    return [
        {
            "product_id": product.product_id,
            "name": product.name,
            "product_type": product.product_type,
            "product_family": product.product_family,
            "mode": product.mode,
            "domain_hint": product.domain_hint,
            "sensing_start": product.sensing_start,
            "sensing_end": product.sensing_end,
            "content_length": product.content_length,
            "content_length_human": human_bytes(product.content_length),
            "content_type": product.content_type,
            "online": product.online,
            "eviction_date": product.eviction_date,
            "s3_path": product.s3_path,
            "geofootprint": product.geofootprint,
            "footprint": product.footprint,
            "source_json": product.source_json,
        }
        for product in products
    ]


def product_target_path(product: Sentinel1Product, sentinel1_root: Path) -> Path:
    return sentinel1_root / product.default_subdir / f"{product.name}.zip"


def products_to_manifest_rows(products: list[Sentinel1Product], *, sentinel1_root: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product in products:
        target_path = product_target_path(product, sentinel1_root) if sentinel1_root is not None else Path()
        pixel_domain = "complex_slc" if product.product_family == "SLC" else "amplitude"
        rows.append(
            {
                "record_type": "product",
                "dataset": "sentinel1",
                "sample_id": product.product_id,
                "split": "all",
                "image_path": "",
                "annotation_path": "",
                "remote_source": f"cdse:{product.product_id}",
                "status": "metadata-only",
                "download_status": "metadata-only",
                "notes": "",
                "product_id": product.product_id,
                "product_name": product.name,
                "product_type": product.product_type,
                "product_family": product.product_family,
                "mode": product.mode,
                "domain_hint": product.domain_hint,
                "sensing_start": product.sensing_start,
                "sensing_end": product.sensing_end,
                "expected_size_bytes": product.content_length,
                "expected_size_human": human_bytes(product.content_length),
                "online": product.online,
                "eviction_date": product.eviction_date,
                "s3_path": product.s3_path,
                "local_target_path": target_path.resolve().as_posix() if sentinel1_root is not None else "",
                "prepared_product_path": "",
                "prepared_image_path": "",
                "measurement_count": "",
                "primary_polarization": "",
                "annotation_xml_path": "",
                "calibration_xml_path": "",
                "noise_xml_path": "",
                "manifest_safe_path": "",
                "metadata_json": {
                    "geofootprint": product.geofootprint,
                    "footprint": product.footprint,
                    "content_type": product.content_type,
                    "pixel_domain": pixel_domain,
                    "product_family": product.product_family,
                },
            }
        )
    return rows


def _row_identity(row: dict[str, Any]) -> str:
    return str(row.get("product_id") or row.get("sample_id") or "").strip()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return value in ([], {})


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return deserialize_json_field(value) or {}


def _has_meaningful_local_state(row: dict[str, Any]) -> bool:
    status = str(row.get("status", "")).strip().lower()
    prepared_status = str(row.get("prepared_status", "")).strip().lower()
    download_status = str(row.get("download_status", "")).strip().lower()
    if prepared_status in {"ready", "failed"}:
        return True
    if download_status in {"complete", "failed", "planned"}:
        return True
    return status in {"partial", "complete", "failed", "external-linked"}


def _merge_notes(existing: Any, incoming: Any) -> str:
    values: list[str] = []
    for candidate in (existing, incoming):
        text = str(candidate or "").strip()
        if text and text not in values:
            values.append(text)
    return " ".join(values)


def merge_manifest_row(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(incoming)
    for key, value in existing.items():
        merged.setdefault(key, value)

    keep_existing_state = _has_meaningful_local_state(existing)
    if keep_existing_state:
        for key in LOCAL_STATE_KEYS:
            existing_value = existing.get(key)
            if not _is_blank(existing_value):
                merged[key] = existing_value

    merged["notes"] = _merge_notes(existing.get("notes"), incoming.get("notes"))
    merged["preparation_notes"] = _merge_notes(existing.get("preparation_notes"), incoming.get("preparation_notes"))

    existing_metadata = _metadata_dict(existing.get("metadata_json"))
    incoming_metadata = _metadata_dict(incoming.get("metadata_json"))
    if existing_metadata or incoming_metadata:
        metadata = dict(existing_metadata)
        metadata.update(incoming_metadata)
        merged["metadata_json"] = metadata

    return merged


def merge_manifest_rows(existing_rows: list[dict[str, Any]], incoming_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_by_id = {_row_identity(row): dict(row) for row in existing_rows if _row_identity(row)}
    ordered_ids = [_row_identity(row) for row in existing_rows if _row_identity(row)]

    for incoming in incoming_rows:
        product_id = _row_identity(incoming)
        if not product_id:
            continue
        if product_id in existing_by_id:
            existing_by_id[product_id] = merge_manifest_row(existing_by_id[product_id], incoming)
        else:
            existing_by_id[product_id] = dict(incoming)
            ordered_ids.append(product_id)

    return [existing_by_id[product_id] for product_id in ordered_ids if product_id in existing_by_id]


def save_search_outputs(
    products: list[Sentinel1Product],
    *,
    json_path: Path,
    manifest_path: Path,
    sentinel1_root: Path | None = None,
) -> None:
    write_json(json_path, products_to_json(products))
    incoming_rows = products_to_manifest_rows(products, sentinel1_root=sentinel1_root)
    existing_rows = read_csv_rows(manifest_path) if manifest_path.exists() else []
    write_csv(manifest_path, merge_manifest_rows(existing_rows, incoming_rows))

def product_from_manifest_row(row: dict[str, Any]) -> Sentinel1Product:
    metadata = _metadata_dict(row.get("metadata_json"))
    expected_size = row.get("expected_size_bytes")
    content_length = int(expected_size) if str(expected_size).strip() else None
    return Sentinel1Product(
        product_id=str(row.get("product_id") or row.get("sample_id") or ""),
        name=str(row.get("product_name", "")),
        product_type=str(row.get("product_type", "")),
        product_family=str(row.get("product_family", "")),
        mode=str(row.get("mode", "")),
        sensing_start=str(row.get("sensing_start", "")),
        sensing_end=str(row.get("sensing_end", "")),
        content_length=content_length,
        content_type=metadata.get("content_type"),
        online=parse_bool(row.get("online"), default=True),
        eviction_date=row.get("eviction_date"),
        s3_path=row.get("s3_path"),
        geofootprint=metadata.get("geofootprint"),
        footprint=metadata.get("footprint"),
        source_json=metadata,
    )


def query_from_mapping(payload: dict[str, Any]) -> Sentinel1Query:
    bbox_value = payload.get("bbox")
    polygon_value = payload.get("polygon")
    return Sentinel1Query(
        product_type=payload.get("product_type"),
        mode=payload.get("mode"),
        start=payload.get("start"),
        end=payload.get("end"),
        bbox=parse_bbox(bbox_value) if bbox_value else None,
        polygon_wkt=polygon_text_to_wkt(polygon_value) if polygon_value else None,
        max_results=int(payload.get("max_results", 10)),
        metadata_only=bool(payload.get("metadata_only", True)),
        order_desc=bool(payload.get("order_desc", True)),
        include_auxiliary=bool(payload.get("include_auxiliary", False)),
    )
