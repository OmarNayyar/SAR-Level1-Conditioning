from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
SLC_BURST_COLLECTION_CONCEPT_ID = "C2709161906-ASF"
DEFAULT_GRD_OUTPUT = "data/external/sentinel1_grd_crosspol"
DEFAULT_SLC_OUTPUT = "data/external/sentinel1_slc_bursts"


def _earthdata_credentials(required: bool) -> tuple[str | None, str | None, list[str]]:
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    warnings: list[str] = []
    if not username or not password:
        message = (
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD are not set. "
            "Set them in PowerShell with: "
            "$env:EARTHDATA_USERNAME='your_username'; $env:EARTHDATA_PASSWORD='your_password'"
        )
        if required:
            raise RuntimeError(message)
        warnings.append(message)
    return username, password, warnings


def _search_params(kind: str, max_results: int) -> dict[str, str | int]:
    if kind == "grd-crosspol":
        return {
            "platform": "Sentinel-1A,Sentinel-1B",
            "processingLevel": "GRD_HD,GRD_MD",
            "beamMode": "IW,EW",
            "polarization": "VV+VH,HH+HV",
            "output": "geojson",
            "maxResults": max_results,
        }
    return {
        "dataset": "SLC-BURST",
        "processingLevel": "BURST",
        "beamMode": "IW",
        "polarization": "VV+VH,HH+HV",
        "output": "geojson",
        "maxResults": max_results,
    }


def _parse_size_bytes(value: object) -> int | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return int(number * 1024 * 1024) if number < 100000 else int(number)
    text = str(value).strip()
    if not text:
        return None
    parts = text.replace(",", "").split()
    try:
        number = float(parts[0])
    except (ValueError, IndexError):
        return None
    unit = parts[1].lower() if len(parts) > 1 else "mb"
    if unit.startswith("gb"):
        return int(number * 1024**3)
    if unit.startswith("kb"):
        return int(number * 1024)
    if unit.startswith("b"):
        return int(number)
    return int(number * 1024**2)


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def _result_properties(item: dict[str, object]) -> dict[str, object]:
    properties = item.get("properties")
    return properties if isinstance(properties, dict) else item


def _download_url(properties: dict[str, object]) -> str:
    for key in ("url", "downloadUrl", "download_url", "productUrl"):
        value = properties.get(key)
        if value:
            return str(value)
    return ""


def _candidate_row(item: dict[str, object]) -> dict[str, object]:
    properties = _result_properties(item)
    size_bytes = _parse_size_bytes(
        properties.get("bytes")
        or properties.get("sizeBytes")
        or properties.get("fileSize")
        or properties.get("sizeMB")
        or properties.get("size")
    )
    return {
        "scene": str(
            properties.get("sceneName")
            or properties.get("granuleName")
            or properties.get("fileID")
            or properties.get("title")
            or ""
        ),
        "platform": str(properties.get("platform") or properties.get("platformName") or ""),
        "date": str(properties.get("startTime") or properties.get("processingDate") or properties.get("stopTime") or ""),
        "beam_mode": str(properties.get("beamMode") or properties.get("beamModeType") or ""),
        "polarization": str(properties.get("polarization") or properties.get("polarizationType") or ""),
        "processing_level": str(properties.get("processingLevel") or ""),
        "size_bytes": size_bytes,
        "size": _human_bytes(size_bytes),
        "url": _download_url(properties),
    }


def _search_asf(kind: str, max_results: int) -> list[dict[str, object]]:
    response = requests.get(ASF_SEARCH_URL, params=_search_params(kind, max_results), timeout=60)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("features"), list):
        items = payload["features"]
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        items = payload["results"]
    else:
        items = []
    rows = [_candidate_row(item) for item in items if isinstance(item, dict)]
    if kind == "slc-burst" and not rows:
        rows = _search_slc_burst_cmr(max_results)
    rows.sort(key=lambda row: row["size_bytes"] if isinstance(row.get("size_bytes"), int) else 10**18)
    return rows[:max_results]


def _search_slc_burst_cmr(max_results: int) -> list[dict[str, object]]:
    response = requests.get(
        CMR_GRANULES_URL,
        params={
            "collection_concept_id": SLC_BURST_COLLECTION_CONCEPT_ID,
            "page_size": max_results,
            "sort_key": "-start_date",
        },
        timeout=60,
    )
    response.raise_for_status()
    entries = response.json().get("feed", {}).get("entry", [])
    rows: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        links = entry.get("links", []) if isinstance(entry.get("links"), list) else []
        data_url = ""
        for link in links:
            if not isinstance(link, dict):
                continue
            href = str(link.get("href", ""))
            rel = str(link.get("rel", ""))
            if href and ("data#" in rel or href.lower().endswith((".tif", ".tiff", ".xml", ".zip"))):
                data_url = href
                break
        title = str(entry.get("producer_granule_id") or entry.get("title") or entry.get("granule_ur") or "")
        polarization = ""
        for token in ("_VV_", "_VH_", "_HH_", "_HV_"):
            if token in title:
                polarization = token.strip("_")
                break
        rows.append(
            {
                "scene": title,
                "platform": "Sentinel-1",
                "date": str(entry.get("time_start") or ""),
                "beam_mode": "IW/EW burst",
                "polarization": polarization,
                "processing_level": "BURST",
                "size_bytes": _parse_size_bytes(entry.get("granule_size")),
                "size": _human_bytes(_parse_size_bytes(entry.get("granule_size"))),
                "url": data_url,
                "source": "CMR fallback for ASF SLC Burst collection",
            }
        )
    return rows


def _download_candidate(
    candidate: dict[str, object],
    *,
    output_root: Path,
    username: str,
    password: str,
    max_download_bytes: int,
    force: bool,
) -> Path:
    url = str(candidate.get("url") or "")
    if not url:
        raise RuntimeError(f"Candidate {candidate.get('scene', '')!r} does not expose a download URL.")
    size_bytes = candidate.get("size_bytes")
    if not isinstance(size_bytes, int) and not force:
        raise RuntimeError("Candidate size is unknown. Re-run with --force if you intentionally want to download it.")
    if isinstance(size_bytes, int) and size_bytes > max_download_bytes and not force:
        raise RuntimeError(
            f"Candidate is {_human_bytes(size_bytes)}, above the {_human_bytes(max_download_bytes)} safety limit. "
            "Re-run with --force or choose a smaller candidate."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name or f"{candidate.get('scene', 'sentinel1_product')}.zip"
    output_path = output_root / filename
    with requests.Session() as session:
        response = session.get(url, auth=(username, password), stream=True, timeout=120)
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return output_path


def _write_sample_manifest(output_root: Path, result: dict[str, object]) -> Path:
    manifest = dict(result)
    downloaded = []
    for item in result.get("downloaded", []) or []:
        path = Path(str(item))
        downloaded.append(
            {
                "path": path.as_posix(),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "size": _human_bytes(path.stat().st_size) if path.exists() else "missing",
            }
        )
    manifest["downloaded_files"] = downloaded
    manifest_path = output_root / "sample_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search and optionally download tiny Sentinel-1 proof-of-concept samples.")
    parser.add_argument("--kind", choices=["slc-burst", "grd-crosspol"], required=True)
    parser.add_argument("--provider", choices=["asf"], default="asf")
    parser.add_argument("--max-results", type=int, default=1)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow unknown or above-threshold download sizes.")
    parser.add_argument("--max-download-gb", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root_text = args.output_root or (DEFAULT_SLC_OUTPUT if args.kind == "slc-burst" else DEFAULT_GRD_OUTPUT)
    output_root = (REPO_ROOT / output_root_text).resolve() if not Path(output_root_text).is_absolute() else Path(output_root_text)
    username, password, warnings = _earthdata_credentials(required=not args.dry_run)
    candidates = _search_asf(args.kind, max(args.max_results, 1))
    result: dict[str, object] = {
        "kind": args.kind,
        "provider": args.provider,
        "dry_run": bool(args.dry_run),
        "output_root": output_root.as_posix(),
        "credential_warnings": warnings,
        "candidates": candidates,
    }
    if args.dry_run:
        if args.kind == "slc-burst":
            result["manual_fallback"] = (
                "ASF SLC-BURST search is automated here. If direct burst download requires extra ASF/asf_search handling, "
                "open the candidate URL in ASF Vertex or use asf_search with Earthdata credentials."
            )
        print(json.dumps(result, indent=2))
        return

    if args.kind == "slc-burst":
        result["status"] = "manual-download-required"
        result["manual_fallback"] = (
            "SLC burst search completed, but direct burst download is intentionally left manual until the desired burst product is confirmed. "
            "Use the candidate metadata/URL with ASF Vertex or asf_search."
        )
        result["manifest_path"] = _write_sample_manifest(output_root, result).as_posix()
        print(json.dumps(result, indent=2))
        return

    downloaded = [
        _download_candidate(
            candidate,
            output_root=output_root,
            username=str(username),
            password=str(password),
            max_download_bytes=int(float(args.max_download_gb) * 1024**3),
            force=bool(args.force),
        ).resolve().as_posix()
        for candidate in candidates
    ]
    result["status"] = "downloaded"
    result["downloaded"] = downloaded
    result["manifest_path"] = _write_sample_manifest(output_root, result).as_posix()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
