from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from .common import StorageEstimate, ensure_storage_guard
from .sentinel1_catalog import Sentinel1Product


TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD_ZIP_URL_TEMPLATE = "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$zip"
DOWNLOAD_VALUE_URL_TEMPLATE = "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
DOWNLOAD_FALLBACK_STATUSES = {400, 404}
DEFAULT_CONNECT_TIMEOUT_SECONDS = 30
DEFAULT_READ_TIMEOUT_SECONDS = 30
DEFAULT_PROGRESS_LOG_INTERVAL_SECONDS = 5.0


logger = logging.getLogger("fetch_sentinel1_subset")


@dataclass(slots=True)
class CDSEAuth:
    username: str | None = None
    password: str | None = None
    access_token: str | None = None
    client_id: str = "cdse-public"
    totp: str | None = None


def resolve_access_token(auth: CDSEAuth, *, timeout_seconds: int = 60) -> str:
    if auth.access_token:
        return auth.access_token
    if not auth.username or not auth.password:
        raise ValueError(
            "Sentinel-1 download requires either an existing access token or CDSE username/password credentials."
        )
    payload = {
        "client_id": auth.client_id,
        "username": auth.username,
        "password": auth.password,
        "grant_type": "password",
    }
    if auth.totp:
        payload["totp"] = auth.totp
    session = requests.Session()
    session.trust_env = False
    response = session.post(TOKEN_URL, data=payload, timeout=timeout_seconds)
    response.raise_for_status()
    token_payload = response.json()
    access_token = token_payload.get("access_token")
    if not access_token:
        raise RuntimeError("CDSE token response did not include an access_token.")
    return str(access_token)


def product_download_estimates(products: Iterable[Sentinel1Product]) -> list[StorageEstimate]:
    estimates: list[StorageEstimate] = []
    for product in products:
        estimates.append(
            StorageEstimate(
                description=product.name,
                size_bytes=product.content_length,
                source=product.product_id,
            )
        )
    return estimates


def _download_candidate_urls(product: Sentinel1Product) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    if product.product_family in {"SLC", "GRD"}:
        urls.append(("native-compressed", DOWNLOAD_ZIP_URL_TEMPLATE.format(product_id=product.product_id)))
    urls.append(("standard", DOWNLOAD_VALUE_URL_TEMPLATE.format(product_id=product.product_id)))
    return urls


def _response_total_bytes(response: requests.Response, existing_size: int, product: Sentinel1Product) -> int | None:
    content_length_header = response.headers.get("Content-Length")
    response_length = int(content_length_header) if content_length_header and content_length_header.isdigit() else None
    if response.status_code == 206 and response_length is not None:
        return existing_size + response_length
    if response_length is not None:
        return response_length
    return product.content_length


def _log_download_progress(
    *,
    product: Sentinel1Product,
    download_mode: str,
    bytes_written: int,
    total_bytes: int | None,
    started_at: float,
    resumed_from: int,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-6)
    speed_bps = (bytes_written - resumed_from) / elapsed
    if total_bytes:
        percent = (bytes_written / total_bytes) * 100.0
        logger.info(
            "%s | %s | %s / %s (%.1f%%) at %.2f MB/s",
            product.product_id,
            download_mode,
            f"{bytes_written / 1024**2:.2f} MB",
            f"{total_bytes / 1024**2:.2f} MB",
            percent,
            speed_bps / 1024**2,
        )
        return
    logger.info(
        "%s | %s | %s downloaded at %.2f MB/s",
        product.product_id,
        download_mode,
        f"{bytes_written / 1024**2:.2f} MB",
        speed_bps / 1024**2,
    )


def download_sentinel1_product(
    product: Sentinel1Product,
    destination: Path,
    *,
    auth: CDSEAuth,
    dry_run: bool = False,
    force: bool = False,
    chunk_size: int = 64 * 1024,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_storage_guard(product_download_estimates([product]), destination.parent, force=force)

    if dry_run:
        return destination

    if destination.exists() and product.content_length and destination.stat().st_size == product.content_length:
        logger.info("Skipping %s because %s already exists with the expected size.", product.product_id, destination)
        return destination

    access_token = resolve_access_token(auth)
    last_error: requests.HTTPError | None = None
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"Authorization": f"Bearer {access_token}"})
    temp_destination = destination.with_suffix(destination.suffix + ".part")

    for download_mode, download_url in _download_candidate_urls(product):
        existing_size = temp_destination.stat().st_size if temp_destination.exists() else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            logger.info(
                "Resuming %s via %s from %.2f MB.",
                product.product_id,
                download_mode,
                existing_size / 1024**2,
            )

        response = session.get(
            download_url,
            headers=headers,
            stream=True,
            timeout=(DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_READ_TIMEOUT_SECONDS),
        )
        if response.ok:
            file_mode = "ab"
            if existing_size > 0 and response.status_code != 206:
                logger.warning(
                    "%s ignored the resume request for %s; restarting from byte 0.",
                    download_mode,
                    product.product_id,
                )
                existing_size = 0
                file_mode = "wb"
            elif existing_size == 0:
                file_mode = "wb"

            total_bytes = _response_total_bytes(response, existing_size, product)
            started_at = time.monotonic()
            last_progress_log = started_at
            bytes_written = existing_size

            try:
                with temp_destination.open(file_mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
                            bytes_written += len(chunk)
                            current_time = time.monotonic()
                            if current_time - last_progress_log >= DEFAULT_PROGRESS_LOG_INTERVAL_SECONDS:
                                _log_download_progress(
                                    product=product,
                                    download_mode=download_mode,
                                    bytes_written=bytes_written,
                                    total_bytes=total_bytes,
                                    started_at=started_at,
                                    resumed_from=existing_size,
                                )
                                last_progress_log = current_time
                _log_download_progress(
                    product=product,
                    download_mode=download_mode,
                    bytes_written=bytes_written,
                    total_bytes=total_bytes,
                    started_at=started_at,
                    resumed_from=existing_size,
                )
                temp_destination.replace(destination)
                return destination
            except OSError:
                if temp_destination.exists() and temp_destination.stat().st_size == 0:
                    temp_destination.unlink()
                raise

        error = requests.HTTPError(
            f"CDSE download failed for {product.product_id} via {download_mode} endpoint "
            f"({response.status_code}): {response.text[:300]}",
            response=response,
        )
        response.close()
        last_error = error

        if download_mode == "native-compressed" and response.status_code in DOWNLOAD_FALLBACK_STATUSES:
            logger.warning(
                "Native compressed Sentinel-1 download unavailable for %s (%s). Falling back to standard $value download.",
                product.product_id,
                response.status_code,
            )
            continue
        raise error

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No download endpoint candidates were available for product {product.product_id}.")
