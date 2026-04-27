from __future__ import annotations

import logging
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.stage1.sentinel1_batch import default_batch_output_root, evaluate_bundle_a_sentinel1_batch

from .common import ensure_data_layout, ensure_storage_guard, read_csv_rows, write_csv, write_json
from .registry import DatasetRegistration, DatasetRegistry, default_registry_path
from .sentinel1_catalog import (
    Sentinel1Product,
    Sentinel1Query,
    merge_manifest_rows,
    product_from_manifest_row,
    product_target_path,
    products_to_manifest_rows,
    search_sentinel1_products,
)
from .sentinel1_fetch import CDSEAuth, download_sentinel1_product, product_download_estimates
from .sentinel1_loader import prepare_sentinel1_record


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EvidenceQuerySpec:
    label: str
    reason: str
    query: Sentinel1Query


@dataclass(slots=True)
class Sentinel1EvidenceSummary:
    fetched_count: int
    ready_count: int
    prepared_ready_count: int
    target_ready_scenes: int
    manifest_path: str
    batch_output_root: str
    blocked_reason: str = ""
    selected_candidates: list[dict[str, Any]] | None = None
    downloaded_product_ids: list[str] | None = None
    batch_topline_metrics: dict[str, Any] | None = None


@dataclass(slots=True)
class Sentinel1EvidencePlan:
    manifest_path: str
    batch_output_root: str
    ready_scene_count: int
    evaluated_scene_count: int
    regime_counts: dict[str, int]
    underrepresented_regimes: list[str]
    recommended_next_scene_count: int
    recommendations: list[str]


DEFAULT_EVIDENCE_QUERY_SPECS = [
    EvidenceQuerySpec(
        label="north_sea_open",
        reason="Likely lower-backscatter open-water North Sea scene.",
        query=Sentinel1Query(
            product_type="GRD",
            mode="IW",
            start="2024-02-01",
            end="2024-02-10",
            bbox=(1.0, 56.0, 4.5, 58.5),
            max_results=3,
            metadata_only=True,
        ),
    ),
    EvidenceQuerySpec(
        label="gibraltar_coastal",
        reason="Coastal clutter and busy shoreline near Gibraltar / Alboran.",
        query=Sentinel1Query(
            product_type="GRD",
            mode="IW",
            start="2024-02-01",
            end="2024-02-10",
            bbox=(-6.5, 35.2, -2.0, 36.8),
            max_results=3,
            metadata_only=True,
        ),
    ),
    EvidenceQuerySpec(
        label="adriatic_mix",
        reason="Semi-enclosed sea with coastal structure and varied background.",
        query=Sentinel1Query(
            product_type="GRD",
            mode="IW",
            start="2024-03-01",
            end="2024-03-10",
            bbox=(13.0, 42.0, 18.8, 45.8),
            max_results=3,
            metadata_only=True,
        ),
    ),
    EvidenceQuerySpec(
        label="east_med_coastal",
        reason="Eastern Mediterranean coastal scene with land-water transitions.",
        query=Sentinel1Query(
            product_type="GRD",
            mode="IW",
            start="2024-03-01",
            end="2024-03-10",
            bbox=(31.0, 31.0, 35.8, 34.8),
            max_results=3,
            metadata_only=True,
        ),
    ),
    EvidenceQuerySpec(
        label="red_sea_open",
        reason="Warm-water maritime scene with another open-water operating regime.",
        query=Sentinel1Query(
            product_type="GRD",
            mode="IW",
            start="2024-03-01",
            end="2024-03-10",
            bbox=(36.0, 20.0, 40.5, 24.5),
            max_results=3,
            metadata_only=True,
        ),
    ),
]


def _load_batch_scene_rows(batch_output_root: Path) -> list[dict[str, Any]]:
    scene_summary_path = batch_output_root / "tables" / "scene_summary.json"
    if not scene_summary_path.exists():
        return []
    try:
        payload = json.loads(scene_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [dict(row) for row in payload.get("scenes", [])]


def plan_sentinel1_evidence(
    *,
    repo_root: Path,
    manifest_path: Path | None = None,
    batch_output_root: Path | None = None,
    target_scene_count: int = 10,
) -> Sentinel1EvidencePlan:
    """Summarize current Sentinel-1 evidence and recommend the next acquisition shape.

    This helper does not download anything.  It gives a manifest-aware plan so
    the next GRD expansion round can stay deliberate rather than ad hoc.
    """

    layout = ensure_data_layout(repo_root)
    resolved_manifest = _manifest_path(layout, manifest_path)
    manifest_rows = _load_manifest_rows(resolved_manifest)
    ready_count = len(_ready_rows(manifest_rows))
    resolved_batch = batch_output_root.resolve() if batch_output_root else default_batch_output_root(repo_root)
    scene_rows = _load_batch_scene_rows(resolved_batch)
    evaluated_rows = [row for row in scene_rows if bool(row.get("scene_evaluated"))]

    regime_counts = {
        "metadata-rich": 0,
        "metadata-poor": 0,
        "structured-artifact": 0,
        "low-backscatter/open-ocean": 0,
        "quiet-background": 0,
        "overview-only": 0,
    }
    for row in evaluated_rows:
        label = str(row.get("regime_label", "")).lower()
        if "metadata-rich" in label:
            regime_counts["metadata-rich"] += 1
        if "metadata-poor" in label:
            regime_counts["metadata-poor"] += 1
        if "structured-artifact" in label:
            regime_counts["structured-artifact"] += 1
        if "low-backscatter" in label or "open-ocean" in label:
            regime_counts["low-backscatter/open-ocean"] += 1
        if "quiet background" in label:
            regime_counts["quiet-background"] += 1
        if "overview-only" in label:
            regime_counts["overview-only"] += 1

    underrepresented: list[str] = []
    recommendations: list[str] = []
    if len(evaluated_rows) < target_scene_count:
        recommendations.append(f"Add about {target_scene_count - len(evaluated_rows)} more maritime GRD scenes to reach {target_scene_count} evaluated scenes.")
    if regime_counts["metadata-poor"] == 0:
        underrepresented.append("metadata-poor or partial-metadata scenes")
        recommendations.append("Keep at least a few metadata-poor/partial-metadata products in the plan so A2/A0 routing is tested honestly.")
    if regime_counts["structured-artifact"] < 2:
        underrepresented.append("structured-artifact/coastal-clutter scenes")
        recommendations.append("Prioritize 1-2 coastal or visually artifact-prone scenes to test whether A3 is justified beyond one-off cases.")
    if regime_counts["low-backscatter/open-ocean"] < 2:
        underrepresented.append("low-backscatter open-ocean scenes")
        recommendations.append("Add 1-2 darker open-ocean scenes to make the A1/A2 comparison less coastline-biased.")
    if not recommendations:
        recommendations.append("Current scene-regime coverage is adequate for the next detector-validation bridge; add more only for diversity.")

    plan = Sentinel1EvidencePlan(
        manifest_path=resolved_manifest.resolve().as_posix(),
        batch_output_root=resolved_batch.resolve().as_posix(),
        ready_scene_count=ready_count,
        evaluated_scene_count=len(evaluated_rows),
        regime_counts=regime_counts,
        underrepresented_regimes=underrepresented,
        recommended_next_scene_count=max(target_scene_count - len(evaluated_rows), 0),
        recommendations=recommendations,
    )
    output_tables = resolved_batch / "tables"
    output_tables.mkdir(parents=True, exist_ok=True)
    write_json(output_tables / "evidence_plan.json", asdict(plan))
    (output_tables / "evidence_plan.md").write_text(
        "# Sentinel-1 evidence expansion plan\n\n"
        f"- Ready local GRD scenes: `{plan.ready_scene_count}`\n"
        f"- Evaluated scenes: `{plan.evaluated_scene_count}`\n"
        f"- Recommended additional scenes: `{plan.recommended_next_scene_count}`\n"
        f"- Underrepresented regimes: `{', '.join(plan.underrepresented_regimes) or 'none flagged'}`\n\n"
        "## Recommendations\n\n"
        + "\n".join(f"- {item}" for item in plan.recommendations)
        + "\n",
        encoding="utf-8",
    )
    return plan


def _non_empty(value: Any) -> bool:
    return value not in {"", None, [], {}}


def _merge_notes(existing: Any, incoming: Any) -> str:
    values: list[str] = []
    for candidate in (existing, incoming):
        text = str(candidate or "").strip()
        if text and text not in values:
            values.append(text)
    return " ".join(values)


def _manifest_path(layout: dict[str, Path | bool], explicit_manifest: Path | None) -> Path:
    if explicit_manifest is not None:
        return explicit_manifest.resolve()
    return Path(layout["manifests"]) / "sentinel1_manifest.csv"


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in read_csv_rows(path)] if path.exists() else []


def _is_grd_row(row: dict[str, Any]) -> bool:
    if str(row.get("record_type", "")).strip().lower() == "placeholder":
        return False
    return str(row.get("dataset", "")).strip().lower() == "sentinel1" and str(row.get("product_family", "")).upper() == "GRD"


def _ready_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _is_grd_row(row) and str(row.get("prepared_status", "")).lower() == "ready"]


def _missing_download_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for row in rows:
        if not _is_grd_row(row):
            continue
        local_target = Path(str(row.get("local_target_path", "")).strip()) if str(row.get("local_target_path", "")).strip() else None
        if local_target is not None and not local_target.exists():
            missing.append(row)
    return missing


def _auth_available(auth: CDSEAuth) -> bool:
    return bool(auth.access_token or (auth.username and auth.password))


def _download_products(
    *,
    manifest_rows: list[dict[str, Any]],
    products: list[Sentinel1Product],
    auth: CDSEAuth,
    sentinel1_root: Path,
    force: bool,
    dry_run: bool,
) -> list[str]:
    if not products:
        return []
    if not dry_run and not _auth_available(auth):
        raise ValueError(
            "Missing CDSE credentials. Set CDSE_USERNAME/CDSE_PASSWORD in the environment."
        )

    warnings = ensure_storage_guard(product_download_estimates(products), sentinel1_root, force=force)
    for warning in warnings:
        LOGGER.info(warning)

    downloaded_ids: list[str] = []
    rows_by_product_id = {
        str(row.get("product_id", "")).strip(): row
        for row in manifest_rows
        if str(row.get("product_id", "")).strip()
    }

    for product in products:
        row = rows_by_product_id[product.product_id]
        destination = product_target_path(product, sentinel1_root)
        row["local_target_path"] = destination.resolve().as_posix()
        try:
            downloaded_path = download_sentinel1_product(
                product,
                destination,
                auth=auth,
                dry_run=dry_run,
                force=force,
            )
            row["image_path"] = downloaded_path.resolve().as_posix()
            row["status"] = "partial" if dry_run else "complete"
            row["download_status"] = "planned" if dry_run else "complete"
            row["notes"] = _merge_notes(row.get("notes", ""), "dry-run only" if dry_run else "")
            downloaded_ids.append(product.product_id)
            LOGGER.info("%s downloaded/planned to %s", product.product_id, downloaded_path)
        except Exception as exc:
            row["status"] = "failed"
            row["download_status"] = "failed"
            row["notes"] = _merge_notes(row.get("notes", ""), str(exc))
            LOGGER.error("Failed to fetch %s (%s): %s", product.product_id, product.name, exc)
    return downloaded_ids


def _best_candidate(products: list[Sentinel1Product], seen_ids: set[str]) -> Sentinel1Product | None:
    ranked = sorted(
        [product for product in products if product.product_id not in seen_ids],
        key=lambda product: (
            0 if "_COG.SAFE" in product.name.upper() else 1,
            0 if product.online else 1,
            product.content_length or 10**18,
            product.name,
        ),
    )
    return ranked[0] if ranked else None


def _select_additional_products(
    *,
    existing_rows: list[dict[str, Any]],
    target_ready_scenes: int,
    max_new_downloads: int,
) -> list[tuple[Sentinel1Product, str, str]]:
    existing_ids = {
        str(row.get("product_id", "")).strip()
        for row in existing_rows
        if str(row.get("product_id", "")).strip()
    }
    additional_needed = max(target_ready_scenes - len(_ready_rows(existing_rows)), 0)
    additional_needed = min(additional_needed, max_new_downloads)
    if additional_needed <= 0:
        return []

    selected: list[tuple[Sentinel1Product, str, str]] = []
    seen_ids = set(existing_ids)
    for spec in DEFAULT_EVIDENCE_QUERY_SPECS:
        if len(selected) >= additional_needed:
            break
        try:
            products = search_sentinel1_products(spec.query)
        except Exception as exc:
            LOGGER.warning("Sentinel-1 evidence query %s failed: %s", spec.label, exc)
            continue
        candidate = _best_candidate(products, seen_ids)
        if candidate is None:
            continue
        selected.append((candidate, spec.label, spec.reason))
        seen_ids.add(candidate.product_id)
    return selected


def _merge_new_products_into_manifest(
    manifest_rows: list[dict[str, Any]],
    products: list[Sentinel1Product],
    *,
    sentinel1_root: Path,
) -> list[dict[str, Any]]:
    if not products:
        return manifest_rows
    incoming_rows = products_to_manifest_rows(products, sentinel1_root=sentinel1_root)
    return merge_manifest_rows(manifest_rows, incoming_rows)


def _prepare_grd_rows(
    *,
    repo_root: Path,
    manifest_rows: list[dict[str, Any]],
    layout: dict[str, Path | bool],
) -> int:
    prepared_count = 0
    for row in manifest_rows:
        if not _is_grd_row(row):
            continue
        prepared = prepare_sentinel1_record(row, repo_root=repo_root)
        updates = prepared.manifest_updates()
        row.update({key: value for key, value in updates.items()})
        existing_notes = str(row.get("notes", ""))
        if prepared.usable and "Local Sentinel-1 path does not exist:" in existing_notes:
            existing_notes = ""
        row["notes"] = _merge_notes(existing_notes, prepared.notes)
        row["prepared_status"] = "ready" if prepared.usable else "failed"
        if prepared.usable and prepared.image_path is not None:
            row["image_path"] = prepared.image_path.resolve().as_posix()
            if str(row.get("status", "")).strip().lower() != "complete":
                row["status"] = "partial"
            prepared_count += 1
    registry = DatasetRegistry(default_registry_path(repo_root))
    registry.upsert(
        DatasetRegistration(
            dataset_name="sentinel1",
            manifest_path=(Path(layout["manifests"]) / "sentinel1_manifest.csv").resolve().as_posix(),
            local_path=(Path(layout["raw"]) / "sentinel1").resolve().as_posix(),
            remote_source="https://dataspace.copernicus.eu/",
            notes=(
                "Sentinel-1 GRD evidence path with local SAFE preparation for Bundle A multi-scene screening."
            ),
            status="partial" if prepared_count > 0 else "metadata-only",
            sample_count=prepared_count,
        )
    )
    registry.save()
    return prepared_count


def expand_sentinel1_evidence(
    *,
    repo_root: Path,
    batch_config_path: Path,
    manifest_path: Path | None = None,
    target_ready_scenes: int = 12,
    max_new_downloads: int = 8,
    force: bool = False,
    dry_run: bool = False,
    batch_output_root: Path | None = None,
) -> Sentinel1EvidenceSummary:
    layout = ensure_data_layout(repo_root)
    resolved_manifest_path = _manifest_path(layout, manifest_path)
    manifest_rows = _load_manifest_rows(resolved_manifest_path)
    sentinel1_root = Path(layout["raw"]) / "sentinel1"
    auth = CDSEAuth(
        username=os.getenv("CDSE_USERNAME"),
        password=os.getenv("CDSE_PASSWORD"),
        access_token=os.getenv("CDSE_ACCESS_TOKEN"),
    )
    blocked_reason = ""
    downloaded_ids: list[str] = []

    missing_rows = _missing_download_rows(manifest_rows)
    missing_products = [product_from_manifest_row(row) for row in missing_rows]
    try:
        downloaded_ids.extend(
            _download_products(
                manifest_rows=manifest_rows,
                products=missing_products,
                auth=auth,
                sentinel1_root=sentinel1_root,
                force=force,
                dry_run=dry_run,
            )
        )
    except Exception as exc:
        blocked_reason = str(exc)
        LOGGER.warning("Sentinel-1 evidence backfill is blocked: %s", exc)

    write_csv(resolved_manifest_path, manifest_rows)
    prepared_ready_count = _prepare_grd_rows(repo_root=repo_root, manifest_rows=manifest_rows, layout=layout)
    write_csv(resolved_manifest_path, manifest_rows)

    additional_selections = _select_additional_products(
        existing_rows=manifest_rows,
        target_ready_scenes=target_ready_scenes,
        max_new_downloads=max_new_downloads,
    )
    selected_candidates = [
        {
            "product_id": product.product_id,
            "product_name": product.name,
            "selection_label": label,
            "selection_reason": reason,
            "expected_size_bytes": product.content_length,
        }
        for product, label, reason in additional_selections
    ]

    if additional_selections:
        additional_products = [product for product, _, _ in additional_selections]
        manifest_rows = _merge_new_products_into_manifest(manifest_rows, additional_products, sentinel1_root=sentinel1_root)
        write_csv(resolved_manifest_path, manifest_rows)
        if not blocked_reason:
            try:
                downloaded_ids.extend(
                    _download_products(
                        manifest_rows=manifest_rows,
                        products=additional_products,
                        auth=auth,
                        sentinel1_root=sentinel1_root,
                        force=force,
                        dry_run=dry_run,
                    )
                )
            except Exception as exc:
                blocked_reason = str(exc)
                LOGGER.warning("Sentinel-1 additional evidence fetch is blocked: %s", exc)
            write_csv(resolved_manifest_path, manifest_rows)

    prepared_ready_count = _prepare_grd_rows(repo_root=repo_root, manifest_rows=manifest_rows, layout=layout)
    write_csv(resolved_manifest_path, manifest_rows)

    resolved_batch_output = batch_output_root.resolve() if batch_output_root else default_batch_output_root(repo_root)
    batch_artifacts = evaluate_bundle_a_sentinel1_batch(
        repo_root=repo_root,
        config_path=batch_config_path.resolve(),
        manifest_path=resolved_manifest_path,
        output_root=resolved_batch_output,
        statuses=("ready", "failed", "metadata-only"),
        additive_submethod=None,
        compare_submethods=True,
    )

    selection_payload = {
        "target_ready_scenes": target_ready_scenes,
        "max_new_downloads": max_new_downloads,
        "blocked_reason": blocked_reason,
        "selected_candidates": selected_candidates,
        "downloaded_product_ids": downloaded_ids,
    }
    write_json(resolved_batch_output / "config" / "evidence_expansion.json", selection_payload)

    return Sentinel1EvidenceSummary(
        fetched_count=len(downloaded_ids),
        ready_count=len(_ready_rows(manifest_rows)),
        prepared_ready_count=prepared_ready_count,
        target_ready_scenes=target_ready_scenes,
        manifest_path=resolved_manifest_path.resolve().as_posix(),
        batch_output_root=resolved_batch_output.as_posix(),
        blocked_reason=blocked_reason,
        selected_candidates=selected_candidates,
        downloaded_product_ids=downloaded_ids,
        batch_topline_metrics=batch_artifacts.topline_metrics,
    )
