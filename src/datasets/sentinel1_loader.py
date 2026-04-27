from __future__ import annotations

import logging
import math
import shutil
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import (
    ManifestDataset,
    deserialize_json_field,
    ensure_data_layout,
    infer_split_from_parts,
    repo_root_from_path,
    write_csv,
)


LOGGER = logging.getLogger(__name__)
PREFERRED_POLARIZATIONS = ("vv", "hh", "vh", "hv")
SAFE_ARCHIVE_SUFFIX = ".safe.zip"
DEFAULT_OVERVIEW_MAX_PIXELS = 8_000_000


@dataclass(slots=True)
class Sentinel1PreparedProduct:
    original_product_path: Path
    prepared_product_path: Path | None
    image_path: Path | None
    measurement_paths: list[Path]
    primary_polarization: str | None
    annotation_xml_path: Path | None
    calibration_xml_path: Path | None
    noise_xml_path: Path | None
    manifest_safe_path: Path | None
    usable: bool
    notes: str
    product_family: str

    def manifest_updates(self) -> dict[str, Any]:
        measurement_paths = [path.resolve().as_posix() for path in self.measurement_paths]
        cache_root = self.prepared_product_path
        noise_vector_path = None
        if cache_root is not None:
            noise_vector_path = cache_root / "annotation" / "calibration" / "noise_vector.npy"
        pixel_domain = "complex_slc" if self.product_family == "SLC" else "amplitude"
        return {
            "image_path": self.image_path.resolve().as_posix() if self.image_path is not None else "",
            "prepared_product_path": self.prepared_product_path.resolve().as_posix()
            if self.prepared_product_path is not None
            else "",
            "prepared_image_path": self.image_path.resolve().as_posix() if self.image_path is not None else "",
            "measurement_paths": measurement_paths,
            "measurement_count": len(self.measurement_paths),
            "primary_polarization": self.primary_polarization or "",
            "annotation_xml_path": self.annotation_xml_path.resolve().as_posix()
            if self.annotation_xml_path is not None
            else "",
            "calibration_xml_path": self.calibration_xml_path.resolve().as_posix()
            if self.calibration_xml_path is not None
            else "",
            "noise_xml_path": self.noise_xml_path.resolve().as_posix() if self.noise_xml_path is not None else "",
            "manifest_safe_path": self.manifest_safe_path.resolve().as_posix()
            if self.manifest_safe_path is not None
            else "",
            "noise_vector_path": noise_vector_path.resolve().as_posix() if noise_vector_path is not None else "",
            "pixel_domain": pixel_domain,
            "preparation_notes": self.notes,
        }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _find_first_with_local_name(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if _local_name(element.tag) == name:
            return element
    return None


def _find_all_with_local_name(root: ET.Element, name: str) -> list[ET.Element]:
    return [element for element in root.iter() if _local_name(element.tag) == name]


def _parse_product_name(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".zip"):
        name = name[:-4]
    return name


def _infer_product_family(path: Path) -> str:
    name = _parse_product_name(path)
    tokens = name.split("_")
    if len(tokens) < 3:
        return "UNKNOWN"
    family = tokens[2].upper()
    if family.startswith("GRD"):
        return "GRD"
    if family.startswith("SLC"):
        return "SLC"
    return family


def _extract_polarization_from_name(path: Path) -> str | None:
    lowered = path.name.lower()
    for polarization in PREFERRED_POLARIZATIONS:
        token = f"-{polarization}-"
        if token in lowered:
            return polarization.upper()
    return None


def _measurement_priority(path: Path) -> tuple[int, str]:
    polarization = (_extract_polarization_from_name(path) or "").lower()
    try:
        index = PREFERRED_POLARIZATIONS.index(polarization)
    except ValueError:
        index = len(PREFERRED_POLARIZATIONS)
    return index, path.name.lower()


def _choose_by_polarization(paths: Iterable[Path], polarization: str | None) -> Path | None:
    items = sorted(Path(path) for path in paths)
    if not items:
        return None
    if not polarization:
        return items[0]
    token = f"-{polarization.lower()}-"
    for path in items:
        if token in path.name.lower():
            return path
    return items[0]


def _discover_measurements(product_root: Path) -> list[Path]:
    measurement_dir = product_root / "measurement"
    candidates = sorted(measurement_dir.glob("*.tif")) + sorted(measurement_dir.glob("*.tiff"))
    if candidates:
        return sorted({path.resolve() for path in candidates}, key=_measurement_priority)
    fallback = sorted(product_root.rglob("*.tif")) + sorted(product_root.rglob("*.tiff"))
    return sorted({path.resolve() for path in fallback}, key=_measurement_priority)


def _safe_archive_root(member_names: list[str], archive_path: Path) -> str:
    top_levels = []
    for member_name in member_names:
        parts = Path(member_name).parts
        if parts:
            top_levels.append(parts[0])
    safe_candidates = [name for name in top_levels if name.upper().endswith(".SAFE")]
    if safe_candidates:
        return safe_candidates[0]
    stem = archive_path.name[:-4] if archive_path.name.lower().endswith(".zip") else archive_path.name
    return stem


def _extract_safe_subset(archive_path: Path, prepared_root: Path, *, force_reextract: bool) -> tuple[Path, str]:
    prepared_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.namelist() if not member.endswith("/")]
        if not members:
            raise RuntimeError(f"{archive_path} does not contain any files.")
        safe_root_name = _safe_archive_root(members, archive_path)
        safe_root = prepared_root / safe_root_name
        selected_members: list[str] = []
        for member_name in members:
            parts = Path(member_name).parts
            if not parts:
                continue
            relative_parts = parts[1:] if parts[0] == safe_root_name else parts
            if not relative_parts:
                continue
            relative_path = Path(*relative_parts)
            first_token = relative_path.parts[0].lower()
            if (
                relative_path.as_posix() == "manifest.safe"
                or first_token in {"measurement", "annotation", "preview"}
            ):
                selected_members.append(member_name)

        if not selected_members:
            raise RuntimeError(
                f"{archive_path} does not contain the expected SAFE measurement / annotation files."
            )

        extracted_count = 0
        for member_name in selected_members:
            parts = Path(member_name).parts
            relative_parts = parts[1:] if parts and parts[0] == safe_root_name else parts
            target = safe_root.joinpath(*relative_parts)
            resolved_target = target.resolve()
            if safe_root.resolve() not in {resolved_target, *resolved_target.parents}:
                raise RuntimeError(f"Refusing to extract archive member outside of {safe_root}: {member_name}")
            if resolved_target.exists() and not force_reextract:
                continue
            resolved_target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member_name) as source_handle, resolved_target.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            extracted_count += 1

    if extracted_count == 0:
        note = f"Re-used existing prepared SAFE cache at {safe_root.as_posix()}."
    else:
        note = f"Extracted {extracted_count} SAFE support files into {safe_root.as_posix()}."
    return safe_root.resolve(), note


def _resolve_local_product_root(
    local_product_path: Path,
    prepared_root: Path,
    *,
    force_reextract: bool = False,
) -> tuple[Path | None, str]:
    if not local_product_path.exists():
        return None, f"Local Sentinel-1 path does not exist: {local_product_path.as_posix()}."
    if local_product_path.is_file() and local_product_path.name.lower() == "manifest.safe":
        return local_product_path.parent.resolve(), "Using the extracted SAFE directory referenced by manifest.safe."
    if local_product_path.is_dir():
        if (local_product_path / "manifest.safe").exists():
            return local_product_path.resolve(), "Using an already extracted SAFE directory."
        measurement_candidates = _discover_measurements(local_product_path)
        if measurement_candidates:
            return local_product_path.resolve(), "Using the provided local Sentinel-1 directory with TIFF measurements."
        return None, f"Directory does not look like a usable Sentinel-1 SAFE product: {local_product_path.as_posix()}."

    suffixes = "".join(local_product_path.suffixes).lower()
    if suffixes.endswith(SAFE_ARCHIVE_SUFFIX) or local_product_path.suffix.lower() == ".zip":
        prepared_product_path, extraction_note = _extract_safe_subset(
            local_product_path,
            prepared_root,
            force_reextract=force_reextract,
        )
        return prepared_product_path, extraction_note

    if local_product_path.suffix.lower() in {".tif", ".tiff"}:
        return local_product_path.resolve().parent, "Using a direct local Sentinel-1 TIFF path."

    return None, f"Unsupported local Sentinel-1 product path: {local_product_path.as_posix()}."


def prepare_local_sentinel1_product(
    local_product_path: Path,
    prepared_root: Path,
    *,
    force_reextract: bool = False,
) -> Sentinel1PreparedProduct:
    local_product_path = local_product_path.expanduser().resolve()
    product_family = _infer_product_family(local_product_path)
    product_root, preparation_note = _resolve_local_product_root(
        local_product_path,
        prepared_root,
        force_reextract=force_reextract,
    )
    if product_root is None:
        return Sentinel1PreparedProduct(
            original_product_path=local_product_path,
            prepared_product_path=None,
            image_path=None,
            measurement_paths=[],
            primary_polarization=None,
            annotation_xml_path=None,
            calibration_xml_path=None,
            noise_xml_path=None,
            manifest_safe_path=None,
            usable=False,
            notes=preparation_note,
            product_family=product_family,
        )

    measurements = _discover_measurements(product_root)
    if not measurements:
        return Sentinel1PreparedProduct(
            original_product_path=local_product_path,
            prepared_product_path=product_root,
            image_path=None,
            measurement_paths=[],
            primary_polarization=None,
            annotation_xml_path=None,
            calibration_xml_path=None,
            noise_xml_path=None,
            manifest_safe_path=(product_root / "manifest.safe") if (product_root / "manifest.safe").exists() else None,
            usable=False,
            notes=f"{preparation_note} No readable measurement TIFFs were found under the local SAFE content.",
            product_family=product_family,
        )

    primary_measurement = sorted(measurements, key=_measurement_priority)[0]
    primary_polarization = _extract_polarization_from_name(primary_measurement)

    annotation_dir = product_root / "annotation"
    calibration_dir = annotation_dir / "calibration"
    annotation_xml_path = _choose_by_polarization(annotation_dir.glob("*.xml"), primary_polarization)
    calibration_xml_path = _choose_by_polarization(calibration_dir.glob("calibration-*.xml"), primary_polarization)
    noise_xml_path = _choose_by_polarization(calibration_dir.glob("noise-*.xml"), primary_polarization)
    manifest_safe_path = product_root / "manifest.safe" if (product_root / "manifest.safe").exists() else None

    note_parts = [preparation_note]
    note_parts.append(
        f"Selected {primary_measurement.name} as the primary local measurement"
        f"{f' ({primary_polarization})' if primary_polarization else ''}."
    )
    if noise_xml_path is None:
        note_parts.append("No local noise XML was found, so Bundle A will only have metadata-free additive fallback.")

    return Sentinel1PreparedProduct(
        original_product_path=local_product_path,
        prepared_product_path=product_root.resolve(),
        image_path=primary_measurement.resolve(),
        measurement_paths=measurements,
        primary_polarization=primary_polarization,
        annotation_xml_path=annotation_xml_path.resolve() if annotation_xml_path is not None else None,
        calibration_xml_path=calibration_xml_path.resolve() if calibration_xml_path is not None else None,
        noise_xml_path=noise_xml_path.resolve() if noise_xml_path is not None else None,
        manifest_safe_path=manifest_safe_path.resolve() if manifest_safe_path is not None else None,
        usable=True,
        notes=" ".join(note_parts).strip(),
        product_family=product_family,
    )


def prepare_sentinel1_record(
    record: dict[str, str],
    *,
    repo_root: Path | None = None,
    force_reextract: bool = False,
) -> Sentinel1PreparedProduct:
    resolved_repo_root = repo_root_from_path(repo_root or Path(__file__))
    layout = ensure_data_layout(resolved_repo_root)
    prepared_root = Path(layout["interim"]) / "sentinel1" / "prepared"
    preferred_path = record.get("prepared_product_path") or record.get("local_target_path") or record.get("image_path") or ""
    if not preferred_path:
        return Sentinel1PreparedProduct(
            original_product_path=prepared_root,
            prepared_product_path=None,
            image_path=None,
            measurement_paths=[],
            primary_polarization=None,
            annotation_xml_path=None,
            calibration_xml_path=None,
            noise_xml_path=None,
            manifest_safe_path=None,
            usable=False,
            notes=f"No local product path is recorded for Sentinel-1 sample {record.get('sample_id', 'unknown')}.",
            product_family=str(record.get("product_family", "UNKNOWN")).upper(),
        )
    return prepare_local_sentinel1_product(Path(preferred_path), prepared_root, force_reextract=force_reextract)


def _parse_noise_profile_element(vector_element: ET.Element, width: int) -> np.ndarray | None:
    pixel_node = _find_first_with_local_name(vector_element, "pixel")
    if pixel_node is None or not (pixel_node.text or "").strip():
        return None
    lut_node = _find_first_with_local_name(vector_element, "noiseRangeLut")
    if lut_node is None:
        lut_node = _find_first_with_local_name(vector_element, "noiseLut")
    if lut_node is None:
        lut_node = _find_first_with_local_name(vector_element, "noiseAzimuthLut")
    if lut_node is None or not (lut_node.text or "").strip():
        return None
    pixels = np.fromstring(pixel_node.text, sep=" ", dtype=np.float32)
    values = np.fromstring(lut_node.text, sep=" ", dtype=np.float32)
    if pixels.size == 0 or values.size == 0 or pixels.size != values.size:
        return None
    if pixels.size == 1:
        return np.full((width,), float(values[0]), dtype=np.float32)
    target_pixels = np.arange(width, dtype=np.float32)
    return np.interp(target_pixels, pixels, values, left=values[0], right=values[-1]).astype(np.float32)


def load_or_build_noise_vector(
    noise_xml_path: Path,
    *,
    width: int,
    cache_path: Path | None = None,
) -> np.ndarray | None:
    if cache_path is not None and cache_path.exists():
        try:
            cached = np.load(cache_path)
            if cached.ndim == 1 and cached.size == width:
                return cached.astype(np.float32)
        except OSError:
            LOGGER.warning("Could not read cached Sentinel-1 noise vector from %s.", cache_path)

    tree = ET.parse(noise_xml_path)
    root = tree.getroot()
    profiles: list[np.ndarray] = []
    for vector_name in ("noiseRangeVector", "noiseVector", "noiseAzimuthVector"):
        for vector in _find_all_with_local_name(root, vector_name):
            profile = _parse_noise_profile_element(vector, width)
            if profile is not None:
                profiles.append(profile)
        if profiles:
            break

    if not profiles:
        return None

    aggregated = np.mean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, aggregated)
        except OSError:
            LOGGER.warning("Could not cache Sentinel-1 noise vector at %s.", cache_path)
    return aggregated


def hydrate_sentinel1_metadata(
    metadata: dict[str, Any],
    prepared: Sentinel1PreparedProduct,
    *,
    image_width: int | None = None,
) -> dict[str, Any]:
    hydrated = dict(metadata)
    hydrated.update(prepared.manifest_updates())
    if image_width is None:
        return hydrated
    noise_xml_path_text = str(hydrated.get("noise_xml_path", "")).strip()
    if not noise_xml_path_text:
        return hydrated
    noise_xml_path = Path(noise_xml_path_text)
    if not noise_xml_path.exists():
        return hydrated
    cache_path_text = str(hydrated.get("noise_vector_path", "")).strip()
    cache_path = Path(cache_path_text) if cache_path_text else None
    noise_vector = load_or_build_noise_vector(noise_xml_path, width=image_width, cache_path=cache_path)
    if noise_vector is not None:
        hydrated["noise_vector"] = noise_vector
        if cache_path is not None:
            hydrated["noise_vector_path"] = cache_path.resolve().as_posix()
    return hydrated


def read_sentinel1_measurement(
    image_path: Path,
    *,
    max_pixels: int = DEFAULT_OVERVIEW_MAX_PIXELS,
) -> tuple[np.ndarray, str]:
    suffix = image_path.suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        raise ValueError(f"Sentinel-1 measurement reader only supports TIFF inputs, got: {image_path}")

    import tifffile

    with tifffile.TiffFile(image_path) as tiff_handle:
        pages = list(tiff_handle.pages)
        if not pages:
            raise RuntimeError(f"No TIFF pages were found in {image_path}.")
        selected_page = pages[0]
        selected_index = 0
        selected_shape = tuple(int(dimension) for dimension in selected_page.shape[:2])
        selected_pixels = int(np.prod(selected_shape))
        if selected_pixels > max_pixels and len(pages) > 1:
            overview_candidates: list[tuple[int, Any, int]] = []
            for index, page in enumerate(pages[1:], start=1):
                shape = tuple(int(dimension) for dimension in page.shape[:2])
                overview_candidates.append((index, page, int(np.prod(shape))))
            fitting = [candidate for candidate in overview_candidates if candidate[2] <= max_pixels]
            if fitting:
                selected_index, selected_page, selected_pixels = fitting[0]
            else:
                selected_index, selected_page, selected_pixels = overview_candidates[-1]
        if selected_index == 0 and selected_pixels > max_pixels:
            stride = max(2, int(math.ceil(math.sqrt(selected_pixels / max_pixels))))
            try:
                memmapped = tifffile.memmap(image_path)
                array = np.asarray(memmapped[::stride, ::stride])
                note = (
                    f"Loaded a stride-{stride} memory-mapped Sentinel-1 TIFF sample with shape {array.shape} "
                    "because no internal overview page was available."
                )
                return array, note
            except Exception as exc:
                raise RuntimeError(
                    f"{image_path} is too large for full-resolution Bundle A processing and no internal overview "
                    f"could be used. Memory-mapped decimation also failed: {exc}"
                ) from exc
        array = np.asarray(selected_page.asarray())
        if selected_index == 0:
            note = f"Loaded the full-resolution local Sentinel-1 measurement TIFF ({selected_pixels} pixels)."
        else:
            note = (
                f"Loaded internal Sentinel-1 COG overview page {selected_index} with shape {array.shape} "
                f"to keep Bundle A memory-safe."
            )
        return array, note


def _candidate_sentinel1_products(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for safe_dir in root.rglob("*.SAFE"):
        if safe_dir.is_dir():
            candidates.add(safe_dir.resolve())
    for archive in root.rglob("*.zip"):
        if ".safe" in archive.name.lower() or archive.name.lower().endswith(".eof.zip"):
            candidates.add(archive.resolve())
    for manifest_safe in root.rglob("manifest.safe"):
        candidates.add(manifest_safe.parent.resolve())
    return sorted(candidates)


def build_sentinel1_manifest(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    repo_root = repo_root_from_path(root)
    rows: list[dict[str, Any]] = []
    for product_path in _candidate_sentinel1_products(root.resolve()):
        product_name = _parse_product_name(product_path)
        family = _infer_product_family(product_path)
        if family not in {"GRD", "SLC"}:
            continue
        prepared = prepare_sentinel1_record(
            {
                "sample_id": product_name,
                "product_family": family,
                "local_target_path": product_path.resolve().as_posix(),
            },
            repo_root=repo_root,
        )
        metadata_json = {
            "pixel_domain": "complex_slc" if family == "SLC" else "amplitude",
            "preparation_notes": prepared.notes,
            "measurement_count": len(prepared.measurement_paths),
            "primary_polarization": prepared.primary_polarization,
        }
        metadata_json.update(prepared.manifest_updates())
        rows.append(
            {
                "record_type": "product",
                "dataset": "sentinel1",
                "sample_id": product_name,
                "split": infer_split_from_parts(product_path),
                "image_path": prepared.image_path.resolve().as_posix() if prepared.image_path is not None else "",
                "annotation_path": "",
                "remote_source": "",
                "status": "partial" if prepared.usable else "metadata-only",
                "download_status": "local",
                "notes": prepared.notes,
                "product_id": "",
                "product_name": product_name,
                "product_type": family,
                "product_family": family,
                "mode": "",
                "domain_hint": "complex_slc" if family == "SLC" else "detected_ground_range",
                "local_target_path": product_path.resolve().as_posix(),
                "prepared_product_path": prepared.prepared_product_path.resolve().as_posix()
                if prepared.prepared_product_path is not None
                else "",
                "prepared_image_path": prepared.image_path.resolve().as_posix()
                if prepared.image_path is not None
                else "",
                "measurement_count": len(prepared.measurement_paths),
                "primary_polarization": prepared.primary_polarization or "",
                "annotation_xml_path": prepared.annotation_xml_path.resolve().as_posix()
                if prepared.annotation_xml_path is not None
                else "",
                "calibration_xml_path": prepared.calibration_xml_path.resolve().as_posix()
                if prepared.calibration_xml_path is not None
                else "",
                "noise_xml_path": prepared.noise_xml_path.resolve().as_posix()
                if prepared.noise_xml_path is not None
                else "",
                "manifest_safe_path": prepared.manifest_safe_path.resolve().as_posix()
                if prepared.manifest_safe_path is not None
                else "",
                "metadata_json": metadata_json,
            }
        )
    write_csv(manifest_path, rows)
    return rows


class Sentinel1Dataset(ManifestDataset):
    def __init__(self, records: list[dict[str, Any]], *, split: str | None = None, sample_limit: int | None = None) -> None:
        normalized_records: list[dict[str, Any]] = []
        for record in records:
            row = dict(record)
            row["metadata"] = deserialize_json_field(row.get("metadata_json"))
            normalized_records.append(row)
        super().__init__(normalized_records, split=split, sample_limit=sample_limit)
