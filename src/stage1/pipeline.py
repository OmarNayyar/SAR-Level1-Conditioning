from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml
from skimage import io as skio

from src.datasets.common import deserialize_json_field, read_csv_rows, repo_root_from_path
from src.datasets.registry import DatasetRegistry, default_registry_path
from src.datasets.sentinel1_loader import (
    hydrate_sentinel1_metadata,
    prepare_sentinel1_record,
    read_sentinel1_measurement,
)
from src.datasets.ssdd_loader import parse_voc_annotation
from src.stage1.viz.side_by_side import prepare_display_image
from src.utils import payload_fingerprint


LOGGER = logging.getLogger(__name__)
COMPLEX_DOMAIN_HINTS = {"complex", "complex_slc", "slc_complex", "slc"}
_BUNDLE_TOP_LEVEL_KEYS = frozenset({"bundle", "profile", "dataset", "processing", "metrics", "statistics", "downstream", "outputs"})
_DETECTION_TOP_LEVEL_KEYS = frozenset({"dataset", "variants", "detector", "outputs"})
_FINAL_SWEEP_TOP_LEVEL_KEYS = frozenset({"detection", "reports", "freeze"})
_FINAL_SWEEP_DETECTION_KEYS = frozenset(
    {
        "config",
        "mode",
        "output_root",
        "datasets",
        "variants",
        "bundle_a_config",
        "bundle_a_conservative_config",
        "bundle_b_config",
        "bundle_d_config",
        "limit_per_split",
        "epochs",
        "imgsz",
        "batch",
        "workers",
        "model",
        "device",
    }
)
_FINAL_SWEEP_REPORT_KEYS = frozenset(
    {
        "generate_surface_packs",
        "surface",
        "generate_demo_index",
        "demo_index_max_examples",
        "summary_output",
    }
)
_FINAL_SWEEP_FREEZE_KEYS = frozenset({"selected_datasets", "selected_variants", "notes"})


@dataclass(slots=True)
class LoadedSample:
    dataset_name: str
    sample_id: str
    split: str
    intensity_image: np.ndarray
    display_image: np.ndarray
    metadata: dict[str, Any]
    annotation: dict[str, Any] | None
    annotation_count: int
    downstream_target: np.ndarray | None
    source_note: str
    complex_image: np.ndarray | None = None
    pixel_domain: str = "unknown"


@dataclass(slots=True)
class BundleProcessResult:
    additive_output: np.ndarray
    final_output: np.ndarray
    additive_applied: bool
    additive_mode: str
    additive_notes: str
    multiplicative_mode: str
    multiplicative_notes: str
    estimated_additive_component: np.ndarray | None = None
    display_output: np.ndarray | None = None
    extra_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    result_metadata: dict[str, Any] = field(default_factory=dict)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_bundle_arg_parser(*, description: str, default_config_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=default_config_path, help="Bundle configuration YAML.")
    parser.add_argument("--dataset", help="Override dataset name from the config.")
    parser.add_argument("--split", help="Override split name from the config.")
    parser.add_argument("--sample-limit", type=int, help="Override sample_limit from the config.")
    parser.add_argument("--manifest", help="Use an explicit manifest path instead of the registry entry.")
    parser.add_argument(
        "--output-root",
        help="Override the results root directory. Otherwise the config root is reused unless dataset/split/sample-limit overrides require a derived outputs/... path.",
    )
    return parser


def _config_path_label(path: Path) -> str:
    return path.resolve().as_posix()


def _ensure_mapping(value: Any, *, section: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{section} in {_config_path_label(path)} must be a mapping.")
    return value


def _validate_allowed_keys(payload: dict[str, Any], *, allowed: set[str] | frozenset[str], section: str, path: Path) -> None:
    unknown = sorted(str(key) for key in payload.keys() if str(key) not in allowed)
    if unknown:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(
            f"{section} in {_config_path_label(path)} contains unsupported key(s): {', '.join(unknown)}. "
            f"Allowed keys: {allowed_text}"
        )


def _ensure_string(value: Any, *, section: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section} in {_config_path_label(path)} must be a non-empty string.")
    return value


def _ensure_int_like(value: Any, *, section: str, path: Path) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{section} in {_config_path_label(path)} must be an integer, not a boolean.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section} in {_config_path_label(path)} must be an integer-like value.") from exc


def _ensure_number(value: Any, *, section: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{section} in {_config_path_label(path)} must be numeric, not a boolean.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section} in {_config_path_label(path)} must be numeric.") from exc


def _ensure_string_list(value: Any, *, section: str, path: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{section} in {_config_path_label(path)} must be a non-empty list of strings.")
    cleaned: list[str] = []
    for index, item in enumerate(value):
        cleaned.append(_ensure_string(item, section=f"{section}[{index}]", path=path))
    return cleaned


def _validate_bundle_config(path: Path, payload: dict[str, Any]) -> None:
    _validate_allowed_keys(payload, allowed=_BUNDLE_TOP_LEVEL_KEYS, section="bundle config", path=path)
    bundle = _ensure_mapping(payload.get("bundle"), section="bundle", path=path)
    dataset = _ensure_mapping(payload.get("dataset"), section="dataset", path=path)
    processing = _ensure_mapping(payload.get("processing"), section="processing", path=path)
    outputs = _ensure_mapping(payload.get("outputs"), section="outputs", path=path)

    bundle_name = _ensure_string(bundle.get("name"), section="bundle.name", path=path)
    if not bundle_name.startswith("bundle_"):
        raise ValueError(f"bundle.name in {_config_path_label(path)} must start with `bundle_`.")
    _ensure_string(dataset.get("name"), section="dataset.name", path=path)
    if dataset.get("split") not in {None, ""}:
        _ensure_string(dataset.get("split"), section="dataset.split", path=path)
    if dataset.get("sample_limit") not in {None, ""}:
        _ensure_int_like(dataset.get("sample_limit"), section="dataset.sample_limit", path=path)
    additive = _ensure_mapping(processing.get("additive"), section="processing.additive", path=path)
    multiplicative = _ensure_mapping(processing.get("multiplicative"), section="processing.multiplicative", path=path)
    if not additive:
        raise ValueError(f"processing.additive in {_config_path_label(path)} cannot be empty.")
    if not multiplicative:
        raise ValueError(f"processing.multiplicative in {_config_path_label(path)} cannot be empty.")
    _ensure_string(outputs.get("root"), section="outputs.root", path=path)

    for optional_section in ("profile", "metrics", "statistics", "downstream"):
        optional_value = payload.get(optional_section)
        if optional_value is not None:
            _ensure_mapping(optional_value, section=optional_section, path=path)


def _validate_detection_config(path: Path, payload: dict[str, Any]) -> None:
    _validate_allowed_keys(payload, allowed=_DETECTION_TOP_LEVEL_KEYS, section="detection config", path=path)
    dataset = _ensure_mapping(payload.get("dataset"), section="dataset", path=path)
    detector = _ensure_mapping(payload.get("detector"), section="detector", path=path)
    outputs = _ensure_mapping(payload.get("outputs"), section="outputs", path=path)

    if dataset.get("name") not in {None, ""}:
        _ensure_string(dataset.get("name"), section="dataset.name", path=path)
    if dataset.get("limit_per_split") not in {None, ""}:
        limit_per_split = dataset.get("limit_per_split")
        if not isinstance(limit_per_split, str):
            _ensure_int_like(limit_per_split, section="dataset.limit_per_split", path=path)
    if dataset.get("val_fraction") not in {None, ""}:
        _ensure_number(dataset.get("val_fraction"), section="dataset.val_fraction", path=path)

    _ensure_string_list(payload.get("variants"), section="variants", path=path)
    _ensure_string(detector.get("backend"), section="detector.backend", path=path)
    _ensure_string(detector.get("model"), section="detector.model", path=path)
    for key in ("epochs", "imgsz", "batch", "workers"):
        if detector.get(key) not in {None, ""}:
            _ensure_int_like(detector.get(key), section=f"detector.{key}", path=path)
    if detector.get("eval_split") not in {None, ""}:
        _ensure_string(detector.get("eval_split"), section="detector.eval_split", path=path)
    if detector.get("device") not in {None, ""}:
        _ensure_string(detector.get("device"), section="detector.device", path=path)
    _ensure_string(outputs.get("root"), section="outputs.root", path=path)


def _validate_final_sweep_config(path: Path, payload: dict[str, Any]) -> None:
    _validate_allowed_keys(payload, allowed=_FINAL_SWEEP_TOP_LEVEL_KEYS, section="final sweep config", path=path)
    detection = _ensure_mapping(payload.get("detection"), section="detection", path=path)
    reports = _ensure_mapping(payload.get("reports"), section="reports", path=path)
    freeze = payload.get("freeze")

    _validate_allowed_keys(detection, allowed=_FINAL_SWEEP_DETECTION_KEYS, section="final sweep detection", path=path)
    _validate_allowed_keys(reports, allowed=_FINAL_SWEEP_REPORT_KEYS, section="final sweep reports", path=path)
    _ensure_string(detection.get("config"), section="detection.config", path=path)
    if detection.get("mode") not in {None, ""}:
        mode = _ensure_string(detection.get("mode"), section="detection.mode", path=path)
        if mode not in {"prepare", "all"}:
            raise ValueError(f"detection.mode in {_config_path_label(path)} must be `prepare` or `all`.")
    if detection.get("output_root") not in {None, ""}:
        _ensure_string(detection.get("output_root"), section="detection.output_root", path=path)
    if detection.get("datasets") is not None:
        _ensure_string_list(detection.get("datasets"), section="detection.datasets", path=path)
    if detection.get("variants") is not None:
        _ensure_string_list(detection.get("variants"), section="detection.variants", path=path)
    for key in ("bundle_a_config", "bundle_a_conservative_config", "bundle_b_config", "bundle_d_config", "model", "device"):
        if detection.get(key) not in {None, ""}:
            _ensure_string(detection.get(key), section=f"detection.{key}", path=path)
    for key in ("epochs", "imgsz", "batch", "workers"):
        if detection.get(key) not in {None, ""}:
            _ensure_int_like(detection.get(key), section=f"detection.{key}", path=path)
    if detection.get("limit_per_split") not in {None, ""} and not isinstance(detection.get("limit_per_split"), str):
        _ensure_int_like(detection.get("limit_per_split"), section="detection.limit_per_split", path=path)

    if reports.get("surface") not in {None, ""}:
        surface = _ensure_string(reports.get("surface"), section="reports.surface", path=path)
        if surface not in {"public", "private", "all"}:
            raise ValueError(f"reports.surface in {_config_path_label(path)} must be `public`, `private`, or `all`.")
    if reports.get("summary_output") not in {None, ""}:
        _ensure_string(reports.get("summary_output"), section="reports.summary_output", path=path)
    if reports.get("demo_index_max_examples") not in {None, ""}:
        _ensure_int_like(reports.get("demo_index_max_examples"), section="reports.demo_index_max_examples", path=path)

    if freeze is not None:
        freeze_mapping = _ensure_mapping(freeze, section="freeze", path=path)
        _validate_allowed_keys(freeze_mapping, allowed=_FINAL_SWEEP_FREEZE_KEYS, section="final sweep freeze", path=path)
        if freeze_mapping.get("selected_datasets") is not None:
            _ensure_string_list(freeze_mapping.get("selected_datasets"), section="freeze.selected_datasets", path=path)
        if freeze_mapping.get("selected_variants") is not None:
            _ensure_string_list(freeze_mapping.get("selected_variants"), section="freeze.selected_variants", path=path)
        if freeze_mapping.get("notes") is not None:
            _ensure_string_list(freeze_mapping.get("notes"), section="freeze.notes", path=path)


def load_yaml(path: Path, *, expected_kind: Literal["bundle", "detection", "final_sweep"] | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} did not contain a mapping at the top level.")
    if expected_kind == "bundle":
        _validate_bundle_config(path, payload)
    elif expected_kind == "detection":
        _validate_detection_config(path, payload)
    elif expected_kind == "final_sweep":
        _validate_final_sweep_config(path, payload)
    elif expected_kind is not None:
        raise ValueError(f"Unsupported config validation kind: {expected_kind}")
    return payload


def slugify_cli_token(text: str) -> str:
    collapsed = "".join(character.lower() if character.isalnum() else "_" for character in str(text))
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_")


def resolve_bundle_output_root(
    repo_root: Path,
    config: dict[str, Any],
    *,
    bundle_name: str,
    dataset_name: str,
    split: str | None,
    sample_limit: int | None,
    extra_tokens: Sequence[str] = (),
) -> Path:
    dataset_cfg = dict(config.get("dataset", {}))
    configured_root = str(config.get("outputs", {}).get("root", f"results/{bundle_name}"))
    override_tokens = [slugify_cli_token(token) for token in extra_tokens if str(token).strip()]
    if dataset_name != dataset_cfg.get("name"):
        override_tokens.append(slugify_cli_token(dataset_name))
    if split and split != dataset_cfg.get("split"):
        override_tokens.append(slugify_cli_token(split))
    configured_limit = dataset_cfg.get("sample_limit")
    if sample_limit is not None and sample_limit != configured_limit:
        override_tokens.append(f"n{int(sample_limit)}")

    if override_tokens:
        return (repo_root / "outputs" / (bundle_name + "_" + "__".join(override_tokens))).resolve()
    configured_root_path = Path(configured_root)
    return configured_root_path.resolve() if configured_root_path.is_absolute() else (repo_root / configured_root_path).resolve()


def bundle_artifact_identity(
    *,
    bundle_name: str,
    dataset_name: str,
    split: str | None,
    sample_limit: int | None,
    manifest_path: Path,
    config: dict[str, Any],
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "artifact_kind": "bundle_run",
        "bundle_name": bundle_name,
        "dataset": dataset_name,
        "split": split or "all",
        "sample_limit": sample_limit if sample_limit is not None else "all",
        "manifest_path": manifest_path.resolve().as_posix(),
        "manifest_mtime": manifest_path.stat().st_mtime if manifest_path.exists() else None,
        "config_hash": payload_fingerprint(config),
    }
    if extra_fields:
        identity.update(extra_fields)
    return identity


def save_config_snapshot(config: dict[str, Any], output_root: Path) -> Path:
    path = output_root / "config_snapshot.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def resolve_manifest_path(repo_root: Path, dataset_name: str, manifest_override: str | None) -> Path:
    if manifest_override:
        return Path(manifest_override).resolve()
    registry = DatasetRegistry(default_registry_path(repo_root))
    registration = registry.get(dataset_name)
    if registration is None or not registration.manifest_path:
        raise FileNotFoundError(
            f"No manifest is registered for dataset '{dataset_name}'. Run the dataset setup flow first."
        )
    return Path(registration.manifest_path)


def load_manifest_records(
    manifest_path: Path,
    *,
    split: str | None,
    sample_limit: int | None,
    dataset_name: str,
    product_family: str | None = None,
) -> list[dict[str, str]]:
    rows = read_csv_rows(manifest_path)
    filtered: list[dict[str, str]] = []
    normalized_split = split.lower() if split else None
    normalized_family = product_family.upper() if product_family else None
    for row in rows:
        if row.get("record_type") == "placeholder":
            continue
        row_split = row.get("split", "all").lower()
        if normalized_split and row_split not in {normalized_split, "all"}:
            continue
        if dataset_name == "sentinel1":
            family = (row.get("product_family") or row.get("product_type") or "").upper()
            if normalized_family and not family.startswith(normalized_family):
                continue
        filtered.append(row)
    if sample_limit is not None:
        filtered = filtered[:sample_limit]
    return filtered


def _read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as payload:
            first_key = next(iter(payload.files), None)
            if first_key is None:
                raise ValueError(f"No arrays were found inside {path}.")
            return payload[first_key]
    return np.asarray(skio.imread(path))


def _extract_complex_image(array: np.ndarray) -> np.ndarray | None:
    if np.iscomplexobj(array):
        return np.asarray(array, dtype=np.complex64)
    if array.ndim == 3:
        if array.shape[-1] == 2:
            return array[..., 0].astype(np.float32) + 1j * array[..., 1].astype(np.float32)
        if array.shape[0] == 2 and array.shape[0] < array.shape[-1]:
            moved = np.moveaxis(array, 0, -1)
            return moved[..., 0].astype(np.float32) + 1j * moved[..., 1].astype(np.float32)
    return None


def _intensity_from_complex(complex_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amplitude = np.abs(complex_image).astype(np.float32)
    intensity = np.square(amplitude, dtype=np.float32)
    display = np.log1p(amplitude).astype(np.float32)
    return intensity, display


def _ensure_sample_domains(
    image: np.ndarray,
    *,
    dataset_name: str,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str, str]:
    array = np.asarray(image)
    pixel_domain = str(metadata.get("pixel_domain", "unknown")).strip().lower()
    complex_image = _extract_complex_image(array)

    if pixel_domain in COMPLEX_DOMAIN_HINTS or complex_image is not None:
        if complex_image is None:
            raise ValueError("Metadata indicates a complex domain, but the array could not be interpreted as complex.")
        intensity, display = _intensity_from_complex(complex_image)
        return intensity, display, complex_image, "Derived intensity power from a complex SLC-like input.", "complex_slc"

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in {1, 2, 3, 4} and array.shape[0] < array.shape[-1]:
        array = np.moveaxis(array, 0, -1)

    if dataset_name == "sen1floods11":
        if array.ndim == 3:
            band_count = min(array.shape[-1], 2)
            db_bands = array[..., :band_count]
            linear_bands = np.power(10.0, db_bands / 10.0, dtype=np.float32)
            intensity = np.mean(linear_bands, axis=-1)
            return (
                intensity.astype(np.float32),
                np.asarray(db_bands[..., 0], dtype=np.float32),
                None,
                "Converted Sen1Floods11 GRD dB chips to linear intensity using the mean VV/VH proxy.",
                "log_db",
            )
        intensity = np.power(10.0, array / 10.0, dtype=np.float32)
        return intensity.astype(np.float32), array.astype(np.float32), None, "Converted single-band Sen1Floods11 dB imagery to linear intensity.", "log_db"

    if pixel_domain in {"amplitude", "detected_ground_range", "ground_range_amplitude"}:
        amplitude = np.maximum(array.astype(np.float32), 0.0)
        intensity = np.square(amplitude, dtype=np.float32)
        display = np.log1p(amplitude).astype(np.float32)
        return (
            intensity.astype(np.float32),
            display,
            None,
            "Derived intensity power from an amplitude-domain detected SAR raster.",
            "amplitude",
        )

    if array.ndim == 3:
        intensity = np.mean(array[..., : min(array.shape[-1], 3)], axis=-1)
        return intensity.astype(np.float32), array.astype(np.float32), None, "Collapsed a multi-channel detected-image product to a single intensity proxy by averaging channels.", pixel_domain or "detected_chip"

    if dataset_name == "sentinel1" and pixel_domain == "log_db":
        intensity = np.power(10.0, array / 10.0, dtype=np.float32)
        return intensity.astype(np.float32), array.astype(np.float32), None, "Converted log-dB Sentinel-1 raster values to linear intensity.", "log_db"

    return array.astype(np.float32), array.astype(np.float32), None, "Used the single-band detected-image values directly as an intensity proxy.", pixel_domain or "intensity"


def _resolve_sentinel1_measurement(record: dict[str, str]) -> tuple[Path | None, str]:
    prepared = prepare_sentinel1_record(
        record,
        repo_root=repo_root_from_path(Path(__file__)),
    )
    return prepared.image_path, prepared.notes


@lru_cache(maxsize=8)
def _load_coco_annotation_index(annotation_path_text: str) -> dict[str, Any]:
    annotation_path = Path(annotation_path_text)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images_by_stem: dict[str, dict[str, Any]] = {}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    categories = {
        int(category["id"]): category.get("name", "unknown")
        for category in payload.get("categories", [])
        if "id" in category
    }
    for image_payload in payload.get("images", []):
        file_name = str(image_payload.get("file_name", ""))
        images_by_stem[Path(file_name).stem] = image_payload
    for annotation in payload.get("annotations", []):
        image_id = int(annotation.get("image_id", -1))
        annotations_by_image[image_id].append(annotation)
    return {
        "images_by_stem": images_by_stem,
        "annotations_by_image": annotations_by_image,
        "categories": categories,
    }


def _load_hrsid_annotation(annotation_path: Path, sample_id: str) -> dict[str, Any] | None:
    if not annotation_path.exists():
        return None
    indexed = _load_coco_annotation_index(annotation_path.resolve().as_posix())
    image_payload = indexed["images_by_stem"].get(sample_id)
    if image_payload is None:
        return None
    image_id = int(image_payload.get("id", -1))
    annotations = indexed["annotations_by_image"].get(image_id, [])
    return {
        "image": image_payload,
        "annotations": annotations,
        "categories": indexed["categories"],
    }


def load_sample(record: dict[str, str], dataset_name: str) -> LoadedSample:
    metadata = deserialize_json_field(record.get("metadata_json")) or {}
    sample_id = record.get("sample_id", "sample")
    split = record.get("split", "all")
    annotation: dict[str, Any] | None = None
    downstream_target: np.ndarray | None = None

    if dataset_name == "sentinel1":
        prepared = prepare_sentinel1_record(
            record,
            repo_root=repo_root_from_path(Path(__file__)),
        )
        measurement_path, sentinel_note = prepared.image_path, prepared.notes
        if measurement_path is None:
            raise RuntimeError(sentinel_note)
        image, raster_note = read_sentinel1_measurement(measurement_path)
        metadata = hydrate_sentinel1_metadata(
            metadata,
            prepared,
            image_width=int(image.shape[1]) if np.asarray(image).ndim >= 2 else None,
        )
        intensity, display, complex_image, source_note, pixel_domain = _ensure_sample_domains(
            image,
            dataset_name=dataset_name,
            metadata=metadata,
        )
        return LoadedSample(
            dataset_name=dataset_name,
            sample_id=sample_id,
            split=split,
            intensity_image=intensity,
            display_image=display,
            metadata=metadata,
            annotation=None,
            annotation_count=int(record.get("annotation_count") or 0),
            downstream_target=None,
            source_note=f"{sentinel_note} {raster_note} {source_note}".strip(),
            complex_image=complex_image,
            pixel_domain=pixel_domain,
        )

    image_path = Path(record.get("image_path", ""))
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    image = _read_image(image_path)
    intensity, display, complex_image, source_note, pixel_domain = _ensure_sample_domains(
        image,
        dataset_name=dataset_name,
        metadata=metadata,
    )

    annotation_count = int(record.get("annotation_count") or 0)
    annotation_path_text = record.get("annotation_path", "")
    if annotation_path_text:
        annotation_path = Path(annotation_path_text)
        if annotation_path.exists():
            if dataset_name in {"ssdd", "ls_ssdd"}:
                annotation = parse_voc_annotation(annotation_path)
                annotation_count = len(annotation.get("objects", []))
            elif dataset_name == "sen1floods11":
                mask = _read_image(annotation_path)
                downstream_target = np.squeeze(mask).astype(np.int32)
                annotation = {
                    "mask_path": annotation_path.as_posix(),
                    "valid_labels": [0, 1],
                    "ignore_value": -1,
                }
            elif dataset_name == "hrsid":
                annotation = _load_hrsid_annotation(annotation_path, sample_id)
                if annotation is not None:
                    annotation_count = len(annotation.get("annotations", []))
            else:
                annotation = {
                    "annotation_path": annotation_path.resolve().as_posix(),
                    "format": annotation_path.suffix.lower(),
                }

    return LoadedSample(
        dataset_name=dataset_name,
        sample_id=sample_id,
        split=split,
        intensity_image=intensity,
        display_image=display,
        metadata=metadata,
        annotation=annotation,
        annotation_count=annotation_count,
        downstream_target=downstream_target,
        source_note=source_note,
        complex_image=complex_image,
        pixel_domain=pixel_domain,
    )


def prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    layout = {
        "root": output_root,
        "config": output_root / "config",
        "metrics": output_root / "metrics",
        "plots": output_root / "plots",
        "panels": output_root / "plots" / "panels",
        "galleries": output_root / "galleries",
        "statistics": output_root / "statistics",
        "tables": output_root / "tables",
        "logs": output_root / "logs",
        "side_by_side": output_root / "plots" / "side_by_side",
        "diagnostic_maps": output_root / "plots" / "diagnostic_maps",
        "arrays": output_root / "logs" / "arrays",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def save_map_figure(output_path: Path, image: np.ndarray, *, title: str, cmap: str = "magma") -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(prepare_display_image(image), cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _safe_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def aggregate_numeric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    numeric_fields: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                numeric_fields.setdefault(key, []).append(float(value))
            elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                numeric_fields.setdefault(key, []).append(float(value))
    aggregate_rows: list[dict[str, Any]] = []
    for key in sorted(numeric_fields):
        values = numeric_fields[key]
        aggregate_rows.append(
            {
                "metric": key,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": int(len(values)),
            }
        )
    return aggregate_rows
