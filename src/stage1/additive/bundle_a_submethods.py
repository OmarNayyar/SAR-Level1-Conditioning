from __future__ import annotations

"""Bundle A additive submethod routing.

Bundle A always operates in detected intensity space before the Refined Lee
speckle step. This module keeps the additive choices explicit:

- A0: control path, no additive correction.
- A1: metadata-driven thermal/noise-vector subtraction.
- A2: image-derived additive floor estimate for metadata-poor chips.
- A3: structured additive artifact correction for stripe-like contamination.

The auto router is intentionally conservative. It uses metadata first, then
structured-artifact evidence, then the image-derived fallback.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .destripe_lowrank_sparse import destripe_lowrank_sparse
from .intensity_floor_estimate import estimate_intensity_floor
from .thermal_noise_subtract import thermal_noise_subtract_intensity


BUNDLE_A_SUBMETHOD_ORDER = ("A0", "A1", "A2", "A3")
ADDITIVE_METADATA_FIELDS = (
    "noise_vector",
    "noise_power",
    "nesz_db",
    "noise_xml_path",
    "calibration_xml_path",
    "manifest_safe_path",
    "primary_polarization",
)


@dataclass(frozen=True, slots=True)
class BundleAAdditiveSpec:
    code: str
    label: str
    description: str
    required_inputs: tuple[str, ...]
    trust_level: str


@dataclass(slots=True)
class BundleAAdditiveOutcome:
    spec: BundleAAdditiveSpec
    corrected_intensity: np.ndarray
    estimated_additive_component: np.ndarray
    additive_applied: bool
    additive_mode: str
    additive_notes: str
    metadata_available: bool
    metadata_fields_present: list[str]
    fallback_used: str | None
    selection_mode: str
    selection_reason: str
    confidence_level: str
    warning: str
    extra_arrays: dict[str, np.ndarray]
    artifact_score: float | None = None
    artifact_orientation: str | None = None

    def to_metadata_fields(self) -> dict[str, Any]:
        return {
            "additive_submethod_code": self.spec.code,
            "additive_submethod_name": self.spec.label,
            "additive_submethod_description": self.spec.description,
            "additive_submethod_required_inputs": " | ".join(self.spec.required_inputs),
            "additive_metadata_available": self.metadata_available,
            "additive_metadata_fields_present": ", ".join(self.metadata_fields_present),
            "additive_fallback_used": self.fallback_used or "",
            "additive_selection_mode": self.selection_mode,
            "additive_selection_reason": self.selection_reason,
            "additive_confidence_level": self.confidence_level,
            "additive_warning": self.warning,
            "additive_artifact_score": self.artifact_score,
            "additive_artifact_orientation": self.artifact_orientation or "",
        }


BUNDLE_A_SUBMETHOD_SPECS: dict[str, BundleAAdditiveSpec] = {
    "A0": BundleAAdditiveSpec(
        code="A0",
        label="A0 - No Additive Correction",
        description="Pass-through baseline with no explicit additive-noise correction.",
        required_inputs=("intensity image only",),
        trust_level="baseline-only",
    ),
    "A1": BundleAAdditiveSpec(
        code="A1",
        label="A1 - Metadata Thermal / Noise-Vector Subtraction",
        description="Subtract additive noise using metadata-derived noise vectors, explicit noise power, or NESZ-style metadata.",
        required_inputs=("noise_vector or noise_power or nesz_db metadata",),
        trust_level="metadata-driven",
    ),
    "A2": BundleAAdditiveSpec(
        code="A2",
        label="A2 - Image-Derived Additive Floor Estimate",
        description="Estimate a constant additive floor directly from the lower-tail intensity distribution.",
        required_inputs=("intensity image only",),
        trust_level="image-derived",
    ),
    "A3": BundleAAdditiveSpec(
        code="A3",
        label="A3 - Structured Additive Artifact Correction",
        description="Apply an artifact-aware additive cleanup for stripe-like or structured intensity-domain contamination.",
        required_inputs=("intensity image", "detectable structured artifact pattern"),
        trust_level="artifact-based",
    ),
}


def available_additive_metadata_fields(metadata: dict[str, Any]) -> list[str]:
    present: list[str] = []
    for field_name in ADDITIVE_METADATA_FIELDS:
        value = metadata.get(field_name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        present.append(field_name)
    return present


def has_metadata_for_a1(metadata: dict[str, Any]) -> bool:
    return any(field in {"noise_vector", "noise_power", "nesz_db"} for field in available_additive_metadata_fields(metadata))


def detect_structured_artifact(
    intensity_image: np.ndarray,
    *,
    smoothing_sigma: float = 5.0,
) -> dict[str, Any]:
    """Return a lightweight row/column artifact score from median profiles."""

    image = np.asarray(intensity_image, dtype=np.float32)
    eps = 1e-6
    image_scale = float(np.nanstd(image) + eps)

    column_profile = np.median(image, axis=0).astype(np.float32)
    row_profile = np.median(image, axis=1).astype(np.float32)
    column_residual = column_profile - gaussian_filter1d(column_profile, sigma=float(smoothing_sigma), mode="reflect")
    row_residual = row_profile - gaussian_filter1d(row_profile, sigma=float(smoothing_sigma), mode="reflect")

    column_score = float(np.std(column_residual) / image_scale)
    row_score = float(np.std(row_residual) / image_scale)
    orientation = "columns" if column_score >= row_score else "rows"
    score = max(column_score, row_score)
    return {
        "orientation": orientation,
        "artifact_score": score,
        "column_artifact_score": column_score,
        "row_artifact_score": row_score,
    }


def _resolve_requested_submethod(additive_cfg: dict[str, Any]) -> tuple[str, str]:
    requested = str(
        additive_cfg.get("forced_submethod")
        or additive_cfg.get("submethod")
        or additive_cfg.get("mode")
        or "auto"
    ).strip().upper()
    if requested in {"", "AUTO"}:
        return "auto", "auto"
    if requested not in BUNDLE_A_SUBMETHOD_SPECS:
        raise ValueError(
            f"Unsupported Bundle A additive submethod {requested!r}. Expected one of: auto, "
            + ", ".join(BUNDLE_A_SUBMETHOD_ORDER)
        )
    return "forced", requested


def _run_a0(intensity_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, str, str, dict[str, np.ndarray]]:
    image = np.asarray(intensity_image, dtype=np.float32)
    return (
        image.copy(),
        np.zeros_like(image, dtype=np.float32),
        False,
        "pass_through",
        "Bundle A baseline A0 left the additive component untouched.",
        {},
    )


def _run_a1(intensity_image: np.ndarray, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, bool, str, str, dict[str, np.ndarray]]:
    result = thermal_noise_subtract_intensity(intensity_image, metadata=metadata)
    return (
        result.corrected_intensity.astype(np.float32),
        result.estimated_noise_power.astype(np.float32),
        bool(result.applied),
        result.mode,
        result.notes,
        {},
    )


def _run_a2(intensity_image: np.ndarray, additive_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, bool, str, str, dict[str, np.ndarray]]:
    floor_cfg = additive_cfg.get("image_floor", {})
    result = estimate_intensity_floor(
        intensity_image,
        floor_quantile=float(floor_cfg.get("floor_quantile", 0.02)),
        quiet_upper_quantile=float(floor_cfg.get("quiet_upper_quantile", 0.25)),
    )
    return (
        result.corrected_intensity.astype(np.float32),
        result.estimated_noise_power.astype(np.float32),
        bool(result.applied),
        result.mode,
        result.notes,
        {},
    )


def _run_a3(
    intensity_image: np.ndarray,
    additive_cfg: dict[str, Any],
    artifact_info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool, str, str, dict[str, np.ndarray]]:
    structured_cfg = additive_cfg.get("structured_artifact", {})
    result = destripe_lowrank_sparse(
        intensity_image,
        domain=str(structured_cfg.get("domain", "intensity")),
        orientation=str(structured_cfg.get("orientation", artifact_info.get("orientation", "columns"))),
        background_sigma=float(structured_cfg.get("background_sigma", 9.0)),
        profile_sigma=float(structured_cfg.get("profile_sigma", 3.0)),
        correction_strength=float(structured_cfg.get("correction_strength", 1.0)),
    )
    return (
        result.corrected_image.astype(np.float32),
        result.stripe_component.astype(np.float32),
        bool(result.applied),
        result.mode,
        result.notes,
        {
            "artifact_lowrank_component": result.lowrank_component.astype(np.float32),
            "artifact_stripe_component": result.stripe_component.astype(np.float32),
        },
    )


def run_bundle_a_additive_submethod(
    intensity_image: np.ndarray,
    metadata: dict[str, Any],
    additive_cfg: dict[str, Any] | None = None,
) -> BundleAAdditiveOutcome:
    """Run the requested Bundle A additive route and record why it was chosen."""

    additive_cfg = additive_cfg or {}
    selection_mode, requested = _resolve_requested_submethod(additive_cfg)
    metadata_fields_present = available_additive_metadata_fields(metadata)
    metadata_available = has_metadata_for_a1(metadata)
    artifact_info = detect_structured_artifact(
        intensity_image,
        smoothing_sigma=float(additive_cfg.get("artifact_smoothing_sigma", 5.0)),
    )
    artifact_threshold = float(additive_cfg.get("artifact_auto_threshold", 0.1))
    fallback_submethod = str(additive_cfg.get("fallback_submethod", "A2")).strip().upper() or "A2"
    if fallback_submethod not in BUNDLE_A_SUBMETHOD_SPECS:
        fallback_submethod = "A2"

    fallback_used: str | None = None
    warning = ""
    selection_reason = ""

    if selection_mode == "auto":
        if metadata_available:
            selected = "A1"
            selection_reason = "Auto-selected A1 because additive metadata was available."
        elif float(artifact_info["artifact_score"]) >= artifact_threshold:
            selected = "A3"
            selection_reason = (
                "Auto-selected A3 because no additive metadata was available and the structured-artifact "
                f"score {artifact_info['artifact_score']:.3f} exceeded the threshold {artifact_threshold:.3f}."
            )
        else:
            selected = "A2"
            selection_reason = "Auto-selected A2 because no additive metadata was available and no strong structured artifact was detected."
    else:
        selected = requested
        selection_reason = f"Forced Bundle A additive submethod {selected} from the run configuration or CLI override."

    if selected == "A1" and not metadata_available:
        fallback_used = fallback_submethod
        warning = (
            f"Requested A1 but additive metadata was unavailable; falling back to {fallback_submethod}."
        )
        selected = fallback_submethod

    spec = BUNDLE_A_SUBMETHOD_SPECS[selected]
    if selected == "A0":
        corrected, additive_component, applied, mode, notes, extra_arrays = _run_a0(intensity_image)
    elif selected == "A1":
        corrected, additive_component, applied, mode, notes, extra_arrays = _run_a1(intensity_image, metadata)
    elif selected == "A2":
        corrected, additive_component, applied, mode, notes, extra_arrays = _run_a2(intensity_image, additive_cfg)
    else:
        corrected, additive_component, applied, mode, notes, extra_arrays = _run_a3(intensity_image, additive_cfg, artifact_info)

    return BundleAAdditiveOutcome(
        spec=spec,
        corrected_intensity=corrected.astype(np.float32),
        estimated_additive_component=additive_component.astype(np.float32),
        additive_applied=applied,
        additive_mode=mode,
        additive_notes=notes,
        metadata_available=metadata_available,
        metadata_fields_present=metadata_fields_present,
        fallback_used=fallback_used,
        selection_mode=selection_mode,
        selection_reason=selection_reason,
        confidence_level=spec.trust_level,
        warning=warning,
        extra_arrays={
            **extra_arrays,
            "artifact_score": np.asarray(float(artifact_info["artifact_score"]), dtype=np.float32),
        },
        artifact_score=float(artifact_info["artifact_score"]),
        artifact_orientation=str(artifact_info["orientation"]),
    )
