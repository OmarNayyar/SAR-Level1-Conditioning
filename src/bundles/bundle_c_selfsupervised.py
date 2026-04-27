from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.bundles.common import run_stage1_bundle
from src.stage1.additive.starlet_complex_denoise import starlet_complex_denoise
from src.stage1.multiplicative.merlin_wrapper import run_merlin_wrapper
from src.stage1.pipeline import BundleProcessResult, LoadedSample


def _resolve_complex_input(sample: LoadedSample, allow_detected_fallback: bool) -> tuple[np.ndarray, list[str]]:
    if sample.complex_image is not None:
        return sample.complex_image.astype(np.complex64), []
    if not allow_detected_fallback:
        raise RuntimeError(
            "Bundle C requires a complex SLC-like input. This sample only exposes detected-domain intensity."
        )
    pseudo_amplitude = np.sqrt(np.maximum(sample.intensity_image, 0.0)).astype(np.float32)
    pseudo_complex = pseudo_amplitude.astype(np.complex64)
    return pseudo_complex, [
        "Used a surrogate complex input derived from detected-domain amplitude because no local complex SLC sample was available."
    ]


def process_bundle_c_sample(sample: LoadedSample, config: dict[str, Any]) -> BundleProcessResult:
    additive_cfg = config.get("processing", {}).get("additive", {})
    multiplicative_cfg = config.get("processing", {}).get("multiplicative", {})
    complex_input, notes = _resolve_complex_input(
        sample,
        allow_detected_fallback=bool(config.get("bundle", {}).get("allow_detected_fallback", True)),
    )
    starlet_result = starlet_complex_denoise(
        complex_input,
        levels=int(additive_cfg.get("levels", 4)),
        threshold_scale=float(additive_cfg.get("threshold_scale", 3.0)),
    )
    merlin_result = run_merlin_wrapper(
        starlet_result.denoised_complex,
        external=multiplicative_cfg.get("external"),
        fallback_levels=int(multiplicative_cfg.get("fallback_levels", 3)),
        fallback_threshold_scale=float(multiplicative_cfg.get("fallback_threshold_scale", 2.5)),
    )
    notes.extend(
        [
            "Bundle C is a feasibility path because local complex-SLC downstream truth is still limited.",
            merlin_result.notes,
        ]
    )
    return BundleProcessResult(
        additive_output=starlet_result.surrogate_intensity.astype("float32"),
        final_output=merlin_result.output_intensity.astype("float32"),
        additive_applied=starlet_result.applied,
        additive_mode=f"starlet_complex_L{starlet_result.levels}",
        additive_notes=starlet_result.notes,
        multiplicative_mode=f"merlin_{merlin_result.backend}",
        multiplicative_notes=merlin_result.notes,
        display_output=np.log1p(np.abs(merlin_result.output_complex)).astype(np.float32),
        extra_arrays={
            "bundle_c_real_output": np.real(merlin_result.output_complex).astype(np.float32),
            "bundle_c_imag_output": np.imag(merlin_result.output_complex).astype(np.float32),
        },
        notes=notes,
    )


def run_bundle_c(
    records: list[dict[str, str]],
    *,
    dataset_name: str,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    return run_stage1_bundle(
        records=records,
        dataset_name=dataset_name,
        config=config,
        output_root=output_root,
        processor=process_bundle_c_sample,
        bundle_name="bundle_c",
        after_title="Bundle C output",
    )
