from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.bundles.common import run_stage1_bundle
from src.stage1.additive.pnp_admm_additive import pnp_admm_additive
from src.stage1.multiplicative.speckle2void_wrapper import run_speckle2void_wrapper
from src.stage1.pipeline import BundleProcessResult, LoadedSample
from src.stage1.statistics import IntensityStatisticsAnalyzer


def process_bundle_d_sample(sample: LoadedSample, config: dict[str, Any]) -> BundleProcessResult:
    additive_cfg = config.get("processing", {}).get("additive", {})
    multiplicative_cfg = config.get("processing", {}).get("multiplicative", {})

    pnp_result = pnp_admm_additive(
        sample.intensity_image,
        iterations=int(additive_cfg.get("iterations", 6)),
        rho=float(additive_cfg.get("rho", 1.0)),
        denoiser_backend=str(additive_cfg.get("denoiser_backend", "wavelet")),
        domain=str(additive_cfg.get("domain", "log_intensity")),
    )
    additive_strength = min(max(float(additive_cfg.get("strength", 1.0)), 0.0), 1.0)
    additive_output = (
        (1.0 - additive_strength) * sample.intensity_image.astype("float32")
        + additive_strength * pnp_result.corrected_image.astype("float32")
    ).astype("float32")
    residual_image = np.clip(sample.intensity_image.astype("float32") - additive_output, 0.0, None).astype("float32")
    s2v_result = run_speckle2void_wrapper(
        additive_output,
        external=multiplicative_cfg.get("external"),
        fallback_sigma=float(multiplicative_cfg.get("fallback_sigma", 1.0)),
    )
    multiplicative_strength = min(max(float(multiplicative_cfg.get("strength", 1.0)), 0.0), 1.0)
    final_output = (
        (1.0 - multiplicative_strength) * additive_output
        + multiplicative_strength * s2v_result.filtered_image.astype("float32")
    ).astype("float32")
    notes = [s2v_result.notes]
    if additive_strength < 1.0:
        notes.append(f"Blended PnP-ADMM additive output with input using additive strength={additive_strength:.2f}.")
    if multiplicative_strength < 1.0:
        notes.append(
            f"Blended Speckle2Void output with additive output using multiplicative strength={multiplicative_strength:.2f}."
        )
    return BundleProcessResult(
        additive_output=additive_output,
        final_output=final_output,
        additive_applied=True,
        additive_mode=f"pnp_admm_{pnp_result.backend}",
        additive_notes=pnp_result.notes,
        multiplicative_mode=f"speckle2void_{s2v_result.backend}",
        multiplicative_notes=s2v_result.notes,
        estimated_additive_component=residual_image,
        notes=notes,
    )


def run_bundle_d(
    records: list[dict[str, str]],
    *,
    dataset_name: str,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    statistics_cfg = config.get("statistics", {})
    analyzer = None
    if bool(statistics_cfg.get("enabled", True)):
        analyzer = IntensityStatisticsAnalyzer(output_root / "statistics", statistics_cfg)
    return run_stage1_bundle(
        records=records,
        dataset_name=dataset_name,
        config=config,
        output_root=output_root,
        processor=process_bundle_d_sample,
        bundle_name="bundle_d",
        after_title="Bundle D output",
        sample_analyzer=analyzer,
    )
