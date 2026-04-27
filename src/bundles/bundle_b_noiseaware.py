from __future__ import annotations

from pathlib import Path
from typing import Any

from src.bundles.common import run_stage1_bundle
from src.stage1.additive.destripe_lowrank_sparse import destripe_lowrank_sparse
from src.stage1.multiplicative.mulog_bm3d import mulog_bm3d
from src.stage1.pipeline import BundleProcessResult, LoadedSample
from src.stage1.statistics import IntensityStatisticsAnalyzer


def process_bundle_b_sample(sample: LoadedSample, config: dict[str, Any]) -> BundleProcessResult:
    additive_cfg = config.get("processing", {}).get("additive", {})
    multiplicative_cfg = config.get("processing", {}).get("multiplicative", {})

    destripe_result = destripe_lowrank_sparse(
        sample.intensity_image,
        domain=str(additive_cfg.get("domain", "log_intensity")),
        orientation=str(additive_cfg.get("orientation", "columns")),
        background_sigma=float(additive_cfg.get("background_sigma", 9.0)),
        profile_sigma=float(additive_cfg.get("profile_sigma", 3.0)),
        correction_strength=float(additive_cfg.get("correction_strength", 1.0)),
    )
    mulog_result = mulog_bm3d(
        destripe_result.corrected_image,
        backend_preference=str(multiplicative_cfg.get("backend_preference", "bm3d")),
    )
    mulog_strength = min(max(float(multiplicative_cfg.get("strength", 1.0)), 0.0), 1.0)
    final_output = (
        (1.0 - mulog_strength) * destripe_result.corrected_image.astype("float32")
        + mulog_strength * mulog_result.filtered_image.astype("float32")
    ).astype("float32")
    notes = []
    if mulog_result.backend != "bm3d":
        notes.append(
            "BM3D was unavailable or not requested, so the MuLoG path fell back to a documented lightweight denoiser."
        )
    if mulog_strength < 1.0:
        notes.append(f"Blended MuLoG output with pre-speckle image using multiplicative strength={mulog_strength:.2f}.")
    return BundleProcessResult(
        additive_output=destripe_result.corrected_image.astype("float32"),
        final_output=final_output,
        additive_applied=destripe_result.applied,
        additive_mode=destripe_result.mode,
        additive_notes=destripe_result.notes,
        multiplicative_mode=f"mulog_{mulog_result.backend}",
        multiplicative_notes=mulog_result.notes,
        estimated_additive_component=destripe_result.stripe_component.astype("float32"),
        extra_arrays={
            "mulog_log_input": mulog_result.log_input.astype("float32"),
            "mulog_log_filtered": mulog_result.log_filtered.astype("float32"),
            "destripe_lowrank_component": destripe_result.lowrank_component.astype("float32"),
        },
        notes=notes,
    )


def run_bundle_b(
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
        processor=process_bundle_b_sample,
        bundle_name="bundle_b",
        after_title="Bundle B output",
        sample_analyzer=analyzer,
    )
