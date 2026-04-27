from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from src.stage1.downstream.proxy_eval import ProxyEvaluation
from src.stage1.pipeline import BundleProcessResult, LoadedSample
from src.stage1.statistics.intensity_statistics import IntensityStatisticsAnalyzer


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_intensity_statistics_analyzer_produces_summary() -> None:
    rng = np.random.default_rng(11)
    target_mask = np.zeros((64, 64), dtype=np.uint8)
    target_mask[20:36, 24:40] = 1
    background = rng.exponential(scale=1.0, size=(64, 64)).astype(np.float32)
    target = rng.lognormal(mean=1.4, sigma=0.35, size=(64, 64)).astype(np.float32)
    input_image = np.where(target_mask.astype(bool), target, background).astype(np.float32)
    filtered_image = np.where(target_mask.astype(bool), target * 0.9, background * 0.8).astype(np.float32)

    sample = LoadedSample(
        dataset_name="ssdd",
        sample_id="synthetic",
        split="train",
        intensity_image=input_image,
        display_image=input_image,
        metadata={},
        annotation={"objects": [{"bbox": {"xmin": "24", "ymin": "20", "xmax": "40", "ymax": "36"}}]},
        annotation_count=1,
        downstream_target=None,
        source_note="synthetic",
        complex_image=None,
        pixel_domain="intensity",
    )
    process_result = BundleProcessResult(
        additive_output=input_image,
        final_output=filtered_image,
        additive_applied=True,
        additive_mode="synthetic",
        additive_notes="synthetic",
        multiplicative_mode="synthetic",
        multiplicative_notes="synthetic",
    )
    proxy_eval = ProxyEvaluation(
        metrics={},
        downstream_row={},
        predicted_mask=target_mask,
        target_mask=target_mask,
    )

    workspace_root = REPO_ROOT / "outputs" / "test_intensity_statistics_fixture"
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    try:
        analyzer = IntensityStatisticsAnalyzer(workspace_root / "statistics")
        payload = analyzer.process_sample(
            sample=sample,
            process_result=process_result,
            proxy_evaluation=proxy_eval,
            metrics_row={},
        )
        assert payload is not None
        summary = analyzer.finalize()
        assert summary["valid_fit_count"] == 1
        assert (workspace_root / "statistics" / "per_sample_statistics.csv").exists()
        assert (workspace_root / "statistics" / "summary.json").exists()
    finally:
        if workspace_root.exists():
            shutil.rmtree(workspace_root)
