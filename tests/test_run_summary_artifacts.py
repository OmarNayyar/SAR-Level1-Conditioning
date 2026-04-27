from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.bundles.common import write_summary_artifacts


def test_write_summary_artifacts_emits_markdown_and_topline_json() -> None:
    output_root = Path.cwd() / "outputs" / "test_run_summary_artifacts"
    if output_root.exists():
        shutil.rmtree(output_root)

    summary = {
        "bundle_name": "bundle_a",
        "dataset": "ssdd",
        "processed_count": 4,
        "skipped_count": 0,
        "input_record_count": 4,
        "downstream_status": "proxy-only",
        "aggregate_metrics": {
            "proxy_enl_before": 5.0,
            "proxy_enl_after": 8.0,
            "proxy_enl_gain": 3.0,
            "edge_sharpness_before": 10.0,
            "edge_sharpness_after": 8.0,
            "edge_sharpness_delta": -2.0,
            "distribution_separability_before": 0.4,
            "distribution_separability_after": 0.6,
            "threshold_f1_before": 0.2,
            "threshold_f1_after": 0.35,
            "additive_metadata_available": 0.0,
        },
        "additive_submethod_counts": {"A2": 4},
        "maturity_note": "Current lead and most interpretable baseline.",
        "interpretation": "This run used A2 additive routing and improved separability.",
        "current_recommendation": "Use A2 as the practical default on metadata-poor public chip data.",
        "warnings": ["Proxy-only downstream evaluation."],
    }

    write_summary_artifacts(output_root, summary)

    topline_path = output_root / "metrics" / "topline_metrics.json"
    markdown_path = output_root / "tables" / "run_summary.md"
    assert topline_path.exists()
    assert markdown_path.exists()

    topline = json.loads(topline_path.read_text(encoding="utf-8"))
    assert topline["dominant_additive_submethod"] == "A2"
    assert topline["distribution_separability_delta"] == 0.19999999999999996
    assert "A2" in markdown_path.read_text(encoding="utf-8")

    shutil.rmtree(output_root)
