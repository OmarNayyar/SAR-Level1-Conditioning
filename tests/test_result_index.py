from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.reporting.result_index import bundle_layout, discover_bundle_runs, load_bundle_run, load_sentinel1_batch_snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_result_index_discovers_standardized_run_layout() -> None:
    fixture_root = Path.cwd() / "outputs" / "test_result_index_repo"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    bundle_root = fixture_root / "results" / "bundle_a"

    _write_json(
        bundle_root / "metrics" / "run_summary.json",
        {
            "bundle_name": "bundle_a",
            "dataset": "ssdd",
            "processed_count": 2,
            "skipped_count": 0,
        },
    )
    _write_json(bundle_root / "metrics" / "aggregate_metrics.json", {"metrics": [{"metric": "proxy_enl_after", "mean": 4.2}]})
    _write_json(bundle_root / "metrics" / "per_sample_metrics.json", {"samples": [{"sample_id": "0001"}]})
    _write_json(bundle_root / "metrics" / "topline_metrics.json", {"proxy_enl_gain": 1.7})
    _write_json(bundle_root / "tables" / "sample_summary.json", {"samples": [{"sample_id": "0001"}]})
    (bundle_root / "tables").mkdir(parents=True, exist_ok=True)
    (bundle_root / "tables" / "run_summary.md").write_text("# summary\n", encoding="utf-8")

    runs = discover_bundle_runs(fixture_root)
    assert len(runs) == 1
    assert runs[0].bundle_name == "bundle_a"

    payload = load_bundle_run(bundle_root)
    layout = bundle_layout(bundle_root)
    assert payload["summary"]["processed_count"] == 2
    assert payload["aggregate_metrics"]["metrics"][0]["mean"] == 4.2
    assert payload["topline_metrics"]["proxy_enl_gain"] == 1.7
    assert payload["run_summary_markdown"] == "# summary\n"
    assert layout["metrics"] == bundle_root / "metrics"

    shutil.rmtree(fixture_root)


def test_result_index_loads_sentinel1_batch_snapshot() -> None:
    fixture_root = Path.cwd() / "outputs" / "test_result_index_batch"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    batch_root = fixture_root / "outputs" / "bundle_a_sentinel1_batch"

    _write_json(batch_root / "metrics" / "topline_metrics.json", {"evaluated_scene_count": 1})
    _write_json(batch_root / "tables" / "scene_summary.json", {"scenes": [{"scene_id": "scene-1"}]})
    _write_json(batch_root / "tables" / "submethod_comparison.json", {"rows": [{"scene_id": "scene-1", "additive_submethod_used": "A1"}]})
    (batch_root / "tables").mkdir(parents=True, exist_ok=True)
    (batch_root / "tables" / "scene_recommendations.md").write_text("# recommendations\n", encoding="utf-8")

    payload = load_sentinel1_batch_snapshot(fixture_root)
    assert payload["topline_metrics"]["evaluated_scene_count"] == 1
    assert payload["scene_summary"]["scenes"][0]["scene_id"] == "scene-1"
    assert payload["submethod_comparison"]["rows"][0]["additive_submethod_used"] == "A1"
    assert payload["scene_recommendations_markdown"] == "# recommendations\n"

    shutil.rmtree(fixture_root)
