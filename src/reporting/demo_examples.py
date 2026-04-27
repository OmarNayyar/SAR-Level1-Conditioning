from __future__ import annotations

"""Curate small visual examples for the Streamlit demo page.

The demo index is intentionally built from existing run outputs. It does not
download data, train models, or create new scientific claims; it simply chooses
representative before/after panels that already exist on disk.
"""

from pathlib import Path
from typing import Any

from src.datasets.common import write_json

from .decision_support import safe_float
from .result_index import discover_bundle_runs, load_bundle_run, load_sentinel1_batch_snapshot


def _sample_score(row: dict[str, Any]) -> float:
    score = safe_float(row.get("decision_score"))
    if score is not None:
        return score
    return safe_float(row.get("bundle_quality_score")) or 0.0


def _valid_image_path(path_text: Any, repo_root: Path) -> str:
    """Return a usable image path or an empty string.

    Older run artifacts accidentally stored an empty `Path()` as the repository
    root.  The demo curator must reject directories and missing files so the app
    never tries to render the repo root as an image.
    """

    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = path if path.exists() else repo_root / path
    try:
        resolved = path.resolve()
    except OSError:
        return ""
    if resolved == repo_root.resolve() or not resolved.is_file():
        return ""
    if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        return ""
    return resolved.as_posix()


def _paths_available(row: dict[str, Any], repo_root: Path) -> bool:
    return bool(_valid_image_path(row.get("before_panel_path"), repo_root) and _valid_image_path(row.get("after_panel_path"), repo_root))


def _example_from_sample(
    run: dict[str, Any],
    sample: dict[str, Any],
    label: str,
    reason: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "label": label,
        "reason": reason,
        "bundle_name": run["bundle_name"],
        "dataset": run["dataset"],
        "sample_id": sample.get("sample_id", ""),
        "additive_submethod": sample.get("additive_submethod_code", sample.get("additive_submethod_used", "")),
        "before_panel_path": _valid_image_path(sample.get("before_panel_path", ""), repo_root),
        "after_panel_path": _valid_image_path(sample.get("after_panel_path", ""), repo_root),
        "difference_panel_path": _valid_image_path(sample.get("difference_panel_path", ""), repo_root),
        "side_by_side_path": _valid_image_path(sample.get("side_by_side_path", ""), repo_root),
        "decision_score": _sample_score(sample),
        "decision_confidence": sample.get("decision_confidence", sample.get("decision_evidence_grade", "")),
        "interpretation": sample.get("decision_rationale", sample.get("source_note", "")),
        "caveat": sample.get("decision_metric_bias_warning", ""),
    }


def collect_demo_examples(repo_root: Path, *, max_examples: int = 8) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for run in discover_bundle_runs(repo_root):
        payload = load_bundle_run(Path(run.output_root))
        samples = [
            sample
            for sample in payload.get("per_sample_metrics", {}).get("samples", [])
            if _paths_available(sample, repo_root)
        ]
        if not samples:
            continue
        run_payload = {"bundle_name": run.bundle_name, "dataset": run.dataset}
        ordered = sorted(samples, key=_sample_score)
        examples.append(
            _example_from_sample(
                run_payload,
                ordered[-1],
                "Strong proxy case",
                "Highest current decision score in this run.",
                repo_root=repo_root,
            )
        )
        examples.append(
            _example_from_sample(
                run_payload,
                ordered[len(ordered) // 2],
                "Representative case",
                "Middle-ranked sample, useful for a typical visual check.",
                repo_root=repo_root,
            )
        )
        examples.append(
            _example_from_sample(
                run_payload,
                ordered[0],
                "Caveat / weak case",
                "Lowest current decision score; inspect for oversmoothing or weak separation.",
                repo_root=repo_root,
            )
        )
        if len(examples) >= max_examples:
            break

    sentinel_payload = load_sentinel1_batch_snapshot(repo_root)
    comparison_rows = sentinel_payload.get("submethod_comparison", {}).get("rows", [])
    if comparison_rows:
        sentinel_rows = [row for row in comparison_rows if _paths_available(row, repo_root)]
        if sentinel_rows:
            best = max(sentinel_rows, key=_sample_score)
            examples.insert(
                0,
                {
                    **_example_from_sample(
                        {
                            "bundle_name": "bundle_a_sentinel1_batch",
                            "dataset": "sentinel1",
                        },
                        best,
                        "Sentinel-1 real-product example",
                        "Best current Sentinel-1 Bundle A submethod row under the balanced proxy heuristic.",
                        repo_root=repo_root,
                    ),
                    "scene_id": best.get("scene_id", ""),
                    "caveat": "Sentinel-1 examples are proxy-only and overview-scale.",
                },
            )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for example in examples:
        key = (str(example.get("bundle_name")), str(example.get("dataset")), str(example.get("sample_id") or example.get("scene_id")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped[:max_examples]


def write_demo_index(repo_root: Path, *, max_examples: int = 8) -> Path:
    """Write the current curated demo index as a small app-friendly JSON file."""

    output_path = repo_root / "outputs" / "demo_examples" / "demo_index.json"
    write_json(output_path, {"examples": collect_demo_examples(repo_root, max_examples=max_examples)})
    return output_path
