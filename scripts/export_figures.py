from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from skimage import io as skio


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.common import read_json, write_csv, write_json
from src.reporting import bundle_layout, resolve_bundle_output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight bundle comparison pack.")
    parser.add_argument(
        "--bundles",
        nargs="+",
        default=["bundle_a", "bundle_b", "bundle_c", "bundle_d"],
        help="Bundle result folders to include.",
    )
    return parser.parse_args()


def _bundle_output_root(bundle_name: str) -> Path:
    return resolve_bundle_output_root(REPO_ROOT, bundle_name)


def _first_existing_image(bundle_root: Path) -> Path | None:
    layout = bundle_layout(bundle_root)
    for candidate in [
        layout["galleries"] / "success_gallery.png",
        layout["galleries"] / "failure_gallery.png",
    ]:
        if candidate.exists():
            return candidate
    side_by_side_dir = layout["side_by_side"]
    if side_by_side_dir.exists():
        candidates = sorted(side_by_side_dir.glob("*.png"))
        if candidates:
            return candidates[0]
    return None


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Bundle Comparison Pack",
        "",
        "| Bundle | Dataset | Processed | Downstream | Evidence | Decision Score | ENL Gain | Edge Delta | Separability Delta | Threshold F1 Delta |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['bundle_name']}` | `{row['dataset']}` | {row['processed_count']} | `{row['downstream_status']}` | "
            f"{row.get('evidence_grade', '')} | {row.get('mean_decision_score', '')} | "
            f"{row.get('mean_proxy_enl_gain', '')} | {row.get('mean_edge_sharpness_delta', '')} | "
            f"{row.get('distribution_separability_delta', '')} | {row.get('threshold_f1_delta', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comparison_rows: list[dict[str, Any]] = []
    figure_inputs: list[tuple[str, Path]] = []

    for bundle_name in args.bundles:
        bundle_root = _bundle_output_root(bundle_name)
        layout = bundle_layout(bundle_root)
        run_summary_path = layout["metrics"] / "run_summary.json"
        if not run_summary_path.exists():
            continue
        run_summary = read_json(run_summary_path)
        aggregate_path = layout["metrics"] / "aggregate_metrics.json"
        aggregate_payload = read_json(aggregate_path) if aggregate_path.exists() else {"metrics": []}
        aggregate_rows = aggregate_payload.get("metrics", [])
        aggregate_by_metric = {row["metric"]: row for row in aggregate_rows if "metric" in row}
        comparison_rows.append(
            {
                "bundle_name": bundle_name,
                "dataset": run_summary.get("dataset", ""),
                "processed_count": run_summary.get("processed_count", 0),
                "skipped_count": run_summary.get("skipped_count", 0),
                "downstream_status": run_summary.get("downstream_status", ""),
                "evidence_grade": run_summary.get("evidence_grade", ""),
                "maturity_note": run_summary.get("maturity_note", ""),
                "current_recommendation": run_summary.get("current_recommendation", ""),
                "mean_decision_score": aggregate_by_metric.get("decision_score", {}).get("mean"),
                "mean_proxy_enl_after": aggregate_by_metric.get("proxy_enl_after", {}).get("mean"),
                "mean_proxy_enl_gain": aggregate_by_metric.get("proxy_enl_gain", {}).get("mean"),
                "mean_edge_sharpness_after": aggregate_by_metric.get("edge_sharpness_after", {}).get("mean"),
                "mean_edge_sharpness_delta": aggregate_by_metric.get("edge_sharpness_delta", {}).get("mean"),
                "distribution_separability_delta": (
                    (aggregate_by_metric.get("distribution_separability_after", {}).get("mean") or 0.0)
                    - (aggregate_by_metric.get("distribution_separability_before", {}).get("mean") or 0.0)
                    if "distribution_separability_after" in aggregate_by_metric
                    and "distribution_separability_before" in aggregate_by_metric
                    else None
                ),
                "threshold_f1_delta": (
                    (aggregate_by_metric.get("threshold_f1_after", {}).get("mean") or 0.0)
                    - (aggregate_by_metric.get("threshold_f1_before", {}).get("mean") or 0.0)
                    if "threshold_f1_after" in aggregate_by_metric and "threshold_f1_before" in aggregate_by_metric
                    else None
                ),
            }
        )
        preview_image = _first_existing_image(bundle_root)
        if preview_image is not None:
            figure_inputs.append((bundle_name, preview_image))

    comparison_root = (REPO_ROOT / "results" / "comparison_tables").resolve()
    figures_root = (REPO_ROOT / "results" / "figures").resolve()
    comparison_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    write_csv(comparison_root / "bundle_comparison.csv", comparison_rows)
    write_json(comparison_root / "bundle_comparison.json", {"bundles": comparison_rows})
    _write_markdown(comparison_root / "bundle_comparison.md", comparison_rows)

    if figure_inputs:
        figure, axes = plt.subplots(len(figure_inputs), 1, figsize=(10, 4 * len(figure_inputs)))
        if len(figure_inputs) == 1:
            axes = [axes]
        for axis, (bundle_name, image_path) in zip(axes, figure_inputs):
            axis.imshow(skio.imread(image_path))
            axis.set_title(bundle_name)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(figures_root / "comparison_contact_sheet.png", dpi=160, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
