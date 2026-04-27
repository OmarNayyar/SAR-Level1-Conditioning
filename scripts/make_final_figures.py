from __future__ import annotations

"""Build final report-ready figures and compact evidence tables.

The script is intentionally read-only with respect to experiments: it consumes
cached denoising metrics and cached detector final-sweep summaries, then writes
small report artifacts. It must not train, download, or regenerate bundles.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.common import write_csv, write_json


VARIANT_ORDER = ["raw", "bundle_a", "bundle_a_conservative", "bundle_b", "bundle_d"]
VARIANT_LABELS = {
    "raw": "Raw",
    "bundle_a": "Bundle A",
    "bundle_a_conservative": "A conservative",
    "bundle_b": "Bundle B",
    "bundle_d": "Bundle D",
}
VARIANT_COLORS = {
    "raw": "#4b5563",
    "bundle_a": "#2563eb",
    "bundle_a_conservative": "#60a5fa",
    "bundle_b": "#059669",
    "bundle_d": "#d97706",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def _round(value: Any, digits: int = 4) -> float | str:
    if value is None or value == "":
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def _denoising_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_variant = {str(row["variant"]): dict(row) for row in aggregate.get("variants", [])}
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        if variant not in rows_by_variant:
            continue
        row = rows_by_variant[variant]
        rows.append(
            {
                "track": "paired_denoising",
                "dataset": "Mendeley SAR despeckling val",
                "variant": variant,
                "label": VARIANT_LABELS.get(variant, variant),
                "sample_count": int(row.get("sample_count", 0) or 0),
                "mean_psnr": _round(row.get("mean_psnr")),
                "mean_ssim": _round(row.get("mean_ssim")),
                "mean_mse": _round(row.get("mean_mse"), 6),
                "mean_nrmse": _round(row.get("mean_nrmse")),
                "mean_edge_preservation_index": _round(row.get("mean_edge_preservation_index")),
            }
        )
    return rows


def _detector_rows(final_sweep: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_summary in final_sweep.get("dataset_summaries", []) or []:
        dataset = str(dataset_summary.get("dataset", "")).upper()
        for row in dataset_summary.get("rows", []) or []:
            variant = str(row.get("variant", ""))
            metrics = row.get("metrics", {}) or {}
            split_counts = row.get("split_counts", {}) or {}
            rows.append(
                {
                    "track": "detector_compatibility",
                    "dataset": dataset,
                    "variant": variant,
                    "label": VARIANT_LABELS.get(variant, variant),
                    "train_count": split_counts.get("train", ""),
                    "val_count": split_counts.get("val", ""),
                    "test_count": split_counts.get("test", ""),
                    "map": _round(metrics.get("map")),
                    "map50": _round(metrics.get("map50")),
                    "map75": _round(metrics.get("map75")),
                    "precision": _round(metrics.get("precision")),
                    "recall": _round(metrics.get("recall")),
                    "f1": _round(metrics.get("f1")),
                }
            )
    return sorted(rows, key=lambda item: (str(item["dataset"]), VARIANT_ORDER.index(item["variant"]) if item["variant"] in VARIANT_ORDER else 99))


def _bar_chart(
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    lower_is_better: bool = False,
) -> None:
    plot_rows = [row for row in rows if row.get(metric) != ""]
    labels = [str(row["label"]) for row in plot_rows]
    values = [float(row[metric]) for row in plot_rows]
    colors = [VARIANT_COLORS.get(str(row["variant"]), "#6b7280") for row in plot_rows]
    figure, axis = plt.subplots(figsize=(8, 4.6))
    bars = axis.bar(labels, values, color=colors)
    axis.set_title(title, fontsize=13, weight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=20)
    best_index = values.index(min(values) if lower_is_better else max(values)) if values else -1
    for index, (bar, value) in enumerate(zip(bars, values)):
        label = f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=9)
        if index == best_index:
            bar.set_edgecolor("#111827")
            bar.set_linewidth(2.0)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _detector_map_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    datasets = sorted({str(row["dataset"]) for row in rows})
    variants = [variant for variant in VARIANT_ORDER if any(row["variant"] == variant for row in rows)]
    width = 0.14
    x_positions = list(range(len(datasets)))
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for offset, variant in enumerate(variants):
        values = []
        for dataset in datasets:
            match = next((row for row in rows if row["dataset"] == dataset and row["variant"] == variant), None)
            values.append(float(match["map"]) if match and match.get("map") != "" else 0.0)
        shift = (offset - (len(variants) - 1) / 2) * width
        bars = axis.bar(
            [position + shift for position in x_positions],
            values,
            width=width,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, "#6b7280"),
        )
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axis.set_title("Detector Compatibility: YOLO mAP50-95", fontsize=13, weight="bold")
    axis.set_ylabel("mAP50-95")
    axis.set_xticks(x_positions, datasets)
    axis.set_ylim(0, max([float(row["map"]) for row in rows if row.get("map") != ""] + [0.1]) * 1.18)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, fontsize=8, frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _write_results_summary(docs_root: Path, denoise_rows: list[dict[str, Any]], detector_rows: list[dict[str, Any]]) -> None:
    denoise_by_variant = {row["variant"]: row for row in denoise_rows}
    detector_by_dataset = {}
    for row in detector_rows:
        detector_by_dataset.setdefault(row["dataset"], {})[row["variant"]] = row
    lines = [
        "# Results Summary",
        "",
        "This summary separates denoising quality from detector compatibility. PSNR/SSIM/MSE/NRMSE evaluate paired denoising against the Mendeley reference target. YOLO mAP evaluates whether a lightweight ship detector benefits from the conditioned images in the current SSDD/HRSID setup.",
        "",
        "## Paired Denoising: Mendeley Validation Split",
        "",
        "Bundle B is currently strongest for paired denoising quality on the local Mendeley validation split. Bundle D is also useful as a structure-preserving candidate because it improves SSIM and edge preservation versus raw noisy input. Raw here means the noisy input compared directly against the pseudo-clean reference.",
        "",
        _markdown_table(
            denoise_rows,
            ["label", "sample_count", "mean_psnr", "mean_ssim", "mean_mse", "mean_nrmse", "mean_edge_preservation_index"],
        ),
        "",
        "## Detector Compatibility: SSDD/HRSID Final Sweep",
        "",
        "Raw imagery remains strongest for the current lightweight YOLO detector setup on both SSDD and HRSID. This is downstream compatibility evidence, not proof that denoising is useless. It likely reflects detector/data-distribution tuning and the fact that conditioning can suppress texture or edge cues used by this detector.",
        "",
        _markdown_table(
            detector_rows,
            ["dataset", "label", "map", "map50", "map75", "precision", "recall", "f1"],
        ),
        "",
        "## Practical Conclusion",
        "",
        "- Bundle B should be prioritized for paired/intensity-domain denoising quality.",
        "- Bundle D should remain a structure-preserving candidate.",
        "- Bundle A remains the interpretable conditioning and screening family, not the best operational detector baseline.",
        "- Raw remains the detector baseline only for the current YOLO setup until a conditioned variant beats raw on the target detector.",
        "- Representative operational SAR samples, product-level clarification, metadata/noise vectors, SLC availability, and downstream task details are needed before deployment-style recommendations.",
        "",
    ]
    (docs_root / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _write_public_summary(results_root: Path, denoise_rows: list[dict[str, Any]], detector_rows: list[dict[str, Any]]) -> None:
    root = results_root / "public"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "surface": "public",
        "title": "SAR Stage-1 Conditioning: Denoising and Validation for Maritime SAR Imagery",
        "positioning": "Public-data validation repo for SAR Stage-1 conditioning, paired denoising metrics, and detector compatibility checks.",
        "denoising_result": "Bundle B improved paired PSNR/SSIM/MSE versus raw noisy input on the Mendeley validation split.",
        "detector_result": "Raw imagery remained strongest for the current lightweight YOLO detector setup on SSDD and HRSID.",
        "caveat": "Detector compatibility is not the same as denoising quality; no claim is made for non-public or operational data without representative validation.",
        "denoising_rows": denoise_rows,
        "detector_rows": detector_rows,
    }
    write_json(root / "project_summary.json", payload)
    write_csv(root / "final_denoising_metrics.csv", denoise_rows)
    write_csv(root / "final_detector_metrics.csv", detector_rows)
    write_csv(root / "bundle_matrix.csv", _bundle_matrix_rows())
    lines = [
        "# SAR Stage-1 Conditioning Public Summary",
        "",
        "**Public-safe conclusion:** Bundle B is currently strongest on paired public denoising metrics, while raw imagery remains strongest for the current lightweight YOLO detector compatibility run.",
        "",
        "This is a public-data validation repo, not a claim that conditioning universally improves downstream maritime SAR detection.",
        "",
        "## Denoising Evidence",
        "",
        _markdown_table(denoise_rows, ["label", "sample_count", "mean_psnr", "mean_ssim", "mean_mse", "mean_edge_preservation_index"]),
        "",
        "## Detector Compatibility Evidence",
        "",
        _markdown_table(detector_rows, ["dataset", "label", "map", "map50", "precision", "recall", "f1"]),
        "",
        "## Interpretation",
        "",
        "- Paired denoising and detector compatibility answer different questions.",
        "- Bundle B improved paired denoising quality on the Mendeley validation split.",
        "- Raw remained best for this YOLO detector setup, so it stays the detector baseline for this experiment.",
        "- No claim is made for non-public operational data without representative validation.",
        "",
    ]
    (root / "project_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _copy_public_figure_pack(output_root: Path, results_root: Path) -> None:
    public_root = results_root / "public"
    figures_root = public_root / "figures"
    tables_root = public_root / "tables"
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    for filename in [
        "denoising_psnr.png",
        "denoising_ssim.png",
        "denoising_mse.png",
        "denoising_edge_preservation.png",
        "detector_map_comparison.png",
    ]:
        source = output_root / filename
        if source.exists():
            shutil.copy2(source, figures_root / filename)
    for filename in [
        "final_denoising_metrics.csv",
        "final_detector_metrics.csv",
        "final_evidence_summary.json",
        "final_evidence_table.csv",
        "final_evidence_table.md",
    ]:
        source = output_root / filename
        if source.exists():
            shutil.copy2(source, tables_root / filename)


def _write_private_summary(results_root: Path, denoise_rows: list[dict[str, Any]], detector_rows: list[dict[str, Any]]) -> None:
    handoff_org = os.environ.get("SAR_HANDOFF_ORG", "private team")
    root = results_root / "handoff"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "surface": "private",
        "current_recommendation": f"Prioritize representative {handoff_org} sample validation before adopting any conditioning route. Use raw as the detector baseline, Bundle B as the primary paired/intensity-domain denoising candidate, Bundle D as a structure-preserving candidate, and Bundle A as the interpretable screening family.",
        "required_inputs": [
            "representative raw/noisy samples",
            "product level and pixel domain",
            "noise vectors / calibration metadata",
            "SLC availability",
            "downstream detector/task details",
        ],
        "denoising_rows": denoise_rows,
        "detector_rows": detector_rows,
        "bundle_matrix": _bundle_matrix_rows(),
    }
    write_json(root / "project_recommendations.json", payload)
    write_csv(root / "final_denoising_metrics.csv", denoise_rows)
    write_csv(root / "final_detector_metrics.csv", detector_rows)
    write_csv(root / "bundle_matrix.csv", _bundle_matrix_rows())
    lines = [
        "# SAR Stage-1 Conditioning Handoff Recommendation",
        "",
        f"**Current recommendation:** Run a representative {handoff_org} validation slice before adopting any conditioning route. Raw remains the detector baseline for the current YOLO evidence; Bundle B is the first paired/intensity-domain denoising candidate; Bundle D stays as a structure-preserving candidate; Bundle A remains the interpretable screening family.",
        "",
        "## Evidence Now",
        "",
        _markdown_table(denoise_rows, ["label", "sample_count", "mean_psnr", "mean_ssim", "mean_mse", "mean_edge_preservation_index"]),
        "",
        _markdown_table(detector_rows, ["dataset", "label", "map", "map50", "precision", "recall", "f1"]),
        "",
        f"## Ask {handoff_org} For",
        "",
        "- Representative raw/noisy products.",
        "- Product level and pixel-domain clarification: GRD, SLC, detected intensity, amplitude, or log-intensity chips.",
        "- Noise vectors, calibration metadata, NESZ-style metadata, or equivalent.",
        "- SLC availability for the Bundle C / MERLIN-style path.",
        "- Downstream detector/task details, target class, annotation format, metric priority, and preprocessing latency constraints.",
        "",
        "## Candidate Routing",
        "",
        "- Bundle B: prioritize for paired/intensity-domain denoising validation.",
        "- Bundle D: keep as a structure-preserving candidate.",
        "- Bundle A: keep as interpretable screening and ablation baseline.",
        "- Bundle C/SLC: future work pending genuine SLC access.",
        "",
    ]
    (root / "project_recommendations.md").write_text("\n".join(lines), encoding="utf-8")


def _bundle_matrix_rows() -> list[dict[str, str]]:
    return [
        {
            "bundle": "A",
            "role": "interpretable conditioning/screening family",
            "domain": "intensity / power",
            "current_reading": "useful for transparent ablations; not the best detector baseline",
        },
        {
            "bundle": "B",
            "role": "primary paired denoising candidate",
            "domain": "log-intensity / structured cleanup",
            "current_reading": "strongest paired Mendeley PSNR/SSIM/MSE result",
        },
        {
            "bundle": "C",
            "role": "future complex-domain candidate",
            "domain": "complex SLC preferred",
            "current_reading": "defer until genuine SLC data are available",
        },
        {
            "bundle": "D",
            "role": "structure-preserving candidate",
            "domain": "intensity / log-intensity",
            "current_reading": "improves SSIM and edge preservation versus raw on paired validation",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final figures, tables, and result summaries from cached artifacts.")
    parser.add_argument("--denoising-root", default="outputs/denoising_quality", help="Root containing denoising quality metrics.")
    parser.add_argument("--final-sweep-summary", default="outputs/final_sweep/final_sweep_summary.json")
    parser.add_argument("--output-root", default="outputs/final_figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    denoising_root = (REPO_ROOT / args.denoising_root).resolve()
    final_sweep_summary = (REPO_ROOT / args.final_sweep_summary).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    results_root = REPO_ROOT / "results"
    docs_root = REPO_ROOT / "docs"

    denoise_payload = _load_json(denoising_root / "metrics" / "aggregate_metrics.json")
    detector_payload = _load_json(final_sweep_summary)
    denoise_rows = _denoising_rows(denoise_payload)
    detector_rows = _detector_rows(detector_payload)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "final_denoising_metrics.csv", denoise_rows)
    write_csv(output_root / "final_detector_metrics.csv", detector_rows)
    write_json(
        output_root / "final_evidence_summary.json",
        {
            "denoising": denoise_rows,
            "detector": detector_rows,
            "conclusion": {
                "paired_denoising": "Bundle B is strongest on paired Mendeley validation metrics.",
                "detector": "Raw is strongest for the current lightweight YOLO detector setup.",
                "interpretation": "Detector compatibility and denoising quality are separate evidence tracks.",
            },
        },
    )
    evidence_rows = denoise_rows + detector_rows
    write_csv(output_root / "final_evidence_table.csv", evidence_rows)
    (output_root / "final_evidence_table.md").write_text(
        "# Final Evidence Table\n\n"
        + _markdown_table(evidence_rows, sorted({key for row in evidence_rows for key in row.keys()}))
        + "\n",
        encoding="utf-8",
    )

    _bar_chart(denoise_rows, "mean_psnr", "Paired Denoising: PSNR", "PSNR (dB, higher is better)", output_root / "denoising_psnr.png")
    _bar_chart(denoise_rows, "mean_ssim", "Paired Denoising: SSIM", "SSIM (higher is better)", output_root / "denoising_ssim.png")
    _bar_chart(
        denoise_rows,
        "mean_mse",
        "Paired Denoising: MSE",
        "MSE (lower is better)",
        output_root / "denoising_mse.png",
        lower_is_better=True,
    )
    _bar_chart(
        denoise_rows,
        "mean_edge_preservation_index",
        "Paired Denoising: Edge Preservation",
        "Edge preservation index",
        output_root / "denoising_edge_preservation.png",
    )
    _detector_map_chart(detector_rows, output_root / "detector_map_comparison.png")

    _write_results_summary(docs_root, denoise_rows, detector_rows)
    _write_public_summary(results_root, denoise_rows, detector_rows)
    _write_private_summary(results_root, denoise_rows, detector_rows)
    _copy_public_figure_pack(output_root, results_root)

    print(
        json.dumps(
            {
                "output_root": output_root.as_posix(),
                "results_public": (results_root / "public").as_posix(),
                "results_handoff": (results_root / "handoff").as_posix(),
                "docs_results_summary": (docs_root / "RESULTS_SUMMARY.md").as_posix(),
                "denoising_rows": len(denoise_rows),
                "detector_rows": len(detector_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
