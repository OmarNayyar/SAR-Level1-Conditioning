from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from skimage import io as skio


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bundles.bundle_a_classical import process_bundle_a_sample
from src.bundles.bundle_b_noiseaware import process_bundle_b_sample
from src.bundles.bundle_d_inverse_problem import process_bundle_d_sample
from src.datasets.mendeley_despeckling import MendeleyPair, discover_mendeley_pairs
from src.evaluation.denoising_metrics import compute_denoising_metrics, match_reference_shape, normalize_paired_images
from src.stage1.pipeline import LoadedSample, load_yaml


VARIANT_CONFIGS = {
    "bundle_a": "configs/bundle_a.yaml",
    "bundle_a_conservative": "configs/bundle_a_conservative.yaml",
    "bundle_b": "configs/bundles/profiles/bundle_b_balanced.yaml",
    "bundle_d": "configs/bundles/profiles/bundle_d_conservative.yaml",
}
VARIANT_PROCESSORS: dict[str, Callable[[LoadedSample, dict[str, Any]], Any]] = {
    "bundle_a": process_bundle_a_sample,
    "bundle_a_conservative": process_bundle_a_sample,
    "bundle_b": process_bundle_b_sample,
    "bundle_d": process_bundle_d_sample,
}
DEFAULT_VARIANTS = ["raw", "bundle_a", "bundle_a_conservative", "bundle_b", "bundle_d"]


def _parse_variants(text: str) -> list[str]:
    variants = [item.strip() for item in text.replace(" ", ",").split(",") if item.strip()]
    unknown = sorted(set(variants).difference(["raw", *VARIANT_CONFIGS]))
    if unknown:
        raise ValueError(f"Unsupported variants: {unknown}. Expected raw or one of {sorted(VARIANT_CONFIGS)}.")
    return variants


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_image(path: Path) -> np.ndarray:
    return np.asarray(skio.imread(path))


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array, dtype=np.float32))


def _save_preview(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(array, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)


def _sample(pair: MendeleyPair, noisy: np.ndarray) -> LoadedSample:
    return LoadedSample(
        dataset_name="mendeley",
        sample_id=pair.pair_id,
        split=pair.split,
        intensity_image=np.asarray(noisy, dtype=np.float32),
        display_image=np.asarray(noisy, dtype=np.float32),
        metadata={
            "pixel_domain": "intensity",
            "source_dataset": "mendeley_sar_despeckling",
            "paired_reference_available": True,
        },
        annotation=None,
        annotation_count=0,
        downstream_target=None,
        source_note=pair.noisy_path.as_posix(),
    )


def _load_or_run_variant(
    *,
    variant: str,
    pair: MendeleyPair,
    noisy: np.ndarray,
    reference: np.ndarray,
    output_root: Path,
    configs: dict[str, dict[str, Any]],
    resume: bool,
    force: bool,
) -> np.ndarray:
    output_path = output_root / "variants" / variant / f"{pair.pair_id}.npy"
    preview_path = output_root / "previews" / variant / f"{pair.pair_id}.png"
    if variant == "raw":
        candidate = noisy
    elif output_path.exists() and resume and not force:
        return match_reference_shape(np.load(output_path), reference)
    else:
        result = VARIANT_PROCESSORS[variant](_sample(pair, noisy), configs[variant])
        candidate = result.final_output
    candidate = match_reference_shape(candidate, reference)
    _save_array(output_path, candidate)
    _save_preview(preview_path, candidate)
    return candidate


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {"variant": variant, "sample_count": len(variant_rows)}
        for metric in ("mse", "nrmse", "psnr", "ssim", "edge_preservation_index"):
            values = [float(row[metric]) for row in variant_rows if row.get(metric) not in {"", None}]
            aggregate[f"mean_{metric}"] = float(np.mean(values)) if values else ""
            aggregate[f"median_{metric}"] = float(np.median(values)) if values else ""
        aggregate_rows.append(aggregate)
    return aggregate_rows, {"variants": aggregate_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate paired SAR denoising quality on local datasets.")
    parser.add_argument("--dataset", choices=["mendeley"], default="mendeley")
    parser.add_argument("--input-root", default="data/raw/Mendeley SAR dataset")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output-root", default="outputs/denoising_quality")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = (REPO_ROOT / args.input_root).resolve() if not Path(args.input_root).is_absolute() else Path(args.input_root)
    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    variants = _parse_variants(args.variants)
    pairs = discover_mendeley_pairs(input_root, split=args.split)
    selected_pairs = pairs[: max(args.max_samples, 0)]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "dataset": args.dataset,
                    "input_root": input_root.as_posix(),
                    "split": args.split,
                    "matched_pair_count": len(pairs),
                    "selected_pair_count": len(selected_pairs),
                    "variants": variants,
                    "output_root": output_root.as_posix(),
                    "sample_pair_ids": [pair.pair_id for pair in selected_pairs],
                },
                indent=2,
            )
        )
        return

    configs = {
        variant: load_yaml((REPO_ROOT / VARIANT_CONFIGS[variant]).resolve(), expected_kind="bundle")
        for variant in variants
        if variant != "raw"
    }
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for pair in selected_pairs:
        noisy_raw = _safe_image(pair.noisy_path)
        reference_raw = _safe_image(pair.reference_path)
        noisy, reference = normalize_paired_images(noisy_raw, reference_raw)
        pair_rows.append(pair.to_row())
        _save_array(output_root / "inputs" / "noisy" / f"{pair.pair_id}.npy", noisy)
        _save_array(output_root / "inputs" / "reference" / f"{pair.pair_id}.npy", reference)
        _save_preview(output_root / "previews" / "noisy" / f"{pair.pair_id}.png", noisy)
        _save_preview(output_root / "previews" / "reference" / f"{pair.pair_id}.png", reference)

        for variant in variants:
            candidate = _load_or_run_variant(
                variant=variant,
                pair=pair,
                noisy=noisy,
                reference=reference,
                output_root=output_root,
                configs=configs,
                resume=args.resume,
                force=args.force,
            )
            metrics = compute_denoising_metrics(candidate, reference)
            row = {
                "dataset": args.dataset,
                "split": args.split,
                "pair_id": pair.pair_id,
                "variant": variant,
                "noisy_path": pair.noisy_path.resolve().as_posix(),
                "reference_path": pair.reference_path.resolve().as_posix(),
                "candidate_path": (output_root / "variants" / variant / f"{pair.pair_id}.npy").resolve().as_posix(),
            }
            row.update(metrics.to_dict())
            rows.append(row)

    aggregate_rows, aggregate_payload = _aggregate(rows)
    _write_csv(output_root / "metrics" / "per_image_metrics.csv", rows)
    _write_csv(output_root / "metrics" / "aggregate_metrics.csv", aggregate_rows)
    _write_json(
        output_root / "metrics" / "aggregate_metrics.json",
        {
            "dataset": args.dataset,
            "input_root": input_root.as_posix(),
            "split": args.split,
            "matched_pair_count": len(pairs),
            "evaluated_pair_count": len(selected_pairs),
            "variants": variants,
            **aggregate_payload,
        },
    )
    _write_csv(output_root / "metrics" / "pair_manifest.csv", pair_rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "matched_pair_count": len(pairs),
                "evaluated_pair_count": len(selected_pairs),
                "variants": variants,
                "metrics_root": (output_root / "metrics").resolve().as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

