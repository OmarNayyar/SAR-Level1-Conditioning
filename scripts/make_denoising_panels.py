from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PANEL_VARIANTS = [
    ("noisy", "Noisy input"),
    ("reference", "Reference / pseudo-clean target"),
    ("bundle_a", "Bundle A"),
    ("bundle_a_conservative", "Bundle A conservative"),
    ("bundle_b", "Bundle B"),
    ("bundle_d", "Bundle D"),
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["pair_id"], row["variant"]): row for row in rows}


def _load_image(output_root: Path, pair_id: str, variant: str) -> np.ndarray | None:
    if variant == "noisy":
        path = output_root / "inputs" / "noisy" / f"{pair_id}.npy"
    elif variant == "reference":
        path = output_root / "inputs" / "reference" / f"{pair_id}.npy"
    else:
        path = output_root / "variants" / variant / f"{pair_id}.npy"
    if not path.exists():
        return None
    return np.clip(np.load(path), 0.0, 1.0)


def _subtitle(metrics: dict[tuple[str, str], dict[str, str]], pair_id: str, variant: str) -> str:
    if variant in {"noisy", "reference"}:
        return ""
    row = metrics.get((pair_id, variant), {})
    psnr = row.get("psnr", "")
    ssim = row.get("ssim", "")
    if not psnr or not ssim:
        return "PSNR n/a | SSIM n/a"
    return f"PSNR {float(psnr):.2f} | SSIM {float(ssim):.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create meeting-ready denoising evidence panels from cached metrics.")
    parser.add_argument("--output-root", default="outputs/denoising_quality")
    parser.add_argument("--max-panels", type=int, default=5)
    parser.add_argument("--panels-root", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    metrics_path = output_root / "metrics" / "per_image_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing denoising metrics CSV: {metrics_path.as_posix()}")
    rows = _read_rows(metrics_path)
    metrics = _metric_lookup(rows)
    pair_ids = sorted({row["pair_id"] for row in rows})[: max(args.max_panels, 0)]
    panels_root = (
        (REPO_ROOT / args.panels_root).resolve()
        if args.panels_root and not Path(args.panels_root).is_absolute()
        else Path(args.panels_root).resolve()
        if args.panels_root
        else output_root / "panels"
    )
    panels_root.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for pair_id in pair_ids:
        fig, axes = plt.subplots(1, len(PANEL_VARIANTS), figsize=(18, 4.2), constrained_layout=True)
        fig.suptitle(pair_id, fontsize=12)
        for axis, (variant, title) in zip(axes, PANEL_VARIANTS):
            image = _load_image(output_root, pair_id, variant)
            axis.axis("off")
            axis.set_title(title, fontsize=10)
            if image is None:
                axis.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)
                continue
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            subtitle = _subtitle(metrics, pair_id, variant)
            if subtitle:
                axis.text(
                    0.5,
                    -0.08,
                    subtitle,
                    ha="center",
                    va="top",
                    transform=axis.transAxes,
                    fontsize=9,
                )
        output_path = panels_root / f"{pair_id}_denoising_panel.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(output_path.resolve().as_posix())
    print("\n".join(written))


if __name__ == "__main__":
    main()

