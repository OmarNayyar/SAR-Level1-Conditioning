from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.integrate import trapezoid

from src.datasets.common import write_csv, write_json
from src.stage1.downstream import ProxyEvaluation, annotation_to_mask
from src.stage1.pipeline import BundleProcessResult, LoadedSample
from src.stage1.viz.side_by_side import prepare_display_image


@dataclass(slots=True)
class RegionSelection:
    target_mask: np.ndarray | None
    background_mask: np.ndarray | None
    target_source: str
    background_source: str
    note: str


@dataclass(slots=True)
class DistributionFit:
    family: str
    parameters: dict[str, float]
    sample_count: int


def _safe_positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array) & (array > 0.0)]


def _fit_exponential(values: np.ndarray) -> DistributionFit | None:
    positive = _safe_positive(values)
    if positive.size < 8:
        return None
    scale = float(np.mean(positive))
    rate = float(1.0 / max(scale, 1e-12))
    return DistributionFit(
        family="exponential",
        parameters={"rate": rate, "scale": scale},
        sample_count=int(positive.size),
    )


def _fit_lognormal(values: np.ndarray) -> DistributionFit | None:
    positive = _safe_positive(values)
    if positive.size < 8:
        return None
    logs = np.log(positive)
    mu = float(np.mean(logs))
    sigma = float(np.std(logs))
    sigma = max(sigma, 1e-6)
    return DistributionFit(
        family="lognormal",
        parameters={"mu": mu, "sigma": sigma, "scale": float(np.exp(mu))},
        sample_count=int(positive.size),
    )


def _exponential_pdf(x_values: np.ndarray, fit: DistributionFit | None) -> np.ndarray:
    if fit is None:
        return np.zeros_like(x_values, dtype=np.float64)
    rate = fit.parameters["rate"]
    return rate * np.exp(-rate * x_values)


def _lognormal_pdf(x_values: np.ndarray, fit: DistributionFit | None) -> np.ndarray:
    if fit is None:
        return np.zeros_like(x_values, dtype=np.float64)
    mu = fit.parameters["mu"]
    sigma = fit.parameters["sigma"]
    safe_x = np.maximum(x_values, 1e-12)
    return (
        np.exp(-((np.log(safe_x) - mu) ** 2) / (2.0 * sigma**2))
        / (safe_x * sigma * math.sqrt(2.0 * math.pi))
    )


def _overlap_and_threshold(
    background_values: np.ndarray,
    target_values: np.ndarray,
    background_fit: DistributionFit | None,
    target_fit: DistributionFit | None,
) -> dict[str, float | None]:
    background_positive = _safe_positive(background_values)
    target_positive = _safe_positive(target_values)
    if background_positive.size < 8 or target_positive.size < 8 or background_fit is None or target_fit is None:
        return {
            "distribution_overlap": None,
            "distribution_separability": None,
            "theoretical_threshold": None,
            "empirical_threshold": None,
            "threshold_balanced_accuracy": None,
            "threshold_precision": None,
            "threshold_recall": None,
            "threshold_f1": None,
        }

    grid_min = float(max(min(np.min(background_positive), np.min(target_positive)), 1e-8))
    grid_max = float(max(np.max(background_positive), np.max(target_positive)))
    if grid_max <= grid_min:
        grid_max = grid_min * 1.1
    grid = np.geomspace(grid_min, grid_max, 512)
    background_pdf = _exponential_pdf(grid, background_fit)
    target_pdf = _lognormal_pdf(grid, target_fit)

    overlap = float(trapezoid(np.minimum(background_pdf, target_pdf), grid))
    overlap = min(max(overlap, 0.0), 1.0)
    separability = float(1.0 - overlap)
    theoretical_index = int(np.argmin(np.abs(background_pdf - target_pdf)))
    theoretical_threshold = float(grid[theoretical_index])

    empirical_values = np.concatenate([background_positive, target_positive])
    empirical_labels = np.concatenate(
        [np.zeros(background_positive.shape[0], dtype=np.int32), np.ones(target_positive.shape[0], dtype=np.int32)]
    )
    thresholds = np.quantile(empirical_values, np.linspace(0.05, 0.95, 31))
    best_payload: dict[str, float | None] = {
        "empirical_threshold": None,
        "threshold_balanced_accuracy": None,
        "threshold_precision": None,
        "threshold_recall": None,
        "threshold_f1": None,
    }
    best_score = -1.0
    for threshold in thresholds:
        predictions = empirical_values >= float(threshold)
        true_positive = float(np.sum((predictions == 1) & (empirical_labels == 1)))
        true_negative = float(np.sum((predictions == 0) & (empirical_labels == 0)))
        false_positive = float(np.sum((predictions == 1) & (empirical_labels == 0)))
        false_negative = float(np.sum((predictions == 0) & (empirical_labels == 1)))
        tpr = true_positive / max(true_positive + false_negative, 1.0)
        tnr = true_negative / max(true_negative + false_positive, 1.0)
        balanced_accuracy = 0.5 * (tpr + tnr)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = tpr
        f1 = 0.0 if (precision + recall) <= 0 else 2.0 * precision * recall / (precision + recall)
        if balanced_accuracy > best_score:
            best_score = float(balanced_accuracy)
            best_payload = {
                "empirical_threshold": float(threshold),
                "threshold_balanced_accuracy": float(balanced_accuracy),
                "threshold_precision": float(precision),
                "threshold_recall": float(recall),
                "threshold_f1": float(f1),
            }

    return {
        "distribution_overlap": overlap,
        "distribution_separability": separability,
        "theoretical_threshold": theoretical_threshold,
        **best_payload,
    }


def _save_region_overlay(
    path: Path,
    display_image: np.ndarray,
    target_mask: np.ndarray | None,
    background_mask: np.ndarray | None,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(prepare_display_image(display_image), cmap="gray")
    if background_mask is not None and np.any(background_mask):
        background_alpha = np.zeros((*background_mask.shape, 4), dtype=np.float32)
        background_alpha[background_mask.astype(bool)] = np.array([0.2, 0.6, 1.0, 0.22], dtype=np.float32)
        axis.imshow(background_alpha)
    if target_mask is not None and np.any(target_mask):
        target_alpha = np.zeros((*target_mask.shape, 4), dtype=np.float32)
        target_alpha[target_mask.astype(bool)] = np.array([1.0, 0.2, 0.2, 0.35], dtype=np.float32)
        axis.imshow(target_alpha)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_histogram_plot(
    path: Path,
    *,
    background_values: np.ndarray,
    target_values: np.ndarray,
    background_fit: DistributionFit | None,
    target_fit: DistributionFit | None,
    title: str,
    log_x: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    background_positive = _safe_positive(background_values)
    target_positive = _safe_positive(target_values)
    if background_positive.size == 0 or target_positive.size == 0:
        return

    lower = float(max(min(np.min(background_positive), np.min(target_positive)), 1e-8))
    upper = float(max(np.max(background_positive), np.max(target_positive)))
    if upper <= lower:
        upper = lower * 1.1

    figure, axis = plt.subplots(figsize=(7, 4.5))
    if log_x:
        bins = np.geomspace(lower, upper, 40)
        axis.set_xscale("log")
    else:
        bins = np.linspace(lower, upper, 40)
    axis.hist(background_positive, bins=bins, density=True, alpha=0.45, label="background", color="#3b82f6")
    axis.hist(target_positive, bins=bins, density=True, alpha=0.45, label="ship-like target", color="#ef4444")

    grid = np.geomspace(lower, upper, 400) if log_x else np.linspace(lower, upper, 400)
    axis.plot(grid, _exponential_pdf(grid, background_fit), color="#1d4ed8", lw=2.0, label="exp fit")
    axis.plot(grid, _lognormal_pdf(grid, target_fit), color="#b91c1c", lw=2.0, label="log-normal fit")
    axis.set_title(title)
    axis.set_xlabel("Intensity")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def select_target_background_regions(
    *,
    sample: LoadedSample,
    proxy_evaluation: ProxyEvaluation,
    reference_image: np.ndarray,
    dilation_radius: int = 4,
    background_lower_quantile: float = 0.05,
    background_upper_quantile: float = 0.5,
) -> RegionSelection:
    target_mask = None
    target_source = "missing"
    note = ""

    if sample.downstream_target is not None:
        target_mask = np.asarray(sample.downstream_target == 1, dtype=bool)
        target_source = "label_mask"
        note = "Used the real segmentation label mask for the target region."
    else:
        annotation_mask = annotation_to_mask(sample.annotation, reference_image.shape)
        if annotation_mask is not None:
            target_mask = annotation_mask.astype(bool)
            target_source = "annotation_mask"
            note = "Used annotation-derived ship region(s) for the target mask."
        elif proxy_evaluation.predicted_mask is not None and np.any(proxy_evaluation.predicted_mask):
            target_mask = np.asarray(proxy_evaluation.predicted_mask, dtype=bool)
            target_source = "proxy_detection_mask"
            note = "Used the proxy detection mask because no real ship label was available."

    if target_mask is None or not np.any(target_mask):
        return RegionSelection(
            target_mask=None,
            background_mask=None,
            target_source=target_source,
            background_source="missing",
            note="No usable target region could be selected for statistical fitting.",
        )

    exclusion_mask = binary_dilation(target_mask, iterations=max(int(dilation_radius), 0))
    candidate_background = ~exclusion_mask
    candidate_values = reference_image[candidate_background]
    positive_candidate = _safe_positive(candidate_values)
    if positive_candidate.size < 16:
        return RegionSelection(
            target_mask=target_mask,
            background_mask=None,
            target_source=target_source,
            background_source="missing",
            note="A target region was available, but too few background pixels remained after exclusion.",
        )

    lower_value = float(np.quantile(positive_candidate, background_lower_quantile))
    upper_value = float(np.quantile(positive_candidate, background_upper_quantile))
    background_mask = candidate_background & (reference_image >= lower_value) & (reference_image <= upper_value)
    background_source = "quiet_background_subset"
    note = (
        f"{note} Selected background pixels from the non-target quiet-intensity subset "
        f"between the {background_lower_quantile:.2f} and {background_upper_quantile:.2f} quantiles."
    ).strip()
    return RegionSelection(
        target_mask=target_mask.astype(bool),
        background_mask=background_mask.astype(bool),
        target_source=target_source,
        background_source=background_source,
        note=note,
    )


class IntensityStatisticsAnalyzer:
    def __init__(self, output_root: Path, config: dict[str, Any] | None = None) -> None:
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.plots_root = self.output_root / "plots"
        self.region_root = self.output_root / "regions"
        self.plots_root.mkdir(parents=True, exist_ok=True)
        self.region_root.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.rows: list[dict[str, Any]] = []
        self._pooled_background_before: list[np.ndarray] = []
        self._pooled_target_before: list[np.ndarray] = []
        self._pooled_background_after: list[np.ndarray] = []
        self._pooled_target_after: list[np.ndarray] = []

    def _subsample(self, values: np.ndarray) -> np.ndarray:
        max_points = int(self.config.get("max_points_per_sample", 15000))
        positive = _safe_positive(values)
        if positive.size <= max_points:
            return positive
        rng = np.random.default_rng(7)
        indices = rng.choice(positive.size, size=max_points, replace=False)
        return positive[indices]

    def process_sample(
        self,
        *,
        sample: LoadedSample,
        process_result: BundleProcessResult,
        proxy_evaluation: ProxyEvaluation,
        metrics_row: dict[str, Any],
    ) -> dict[str, Any] | None:
        region_selection = select_target_background_regions(
            sample=sample,
            proxy_evaluation=proxy_evaluation,
            reference_image=process_result.final_output,
            dilation_radius=int(self.config.get("dilation_radius", 4)),
            background_lower_quantile=float(self.config.get("background_lower_quantile", 0.05)),
            background_upper_quantile=float(self.config.get("background_upper_quantile", 0.5)),
        )
        target_mask = region_selection.target_mask
        background_mask = region_selection.background_mask
        if target_mask is None or background_mask is None or not np.any(target_mask) or not np.any(background_mask):
            row = {
                "dataset": sample.dataset_name,
                "sample_id": sample.sample_id,
                "split": sample.split,
                "statistics_status": "skipped",
                "statistics_note": region_selection.note,
                "target_mask_source": region_selection.target_source,
                "background_mask_source": region_selection.background_source,
            }
            self.rows.append(row)
            return {"metrics": row}

        background_before = sample.intensity_image[background_mask]
        target_before = sample.intensity_image[target_mask]
        background_after = process_result.final_output[background_mask]
        target_after = process_result.final_output[target_mask]

        fit_background_before = _fit_exponential(background_before)
        fit_target_before = _fit_lognormal(target_before)
        fit_background_after = _fit_exponential(background_after)
        fit_target_after = _fit_lognormal(target_after)
        separability_before = _overlap_and_threshold(
            background_before,
            target_before,
            fit_background_before,
            fit_target_before,
        )
        separability_after = _overlap_and_threshold(
            background_after,
            target_after,
            fit_background_after,
            fit_target_after,
        )

        sample_prefix = f"{sample.dataset_name}_{sample.sample_id}"
        _save_region_overlay(
            self.region_root / f"{sample_prefix}_regions.png",
            sample.display_image,
            target_mask,
            background_mask,
            title=f"{sample.dataset_name} | {sample.sample_id} regions",
        )
        _save_histogram_plot(
            self.plots_root / f"{sample_prefix}_before_linear.png",
            background_values=background_before,
            target_values=target_before,
            background_fit=fit_background_before,
            target_fit=fit_target_before,
            title=f"{sample.sample_id} | input intensity fit",
            log_x=False,
        )
        _save_histogram_plot(
            self.plots_root / f"{sample_prefix}_after_linear.png",
            background_values=background_after,
            target_values=target_after,
            background_fit=fit_background_after,
            target_fit=fit_target_after,
            title=f"{sample.sample_id} | Bundle A output fit",
            log_x=False,
        )
        _save_histogram_plot(
            self.plots_root / f"{sample_prefix}_after_logx.png",
            background_values=background_after,
            target_values=target_after,
            background_fit=fit_background_after,
            target_fit=fit_target_after,
            title=f"{sample.sample_id} | Bundle A output fit (log-x)",
            log_x=True,
        )

        self._pooled_background_before.append(self._subsample(background_before))
        self._pooled_target_before.append(self._subsample(target_before))
        self._pooled_background_after.append(self._subsample(background_after))
        self._pooled_target_after.append(self._subsample(target_after))

        row = {
            "dataset": sample.dataset_name,
            "sample_id": sample.sample_id,
            "split": sample.split,
            "statistics_status": "ok",
            "statistics_note": region_selection.note,
            "target_mask_source": region_selection.target_source,
            "background_mask_source": region_selection.background_source,
            "target_pixel_count": int(np.sum(target_mask)),
            "background_pixel_count": int(np.sum(background_mask)),
            "background_exp_rate_before": fit_background_before.parameters["rate"] if fit_background_before else None,
            "background_exp_scale_before": fit_background_before.parameters["scale"] if fit_background_before else None,
            "target_lognorm_mu_before": fit_target_before.parameters["mu"] if fit_target_before else None,
            "target_lognorm_sigma_before": fit_target_before.parameters["sigma"] if fit_target_before else None,
            "distribution_overlap_before": separability_before["distribution_overlap"],
            "distribution_separability_before": separability_before["distribution_separability"],
            "threshold_balanced_accuracy_before": separability_before["threshold_balanced_accuracy"],
            "threshold_f1_before": separability_before["threshold_f1"],
            "theoretical_threshold_before": separability_before["theoretical_threshold"],
            "empirical_threshold_before": separability_before["empirical_threshold"],
            "background_exp_rate_after": fit_background_after.parameters["rate"] if fit_background_after else None,
            "background_exp_scale_after": fit_background_after.parameters["scale"] if fit_background_after else None,
            "target_lognorm_mu_after": fit_target_after.parameters["mu"] if fit_target_after else None,
            "target_lognorm_sigma_after": fit_target_after.parameters["sigma"] if fit_target_after else None,
            "distribution_overlap_after": separability_after["distribution_overlap"],
            "distribution_separability_after": separability_after["distribution_separability"],
            "threshold_balanced_accuracy_after": separability_after["threshold_balanced_accuracy"],
            "threshold_f1_after": separability_after["threshold_f1"],
            "theoretical_threshold_after": separability_after["theoretical_threshold"],
            "empirical_threshold_after": separability_after["empirical_threshold"],
        }
        self.rows.append(row)
        return {"metrics": row}

    def finalize(self) -> dict[str, Any]:
        write_csv(self.output_root / "per_sample_statistics.csv", self.rows)
        write_json(self.output_root / "per_sample_statistics.json", {"samples": self.rows})
        valid_rows = [row for row in self.rows if row.get("statistics_status") == "ok"]
        summary: dict[str, Any] = {
            "sample_count": len(self.rows),
            "valid_fit_count": len(valid_rows),
            "output_root": self.output_root.as_posix(),
        }
        if valid_rows:
            numeric_keys = [
                "distribution_separability_before",
                "distribution_separability_after",
                "threshold_balanced_accuracy_before",
                "threshold_balanced_accuracy_after",
                "threshold_f1_before",
                "threshold_f1_after",
            ]
            summary["mean_metrics"] = {
                key: float(np.mean([row[key] for row in valid_rows if row.get(key) is not None]))
                for key in numeric_keys
                if any(row.get(key) is not None for row in valid_rows)
            }

        pooled = self._build_pooled_summary()
        summary["pooled"] = pooled
        write_json(self.output_root / "summary.json", summary)
        write_csv(self.output_root / "summary.csv", [summary.get("mean_metrics", {})] if summary.get("mean_metrics") else [])
        return summary

    def _build_pooled_summary(self) -> dict[str, Any]:
        if not self._pooled_background_after or not self._pooled_target_after:
            return {"status": "no_valid_samples"}

        background_before = np.concatenate(self._pooled_background_before) if self._pooled_background_before else np.array([], dtype=np.float64)
        target_before = np.concatenate(self._pooled_target_before) if self._pooled_target_before else np.array([], dtype=np.float64)
        background_after = np.concatenate(self._pooled_background_after) if self._pooled_background_after else np.array([], dtype=np.float64)
        target_after = np.concatenate(self._pooled_target_after) if self._pooled_target_after else np.array([], dtype=np.float64)

        fit_background_before = _fit_exponential(background_before)
        fit_target_before = _fit_lognormal(target_before)
        fit_background_after = _fit_exponential(background_after)
        fit_target_after = _fit_lognormal(target_after)
        summary_before = _overlap_and_threshold(background_before, target_before, fit_background_before, fit_target_before)
        summary_after = _overlap_and_threshold(background_after, target_after, fit_background_after, fit_target_after)

        _save_histogram_plot(
            self.output_root / "pooled_before_linear.png",
            background_values=background_before,
            target_values=target_before,
            background_fit=fit_background_before,
            target_fit=fit_target_before,
            title="Pooled input intensity fit",
            log_x=False,
        )
        _save_histogram_plot(
            self.output_root / "pooled_after_linear.png",
            background_values=background_after,
            target_values=target_after,
            background_fit=fit_background_after,
            target_fit=fit_target_after,
            title="Pooled Bundle A output fit",
            log_x=False,
        )
        _save_histogram_plot(
            self.output_root / "pooled_after_logx.png",
            background_values=background_after,
            target_values=target_after,
            background_fit=fit_background_after,
            target_fit=fit_target_after,
            title="Pooled Bundle A output fit (log-x)",
            log_x=True,
        )

        return {
            "status": "ok",
            "target_pixel_count": int(target_after.size),
            "background_pixel_count": int(background_after.size),
            "distribution_separability_before": summary_before["distribution_separability"],
            "distribution_separability_after": summary_after["distribution_separability"],
            "threshold_balanced_accuracy_before": summary_before["threshold_balanced_accuracy"],
            "threshold_balanced_accuracy_after": summary_after["threshold_balanced_accuracy"],
            "threshold_f1_before": summary_before["threshold_f1"],
            "threshold_f1_after": summary_after["threshold_f1"],
        }
