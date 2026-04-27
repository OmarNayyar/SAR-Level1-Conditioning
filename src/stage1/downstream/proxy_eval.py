from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.draw import polygon

from src.stage1.metrics.segmentation_miou import compute_segmentation_miou


@dataclass(slots=True)
class ProxyEvaluation:
    metrics: dict[str, Any]
    downstream_row: dict[str, Any]
    predicted_mask: np.ndarray
    target_mask: np.ndarray | None


def _clip_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    height, width = shape
    x0 = max(0, min(width, int(np.floor(x_min))))
    y0 = max(0, min(height, int(np.floor(y_min))))
    x1 = max(0, min(width, int(np.ceil(x_max))))
    y1 = max(0, min(height, int(np.ceil(y_max))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _apply_bbox_mask(mask: np.ndarray, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
    clipped = _clip_bbox(x_min, y_min, x_max, y_max, mask.shape)
    if clipped is None:
        return
    x0, y0, x1, y1 = clipped
    mask[y0:y1, x0:x1] = True


def _apply_polygon_mask(mask: np.ndarray, xs: list[float], ys: list[float]) -> None:
    if len(xs) < 3 or len(ys) < 3:
        return
    rr, cc = polygon(np.asarray(ys, dtype=np.float32), np.asarray(xs, dtype=np.float32), shape=mask.shape)
    mask[rr, cc] = True


def annotation_to_mask(annotation: dict[str, Any] | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not annotation:
        return None
    mask = np.zeros(shape, dtype=bool)

    if "objects" in annotation:
        for object_payload in annotation.get("objects", []):
            segmentation = object_payload.get("segmentation")
            if isinstance(segmentation, list):
                xs: list[float] = []
                ys: list[float] = []
                for point in segmentation:
                    if "x" not in point or "y" not in point:
                        continue
                    xs.append(float(point["x"]))
                    ys.append(float(point["y"]))
                if len(xs) >= 3:
                    _apply_polygon_mask(mask, xs, ys)
                    continue
            bbox = object_payload.get("bbox")
            if not bbox:
                continue
            _apply_bbox_mask(
                mask,
                float(bbox.get("xmin", 0.0)),
                float(bbox.get("ymin", 0.0)),
                float(bbox.get("xmax", 0.0)),
                float(bbox.get("ymax", 0.0)),
            )
        return mask if np.any(mask) else None

    if "annotations" in annotation:
        for item in annotation.get("annotations", []):
            segmentation = item.get("segmentation")
            if isinstance(segmentation, list):
                polygon_groups = segmentation
                if polygon_groups and isinstance(polygon_groups[0], (int, float)):
                    polygon_groups = [polygon_groups]
                applied = False
                for group in polygon_groups:
                    if not isinstance(group, list) or len(group) < 6:
                        continue
                    xs = [float(group[index]) for index in range(0, len(group), 2)]
                    ys = [float(group[index]) for index in range(1, len(group), 2)]
                    _apply_polygon_mask(mask, xs, ys)
                    applied = True
                if applied:
                    continue
            bbox = item.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                _apply_bbox_mask(
                    mask,
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[0]) + float(bbox[2]),
                    float(bbox[1]) + float(bbox[3]),
                )
        return mask if np.any(mask) else None

    return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def evaluate_proxy_outputs(
    *,
    dataset_name: str,
    sample_id: str,
    split: str,
    annotation: dict[str, Any] | None,
    annotation_count: int,
    downstream_target: np.ndarray | None,
    detection_map: np.ndarray,
    threshold_quantile: float = 0.98,
) -> ProxyEvaluation:
    threshold = float(np.quantile(detection_map, threshold_quantile))
    predicted_mask = detection_map >= threshold
    target_mask = None

    if downstream_target is not None:
        target_mask = np.asarray(downstream_target == 1, dtype=bool)
    else:
        target_mask = annotation_to_mask(annotation, predicted_mask.shape)

    task = "proxy_only"
    status = "proxy_only_no_target"
    note = (
        "Saved a thresholded detection-proxy mask for downstream inspection. "
        "No trained detector/segmenter is wired here, so mAP is intentionally not reported."
    )

    metrics: dict[str, Any] = {
        "proxy_threshold_quantile": float(threshold_quantile),
        "proxy_threshold_value": threshold,
        "proxy_positive_fraction": float(np.mean(predicted_mask)),
        "proxy_detection_iou": None,
        "proxy_detection_precision": None,
        "proxy_detection_recall": None,
        "proxy_detection_f1": None,
        "proxy_segmentation_miou": None,
    }

    if target_mask is not None:
        target_mask = np.asarray(target_mask, dtype=bool)
        true_positive = float(np.sum(predicted_mask & target_mask))
        false_positive = float(np.sum(predicted_mask & ~target_mask))
        false_negative = float(np.sum(~predicted_mask & target_mask))
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        iou = _safe_ratio(true_positive, true_positive + false_positive + false_negative)
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        metrics.update(
            {
                "proxy_detection_iou": iou,
                "proxy_detection_precision": precision,
                "proxy_detection_recall": recall,
                "proxy_detection_f1": f1,
            }
        )
        task = "ship_detection_proxy_mask"
        status = "proxy_detection_mask_available"
        note = (
            "Computed a thresholded detection-proxy mask against the available annotations. "
            "This is a sanity-check only, not a trained detector."
        )
        if downstream_target is not None:
            segmentation_metrics = compute_segmentation_miou(
                predicted_mask.astype(np.int32),
                np.asarray(downstream_target, dtype=np.int32),
                valid_labels=(0, 1),
                ignore_value=-1,
            )
            metrics["proxy_segmentation_miou"] = segmentation_metrics["miou"]
            task = "segmentation_proxy_threshold"
            status = "proxy_segmentation_available"
            note = (
                "Computed a thresholded segmentation proxy against the available mask. "
                "This is a sanity-check only, not a trained segmentation model."
            )

    downstream_row = {
        "dataset": dataset_name,
        "sample_id": sample_id,
        "split": split,
        "task": task,
        "status": status,
        "annotation_count": int(annotation_count),
        "proxy_detection_iou": metrics["proxy_detection_iou"],
        "proxy_detection_precision": metrics["proxy_detection_precision"],
        "proxy_detection_recall": metrics["proxy_detection_recall"],
        "proxy_detection_f1": metrics["proxy_detection_f1"],
        "proxy_segmentation_miou": metrics["proxy_segmentation_miou"],
        "note": note,
    }
    return ProxyEvaluation(
        metrics=metrics,
        downstream_row=downstream_row,
        predicted_mask=predicted_mask.astype(np.uint8),
        target_mask=target_mask.astype(np.uint8) if target_mask is not None else None,
    )
