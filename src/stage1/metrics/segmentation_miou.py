from __future__ import annotations

import numpy as np


def compute_segmentation_miou(
    predicted_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    valid_labels: tuple[int, ...] = (0, 1),
    ignore_value: int | None = None,
) -> dict[str, float]:
    prediction = np.asarray(predicted_mask)
    target = np.asarray(target_mask)
    if prediction.shape != target.shape:
        raise ValueError("Predicted and target masks must share the same shape.")

    valid_mask = np.ones_like(target, dtype=bool)
    if ignore_value is not None:
        valid_mask &= target != ignore_value

    ious: list[float] = []
    for label in valid_labels:
        pred_label = (prediction == label) & valid_mask
        target_label = (target == label) & valid_mask
        union = np.logical_or(pred_label, target_label).sum()
        if union == 0:
            continue
        intersection = np.logical_and(pred_label, target_label).sum()
        ious.append(float(intersection / union))

    return {
        "miou": float(np.mean(ious)) if ious else float("nan"),
        "class_count": float(len(ious)),
    }
