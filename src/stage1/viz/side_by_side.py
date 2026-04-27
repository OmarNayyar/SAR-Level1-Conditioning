from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def prepare_display_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in {1, 2, 3, 4} and array.shape[0] < array.shape[-1]:
        array = np.moveaxis(array, 0, -1)

    if array.ndim == 3 and array.shape[-1] == 2:
        vv = array[..., 0]
        vh = array[..., 1]
        array = np.stack([vv, vh, (vv + vh) / 2.0], axis=-1)
    elif array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]

    if array.ndim == 2:
        finite = np.isfinite(array)
        if not np.any(finite):
            return np.zeros_like(array, dtype=np.float32)
        lo, hi = np.percentile(array[finite], [2, 98])
        normalized = np.clip((array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return normalized.astype(np.float32)

    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros_like(array[..., :3], dtype=np.float32)
    lo, hi = np.percentile(array[finite], [2, 98])
    normalized = np.clip((array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return normalized.astype(np.float32)


def save_side_by_side(
    output_path: Path,
    *,
    before: np.ndarray,
    after: np.ndarray,
    before_title: str = "Before",
    after_title: str = "After",
    difference: np.ndarray | None = None,
    caption: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [prepare_display_image(before), prepare_display_image(after)]
    titles = [before_title, after_title]
    if difference is not None:
        panels.append(prepare_display_image(difference))
        titles.append("Difference")

    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for axis, panel, title in zip(axes, panels, titles):
        axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    if caption:
        figure.suptitle(caption)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path
