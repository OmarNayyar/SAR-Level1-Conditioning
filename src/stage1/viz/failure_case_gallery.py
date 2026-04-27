from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .side_by_side import prepare_display_image


def save_failure_case_gallery(
    output_path: Path,
    *,
    cases: Sequence[dict[str, object]],
    title: str,
    columns: int = 3,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cases:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.text(0.5, 0.5, "No cases available", ha="center", va="center")
        axis.axis("off")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return output_path

    rows = int(np.ceil(len(cases) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes_array.flat:
        axis.axis("off")

    for axis, case in zip(axes_array.flat, cases):
        image = prepare_display_image(np.asarray(case["image"]))
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(str(case.get("title", "Case")))
        subtitle = str(case.get("subtitle", ""))
        if subtitle:
            axis.text(0.5, -0.08, subtitle, transform=axis.transAxes, ha="center", va="top", fontsize=8)
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path
