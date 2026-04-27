from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
from skimage import io as skio

from src.datasets.mendeley_despeckling import discover_mendeley_pairs
from src.evaluation.denoising_metrics import compute_denoising_metrics, normalize_paired_images


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_discover_mendeley_pairs_matches_by_stem() -> None:
    workspace = REPO_ROOT / "outputs" / "test-workspaces" / uuid4().hex
    root = workspace / "Mendeley SAR dataset"
    try:
        for folder in ("Noisy", "GTruth", "Noisy_val", "GTruth_val"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        image = np.full((8, 8), 128, dtype=np.uint8)
        skio.imsave(root / "Noisy" / "chip_001.png", image)
        skio.imsave(root / "GTruth" / "chip_001.png", image)
        skio.imsave(root / "Noisy_val" / "chip_002.tiff", image)
        skio.imsave(root / "GTruth_val" / "chip_002.tiff", image)

        train_pairs = discover_mendeley_pairs(root, split="train")
        val_pairs = discover_mendeley_pairs(root, split="val")

        assert len(train_pairs) == 1
        assert train_pairs[0].pair_id == "train_chip_001"
        assert len(val_pairs) == 1
        assert val_pairs[0].pair_id == "val_chip_002"
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def test_denoising_metrics_reward_closer_candidate() -> None:
    reference = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (16, 1))
    noisy = np.clip(reference + 0.2, 0.0, 1.0)
    noisy_norm, reference_norm = normalize_paired_images(noisy, reference)
    bad = compute_denoising_metrics(noisy_norm, reference_norm)
    good = compute_denoising_metrics(reference_norm, reference_norm)

    assert good.mse < bad.mse
    assert good.psnr > bad.psnr
    assert good.ssim > bad.ssim

