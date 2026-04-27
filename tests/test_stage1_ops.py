from __future__ import annotations

import numpy as np

from src.stage1.additive.destripe_lowrank_sparse import destripe_lowrank_sparse
from src.stage1.additive.pnp_admm_additive import pnp_admm_additive
from src.stage1.additive.starlet_complex_denoise import starlet_complex_denoise
from src.stage1.downstream.proxy_eval import annotation_to_mask
from src.stage1.multiplicative.merlin_wrapper import run_merlin_wrapper
from src.stage1.multiplicative.mulog_bm3d import mulog_bm3d
from src.stage1.multiplicative.speckle2void_wrapper import run_speckle2void_wrapper


def test_stage1_bundle_operators_return_expected_shapes() -> None:
    rng = np.random.default_rng(7)
    intensity = np.abs(rng.normal(loc=2.0, scale=0.5, size=(64, 64))).astype(np.float32)
    complex_image = intensity.astype(np.float32) + 1j * rng.normal(scale=0.1, size=(64, 64)).astype(np.float32)

    destripe_result = destripe_lowrank_sparse(intensity)
    assert destripe_result.corrected_image.shape == intensity.shape

    mulog_result = mulog_bm3d(destripe_result.corrected_image, backend_preference="wavelet")
    assert mulog_result.filtered_image.shape == intensity.shape

    starlet_result = starlet_complex_denoise(complex_image, levels=3)
    assert starlet_result.denoised_complex.shape == complex_image.shape

    merlin_result = run_merlin_wrapper(starlet_result.denoised_complex, external=None)
    assert merlin_result.output_intensity.shape == intensity.shape
    assert merlin_result.fallback_used is True

    pnp_result = pnp_admm_additive(intensity, iterations=3)
    assert pnp_result.corrected_image.shape == intensity.shape

    s2v_result = run_speckle2void_wrapper(pnp_result.corrected_image, external=None)
    assert s2v_result.filtered_image.shape == intensity.shape


def test_annotation_to_mask_supports_voc_boxes() -> None:
    annotation = {
        "objects": [
            {
                "bbox": {
                    "xmin": "4",
                    "ymin": "5",
                    "xmax": "12",
                    "ymax": "13",
                }
            }
        ]
    }
    mask = annotation_to_mask(annotation, (20, 20))
    assert mask is not None
    assert int(mask.sum()) > 0
