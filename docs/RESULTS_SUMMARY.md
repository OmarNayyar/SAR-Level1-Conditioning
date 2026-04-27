# Results Summary

This summary separates denoising quality from detector compatibility. PSNR/SSIM/MSE/NRMSE evaluate paired denoising against the Mendeley reference target. YOLO mAP evaluates whether a lightweight ship detector benefits from the conditioned images in the current SSDD/HRSID setup.

## Paired Denoising: Mendeley Validation Split

Bundle B is currently strongest for paired denoising quality on the local Mendeley validation split. Bundle D is also useful as a structure-preserving candidate because it improves SSIM and edge preservation versus raw noisy input. Raw here means the noisy input compared directly against the pseudo-clean reference.

| label | sample_count | mean_psnr | mean_ssim | mean_mse | mean_nrmse | mean_edge_preservation_index |
| --- | --- | --- | --- | --- | --- | --- |
| Raw | 100 | 18.0782 | 0.5261 | 0.0165 | 0.3256 | 0.5583 |
| Bundle A | 100 | 19.4577 | 0.5651 | 0.012262 | 0.2751 | 0.539 |
| A conservative | 100 | 18.861 | 0.5544 | 0.013822 | 0.2971 | 0.5647 |
| Bundle B | 100 | 20.3082 | 0.5974 | 0.009829 | 0.2513 | 0.588 |
| Bundle D | 100 | 19.4313 | 0.5825 | 0.012149 | 0.2794 | 0.5871 |

## Detector Compatibility: SSDD/HRSID Final Sweep

Raw imagery remains strongest for the current lightweight YOLO detector setup on both SSDD and HRSID. This is downstream compatibility evidence, not proof that denoising is useless. It likely reflects detector/data-distribution tuning and the fact that conditioning can suppress texture or edge cues used by this detector.

| dataset | label | map | map50 | map75 | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HRSID | Raw | 0.6486 | 0.933 | 0.7805 | 0.9628 | 0.8695 | 0.9138 |
| HRSID | Bundle A | 0.5277 | 0.8931 | 0.5625 | 0.9555 | 0.8072 | 0.8751 |
| HRSID | A conservative | 0.5229 | 0.8872 | 0.5521 | 0.9415 | 0.7992 | 0.8645 |
| HRSID | Bundle B | 0.4742 | 0.8694 | 0.4571 | 0.9273 | 0.7791 | 0.8468 |
| HRSID | Bundle D | 0.5325 | 0.8993 | 0.5811 | 0.9582 | 0.8293 | 0.8891 |
| SSDD | Raw | 0.4894 | 0.7889 | 0.557 | 0.797 | 0.7051 | 0.7482 |
| SSDD | Bundle A | 0.3426 | 0.7018 | 0.286 | 0.8228 | 0.6117 | 0.7017 |
| SSDD | A conservative | 0.3766 | 0.7397 | 0.3295 | 0.7907 | 0.6643 | 0.722 |
| SSDD | Bundle B | 0.3661 | 0.715 | 0.3285 | 0.7916 | 0.6337 | 0.7039 |
| SSDD | Bundle D | 0.4033 | 0.7357 | 0.4123 | 0.8399 | 0.6342 | 0.7227 |

## Practical Conclusion

- Bundle B should be prioritized for paired/intensity-domain denoising quality.
- Bundle D should remain a structure-preserving candidate.
- Bundle A remains the interpretable conditioning and screening family, not the best operational detector baseline.
- Raw remains the detector baseline only for the current YOLO setup until a conditioned variant beats raw on the target detector.
- Representative operational SAR samples, product-level clarification, metadata/noise vectors, SLC availability, and downstream task details are needed before deployment-style recommendations.
