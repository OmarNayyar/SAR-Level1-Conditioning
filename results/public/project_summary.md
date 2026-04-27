# SAR Stage-1 Conditioning Public Summary

**Public-safe conclusion:** Bundle B is currently strongest on paired public denoising metrics, while raw imagery remains strongest for the current lightweight YOLO detector compatibility run.

This is a public-data validation repo, not a claim that conditioning universally improves downstream maritime SAR detection.

## Denoising Evidence

| label | sample_count | mean_psnr | mean_ssim | mean_mse | mean_edge_preservation_index |
| --- | --- | --- | --- | --- | --- |
| Raw | 100 | 18.0782 | 0.5261 | 0.0165 | 0.5583 |
| Bundle A | 100 | 19.4577 | 0.5651 | 0.012262 | 0.539 |
| A conservative | 100 | 18.861 | 0.5544 | 0.013822 | 0.5647 |
| Bundle B | 100 | 20.3082 | 0.5974 | 0.009829 | 0.588 |
| Bundle D | 100 | 19.4313 | 0.5825 | 0.012149 | 0.5871 |

## Detector Compatibility Evidence

| dataset | label | map | map50 | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| HRSID | Raw | 0.6486 | 0.933 | 0.9628 | 0.8695 | 0.9138 |
| HRSID | Bundle A | 0.5277 | 0.8931 | 0.9555 | 0.8072 | 0.8751 |
| HRSID | A conservative | 0.5229 | 0.8872 | 0.9415 | 0.7992 | 0.8645 |
| HRSID | Bundle B | 0.4742 | 0.8694 | 0.9273 | 0.7791 | 0.8468 |
| HRSID | Bundle D | 0.5325 | 0.8993 | 0.9582 | 0.8293 | 0.8891 |
| SSDD | Raw | 0.4894 | 0.7889 | 0.797 | 0.7051 | 0.7482 |
| SSDD | Bundle A | 0.3426 | 0.7018 | 0.8228 | 0.6117 | 0.7017 |
| SSDD | A conservative | 0.3766 | 0.7397 | 0.7907 | 0.6643 | 0.722 |
| SSDD | Bundle B | 0.3661 | 0.715 | 0.7916 | 0.6337 | 0.7039 |
| SSDD | Bundle D | 0.4033 | 0.7357 | 0.8399 | 0.6342 | 0.7227 |

## Interpretation

- Paired denoising and detector compatibility answer different questions.
- Bundle B improved paired denoising quality on the Mendeley validation split.
- Raw remained best for this YOLO detector setup, so it stays the detector baseline for this experiment.
- No claim is made for non-public operational data without representative validation.
