# Final Evidence Table

| dataset | f1 | label | map | map50 | map75 | mean_edge_preservation_index | mean_mse | mean_nrmse | mean_psnr | mean_ssim | precision | recall | sample_count | test_count | track | train_count | val_count | variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mendeley SAR despeckling val |  | Raw |  |  |  | 0.5583 | 0.0165 | 0.3256 | 18.0782 | 0.5261 |  |  | 100 |  | paired_denoising |  |  | raw |
| Mendeley SAR despeckling val |  | Bundle A |  |  |  | 0.539 | 0.012262 | 0.2751 | 19.4577 | 0.5651 |  |  | 100 |  | paired_denoising |  |  | bundle_a |
| Mendeley SAR despeckling val |  | A conservative |  |  |  | 0.5647 | 0.013822 | 0.2971 | 18.861 | 0.5544 |  |  | 100 |  | paired_denoising |  |  | bundle_a_conservative |
| Mendeley SAR despeckling val |  | Bundle B |  |  |  | 0.588 | 0.009829 | 0.2513 | 20.3082 | 0.5974 |  |  | 100 |  | paired_denoising |  |  | bundle_b |
| Mendeley SAR despeckling val |  | Bundle D |  |  |  | 0.5871 | 0.012149 | 0.2794 | 19.4313 | 0.5825 |  |  | 100 |  | paired_denoising |  |  | bundle_d |
| HRSID | 0.9138 | Raw | 0.6486 | 0.933 | 0.7805 |  |  |  |  |  | 0.9628 | 0.8695 |  | 256 | detector_compatibility | 256 | 256 | raw |
| HRSID | 0.8751 | Bundle A | 0.5277 | 0.8931 | 0.5625 |  |  |  |  |  | 0.9555 | 0.8072 |  | 256 | detector_compatibility | 256 | 256 | bundle_a |
| HRSID | 0.8645 | A conservative | 0.5229 | 0.8872 | 0.5521 |  |  |  |  |  | 0.9415 | 0.7992 |  | 256 | detector_compatibility | 256 | 256 | bundle_a_conservative |
| HRSID | 0.8468 | Bundle B | 0.4742 | 0.8694 | 0.4571 |  |  |  |  |  | 0.9273 | 0.7791 |  | 256 | detector_compatibility | 256 | 256 | bundle_b |
| HRSID | 0.8891 | Bundle D | 0.5325 | 0.8993 | 0.5811 |  |  |  |  |  | 0.9582 | 0.8293 |  | 256 | detector_compatibility | 256 | 256 | bundle_d |
| SSDD | 0.7482 | Raw | 0.4894 | 0.7889 | 0.557 |  |  |  |  |  | 0.797 | 0.7051 |  | 232 | detector_compatibility | 256 | 186 | raw |
| SSDD | 0.7017 | Bundle A | 0.3426 | 0.7018 | 0.286 |  |  |  |  |  | 0.8228 | 0.6117 |  | 232 | detector_compatibility | 256 | 186 | bundle_a |
| SSDD | 0.722 | A conservative | 0.3766 | 0.7397 | 0.3295 |  |  |  |  |  | 0.7907 | 0.6643 |  | 232 | detector_compatibility | 256 | 186 | bundle_a_conservative |
| SSDD | 0.7039 | Bundle B | 0.3661 | 0.715 | 0.3285 |  |  |  |  |  | 0.7916 | 0.6337 |  | 232 | detector_compatibility | 256 | 186 | bundle_b |
| SSDD | 0.7227 | Bundle D | 0.4033 | 0.7357 | 0.4123 |  |  |  |  |  | 0.8399 | 0.6342 |  | 232 | detector_compatibility | 256 | 186 | bundle_d |
