# Final Results Interpretation

## Short Version

This project has two evidence tracks, and they answer different questions.

| Track | Question | Current answer |
| --- | --- | --- |
| Paired denoising | Did conditioning move noisy SAR imagery closer to a reference target? | Bundle B is strongest on Mendeley PSNR/SSIM/MSE |
| Detector compatibility | Did conditioning help this YOLO detector? | Raw is strongest on the current SSDD/HRSID sweep |

That is not a contradiction. Denoising can make an image closer to a reference while also changing the texture statistics a detector learned to use.

## Beginner Explanation

Think of the detector as a person trained to recognize ships in noisy images. If we clean the image, it may look better to us and score better against a clean reference, but the detector may no longer see exactly the cues it was trained on. That means the detector may prefer raw imagery even when denoising metrics improve.

So the right conclusion is:

- Bundle B currently works best for paired denoising quality.
- Raw currently works best for this YOLO detector setup.
- A future detector trained or tuned on conditioned imagery could behave differently.

## Mathematical Explanation

For paired denoising, the repo compares output image `x_hat` against reference image `x`:

```text
MSE = mean((x_hat - x)^2)
PSNR = 10 log10(data_range^2 / MSE)
SSIM = structural similarity between x_hat and x
```

Lower MSE and higher PSNR/SSIM mean the conditioned output is closer to the paired reference target.

Detector validation is different. YOLO mAP measures bounding-box detection quality:

```text
mAP = area under precision-recall behavior across IoU thresholds
```

The detector does not directly optimize PSNR or SSIM. It optimizes detection behavior on the image distribution it sees during training and evaluation.

## Mendeley Paired Denoising Result

| Variant | Mean PSNR | Mean SSIM | Mean MSE | Mean NRMSE | Edge preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 18.0782 | 0.5261 | 0.016500 | 0.3256 | 0.5583 |
| Bundle A | 19.4577 | 0.5651 | 0.012262 | 0.2751 | 0.5390 |
| Bundle A conservative | 18.8610 | 0.5544 | 0.013822 | 0.2971 | 0.5647 |
| Bundle B | 20.3082 | 0.5974 | 0.009829 | 0.2513 | 0.5880 |
| Bundle D | 19.4313 | 0.5825 | 0.012149 | 0.2794 | 0.5871 |

Interpretation:

- Bundle B is the strongest paired denoising route in the current evidence.
- Bundle D is useful as a structure-preserving candidate.
- Bundle A remains valuable because it is interpretable, not because it wins every metric.

## SSDD/HRSID Detector Compatibility Result

| Dataset | Current detector winner | Interpretation |
| --- | --- | --- |
| SSDD | Raw | Raw best matches this YOLO setup |
| HRSID | Raw | Raw best matches this YOLO setup |

Conditioned variants remain usable, but they do not replace raw for this detector unless they beat raw on the target detector metric.

## Sentinel-1 Status

| Item | Status |
| --- | --- |
| GRD cross-pol search | Dry-run search works and found a recent IW `VV+VH` candidate under 1.5 GB |
| GRD download | Attempted but blocked by local `data/external` filesystem permission before bytes were downloaded |
| SLC burst search | Works through CMR fallback and returns a single burst TIFF candidate |
| Full SLC download | Not run; intentionally requires explicit approval |

Sentinel-1 GRD is useful for additive thermal/banding/scalloping demonstrations. SLC is needed for serious complex-domain Bundle C/MERLIN-style work.

## What Works Where

| Need | Recommended starting route | Why |
| --- | --- | --- |
| Detector baseline for current YOLO | Raw | Best current mAP |
| Paired denoising quality | Bundle B | Best PSNR/SSIM/MSE |
| Structure-preserving denoising | Bundle D | Strong SSIM/edge preservation |
| Explainable screening | Bundle A | Clearest additive submethod routing |
| Safer A-family ablation | Bundle A conservative | Milder conditioning |
| Complex/SLC research | Bundle C | Correct future domain, but needs SLC |

## Extra Combination Pass Review

No chained bundle experiment was added before release. The obvious combinations, such as Bundle B then D, D then B, or A conservative then B, would create new unregistered variants outside the frozen A/B/C/D framing. That is not a harmless polish step: it changes the experiment surface and needs explicit configuration, parameter control, and a clean comparison table. The safe next step is to add named extra profiles under `outputs/denoising_quality_extra/` after the current meeting/release package is frozen.

## What Still Needs Representative Data

The repo cannot make final non-public or operational claims until representative samples arrive. Needed inputs:

- representative raw/noisy products
- product level: raw DN, calibrated GRD, SLC, or internally processed products
- sensor, mode, polarization, and product metadata
- noise vectors, NESZ, or equivalent calibration/noise metadata
- downstream task details and metric priorities
- clarification of whether the objective is visual quality, detection, compression, radiometry, or all of them
