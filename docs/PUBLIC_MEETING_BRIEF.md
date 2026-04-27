# Public Meeting Brief

This is the compact public-safe talk track for the project.

## Two-Minute Overview

This repo implements a Stage-1 SAR conditioning screen between standard image formation/calibration and downstream AI. It compares four conditioning families against raw imagery instead of assuming that denoising always helps.

The current evidence separates two questions:

| Question | Current answer |
| --- | --- |
| Which route improves paired denoising metrics on public SAR despeckling data? | Bundle B is strongest on Mendeley validation PSNR/SSIM/MSE. |
| Which input works best for the current lightweight YOLO ship detector? | Raw imagery is strongest on SSDD/HRSID mAP. |

Those are not contradictory results. Denoising metrics measure image similarity to a reference target. Detector mAP measures whether a specific detector benefits from the changed image distribution.

## Bundle Routing

| Situation | Start with | Reason |
| --- | --- | --- |
| Current detector baseline | Raw | Best mAP in the current YOLO sweep. |
| Paired intensity-domain denoising | Bundle B | Best Mendeley PSNR/SSIM/MSE. |
| Explainable metadata/noise-floor screening | Bundle A | Most interpretable additive routing and Lee-style speckle filtering. |
| Milder explainable conditioning | Bundle A conservative | Lower-risk A-family ablation. |
| Structure preservation | Bundle D | Strong SSIM and edge preservation behavior. |
| Complex/SLC validation | Bundle C | Planned SLC-first starlet + MERLIN-style route, pending SLC data. |

## What The Evidence Proves

- Bundle B improves paired denoising quality on the local Mendeley validation split.
- Bundle D is a useful structure-preserving candidate.
- Raw imagery remains the current detector baseline for the lightweight YOLO setup.
- The repo can run public denoising, detector compatibility, public/private surface checks, and final-sweep dry-runs without retraining by accident.

## What The Evidence Does Not Prove

- It does not prove conditioning is universally useful for every downstream detector.
- It does not prove raw imagery is universally best for denoising or analysis.
- It does not prove Bundle C/MERLIN performance without genuine SLC data.
- It does not support operational claims on non-public data until representative products are tested.

## Data Needed Next

- representative raw/noisy SAR products
- product level: raw DN, calibrated GRD, SLC, or internally processed products
- sensor, mode, polarization, look count, and pixel-domain details
- calibration/noise vectors, NESZ, or equivalent noise metadata
- downstream task and metric priorities
- clarification of whether the goal is visual quality, detection, segmentation, compression, radiometry, or a combination

