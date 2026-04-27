# Bundle Method Guide

This guide explains what each route is meant to do, what evidence currently supports it, and where it should not be overused. The bundle names follow the original Stage-1 conditioning brief: metadata-first thermal/noise-vector baseline, artifact-aware destriping plus log-domain despeckling, SLC-first starlet/MERLIN path, and intensity-only inverse/blind-spot path.

## Quick Routing Table

| Situation | Start with | Why | Do not overclaim |
| --- | --- | --- | --- |
| Current YOLO ship detector baseline | Raw | Best mAP on the current SSDD/HRSID final sweep | This does not make denoising invalid |
| Paired denoising quality | Bundle B | Best Mendeley PSNR/SSIM/MSE among tested variants | Validate on representative products before deployment claims |
| Explainable screening / ablation | Bundle A | Most interpretable additive routing family | Not the best detector baseline in current evidence |
| Milder explainable conditioning | Bundle A conservative | Less aggressive than default A | Still needs detector proof before adoption |
| Structure preservation | Bundle D | Strong SSIM and edge preservation | Still trails raw in current detector mAP |
| Complex-domain future path | Bundle C | SLC/MERLIN-oriented route | Not claim-grade without genuine SLC access |

## Raw

Purpose: raw/noisy input baseline.

Input domain: whatever the benchmark or product loader provides after standard local loading.

Noise regime: none removed; used as the control.

Strengths:
- Preserves the detector training distribution.
- Best current YOLO detector mAP on SSDD and HRSID.
- Essential baseline for every task-specific comparison.

Weaknesses:
- Does not reduce paired denoising error on Mendeley.
- Can retain speckle, thermal floor, striping, or other product artifacts.

Current evidence:
- Strongest current detector compatibility result.
- Worse than Bundle B on paired Mendeley PSNR/SSIM/MSE.

Recommendation: use raw as the detector baseline for the current YOLO setup. Do not treat raw as universally best for denoising.

## Bundle A

Purpose: interpretable additive-noise/noise-floor conditioning plus classical speckle filtering.

Input domain: detected intensity / power style chips and GRD-like imagery.

Noise targeted:
- additive thermal floor or noise-vector-style contamination
- image-derived low-intensity floor
- structured additive artifacts when visible
- multiplicative speckle through Refined Lee filtering

Method family:
- A0: no additive correction control
- A1: metadata thermal/noise-vector subtraction
- A2: image-derived additive floor estimate
- A3: structured additive artifact correction
- multiplicative step: Refined Lee

Paper/source inspiration:
- Sentinel-1 thermal denoising and noise-vector literature
- dynamic least-squares additive noise removal
- efficient Sentinel-1 thermal noise removal
- classical SAR speckle filtering / Refined Lee style filtering
- Lee / refined Lee / Gamma-MAP family references

Strengths:
- Easiest route to explain.
- Best for auditing metadata availability and additive-submethod routing.
- Good screening baseline because every submethod records what it used and why.

Weaknesses:
- Default A can suppress detector-useful texture/edge cues.
- It is not the best paired denoising result and not the current detector winner.

Current evidence:
- Improves Mendeley PSNR/SSIM/MSE versus raw.
- Trails raw detector mAP in current SSDD/HRSID YOLO sweep.

Recommendation: use as the interpretable metadata/noise-floor conditioning and screening family, not as the operational detector baseline.

## Bundle A Conservative

Purpose: milder Bundle A-style conditioning.

Input domain: detected intensity / power style chips and GRD-like imagery.

Noise targeted: same broad regime as Bundle A, but with less aggressive correction.

Strengths:
- Lower risk of over-smoothing than default A.
- Useful when detection compatibility matters but explainability is still needed.

Weaknesses:
- Still trails raw detector mAP in the current YOLO sweep.
- Not as strong as Bundle B on paired denoising metrics.

Current evidence:
- Improves Mendeley PSNR/SSIM/MSE versus raw.
- Does not beat raw on detector mAP.

Recommendation: keep as a safer A-family ablation and screening route.

## Bundle B

Purpose: structured additive cleanup plus log-domain denoising.

Input domain: log-intensity / intensity-derived imagery.

Noise targeted:
- stripe-like additive structure
- banding or row/column profile artifacts
- harder additive contamination
- multiplicative speckle through log-domain denoising

Method family:
- low-rank/sparse-inspired destriping
- MuLoG-style log-domain denoising with practical fallbacks such as wavelet/BM3D-like smoothing paths where available

Paper/source inspiration:
- low-rank directional sparse destriping
- MuLoG/BM3D-style log-domain SAR denoising ideas
- BM3D denoising literature

Strengths:
- Best current paired denoising result on Mendeley validation.
- Strongest PSNR/SSIM/MSE among raw/A/A-conservative/B/D.
- Plausible first candidate when the objective is denoising quality rather than detector compatibility.

Weaknesses:
- Current YOLO detector mAP still trails raw.
- Log-domain conditioning can shift texture statistics enough to require detector retuning.

Current evidence:
- Mendeley validation: best PSNR, SSIM, and MSE.
- Detector final sweep: not the detector winner.

Recommendation: prioritize for paired/intensity-domain denoising validation. For detection, test against raw and expect possible detector retuning.

## Bundle C / SLC Future Path

Purpose: complex-domain and self-supervised future path.

Input domain: genuine complex SLC preferred. Intensity fallback is only a demo/feasibility mode.

Noise targeted:
- complex-valued noise regimes
- self-supervised denoising opportunities when clean references do not exist

Method family:
- starlet sparse shrinkage
- MERLIN-style wrapper / documented fallback

Paper/source inspiration:
- MERLIN
- Noise2Noise
- PolMERLIN
- starlet/radio astronomy sparse denoising

Strengths:
- Most relevant path if complex SLC data become available.
- Better scientific fit for phase-aware complex-domain work than forcing intensity-only methods.

Weaknesses:
- Not claim-grade without true SLC data.
- Should not be presented as solved from detected intensity chips.

Current evidence:
- Feasibility only.

Recommendation: defer claims until genuine SLC samples and complex-domain validation are available.

## Bundle D

Purpose: metadata-poor inverse/self-supervised style candidate.

Input domain: intensity or log-intensity.

Noise targeted:
- additive contamination without trusted metadata
- speckle-like degradation where a conservative blind/self-supervised fallback may help
- structure preservation under moderate cleanup

Method family:
- plug-and-play ADMM-style additive cleanup
- Speckle2Void-style wrapper / fallback

Paper/source inspiration:
- plug-and-play priors for model-based reconstruction
- Speckle2Void
- ADAM/joint compression-despeckling ideas where inverse-problem framing is useful

Strengths:
- Strong SSIM and edge preservation on Mendeley validation.
- Useful alternative when Bundle B is too smoothing or when structure preservation is prioritized.

Weaknesses:
- Does not beat Bundle B on paired Mendeley PSNR/SSIM/MSE.
- Does not beat raw on current YOLO detector mAP.

Current evidence:
- Improves paired denoising metrics versus raw.
- Strong edge preservation.
- Trails raw detector mAP.

Recommendation: keep as a structure-preserving candidate and compare against Bundle B on representative data.
