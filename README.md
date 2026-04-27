# SAR Stage-1 Conditioning: Denoising and Validation for Maritime SAR Imagery

`sar-stage1-conditioning` is a practical SAR Stage-1 conditioning repo for downstream segmentation and maritime object-analysis screening.

The repo is not trying to build one universal denoiser. It compares practical conditioning routes that keep three pieces explicit:
- additive-noise handling
- multiplicative-speckle handling
- downstream sanity checks

Domain distinctions stay explicit throughout the code and outputs:
- `complex_slc`
- `amplitude`
- `intensity / power`
- `log domain`

## What This Repo Does

This repo is designed to answer a practical screening question:

Which Stage-1 conditioning route gives the cleanest, most defensible improvement for downstream SAR analysis under realistic metadata and storage constraints?

The original project brief positions Stage-1 conditioning after focusing/basic calibration and before downstream AI. This repo keeps that scope: it uses realistically available product metadata where possible, falls back to image-fitted estimates where metadata is missing, and keeps additive noise, multiplicative speckle, and combined SAR noise separate.

It supports:
- multiple bundles
- multiple additive submethods inside Bundle A
- metadata-rich and metadata-poor inputs
- public benchmarks plus external/local datasets
- honest proxy evaluation plus a separate real detector-baseline path

## Bootstrap

Windows-friendly local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[app,dev]"
```

Optional extras:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[detection]"
.\.venv\Scripts\python.exe -m pip install -e ".[notebooks]"
```

If you prefer a single requirements file for local work:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Fast validation after install:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Editable install also exposes console commands for the main repo flows:
- `sar-stage1-bundle-a`
- `sar-stage1-bundle-b`
- `sar-stage1-bundle-c`
- `sar-stage1-bundle-d`
- `sar-stage1-detection`
- `sar-stage1-final-sweep`
- `sar-stage1-denoise-eval`
- `sar-stage1-denoise-panels`
- `sar-stage1-final-figures`
- `sar-stage1-sentinel-samples`
- `sar-stage1-audit`
- `sar-stage1-demo-index`
- `sar-stage1-surface-check`

## Current Readiness

### Dataset snapshot

| Dataset | Status | Current local snapshot | Practical role now |
| --- | --- | --- | --- |
| `ssdd` | `partial` | `1160` chips (`train: 928`, `test: 232`) | main ship-detection sanity-check dataset |
| `hrsid` | `partial` | `5604` chips (`train: 3642`, `test: 1962`) | second ship-detection sanity-check dataset |
| `sentinel1` | `partial` | `6` locally prepared/evaluated GRD products in the current evidence cache | metadata-driven and proxy-style SAR scene path |
| `sen1floods11` | `partial` | `6` smoke samples (`2/2/2`) | segmentation smoke path only |
| `ai4arctic` | `metadata-only` | `0` local samples | not ready for serious local experiments |
| `ls_ssdd` | `metadata-only` | `0` local samples | blocked by source portal, ignored for now |

Audit outputs:
- `results/data_audit/audit_summary.json`
- `results/data_audit/audit_summary.md`
- local-only audit notes are excluded from the public export

### Bundle maturity

| Bundle | Additive method | Multiplicative method | Domain | Current maturity | Best use case |
| --- | --- | --- | --- | --- | --- |
| `A` | `A0/A1/A2/A3` family | Refined Lee | intensity | interpretable screening family | interpretable baseline and supervisor-facing comparison path |
| `B` | structured cleanup / destriping | MuLoG-style denoising | log-intensity | usable secondary | harder additive artifacts and alternative noise-aware route |
| `C` | starlet shrinkage | MERLIN wrapper / fallback | complex SLC preferred | feasibility only | demo path when genuine complex SLC is available |
| `D` | PnP-ADMM additive cleanup | Speckle2Void wrapper / fallback | intensity / log-intensity | usable secondary | alternative inverse-problem route |

Practical interpretation:
- `Bundle A` is the main interpretable screening family right now, not the operational detector baseline.
- `Bundle B` is currently the strongest paired denoising candidate on the local Mendeley validation split.
- `Bundle D` is a useful structure-preserving candidate.
- `Bundle C` should be treated as feasibility-grade unless real complex SLC support is present.
- Current Sentinel-1 scene rankings are still proxy-only and overview-scale; they are useful for routing decisions, not final detector claims.
- A lightweight YOLO downstream detector path is wired on SSDD/HRSID. The completed final sweep shows raw imagery remains strongest for the current detector setup, while conditioned bundles remain candidates for separate denoising-quality and task-specific validation.

### Current result summary

The repo now has two separate evidence tracks:

| Track | Current result | Interpretation |
| --- | --- | --- |
| Paired denoising, Mendeley validation split | Bundle B has the best PSNR/SSIM/MSE among the tested variants. | This is denoising-quality evidence. |
| Detector compatibility, SSDD/HRSID YOLO sweep | Raw imagery has the best detector mAP in the current setup. | This is detector/data-distribution compatibility evidence. |

Do not collapse these into one claim. A detector preferring raw imagery does not make denoising invalid; it means the current detector setup is still best matched to raw texture/edge statistics.

## Bundle A Additive Submethods

Bundle A is now a family of additive submethods with a fixed multiplicative step:
- additive: `A0 / A1 / A2 / A3`
- multiplicative: `refined_lee`

### Submethod table

| Code | Name | What it does | Requires | Trust level |
| --- | --- | --- | --- | --- |
| `A0` | No additive correction | pass-through baseline | intensity image only | `baseline-only` |
| `A1` | Metadata thermal / noise-vector subtraction | subtract additive noise using `noise_vector`, `noise_power`, or `nesz_db` style metadata | product metadata | `metadata-driven` |
| `A2` | Image-derived additive floor estimate | estimate a constant additive floor from the lower-tail intensity distribution | intensity image only | `image-derived` |
| `A3` | Structured additive artifact correction | apply artifact-aware destriping / structured cleanup in intensity space | intensity image with detectable structured artifact pattern | `artifact-based` |

Bundle A can run in:
- `auto` mode: choose the most practical additive submethod for the current sample
- forced mode: explicitly run `A0`, `A1`, `A2`, or `A3`

Every Bundle A run records:
- additive submethod code and human-readable name
- description
- required inputs
- whether useful additive metadata was actually available
- what fallback was used if metadata was absent
- confidence / trust level

### Current Bundle A routing rules

- `A0` is the control. It is preferred when the scene looks clean enough that additive correction is not justified.
- `A1` is the metadata-driven route. It should be trusted when noise vectors or equivalent metadata exist, but overview-scale proxy metrics may under-credit its benefit.
- `A2` is the practical metadata-poor fallback. It estimates an additive floor from the image itself.
- `A3` is a specialist for visible stripe-like or structured additive artifacts.

## Evidence Grades and Proxy Scoring

The repo now separates three ideas:

- `proxy metrics`: ENL, edge sharpness, target/background separability, and threshold F1.
- `decision heuristics`: a balanced score used to rank methods for screening.
- `evidence grade`: how much confidence the current data supports.

The balanced decision score intentionally caps ENL reward and penalizes edge loss so over-smoothed outputs do not win just because they look homogeneous. It also records caveats such as `A1 may be under-credited by overview-scale proxy metrics`.

Common grades:

| Grade | Meaning |
| --- | --- |
| `claim-grade` | A real downstream detector/segmenter evaluation is wired and reported. |
| `screening-grade / proxy-only` | Useful for choosing what to test next, but not final mAP/IoU evidence. |
| `proxy-only / overview-scale` | Sentinel-1 scene evidence from memory-safe overviews or decimated products. |
| `feasibility-grade` | Runnable path, but the right data regime is still too weak for claims. |
| `thin`, `thin-but-improving`, `developing` | Evidence breadth labels based mostly on scene/sample count and caveats. |

## Commands

## Repo Map

| Path | Purpose |
| --- | --- |
| `configs/` | Bundle, dataset, and downstream baseline configs. |
| `src/stage1/` | Additive, multiplicative, metric, statistics, Sentinel-1 batch, and visualization code. |
| `src/bundles/` | Bundle A/B/C/D orchestration. |
| `src/datasets/` | Dataset manifests, loaders, registry, Sentinel-1 fetch/prepare/evidence helpers. |
| `src/downstream/detection/` | YOLO-format dataset preparation and optional Ultralytics detector runner. |
| `src/reporting/` | Decision scoring, result indexes, demo curation, and app-facing summaries. |
| `scripts/` | Main command-line entry points for bundles, datasets, detector baseline, audit, and reporting. |
| `scripts/sentinel1/` | Sentinel-1 inspection, planning, expansion, and batch comparison commands. |
| `scripts/downstream/` | Namespaced downstream-evaluation wrappers; root commands remain for compatibility. |
| `apps/` | Streamlit decision/demo app. |
| `results/` | Lightweight commit-safe summaries plus ignored visual artifacts. |
| `outputs/` | Local generated caches, detector-prepared datasets, Sentinel-1 batch outputs; ignored by git. |

Documentation:
- `docs/PDF_ALIGNMENT_AUDIT.md`
- `docs/BUNDLE_METHOD_GUIDE.md`
- `docs/RESULTS_SUMMARY.md`
- `docs/FINAL_RESULTS_INTERPRETATION.md`
- `docs/GIT_RELEASE_CHECKLIST.md`
- `docs/DATA_DOWNLOAD_GUIDE_DENOISING.md`
- `docs/PUBLIC_MEETING_BRIEF.md`

### Bundle runs

Reuse-first note:
- Bundle, detector, and Sentinel-1 batch commands now default to cache reuse.
- If matching artifacts already exist, the command will load them.
- If artifacts are missing, the command will stop with a clear message instead of silently recomputing.
- Use `--allow-conditioning`, `--allow-prepare`, `--allow-train`, and `--allow-eval` only when you intentionally want heavy work.

Run Bundle A in auto mode:

```powershell
python scripts/run_bundle_a.py --config configs/bundle_a.yaml
```

Force a specific Bundle A additive submethod:

```powershell
python scripts/run_bundle_a.py --config configs/bundle_a.yaml --additive-submethod A0
python scripts/run_bundle_a.py --config configs/bundle_a.yaml --additive-submethod A1
python scripts/run_bundle_a.py --config configs/bundle_a.yaml --additive-submethod A2
python scripts/run_bundle_a.py --config configs/bundle_a.yaml --additive-submethod A3
```

Note:
- default auto runs write to `results/bundle_a`
- forced or dataset-overridden Bundle A runs now default to separate `outputs/bundle_a_...` folders unless you provide `--output-root`
- add `--allow-conditioning` only when you intentionally want to regenerate a bundle run

Run Bundle B:

```powershell
python scripts/run_bundle_b.py --config configs/bundle_b.yaml
```

Run Bundle C:

```powershell
python scripts/run_bundle_c.py --config configs/bundle_c.yaml
```

Run Bundle D:

```powershell
python scripts/run_bundle_d.py --config configs/bundle_d.yaml
```

Run tuning profiles without editing code:

```powershell
.\.venv\Scripts\python.exe scripts/run_bundle_a.py --config configs/bundles/profiles/bundle_a_conservative.yaml
.\.venv\Scripts\python.exe scripts/run_bundle_b.py --config configs/bundles/profiles/bundle_b_balanced.yaml
.\.venv\Scripts\python.exe scripts/run_bundle_d.py --config configs/bundles/profiles/bundle_d_conservative.yaml
```

### Downstream detector baseline

Prepare full raw SSDD/HRSID splits for YOLO without training. This verifies that the local data and annotations are usable before spending time on detector training:

```powershell
.\.venv\Scripts\python.exe scripts/run_detection_baseline.py --dataset ssdd --variants raw --limit-per-split none --mode prepare --output-root outputs\downstream_detection_fullprep --allow-prepare
.\.venv\Scripts\python.exe scripts/run_detection_baseline.py --dataset hrsid --variants raw --limit-per-split none --mode prepare --output-root outputs\downstream_detection_fullprep --allow-prepare
```

Run the current validation-scale detector comparison. This compares raw imagery against default Bundle A, Bundle D, and a conservative Bundle A variant:

```powershell
pip install ultralytics
.\.venv\Scripts\python.exe scripts/run_detection_baseline.py --dataset ssdd --variants raw bundle_a bundle_d bundle_a_conservative --limit-per-split 64 --mode all --epochs 2 --imgsz 416 --batch 4 --workers 0 --output-root outputs\downstream_detection_validation_trained --allow-prepare --allow-train --allow-eval
.\.venv\Scripts\python.exe scripts/run_detection_baseline.py --dataset hrsid --variants raw bundle_a bundle_d bundle_a_conservative --limit-per-split 64 --mode all --epochs 2 --imgsz 416 --batch 4 --workers 0 --output-root outputs\downstream_detection_validation_trained --allow-prepare --allow-train --allow-eval
```

For a stronger full experiment, start from `configs/downstream/yolo_baseline.yaml` and deliberately increase the split size and epochs:

```powershell
.\.venv\Scripts\python.exe scripts/run_detection_baseline.py --config configs/downstream/yolo_medium.yaml --dataset ssdd --mode all --allow-prepare --allow-train --allow-eval
.\.venv\Scripts\python.exe scripts/run_detection_baseline.py --config configs/downstream/yolo_medium.yaml --dataset hrsid --mode all --allow-prepare --allow-train --allow-eval
```

Bundle profiles can also be injected into detector preparation/training:

```powershell
.\.venv\Scripts\python.exe scripts/run_detection_baseline.py `
  --config configs/downstream/yolo_medium.yaml `
  --dataset ssdd `
  --variants raw bundle_b bundle_d `
  --bundle-b-config configs/bundles/profiles/bundle_b_conservative.yaml `
  --bundle-d-config configs/bundles/profiles/bundle_d_conservative.yaml `
  --mode all `
  --allow-prepare --allow-train --allow-eval
```

What this means:
- `mode prepare` is a real dataset-conversion check but does not report mAP.
- `mode all` trains/evaluates a compact YOLO baseline and writes mAP / precision / recall / F1 when Ultralytics completes.
- `--limit-per-split none` uses all available records for the selected dataset/variant. Use it deliberately; it is meant for preparation checks or longer training sessions.
- `--workers 0` is the safest Windows default and avoids local multiprocessing/cache permission failures.
- SSDD and HRSID are the supported public ship-detection datasets for this first downstream path.
- This detector path is separate from proxy-only bundle screening.
- Current detector findings should be read as validation evidence, not a tuned SOTA claim. The key finding so far is not "Bundle A is bad forever"; it is that this detector setup appears to depend on raw edge/texture cues that default conditioning suppresses.

Detector outputs:
- `outputs/downstream_detection_validation_trained/metrics/downstream_comparison.csv`
- `outputs/downstream_detection_validation_trained/metrics/variant_deltas.csv`
- `outputs/downstream_detection_validation_trained/metrics/diagnostic_summary.csv`
- `outputs/downstream_detection_validation_trained/tables/diagnostic_summary.md`

Generate the comparison pack:

```powershell
python scripts/export_figures.py --bundles bundle_a bundle_b bundle_c bundle_d
```

Build the Streamlit demo index from existing visual outputs:

```powershell
python scripts/build_demo_index.py --max-examples 8
```

### Paired denoising evidence

Run the paired Mendeley validation track:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_denoising_quality.py --dataset mendeley --input-root "data/raw/Mendeley SAR dataset" --split val --variants raw,bundle_a,bundle_a_conservative,bundle_b,bundle_d --max-samples 100 --output-root outputs/denoising_quality
```

Generate presentation panels:

```powershell
.\.venv\Scripts\python.exe scripts/make_denoising_panels.py --output-root outputs\denoising_quality --max-panels 12
```

Build final report figures and public/private result summaries from cached artifacts:

```powershell
.\.venv\Scripts\python.exe scripts/make_final_figures.py
```

Generate the public-safe summary pack:

```powershell
python scripts/generate_handoff_pack.py --surface public
```

Check the intended public export surface:

```powershell
python scripts/check_repo_surface.py --surface public
```

### Final intentional sweep

Dry-run the frozen final sweep before any expensive rerun:

```powershell
.\.venv\Scripts\python.exe scripts/run_final_sweep.py --dry-run
```

Dry-run metadata is written to `outputs/final_sweep/final_sweep_summary_dry_run.json` so it does not overwrite the completed sweep summary.

The completed final sweep artifacts live under `outputs/final_sweep/`. Only rerun the heavy sweep intentionally if you are replacing those artifacts:

```powershell
.\.venv\Scripts\python.exe scripts/run_final_sweep.py
```

Final-sweep references:
- `configs/final_sweep.yaml`
- `docs/RESULTS_SUMMARY.md`
- `docs/FINAL_RESULTS_INTERPRETATION.md`

### Streamlit app

Launch the public-safe local result browser:

```powershell
$env:SAR_APP_SURFACE="public"
.\.venv\Scripts\streamlit.exe run apps/streamlit_app.py
```

Local launch note:
- the app still bootstraps the repo root itself for direct script execution
- editable install via `pip install -e ".[app,dev]"` is the cleanest way to use the repo
- if `SAR_APP_SURFACE` is unset, the app defaults to `public`

### Dataset audit

Regenerate the dataset audit:

```powershell
python scripts/audit_datasets.py --preview-count 3
```

## Sentinel-1 Workflow

GRD dry-run:

```powershell
python scripts/fetch_sentinel1_subset.py --config configs/datasets/sentinel1_grd_small.yaml --dry-run --force
```

Selective fetch:

```powershell
$env:CDSE_USERNAME="your_username"
$env:CDSE_PASSWORD="your_password"
python scripts/fetch_sentinel1_subset.py `
  --config configs/datasets/sentinel1_grd_small.yaml `
  --download-count 1 `
  --force
```

Prepare local SAFE content:

```powershell
python scripts/prepare_sentinel1_local.py --product-family GRD
```

Run Bundle A on Sentinel-1:

```powershell
python scripts/run_bundle_a.py `
  --config configs/bundle_a.yaml `
  --dataset sentinel1 `
  --split all `
  --sample-limit 1
```

Run the Sentinel-1 multi-scene Bundle A batch comparison:

```powershell
python scripts/sentinel1/run_bundle_a_sentinel1_batch.py --compare-submethods
```

Expand Sentinel-1 GRD evidence, prepare new scenes locally, and rerun the batch comparison:

```powershell
python scripts/sentinel1/expand_sentinel1_evidence.py --target-ready-scenes 12 --max-new-downloads 8 --force
```

If a session cannot write to `data/`, use the lightweight override cache before running Sentinel-1 inspection, expansion, or the app:

```powershell
$env:SAR_DATA_LAYOUT_ROOT=(Resolve-Path "outputs\dataset_state")
```

Force one additive submethod across the currently local Sentinel-1 GRD scenes:

```powershell
python scripts/sentinel1/run_bundle_a_sentinel1_batch.py --additive-submethod A0
python scripts/sentinel1/run_bundle_a_sentinel1_batch.py --additive-submethod A1
python scripts/sentinel1/run_bundle_a_sentinel1_batch.py --additive-submethod A2
python scripts/sentinel1/run_bundle_a_sentinel1_batch.py --additive-submethod A3
```

Inspect local Sentinel-1 readiness and missing rows:

```powershell
python scripts/sentinel1/inspect_sentinel1_local.py
```

Plan the next Sentinel-1 evidence round without downloading anything:

```powershell
python scripts/sentinel1/plan_sentinel1_evidence.py --target-scene-count 10
```

Notes:
- AUX / OPOD / PREORB / EOF products are excluded unless explicitly requested.
- Zipped SAFE products are prepared into `data/interim/sentinel1/prepared/`.
- Bundle A uses a memory-safe COG overview for very large local GRD products.
- Some Copernicus COG TIFFs need `imagecodecs` for local decode support.
- Sentinel-1 batch outputs are written under `outputs/bundle_a_sentinel1_batch/` with compact scene summaries, submethod comparisons, and per-scene recommendations.

## Output Layout

New runs use a standardized structure:

- `config/`
- `metrics/`
- `plots/`
- `galleries/`
- `statistics/`
- `tables/`
- `logs/`

Bundle A writes the main human-readable tables under:
- `results/bundle_a/tables/sample_summary.csv`
- `results/bundle_a/tables/run_overview.csv`
- `results/bundle_a/tables/run_summary.md`
- `results/bundle_a/tables/submethod_summary.csv`
- `results/bundle_a/tables/submethod_aggregate.csv`

Bundle A metrics and machine-readable summaries live under:
- `results/bundle_a/metrics/run_summary.json`
- `results/bundle_a/metrics/topline_metrics.json`
- `results/bundle_a/metrics/per_sample_metrics.csv`
- `results/bundle_a/metrics/aggregate_metrics.csv`

Bundle A statistical baseline outputs live under:
- `results/bundle_a/statistics/`

Important honesty rule:
- these are Stage-1 screening and proxy-evaluation outputs
- they are not real detector mAP claims unless a real downstream detector path is actually wired

Downstream detector artifacts live under:
- `outputs/downstream_detection/<dataset>/prepared/`
- `outputs/downstream_detection/<dataset>/metrics/downstream_comparison.csv`
- `outputs/downstream_detection/<dataset>/metrics/variant_deltas.csv`
- `outputs/downstream_detection/metrics/downstream_comparison.csv`
- `outputs/downstream_detection/metrics/variant_deltas.csv`
- `outputs/downstream_detection/metrics/run_summary.json`

## Streamlit App Pages

The local Streamlit app supports:
- Start Here page
- public-safe mode by default
- about / glossary page
- overview page
- bundle results page
- visual comparison page
- demo / try-a-scene page
- downstream detection page
- Bundle A submethod page
- statistics page
- Sentinel-1 readiness page
- dataset audit page

The app is designed as a decision cockpit rather than a raw data browser:
- conclusions first
- key metrics second
- visuals third
- raw JSON behind expanders

The Sentinel-1 page also includes a Bundle A scene-comparison section that shows:
- all currently usable GRD scenes
- which A0 / A1 / A2 / A3 runs were completed
- per-scene recommendations
- metadata availability and overview-only warnings
- confidence/evidence labels and decision-basis notes

The demo page uses existing local output images. It does not upload data, train a model, or create new claims; it simply curates representative before/after/difference panels for quick explanation.

The app handles missing data gracefully and reads both the newer standardized layout and older legacy result folders.

## External Or Local Data

The registry-based dataset workflow remains intact for future external or local datasets.

External/local data may lack noise XML, have partial metadata, live outside the repo, or arrive with COCO/YOLO/bounding-box CSV annotations. That is expected. The repo supports registry-based dataset handling, custom local registration, external path registration, and later data additions without large bundle refactors.

Register a future local dataset:

```powershell
python scripts/register_local_dataset.py `
  --dataset-name local_ship_detection_v1 `
  --path "C:\path\to\dataset_root" `
  --owner "local team" `
  --remote-source "external/local source" `
  --source-access local `
  --pixel-domain intensity `
  --annotation-match stem
```

Validate a future external detection dataset before trying to run bundles or detectors:

```powershell
python scripts/validate_external_detection_dataset.py --config configs/datasets/external_detection_template.yaml
```

The external detection adapter currently supports `COCO`, `YOLO`, and simple bounding-box CSV-style manifests. It validates paths, image counts, annotation presence, and box counts so future dataset handoffs can fail loudly instead of quietly producing empty detector runs.

## Optional External Methods

These remain optional integrations, not hard repo requirements:

| Method | Bundle | Status in this repo |
| --- | --- | --- |
| `BM3D` | `bundle_b` | optional |
| `MERLIN` | `bundle_c` | optional |
| `Speckle2Void` | `bundle_d` | optional |

Example optional install:

```powershell
pip install bm3d
```

## GitHub Readiness

The repo is structured to stay GitHub-ready:
- raw data is ignored
- prepared SAFE caches and large outputs are ignored
- configs, manifests, docs, code, tables, and lightweight JSON/CSV summaries stay commit-safe

This repo should contain reproducible code, configs, manifests, audit outputs, result tables, and small summary artifacts, not benchmark archives or heavyweight local caches.
