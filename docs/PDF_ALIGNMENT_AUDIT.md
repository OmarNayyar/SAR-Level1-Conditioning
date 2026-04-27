# PDF Alignment Audit

Source brief: `SAR Stage-1 Conditioning Screening Report for Downstream Semantic Segmentation and Maritime Object Analysis`.

This audit checks whether the repo still implements the original Stage-1 conditioning brief rather than drifting into a generic denoising demo. Verdict: the repo is aligned with the brief, with two deliberate limitations. Bundle C remains a future SLC route because no genuine complex SLC validation set is available, and closed-data/operational claims remain deferred until representative partner data are tested.

## Alignment Matrix

| Brief requirement | Repo status | Files that satisfy it | Notes |
| --- | --- | --- | --- |
| Stage-1 positioning after focusing/basic calibration and before downstream AI | Matched | `README.md`, `apps/streamlit_app.py`, `docs/BUNDLE_METHOD_GUIDE.md`, `docs/FINAL_RESULTS_INTERPRETATION.md` | The repo frames conditioning as a screening layer, not an automatic replacement for raw imagery. |
| Additive, multiplicative, and combined SAR noise model | Matched | `README.md`, `docs/BUNDLE_METHOD_GUIDE.md`, `src/stage1/additive/`, `src/stage1/multiplicative/`, `src/bundles/` | Bundle implementations keep additive and multiplicative stages explicit. |
| Domain labels: complex SLC, amplitude, intensity/power, log-domain | Matched | `README.md`, `docs/BUNDLE_METHOD_GUIDE.md`, `configs/bundle_*.yaml`, `src/stage1/pipeline.py` | Public docs explicitly distinguish detected intensity/log-intensity routes from the future complex SLC route. |
| Metadata-driven thermal/noise-vector correction and NESZ/noise-floor logic | Matched with public-data caveat | `src/stage1/additive/thermal_noise_subtract.py`, `src/stage1/additive/intensity_floor_estimate.py`, `src/stage1/additive/bundle_a_submethods.py`, `configs/bundle_a.yaml` | The code supports metadata-first routing, but most public chips lack product noise vectors. Image-fitted fallback is therefore expected. |
| Empirical additive destriping / low-rank / frequency-style correction | Matched | `src/stage1/additive/destripe_lowrank_sparse.py`, `src/bundles/bundle_b_noiseaware.py`, `configs/bundle_b.yaml` | Used as the structured-artifact component of Bundle B. |
| Starlet/radio-astronomy-inspired complex denoising | Matched as SLC-ready path | `src/stage1/additive/starlet_complex_denoise.py`, `src/bundles/bundle_c_selfsupervised.py`, `configs/bundle_c.yaml` | Present, but not claim-grade until genuine SLC data are available. |
| Plug-and-play ADMM additive cleanup | Matched | `src/stage1/additive/pnp_admm_additive.py`, `src/bundles/bundle_d_inverse_problem.py`, `configs/bundle_d.yaml` | Used in Bundle D for metadata-poor intensity workflows. |
| Lee / refined Lee / Gamma-MAP style speckle filtering family | Partly matched | `src/stage1/multiplicative/refined_lee.py`, `src/bundles/bundle_a_classical.py`, `docs/BUNDLE_METHOD_GUIDE.md` | Refined Lee is implemented. Gamma-MAP remains a documented family reference, not a separate tuned implementation. |
| MuLoG / BM3D / log-domain despeckling | Matched with optional dependency fallback | `src/stage1/multiplicative/mulog_bm3d.py`, `src/bundles/bundle_b_noiseaware.py`, `configs/bundle_b.yaml` | Bundle B is the current strongest paired denoising route on Mendeley validation. |
| MERLIN / SLC self-supervised path | Matched as future path | `src/stage1/multiplicative/merlin_wrapper.py`, `src/bundles/bundle_c_selfsupervised.py`, `docs/DATA_DOWNLOAD_GUIDE_DENOISING.md` | The repo does not overclaim MERLIN results without SLC data. |
| Speckle2Void / blind-spot intensity path | Matched with fallback | `src/stage1/multiplicative/speckle2void_wrapper.py`, `src/bundles/bundle_d_inverse_problem.py`, `configs/bundle_d.yaml` | Used as the intensity-only self-supervised candidate in Bundle D. |
| Four coherent bundles A/B/C/D | Matched | `src/bundles/`, `scripts/run_bundle_*.py`, `configs/bundle_*.yaml`, `docs/BUNDLE_METHOD_GUIDE.md` | Bundle names and roles match the original brief. |
| Mendeley paired denoising strategy | Matched | `src/datasets/mendeley_despeckling.py`, `scripts/evaluate_denoising_quality.py`, `scripts/make_denoising_panels.py`, `docs/RESULTS_SUMMARY.md` | Main denoising-quality evidence path. |
| SSDD/HRSID detector compatibility strategy | Matched | `scripts/run_final_sweep.py`, `scripts/run_detection_baseline.py`, `src/downstream/detection/`, `docs/RESULTS_SUMMARY.md` | Current detector result favors raw, framed only as YOLO compatibility evidence. |
| Sentinel-1 GRD/SLC readiness | Matched as preparation path | `scripts/download_sentinel1_samples.py`, `scripts/prepare_sentinel1_local.py`, `src/datasets/sentinel1_*.py`, `docs/DATA_DOWNLOAD_GUIDE_DENOISING.md` | GRD is ready for small additive/noise-floor demos; full SLC validation remains pending. |
| Representative partner-data request | Matched without public-surface leakage | `docs/DATA_DOWNLOAD_GUIDE_DENOISING.md`, `docs/PUBLIC_MEETING_BRIEF.md`, non-public handoff docs excluded from public manifest | Public docs ask for representative product characteristics without exposing internal operational language. |
| Decision matrix by ingest type and metadata availability | Matched | `docs/BUNDLE_METHOD_GUIDE.md`, `apps/streamlit_app.py`, `results/public/bundle_matrix.csv` | The routing table separates raw detector baseline, Bundle B denoising, Bundle D structure preservation, Bundle A screening, and Bundle C SLC future work. |

## Fixed During Final Alignment

- Public docs now include this audit so reviewers can connect the repo back to the original Stage-1 screening report.
- Public-facing wording avoids the misleading shorthand that detector results prove denoising quality.
- Bundle A is framed as the interpretable metadata/noise-floor screening family, not as the best operational detector baseline.
- Bundle C is explicitly described as the planned SLC route pending SLC access, not as a completed SLC result.
- The public manifest was tightened to prioritize core docs over session handoff notes.

## Deferred Items

| Deferred item | Why it is deferred |
| --- | --- |
| Full SLC/MERLIN validation | Requires genuine complex SLC samples or burst-level access. |
| Separate Gamma-MAP implementation | Refined Lee covers the committed classical speckle-filter path; Gamma-MAP can be added later if representative data justify it. |
| Closed-data operational recommendation | Requires representative private/internal SAR products, metadata/noise vectors, and downstream task details. |
| Detector retuning on conditioned imagery | Current detector evidence is compatibility screening only; a tuned detector pass would be a separate experiment. |
