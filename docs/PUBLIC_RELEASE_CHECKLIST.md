# Public Release Checklist

Use this before exporting or publishing the public repository.

## Required Before Public Export

- [x] Run `scripts/check_repo_surface.py --surface public`.
- [x] Confirm no private/client-specific language appears in public files.
- [x] Confirm restricted handoff docs are excluded.
- [x] Confirm raw data, prepared datasets, model outputs, and local caches are excluded.
- [x] Confirm public docs use public datasets only: SSDD, HRSID, Mendeley, Sentinel-1 examples, and clearly labeled smoke/metadata-only datasets.
- [x] Confirm raw imagery is described as the current detector baseline for the YOLO experiment.
- [x] Confirm Bundle A is described as an interpretable conditioning family, not the operational detector baseline.
- [x] Confirm paired denoising improvements are not described as detector gains.
- [ ] Confirm authored report/paper assets are safe to redistribute.
- [ ] Confirm a license has been chosen before public release.

## Final Public Artifacts

- `docs/RESULTS_SUMMARY.md`
- `results/public/project_summary.md`
- `results/public/project_summary.json`
- `results/public/final_denoising_metrics.csv`
- `results/public/final_detector_metrics.csv`
- `outputs/final_figures/` for local figure export; figures are intentionally ignored by git.

## Still Wait For Human Review

- final LinkedIn post wording
- license choice
- which generated figures to manually copy into a paper/report

## Public Export Command Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_repo_surface.py --surface public
.\.venv\Scripts\python.exe scripts/run_final_sweep.py --dry-run
```

Do not run the heavy sweep from this checklist.
