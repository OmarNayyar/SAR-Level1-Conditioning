# Git Release Checklist

Use this before the first public commit.

## Current State

- Repo display title: SAR Stage-1 Conditioning: Denoising and Validation for Maritime SAR Imagery.
- Exact repo folder/name: `SAR-Stage1-Conditioning`.
- Public repo posture: open-source style public-data validation and portfolio/research repo.
- First commit message: `Finalize SAR Stage-1 conditioning validation pipeline`.

## Must Pass

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_repo_surface.py --surface public
.\.venv\Scripts\python.exe scripts/check_repo_surface.py --surface private
.\.venv\Scripts\python.exe scripts/run_final_sweep.py --dry-run
.\.venv\Scripts\python.exe scripts/evaluate_denoising_quality.py --dataset mendeley --input-root "data/raw/Mendeley SAR dataset" --split val --variants raw,bundle_a,bundle_a_conservative,bundle_b,bundle_d --max-samples 5 --output-root outputs/denoising_quality_smoke --force
```

## Safe To Commit

- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `docs/`
- `manifests/`
- `results/public/*.md`
- `results/public/*.json`
- `results/public/*.csv`
- `results/handoff/*.md`
- `results/handoff/*.json`
- `results/handoff/*.csv`
- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`

## Do Not Commit

- `data/raw/`
- `data/external/`
- `data/interim/`
- `data/processed/`
- `outputs/`
- `runs/`
- `.venv/`
- `.codex_tmp/`
- `.idea/`
- `*.pt`, `*.pth`, `*.onnx`
- Earthdata cookies, `.netrc`, `_netrc`, `.urs_cookies`, or any credential file

## Human Checks Before Publishing

- Choose and add the license.
- Rotate Earthdata credentials if they were exposed in any screenshot or log.
- Review the public README and `results/public/project_summary.md` for tone.
- Decide which ignored PNG figures from `outputs/final_figures/` should be embedded in the paper/report.
- Confirm the public repo does not include credentials, raw data, model weights, non-public notes, or heavyweight generated outputs.
