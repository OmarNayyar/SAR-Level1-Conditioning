# Data Download Guide For Denoising Evidence

## Already Local

Do not download the Mendeley SAR despeckling dataset again.

Use:

```text
data/raw/Mendeley SAR dataset/
```

Expected folders:

```text
GTruth/
GTruth_val/
Noisy/
Noisy_val/
```

## Credentials

Do not hardcode Earthdata or Copernicus credentials.

If Earthdata credentials were exposed in a screenshot, terminal log, or shared chat, treat them as compromised and rotate them before using them again.

Use environment variables:

```powershell
$env:EARTHDATA_USERNAME="your_username"
$env:EARTHDATA_PASSWORD="your_password"
```

If you used `setx`, close/reopen the terminal or set the current session variables as shown above before running a download command.

Optional future Copernicus variables:

```powershell
$env:CDSE_USERNAME="your_username"
$env:CDSE_PASSWORD="your_password"
```

## Sentinel-1 GRD Cross-Pol Search

Use this for additive thermal noise, banding, scalloping, and cross-pol behavior demos.

Dry-run search only:

```powershell
.\.venv\Scripts\python.exe scripts/download_sentinel1_samples.py --kind grd-crosspol --provider asf --max-results 1 --dry-run
```

Real download is intentionally gated by Earthdata credentials and size checks:

```powershell
.\.venv\Scripts\python.exe scripts/download_sentinel1_samples.py --kind grd-crosspol --provider asf --max-results 1 --output-root data/external/sentinel1_grd_crosspol
```

Current finalization note: dry-run search succeeded with a recent Sentinel-1 IW `VV+VH` GRD candidate under 1.5 GB. The real download was attempted, but this local workspace denied creation under `data/external/sentinel1_grd_crosspol` before any product bytes were written. This is a local filesystem/storage-permission issue, not a credential or ASF search failure.

## Sentinel-1 SLC Burst Search

Use this for future complex-domain and MERLIN-style work.

Dry-run search only:

```powershell
.\.venv\Scripts\python.exe scripts/download_sentinel1_samples.py --kind slc-burst --provider asf --max-results 1 --dry-run
```

SLC burst search is automated. Direct burst download is left as a manual fallback until the exact burst product and ASF/asf_search handling are confirmed.

Manual fallback:

- use the printed ASF candidate metadata
- open the candidate in ASF Vertex
- or use `asf_search` with Earthdata credentials

Official ASF Search API reference:

- https://docs.asf.alaska.edu/api/basics/
- https://docs.asf.alaska.edu/api/keywords/

## Safety Rules

- Do not download huge Sentinel-1 products by default.
- Use `--dry-run` first.
- Keep raw downloaded samples under `data/external/` or another ignored/external path.
- Do not commit raw data, credentials, or downloaded products.
