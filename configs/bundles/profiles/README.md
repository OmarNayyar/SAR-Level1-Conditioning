# Bundle Tuning Profiles

Profiles make bundle severity explicit without code edits.

| Profile | Meaning | Typical Use |
| --- | --- | --- |
| `conservative` | Prioritize detector edge/texture preservation. | First choice for downstream detection validation. |
| `balanced` | Default screening behavior. | Proxy comparison and standard bundle runs. |
| `aggressive` | Strong cleanup stress test. | Only when contamination is obvious; validate against raw. |

Run a profile with the normal bundle scripts:

```powershell
.\.venv\Scripts\python.exe scripts/run_bundle_a.py --config configs/bundles/profiles/bundle_a_conservative.yaml
.\.venv\Scripts\python.exe scripts/run_bundle_b.py --config configs/bundles/profiles/bundle_b_balanced.yaml
.\.venv\Scripts\python.exe scripts/run_bundle_d.py --config configs/bundles/profiles/bundle_d_conservative.yaml
```

Use `scripts/run_detection_baseline.py --bundle-*-config ...` when a detector run should use one of these profiles for a conditioned variant.
