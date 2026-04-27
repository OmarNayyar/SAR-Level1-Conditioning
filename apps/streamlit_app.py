from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reporting import (
    BUNDLE_A_SUBMETHOD_DETAILS,
    BUNDLE_EVIDENCE_ROWS,
    BUNDLE_MATRIX_ROWS,
    CONFIDENCE_BADGE_COLORS,
    DATA_STRATEGY_ROWS,
    DATASET_USAGE_RECOMMENDATIONS,
    GLOSSARY_ROWS,
    TUNING_PROFILE_ROWS,
    collect_demo_examples,
    dataset_status_help_rows,
    discover_bundle_runs,
    load_bundle_run,
    load_detection_baseline_snapshot,
    load_handoff_snapshot,
    load_surface_pack_snapshot,
    load_sentinel1_batch_snapshot,
    load_dataset_audit_snapshot,
    load_dataset_registry_snapshot,
    load_sentinel1_manifest_rows,
    main_blocker_text,
    next_action_text,
    run_snapshot,
    safe_float,
    sentinel1_readiness_text,
    shorten_path,
    strongest_bundle_name,
    strongest_public_chip_submethod,
)


st.set_page_config(page_title="SAR Stage-1 Conditioning", layout="wide")


APP_SURFACE = os.environ.get("SAR_APP_SURFACE", "public").strip().lower()
if APP_SURFACE == "handoff":
    APP_SURFACE = "private"
if APP_SURFACE not in {"public", "private"}:
    APP_SURFACE = "public"


BUNDLE_METHOD_ROWS = [
    {
        "Bundle": "A",
        "Additive": "A0 / A1 / A2 / A3 family",
        "Multiplicative": "Refined Lee",
        "Domain": "Intensity",
        "Maturity": "Interpretable screening family",
        "Best use case": "Explainable additive/noise-floor routing and ablation against raw",
    },
    {
        "Bundle": "B",
        "Additive": "Structured cleanup / destriping",
        "Multiplicative": "MuLoG-style denoising",
        "Domain": "Log-intensity",
        "Maturity": "Best paired denoising result",
        "Best use case": "Paired/intensity-domain denoising when PSNR/SSIM/MSE matter",
    },
    {
        "Bundle": "C",
        "Additive": "Starlet shrinkage",
        "Multiplicative": "MERLIN wrapper / fallback",
        "Domain": "Complex SLC preferred",
        "Maturity": "Feasibility only",
        "Best use case": "Demo path when genuine complex SLC support exists",
    },
    {
        "Bundle": "D",
        "Additive": "PnP-ADMM additive cleanup",
        "Multiplicative": "Speckle2Void wrapper / fallback",
        "Domain": "Intensity / log-intensity",
        "Maturity": "Structure-preserving candidate",
        "Best use case": "Metadata-poor scenes where SSIM/edge preservation are priorities",
    },
]


BUNDLE_ROUTING_ROWS = [
    {
        "Situation": "Need current detector baseline",
        "Use first": "Raw",
        "Why": "Best mAP in current SSDD/HRSID YOLO sweep.",
        "Caveat": "Detector-specific; not a denoising-quality claim.",
    },
    {
        "Situation": "Need paired denoising quality",
        "Use first": "Bundle B",
        "Why": "Best Mendeley validation PSNR/SSIM/MSE among tested variants.",
        "Caveat": "Validate on representative SAR products before operational adoption.",
    },
    {
        "Situation": "Need interpretable screening",
        "Use first": "Bundle A / A conservative",
        "Why": "Clear additive/noise-floor routing and milder variants.",
        "Caveat": "Not the best detector baseline in current evidence.",
    },
    {
        "Situation": "Need structure preservation",
        "Use first": "Bundle D",
        "Why": "Strong SSIM and edge preservation on paired validation.",
        "Caveat": "Still trails raw in current detector mAP.",
    },
    {
        "Situation": "Need complex-domain/SLC path",
        "Use first": "Bundle C",
        "Why": "Designed for future MERLIN-style SLC experiments.",
        "Caveat": "Do not claim until genuine SLC data are available.",
    },
]

STATUS_BADGE_COLORS = {
    "full": "#166534",
    "complete": "#166534",
    "partial": "#0f766e",
    "metadata-only": "#92400e",
    "external-linked": "#2563eb",
    "missing": "#6b7280",
    "ready": "#166534",
    "failed": "#b91c1c",
}


def _bundle_label(bundle_name: str) -> str:
    return bundle_name.replace("_", " ").title()


def _as_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _run_label(run: dict[str, Any]) -> str:
    timestamp = datetime.fromtimestamp(float(run["modified_timestamp"])).strftime("%Y-%m-%d %H:%M")
    return f"{_bundle_label(run['bundle_name'])} | {run['dataset']} | processed={run['processed_count']} | {timestamp}"


def _load_runs() -> list[dict[str, Any]]:
    return [asdict(run) for run in discover_bundle_runs(REPO_ROOT)]


def _load_run_payload(run: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return load_bundle_run(Path(run["output_root"]))
    except Exception as exc:  # pragma: no cover - UI safety
        st.error(f"Could not load run data from {run['output_root']}: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return None


def _page_intro(title: str, description: str, how_to_read: str) -> None:
    st.title(title)
    st.write(description)
    st.caption("How to read this: " + how_to_read)


def _help_expander(title: str, text: str) -> None:
    with st.expander(title):
        st.write(text)


def _show_text_card(title: str, value: str) -> None:
    st.markdown(f"**{title}**")
    st.write(value)


def _badge_html(label: str, color: str) -> str:
    return (
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;"
        f"background:{color}18;color:{color};border:1px solid {color}55;font-size:0.85rem;font-weight:600;'>"
        f"{label}</span>"
    )


def _status_badge(label: str) -> str:
    color = STATUS_BADGE_COLORS.get(label.lower(), "#374151")
    return _badge_html(label, color)


def _render_badge_line(label: str, color: str) -> None:
    st.markdown(_badge_html(label, color), unsafe_allow_html=True)


def _format_number(value: Any, precision: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{precision}f}"


def _format_delta(value: Any, precision: int = 3) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:+.{precision}f}"


def _show_image_if_exists(path_text: str, caption: str) -> None:
    path = Path(path_text) if path_text else None
    if path and path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"No image available for {caption}.")


def _raw_json_expander(title: str, payload: Any) -> None:
    with st.expander(title):
        st.json(payload)


def _dataset_recommendation(dataset_name: str) -> str:
    return DATASET_USAGE_RECOMMENDATIONS.get(dataset_name, "Use according to registry and manifest readiness.")


def _run_metric_cards(snapshot: dict[str, Any]) -> None:
    columns = st.columns(6)
    columns[0].metric("Samples", str(snapshot.get("processed_count", 0)))
    columns[1].metric("Additive routing", snapshot.get("additive_submethod_label", "n/a"))
    columns[2].metric("Proxy ENL gain", _format_delta(snapshot.get("proxy_enl_gain")))
    columns[3].metric("Edge sharpness delta", _format_delta(snapshot.get("edge_sharpness_delta")))
    columns[4].metric("Separability delta", _format_delta(snapshot.get("distribution_separability_delta")))
    columns[5].metric("Threshold F1 delta", _format_delta(snapshot.get("threshold_f1_delta")))
    if snapshot.get("evidence_grade") or snapshot.get("decision_basis"):
        st.caption(f"Evidence grade: {snapshot.get('evidence_grade', 'n/a')}")
        with st.expander("How these decision metrics are used"):
            st.write(
                snapshot.get(
                    "decision_basis",
                    "Scores are proxy screening heuristics. They help compare runs but are not trained detector or segmentation metrics.",
                )
            )


def _friendly_sample_rows(rows: list[dict[str, Any]], compact: bool = True) -> list[dict[str, Any]]:
    friendly_rows: list[dict[str, Any]] = []
    for row in rows:
        friendly_row = {
            "Dataset": row.get("dataset", ""),
            "Sample": row.get("sample_id", ""),
            "Split": row.get("split", ""),
            "Additive submethod": row.get("additive_submethod_name", row.get("additive_submethod_code", "")),
            "Metadata available": "Yes" if row.get("metadata_available", row.get("additive_metadata_available")) else "No",
            "Correction applied": "Yes" if row.get("additive_correction_applied", row.get("additive_applied")) else "No",
            "Confidence": row.get("confidence_level", row.get("additive_confidence_level", "")),
            "ENL before": row.get("proxy_enl_before", ""),
            "ENL after": row.get("proxy_enl_after", ""),
            "Edge before": row.get("edge_sharpness_before", ""),
            "Edge after": row.get("edge_sharpness_after", ""),
            "Separability before": row.get("separability_before", row.get("distribution_separability_before", "")),
            "Separability after": row.get("separability_after", row.get("distribution_separability_after", "")),
            "Threshold F1 before": row.get("threshold_f1_before", ""),
            "Threshold F1 after": row.get("threshold_f1_after", ""),
            "Notes": row.get("notes_warnings", row.get("source_note", "")),
        }
        if not compact:
            friendly_row["Metadata fields"] = row.get("metadata_fields_present", row.get("additive_metadata_fields_present", ""))
            friendly_row["Fallback"] = row.get("fallback_used", row.get("additive_fallback_used", ""))
        friendly_rows.append(friendly_row)
    return friendly_rows


def _friendly_sentinel1_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    friendly_rows: list[dict[str, Any]] = []
    for row in rows:
        friendly_rows.append(
            {
                "Scene": row.get("scene_id", ""),
                "Submethod": row.get("additive_submethod_used", ""),
                "Requested": row.get("requested_additive_submethod", ""),
                "Metadata available": "Yes" if row.get("metadata_available") else "No",
                "Overview fallback": "Yes" if row.get("overview_fallback_used") else "No",
                "ENL gain": row.get("proxy_enl_gain", ""),
                "Edge delta": row.get("edge_sharpness_delta", ""),
                "Separability delta": row.get("distribution_separability_delta", ""),
                "Threshold F1 delta": row.get("threshold_f1_delta", ""),
                "Decision score": row.get("decision_score", ""),
                "Confidence": row.get("decision_confidence", ""),
                "Warnings": row.get("warnings", ""),
            }
        )
    return friendly_rows


def _candidate_groups(sample_index: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> dict[str, list[str]]:
    scored: list[tuple[str, float]] = []
    for sample_id, rows in sample_index.items():
        scores = [safe_float(sample.get("bundle_quality_score")) for _, sample in rows]
        valid_scores = [score for score in scores if score is not None]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        scored.append((sample_id, mean_score))
    scored.sort(key=lambda item: item[1])
    all_ids = [sample_id for sample_id, _ in scored]
    worst = all_ids[: min(5, len(all_ids))]
    best = list(reversed(all_ids[-min(5, len(all_ids)) :]))
    if all_ids:
        midpoint = len(all_ids) // 2
        representative = all_ids[max(0, midpoint - 2) : midpoint + 3]
    else:
        representative = []
    return {
        "Best cases": best,
        "Representative cases": representative,
        "Worst cases": worst,
        "All samples": all_ids,
    }


def _latest_run_by_bundle(runs: list[dict[str, Any]], bundle_name: str) -> dict[str, Any] | None:
    for run in runs:
        if run["bundle_name"] == bundle_name:
            return run
    return None


def _load_json_rows(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get(key, [])


def _read_csv_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - UI safety
        st.warning(f"Could not read {shorten_path(path.as_posix())}: {exc}")
        return pd.DataFrame()


def _first_existing_root(*roots: Path) -> Path | None:
    for root in roots:
        if root.exists():
            return root
    return None


def _show_public_result_figure(filename: str, caption: str) -> None:
    root = _first_existing_root(REPO_ROOT / "results" / "public" / "figures", REPO_ROOT / "outputs" / "final_figures")
    path = root / filename if root else None
    if path is not None and path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not available yet: `{filename}`. Run `python scripts/make_final_figures.py`.")


def _load_sentinel1_batch_payload() -> dict[str, Any] | None:
    payload = load_sentinel1_batch_snapshot(REPO_ROOT)
    if payload.get("topline_metrics") or payload.get("scene_summary", {}).get("scenes"):
        return payload
    return None


def _overview_takeaway_cards(runs: list[dict[str, Any]], sentinel_rows: list[dict[str, Any]]) -> None:
    cards = st.columns(5)
    with cards[0]:
        _show_text_card(
            "Current strongest route",
            "Raw for detector validation" if APP_SURFACE == "public" else strongest_bundle_name(),
        )
    with cards[1]:
        _show_text_card(
            "Best interpretable screening route",
            "Bundle A / A2 fallback" if APP_SURFACE == "public" else strongest_public_chip_submethod(),
        )
    with cards[2]:
        _show_text_card("Sentinel-1 readiness", sentinel1_readiness_text(sentinel_rows))
    with cards[3]:
        _show_text_card("Current main blocker", main_blocker_text(sentinel_rows))
    with cards[4]:
        _show_text_card(
            "Next recommended action",
            "Run medium detector comparisons and validate conditioned variants against raw."
            if APP_SURFACE == "public"
            else next_action_text(),
        )


def page_about() -> None:
    _page_intro(
        "About This Project",
        "This app explains SAR Level-1 conditioning experiments as routing decisions, not as a search for one universal denoiser.",
        "Use this page as the glossary. The rest of the app shows evidence, caveats, and visuals for the current routing choices.",
    )
    st.subheader("What Level-1 conditioning means here")
    st.write(
        "In this repo, Level-1 conditioning means practical image preparation after acquisition/focusing/calibration choices and before downstream AI. "
        "Each bundle keeps additive-noise handling separate from multiplicative speckle handling unless a method explicitly addresses both."
    )
    st.subheader("Method summary")
    st.dataframe(pd.DataFrame(BUNDLE_METHOD_ROWS), width="stretch", hide_index=True)
    st.subheader("Glossary")
    st.dataframe(pd.DataFrame(GLOSSARY_ROWS), width="stretch", hide_index=True)
    st.info(
        "Important honesty rule: current bundle outputs are screening evidence. They become claim-grade only when a real detector or segmenter evaluation path is wired and reported separately."
    )


def page_final_results() -> None:
    _page_intro(
        "Final Results",
        "A compact readout of the current public-data evidence: paired denoising quality and detector compatibility are separate tracks.",
        "Use denoising metrics to judge denoising quality. Use detector mAP only to judge compatibility with this YOLO setup.",
    )

    public_root = REPO_ROOT / "results" / "public"
    denoising = _read_csv_table(public_root / "final_denoising_metrics.csv")
    detector = _read_csv_table(public_root / "final_detector_metrics.csv")

    st.subheader("One-screen conclusion")
    cards = st.columns(4)
    cards[0].metric("Paired denoising lead", "Bundle B")
    cards[1].metric("Structure-preserving candidate", "Bundle D")
    cards[2].metric("Detector baseline", "Raw")
    cards[3].metric("Evidence tracks", "2")
    st.info(
        "Bundle B improves paired Mendeley denoising metrics over raw noisy input. "
        "Raw remains best for the current lightweight YOLO detector sweep. These are not contradictory results."
    )

    st.subheader("Which route should I try first?")
    st.dataframe(pd.DataFrame(BUNDLE_ROUTING_ROWS), width="stretch", hide_index=True)

    st.subheader("Denoising evidence: Mendeley paired validation")
    if denoising.empty:
        st.warning("No public denoising metrics found. Run `python scripts/make_final_figures.py` after the denoising evaluation.")
    else:
        st.dataframe(denoising, width="stretch", hide_index=True)
        fig_cols = st.columns(2)
        with fig_cols[0]:
            _show_public_result_figure("denoising_psnr.png", "PSNR: higher is better")
            _show_public_result_figure("denoising_mse.png", "MSE: lower is better")
        with fig_cols[1]:
            _show_public_result_figure("denoising_ssim.png", "SSIM: higher is better")
            _show_public_result_figure("denoising_edge_preservation.png", "Edge preservation")

    st.subheader("Detector compatibility: SSDD/HRSID YOLO sweep")
    if detector.empty:
        st.warning("No public detector metrics found. Run `python scripts/make_final_figures.py` after final sweep artifacts exist.")
    else:
        st.dataframe(detector, width="stretch", hide_index=True)
        _show_public_result_figure("detector_map_comparison.png", "YOLO detector mAP comparison")
        st.caption(
            "Detector mAP says whether the detector likes the conditioned distribution. It does not by itself measure denoising quality."
        )

    st.subheader("Representative denoising panels")
    panel_root = REPO_ROOT / "outputs" / "denoising_quality" / "panels"
    panels = sorted(panel_root.glob("*_denoising_panel.png"))[:12] if panel_root.exists() else []
    if not panels:
        st.info("No local denoising panels found. Run `python scripts/make_denoising_panels.py --output-root outputs/denoising_quality --max-panels 12`.")
    else:
        selected = st.selectbox("Panel", panels, format_func=lambda path: path.name)
        st.image(str(selected), caption=selected.name, width="stretch")

    st.subheader("SAR noise in plain language")
    st.markdown(
        """
- **Additive noise** is extra signal added by the sensor/electronics or structured artifacts such as striping, banding, or thermal floor effects.
- **Multiplicative speckle** is grain-like interference that scales with the scene return and is intrinsic to coherent SAR imaging.
- **Conditioning** can improve denoising metrics while hurting detector mAP if the detector was trained on raw texture statistics.
"""
    )


def page_start_here() -> None:
    _page_intro(
        "Start Here",
        "A quick orientation for using this project as a decision-support system rather than a raw experiment browser.",
        "Read the recommendation first. Then use the bundle matrix, detector status, and data strategy to choose what to run next.",
    )
    handoff = load_surface_pack_snapshot(REPO_ROOT, APP_SURFACE)
    recommendations = handoff.get("recommendations", {})
    if APP_SURFACE == "public":
        current = recommendations.get(
            "public_recommendation",
            "Use raw imagery as the detector baseline, and evaluate conditioning routes as optional candidates against raw.",
        )
    else:
        current = recommendations.get(
            "current_recommendation",
            "Use raw imagery as the detector baseline, and only hand off a conditioning route after it beats raw on the target detector.",
        )
    st.success(current)

    cols = st.columns(3)
    cols[0].metric("Current detector baseline", "Raw")
    cols[1].metric("Interpretable conditioning family", "Bundle A")
    cols[2].metric("Conditioning candidates", "B / D")

    st.subheader("Current practical reading")
    if APP_SURFACE == "public":
        st.write(
            "This public view presents the project as a reproducible SAR conditioning research/engineering framework. "
            "The central lesson so far is that cleaner-looking imagery is not automatically better for detectors."
        )
    else:
        st.write(
            "Bundle A remains the clearest explainable screening family, but detector validation currently shows raw imagery is stronger for YOLO ship detection. "
            "Stage-1 cleanup should therefore be an opt-in candidate, not a mandatory preprocessing step."
        )

    st.subheader("Bundle roles")
    st.dataframe(pd.DataFrame(BUNDLE_MATRIX_ROWS), width="stretch", hide_index=True)

    st.subheader("What to run next")
    st.markdown(
        """
- Use `configs/downstream/yolo_medium.yaml` for the next meaningful detector run.
- Add Bundle B to detector comparisons when structured artifacts are plausible.
- Keep Sentinel-1 evidence as routing/proxy evidence until full-resolution or labeled downstream validation is available.
- Use the external dataset adapter for future restricted/local datasets before running bundles.
"""
    )


def page_ai_handoff() -> None:
    _page_intro(
        "Handoff Workspace",
        "A concise view of what is usable now, what is exploratory, and what a downstream team should validate before adopting.",
        "Treat this as the project brief for another team. Raw JSON is intentionally hidden unless you open the expander.",
    )
    handoff = load_handoff_snapshot(REPO_ROOT)
    recommendations = handoff.get("recommendations", {})
    if not recommendations:
        st.info("No handoff pack found yet. Run `python scripts/generate_handoff_pack.py` to generate it.")
        recommendations = {
            "current_recommendation": "Use raw imagery as the detector baseline until a conditioned variant beats raw on the target detector.",
            "team_can_use_immediately": [],
            "do_not_overclaim": [],
            "next_validation_steps": [],
        }

    st.subheader("Current recommendation")
    st.success(recommendations.get("current_recommendation", "No recommendation available."))

    left, middle, right = st.columns(3)
    with left:
        st.markdown("**Usable immediately**")
        for item in recommendations.get("team_can_use_immediately", []):
            st.write(f"- {item}")
    with middle:
        st.markdown("**Do not overclaim**")
        for item in recommendations.get("do_not_overclaim", []):
            st.write(f"- {item}")
    with right:
        st.markdown("**Validate next**")
        for item in recommendations.get("next_validation_steps", []):
            st.write(f"- {item}")

    st.subheader("Tuning profiles")
    st.dataframe(pd.DataFrame(handoff.get("tuning_profiles") or TUNING_PROFILE_ROWS), width="stretch", hide_index=True)

    st.subheader("Data strategy")
    st.dataframe(pd.DataFrame(handoff.get("data_strategy") or DATA_STRATEGY_ROWS), width="stretch", hide_index=True)

    if handoff.get("recommendations_markdown"):
        with st.expander("Handoff report markdown"):
            st.markdown(handoff["recommendations_markdown"])
    _raw_json_expander("Handoff raw JSON", recommendations)


def page_overview() -> None:
    _page_intro(
        "SAR Stage-1 Conditioning",
        "A public-safe research cockpit for practical Level-1 SAR conditioning experiments."
        if APP_SURFACE == "public"
        else "A decision cockpit for practical Level-1 SAR conditioning experiments across additive-noise handling, speckle handling, and honest downstream sanity checks.",
        "Read the takeaways first, then the evidence status and latest runs. Use the bundle pages only when you want to drill into details.",
    )

    runs = _load_runs()
    audit_payload = load_dataset_audit_snapshot(REPO_ROOT).get("datasets", {})
    sentinel_rows = load_sentinel1_manifest_rows(REPO_ROOT)

    st.subheader("Current Takeaways")
    _overview_takeaway_cards(runs, sentinel_rows)

    st.subheader("What this repo currently supports")
    if APP_SURFACE == "public":
        st.write(
            "The repo compares multiple SAR conditioning routes on public/locally registered data, tracks domain assumptions, "
            "and separates proxy screening from downstream detector validation."
        )
    else:
        st.write(
            "The repo compares multiple Stage-1 bundle paths, tracks which additive route was used, supports metadata-rich and metadata-poor inputs, "
            "and now includes a separate YOLO detector-baseline workflow so proxy screening and downstream validation stay clearly separated."
        )

    st.subheader("Evidence status")
    st.dataframe(pd.DataFrame(BUNDLE_EVIDENCE_ROWS), width="stretch", hide_index=True)
    _help_expander(
        "What evidence grades mean",
        "Screening-grade and proxy-only results are useful for ranking and debugging conditioning routes, but they are not final mAP/IoU claims. "
        "Feasibility-grade means the code path is runnable but the required data regime is still weak.",
    )

    latest_bundle_a = _latest_run_by_bundle(runs, "bundle_a")
    if latest_bundle_a:
        bundle_a_payload = _load_run_payload(latest_bundle_a)
        if bundle_a_payload is not None:
            snapshot = run_snapshot(bundle_a_payload)
            st.subheader("Current plan")
            st.info(
                "Raw imagery remains the current detector baseline. Bundle A is the most interpretable conditioning family. "
                f"{snapshot.get('current_recommendation', next_action_text())}"
            )
            sentinel_batch_payload = _load_sentinel1_batch_payload()
            if sentinel_batch_payload is not None:
                sentinel_topline = sentinel_batch_payload.get("topline_metrics", {})
                batch_recommendation = str(sentinel_topline.get("current_recommendation", "")).strip()
                if batch_recommendation:
                    st.caption("Sentinel-1 next action: " + batch_recommendation)

    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("Dataset readiness snapshot")
        audit_rows = []
        for dataset_name, payload in sorted(audit_payload.items()):
            audit_rows.append(
                {
                    "Dataset": dataset_name,
                    "Status": payload.get("status", ""),
                    "Total": payload.get("total_count", 0),
                    "Splits": ", ".join(f"{name}:{count}" for name, count in sorted(payload.get("split_counts", {}).items())),
                    "Recommended usage": _dataset_recommendation(dataset_name),
                }
            )
        st.dataframe(_as_dataframe(audit_rows), width="stretch", hide_index=True)
    with right:
        st.subheader("Latest runs")
        if runs:
            latest_rows = [
                {
                    "Bundle": _bundle_label(run["bundle_name"]),
                    "Dataset": run["dataset"],
                    "Processed": run["processed_count"],
                    "Skipped": run["skipped_count"],
                    "Updated": datetime.fromtimestamp(float(run["modified_timestamp"])).strftime("%Y-%m-%d %H:%M"),
                }
                for run in runs[:10]
            ]
            st.dataframe(_as_dataframe(latest_rows), width="stretch", hide_index=True)
        else:
            st.info("No bundle runs were discovered yet.")


def page_bundle_results() -> None:
    _page_intro(
        "Bundle Results",
        "Inspect a single run as a decision summary instead of a raw dump.",
        "Start with the cards and interpretation. Use the tabs only if you need deeper metrics, curated visuals, or raw payloads.",
    )
    runs = _load_runs()
    if not runs:
        st.info("No bundle runs were discovered.")
        return

    bundle_names = sorted({run["bundle_name"] for run in runs})
    selected_bundle = st.selectbox("Bundle", bundle_names, format_func=_bundle_label)
    bundle_runs = [run for run in runs if run["bundle_name"] == selected_bundle]
    dataset_names = sorted({run["dataset"] for run in bundle_runs})
    selected_dataset = st.selectbox("Dataset", dataset_names)
    dataset_runs = [run for run in bundle_runs if run["dataset"] == selected_dataset]
    selected_run = st.selectbox("Run", dataset_runs, format_func=_run_label)
    payload = _load_run_payload(selected_run)
    if payload is None:
        return

    snapshot = run_snapshot(payload)
    _run_metric_cards(snapshot)

    st.info(snapshot.get("interpretation", "No interpretation was available for this run."))
    if snapshot.get("current_recommendation"):
        st.success(snapshot["current_recommendation"])
    if snapshot.get("warnings"):
        for warning in snapshot["warnings"]:
            st.warning(warning)

    tabs = st.tabs(["Overview", "Metrics", "Visuals", "Raw JSON"])

    with tabs[0]:
        left, right = st.columns([1.0, 1.1])
        with left:
            st.markdown("**Run summary**")
            st.write(
                f"Bundle: `{_bundle_label(snapshot.get('bundle_name', ''))}`  \n"
                f"Dataset: `{snapshot.get('dataset', '')}`  \n"
                f"Samples: `{snapshot.get('processed_count', 0)}`  \n"
                f"Additive routing: `{snapshot.get('additive_submethod_label', 'Not explicit')}`  \n"
                f"Downstream status: `{snapshot.get('downstream_status', '')}`"
            )
            if snapshot.get("maturity_note"):
                st.caption(snapshot["maturity_note"])
        with right:
            markdown_summary = payload.get("run_summary_markdown", "").strip()
            if markdown_summary:
                st.markdown(markdown_summary)
            else:
                st.info("No human-readable run summary markdown was available for this run.")

    with tabs[1]:
        aggregate_rows = payload["aggregate_metrics"].get("metrics", [])
        sample_rows = payload["sample_summary"].get("samples") or payload["per_sample_metrics"].get("samples", [])
        split_options = ["all"] + sorted({str(row.get("split", "")) for row in sample_rows if row.get("split")})
        selected_split = st.selectbox("Split", split_options, key="bundle_results_split")
        if selected_split != "all":
            sample_rows = [row for row in sample_rows if str(row.get("split", "")) == selected_split]
        left, right = st.columns(2)
        with left:
            st.markdown("**Aggregate metrics**")
            st.dataframe(_as_dataframe(aggregate_rows), width="stretch", hide_index=True)
        with right:
            st.markdown("**Sample summary**")
            st.dataframe(_as_dataframe(_friendly_sample_rows(sample_rows)), width="stretch", hide_index=True)

    with tabs[2]:
        layout = payload["layout"]
        gallery_cols = st.columns(2)
        with gallery_cols[0]:
            _show_image_if_exists((layout["galleries"] / "success_gallery.png").as_posix(), "Strongest cases")
        with gallery_cols[1]:
            _show_image_if_exists((layout["galleries"] / "failure_gallery.png").as_posix(), "Weakest cases")
        st.caption("These are curated galleries from the run, intended as quick visual checks rather than exhaustive sample browsing.")

    with tabs[3]:
        _raw_json_expander("Run summary JSON", payload["summary"])
        _raw_json_expander("Topline metrics", payload.get("topline_metrics", {}))
        _raw_json_expander("Aggregate metrics JSON", payload["aggregate_metrics"])


def page_visual_comparison() -> None:
    _page_intro(
        "Visual Comparison",
        "Compare bundles or Bundle A submethod runs on the same sample with larger panels and metric deltas.",
        "Pick one dataset, then a small number of runs. Start with representative cases, then inspect best and worst cases.",
    )
    runs = _load_runs()
    if not runs:
        st.info("No bundle runs were discovered.")
        return

    dataset_names = sorted({run["dataset"] for run in runs})
    selected_dataset = st.selectbox("Dataset", dataset_names, key="visual_dataset")
    dataset_runs = [run for run in runs if run["dataset"] == selected_dataset]
    selected_runs = st.multiselect(
        "Runs to compare",
        dataset_runs,
        format_func=_run_label,
        default=dataset_runs[: min(2, len(dataset_runs))],
    )
    if not selected_runs:
        st.info("Select at least one run.")
        return
    if len(selected_runs) > 3:
        st.warning("Visual comparison works best with two or three runs at a time.")

    payloads: dict[str, dict[str, Any]] = {}
    sample_index: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for run in selected_runs:
        payload = _load_run_payload(run)
        if payload is None:
            continue
        payloads[run["output_root"]] = payload
        for sample in payload["per_sample_metrics"].get("samples", []):
            sample_index.setdefault(str(sample.get("sample_id", "")), []).append((run, sample))
    if not sample_index:
        st.info("No per-sample metrics were available for the selected runs.")
        return

    groups = _candidate_groups(sample_index)
    selected_group = st.radio("Sample group", list(groups.keys()), horizontal=True)
    group_samples = groups.get(selected_group, [])
    selected_sample = st.selectbox("Sample", group_samples or sorted(sample_index.keys()))

    st.caption("Metric deltas are reported relative to the raw input for that run.")
    for run, sample in sample_index.get(selected_sample, []):
        snapshot = run_snapshot(payloads[run["output_root"]])
        st.subheader(f"{_bundle_label(run['bundle_name'])} | {selected_sample}")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Additive routing", sample.get("additive_submethod_code", "n/a"))
        metric_cols[1].metric("ENL gain", _format_delta(sample.get("proxy_enl_gain")))
        metric_cols[2].metric("Edge delta", _format_delta(sample.get("edge_sharpness_delta")))
        metric_cols[3].metric(
            "Separability delta",
            _format_delta(
                safe_float(sample.get("distribution_separability_after")) - safe_float(sample.get("distribution_separability_before"))
                if safe_float(sample.get("distribution_separability_after")) is not None
                and safe_float(sample.get("distribution_separability_before")) is not None
                else None
            ),
        )
        metric_cols[4].metric(
            "Threshold F1 delta",
            _format_delta(
                safe_float(sample.get("threshold_f1_after")) - safe_float(sample.get("threshold_f1_before"))
                if safe_float(sample.get("threshold_f1_after")) is not None and safe_float(sample.get("threshold_f1_before")) is not None
                else None
            ),
        )
        st.info(snapshot.get("interpretation", "No interpretation was available for this run."))

        panel_paths = [
            sample.get("before_panel_path", ""),
            sample.get("after_panel_path", ""),
            sample.get("difference_panel_path", ""),
        ]
        if all(Path(path).exists() for path in panel_paths if path):
            columns = st.columns(3)
            with columns[0]:
                _show_image_if_exists(panel_paths[0], f"{run['bundle_name']} | before")
            with columns[1]:
                _show_image_if_exists(panel_paths[1], f"{run['bundle_name']} | after")
            with columns[2]:
                _show_image_if_exists(panel_paths[2], f"{run['bundle_name']} | difference")
        else:
            _show_image_if_exists(sample.get("side_by_side_path", ""), f"{run['bundle_name']} | composite view")

        with st.expander("Additional visuals"):
            extra_cols = st.columns(2)
            with extra_cols[0]:
                _show_image_if_exists(sample.get("detection_map_path", ""), f"{run['bundle_name']} | detection proxy")
            with extra_cols[1]:
                _show_image_if_exists(sample.get("additive_panel_path", ""), f"{run['bundle_name']} | additive estimate")


def page_demo() -> None:
    _page_intro(
        "Demo: Try a Scene",
        "Browse curated examples from existing outputs and see what the conditioning route changed visually.",
        "Pick an example first. The panels show input, output, and difference; the text explains the current recommendation and caveats.",
    )
    examples = collect_demo_examples(REPO_ROOT)
    if not examples:
        st.info("No demo examples were found yet. Run Bundle A or the Sentinel-1 batch comparison to create visual panels.")
        return

    option_labels = [
        f"{example['label']} | {example.get('dataset', '')} | {example.get('sample_id') or example.get('scene_id', '')} | {example.get('bundle_name', '')}"
        for example in examples
    ]
    selected_label = st.selectbox("Curated example", option_labels)
    example = examples[option_labels.index(selected_label)]

    st.subheader(example["label"])
    st.write(example["reason"])
    cards = st.columns(4)
    cards[0].metric("Bundle", str(example.get("bundle_name", "n/a")))
    cards[1].metric("Dataset", str(example.get("dataset", "n/a")))
    cards[2].metric("Submethod", str(example.get("additive_submethod", "n/a") or "n/a"))
    cards[3].metric("Decision score", _format_number(example.get("decision_score")))
    if example.get("decision_confidence"):
        st.caption(f"Confidence: {example['decision_confidence']}")
    if example.get("caveat"):
        st.warning(str(example["caveat"]))
    if example.get("interpretation"):
        st.info(str(example["interpretation"]))

    st.caption(
        "Demo examples are curated from existing local result panels. Missing or stale image paths are skipped by the index builder instead of being shown here."
    )
    image_cols = st.columns(3)
    with image_cols[0]:
        _show_image_if_exists(example.get("before_panel_path", ""), "Input")
    with image_cols[1]:
        _show_image_if_exists(example.get("after_panel_path", ""), "Output")
    with image_cols[2]:
        _show_image_if_exists(example.get("difference_panel_path", ""), "Difference")
    _show_image_if_exists(example.get("side_by_side_path", ""), "Side-by-side")

    comparable = [
        candidate
        for candidate in examples
        if candidate is not example
        and candidate.get("dataset") == example.get("dataset")
        and (candidate.get("sample_id") or candidate.get("scene_id"))
        == (example.get("sample_id") or example.get("scene_id"))
    ]
    if comparable:
        with st.expander("Compare this same sample with another curated run"):
            compare_label = st.selectbox(
                "Comparison example",
                [
                    f"{candidate['label']} | {candidate.get('bundle_name', '')} | {candidate.get('additive_submethod', '')}"
                    for candidate in comparable
                ],
            )
            other = comparable[
                [
                    f"{candidate['label']} | {candidate.get('bundle_name', '')} | {candidate.get('additive_submethod', '')}"
                    for candidate in comparable
                ].index(compare_label)
            ]
            compare_cols = st.columns(2)
            with compare_cols[0]:
                st.markdown("**Selected example output**")
                _show_image_if_exists(example.get("after_panel_path", ""), "Selected output")
            with compare_cols[1]:
                st.markdown("**Comparison output**")
                _show_image_if_exists(other.get("after_panel_path", ""), "Comparison output")
            st.caption(
                "This is a visual comparison only. Use the Bundle Results or Sentinel-1 page for the matching metric caveats."
            )


def page_downstream_detection() -> None:
    _page_intro(
        "Downstream Detection",
        "Track the real ship-detector baseline separately from proxy-only Stage-1 screening.",
        "If status is `prepared`, the YOLO dataset is ready but the optional detector backend has not been run yet. Actual mAP appears only after `--mode all` completes.",
    )
    payload = load_detection_baseline_snapshot(REPO_ROOT)
    rows = payload.get("downstream_comparison", {}).get("rows", [])
    if not rows:
        st.info("No downstream detector artifacts were found yet. Run `python scripts/run_detection_baseline.py --mode prepare` first.")
        return

    summary = payload.get("run_summary", {})
    st.subheader("Detector baseline status")
    cards = st.columns(4)
    datasets = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")})
    completed_rows = [row for row in rows if str(row.get("status", "")).lower() == "completed"]
    prepared_rows = [row for row in rows if str(row.get("status", "")).lower() == "prepared"]
    cards[0].metric("Datasets", ", ".join(datasets) or "n/a")
    cards[1].metric("Completed eval rows", str(len(completed_rows)))
    cards[2].metric("Prepared-only rows", str(len(prepared_rows)))
    cards[3].metric("Rows", str(len(rows)))
    st.warning(
        "Detector numbers are downstream validation evidence only when Ultralytics has actually trained/evaluated. Prepared-only rows are not mAP claims."
    )

    friendly_rows = []
    for row in rows:
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        friendly_rows.append(
            {
                "Dataset": row.get("dataset", ""),
                "Variant": row.get("variant", ""),
                "Status": row.get("status", ""),
                "Input records": row.get("input_record_count", ""),
                "Images": row.get("image_count", 0),
                "Boxes": row.get("box_count", 0),
                "Skipped": row.get("skipped_count", 0),
                "Missing images": row.get("missing_image_count", 0),
                "Missing annotations": row.get("missing_annotation_count", 0),
                "Empty labels": row.get("empty_label_count", 0),
                "mAP": metrics.get("map", ""),
                "Precision": metrics.get("precision", ""),
                "Recall": metrics.get("recall", ""),
                "F1": metrics.get("f1", ""),
                "Dataset YAML": shorten_path(row.get("dataset_yaml", "")),
                "Warnings": " ".join(row.get("warnings", [])),
            }
        )
    st.dataframe(pd.DataFrame(friendly_rows), width="stretch", hide_index=True)

    delta_rows = payload.get("variant_deltas", {}).get("rows", [])
    if delta_rows:
        st.subheader("Raw vs conditioned comparison")
        st.caption(
            "These deltas are real detector comparisons only when both variants completed training/evaluation. "
            "Prepared-only rows are kept visible so missing mAP does not masquerade as a result."
        )
        friendly_deltas = []
        for row in delta_rows:
            friendly_deltas.append(
                {
                    "Dataset": row.get("dataset", ""),
                    "Comparison": f"{row.get('baseline_variant', 'raw')} -> {row.get('comparison_variant', '')}",
                    "Status": row.get("status", ""),
                    "mAP delta": row.get("delta_map", ""),
                    "Precision delta": row.get("delta_precision", ""),
                    "Recall delta": row.get("delta_recall", ""),
                    "F1 delta": row.get("delta_f1", ""),
                    "Interpretation": row.get("interpretation", ""),
                }
            )
        st.dataframe(pd.DataFrame(friendly_deltas), width="stretch", hide_index=True)

    diagnostic_rows = payload.get("diagnostic_summary", {}).get("rows", [])
    if diagnostic_rows:
        st.subheader("Why performance changed")
        st.caption(
            "These target-local diagnostics explain whether conditioning preserved detector-useful edge and texture cues around labeled ships."
        )
        st.dataframe(pd.DataFrame(diagnostic_rows), width="stretch", hide_index=True)
        if payload.get("diagnostic_summary_markdown"):
            with st.expander("Diagnostic summary interpretation"):
                st.markdown(payload["diagnostic_summary_markdown"])

    completed_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "")).lower() != "completed":
            continue
        completed_by_dataset.setdefault(str(row.get("dataset", "")), {})[str(row.get("variant", ""))] = row
    for dataset_name, variants in completed_by_dataset.items():
        raw = variants.get("raw")
        conditioned = variants.get("bundle_a")
        if not raw or not conditioned:
            continue
        raw_map = safe_float(raw.get("metrics", {}).get("map"))
        conditioned_map = safe_float(conditioned.get("metrics", {}).get("map"))
        if raw_map is None or conditioned_map is None:
            continue
        delta = conditioned_map - raw_map
        st.info(
            f"{dataset_name}: Bundle A {'helped' if delta > 0 else 'hurt' if delta < 0 else 'matched'} YOLO mAP by {delta:+.3f} on this run. "
            "Treat this as preliminary if the split or epoch count is small."
        )
    if not completed_rows:
        st.info(
            "Current artifacts are prepared-only or dependency-missing. Run with `--mode all` after installing Ultralytics to produce actual mAP / precision / recall."
        )
    if payload.get("run_summary_markdown"):
        with st.expander("Human-readable detector summary"):
            st.markdown(payload["run_summary_markdown"])
    _raw_json_expander("Detector raw JSON", payload)


def page_bundle_a_submethods() -> None:
    _page_intro(
        "Bundle A Submethods",
        "Compare the additive routes inside Bundle A and see which one is currently recommended for each data condition.",
        "Start with the explanation cards, then read the recommendation block. Use the aggregate table before drilling into per-sample details.",
    )

    card_columns = st.columns(4)
    for column, code in zip(card_columns, ["A0", "A1", "A2", "A3"]):
        details = BUNDLE_A_SUBMETHOD_DETAILS[code]
        with column:
            st.markdown(f"**{details['label']}**")
            _render_badge_line(details["trust"], CONFIDENCE_BADGE_COLORS[details["trust"]])
            st.caption(f"Requires: {details['requires']}")
            st.write(details["note"])

    st.info(
        "A1 requires metadata, A2 is the realistic fallback when metadata is missing, A3 is for structured artifact cases, and A0 is the baseline control."
    )
    _help_expander(
        "How to interpret A1 with zero current wins",
        "A1 should not be dismissed just because overview-scale proxy rankings currently favor A0 or A3. "
        "Metadata-driven thermal correction can be subtle, especially on overview imagery, so the app flags possible under-crediting when metadata exists.",
    )

    runs = [run for run in _load_runs() if run["bundle_name"] == "bundle_a"]
    if not runs:
        st.info("No Bundle A runs were discovered.")
        return

    selected_runs = st.multiselect("Bundle A runs", runs, format_func=_run_label, default=runs[: min(2, len(runs))])
    if not selected_runs:
        st.info("Select at least one Bundle A run.")
        return

    sample_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for run in selected_runs:
        payload = _load_run_payload(run)
        if payload is None:
            continue
        sample_rows.extend(payload["submethod_summary"].get("samples", []))
        aggregate_rows.extend(payload["submethod_aggregate"].get("groups", []))

    st.subheader("Current recommendation")
    st.success(
        "Current recommendation on public chip datasets: use A2 as the default metadata-poor fallback, keep A0 as the control, trust A1 when real metadata exists, and use A3 only when structured artifacts are clearly present."
    )

    compact_mode = st.checkbox("Compact table mode", value=True)
    show_raw_columns = st.checkbox("Show raw columns", value=False)

    if aggregate_rows:
        friendly_aggregate_rows = []
        for row in aggregate_rows:
            friendly_row = {
                "Submethod": row.get("additive_submethod_name", row.get("additive_submethod_code", "")),
                "Confidence": row.get("confidence_level", ""),
                "Samples": row.get("sample_count", 0),
                "Metadata available rate": row.get("metadata_available_rate", ""),
                "Additive applied rate": row.get("additive_applied_rate", ""),
                "Mean ENL before": row.get("mean_proxy_enl_before", ""),
                "Mean ENL after": row.get("mean_proxy_enl_after", ""),
                "Mean edge before": row.get("mean_edge_sharpness_before", ""),
                "Mean edge after": row.get("mean_edge_sharpness_after", ""),
                "Mean separability before": row.get("mean_distribution_separability_before", ""),
                "Mean separability after": row.get("mean_distribution_separability_after", ""),
                "Mean threshold F1 before": row.get("mean_threshold_f1_before", ""),
                "Mean threshold F1 after": row.get("mean_threshold_f1_after", ""),
            }
            if not compact_mode:
                friendly_row["Description"] = row.get("description", "")
                friendly_row["Required inputs"] = row.get("required_inputs", "")
            friendly_aggregate_rows.append(friendly_row)
        st.subheader("Aggregate comparison")
        st.dataframe(_as_dataframe(friendly_aggregate_rows), width="stretch", hide_index=True)

    if sample_rows:
        st.subheader("Per-sample view")
        if show_raw_columns:
            st.dataframe(_as_dataframe(sample_rows), width="stretch", hide_index=True)
        else:
            st.dataframe(_as_dataframe(_friendly_sample_rows(sample_rows, compact=compact_mode)), width="stretch", hide_index=True)


def page_statistics() -> None:
    _page_intro(
        "Statistics",
        "Interpret the intensity-domain statistical baseline rather than reading fit files directly.",
        "Higher separability is better, lower overlap is better, and threshold F1 is a simple sanity check rather than a full detector metric.",
    )
    runs = _load_runs()
    if not runs:
        st.info("No runs were discovered.")
        return

    selected_run = st.selectbox("Run", runs, format_func=_run_label)
    payload = _load_run_payload(selected_run)
    if payload is None:
        return
    summary = payload.get("statistics_summary", {})
    if not summary:
        st.info("No statistics summary was available for this run.")
        return

    st.write(
        "Background intensity pixels are modeled as exponential, ship-like target intensity pixels are modeled as log-normal, lower overlap is better, and higher separability is better."
    )

    pooled = summary.get("pooled", {})
    mean_metrics = summary.get("mean_metrics", {})
    card_cols = st.columns(6)
    card_cols[0].metric("Separability before", _format_number(mean_metrics.get("distribution_separability_before")))
    card_cols[1].metric("Separability after", _format_number(mean_metrics.get("distribution_separability_after")))
    card_cols[2].metric("Balanced acc. before", _format_number(mean_metrics.get("threshold_balanced_accuracy_before")))
    card_cols[3].metric("Balanced acc. after", _format_number(mean_metrics.get("threshold_balanced_accuracy_after")))
    card_cols[4].metric("Threshold F1 before", _format_number(mean_metrics.get("threshold_f1_before")))
    card_cols[5].metric("Threshold F1 after", _format_number(mean_metrics.get("threshold_f1_after")))

    sep_before = safe_float(mean_metrics.get("distribution_separability_before"))
    sep_after = safe_float(mean_metrics.get("distribution_separability_after"))
    f1_before = safe_float(mean_metrics.get("threshold_f1_before"))
    f1_after = safe_float(mean_metrics.get("threshold_f1_after"))
    if sep_before is not None and sep_after is not None and f1_before is not None and f1_after is not None:
        st.info(
            "Current interpretation: "
            + (
                "the statistical separation improved."
                if sep_after >= sep_before and f1_after >= f1_before
                else "the statistical separation did not improve cleanly across all metrics."
            )
        )

    st.caption(
        "Pooled plots aggregate target and background pixels across the current run, so they show the overall trend rather than one special-case sample."
    )
    plot_candidates = [
        payload["layout"]["statistics"] / "pooled_before_linear.png",
        payload["layout"]["statistics"] / "pooled_after_linear.png",
        payload["layout"]["statistics"] / "pooled_after_logx.png",
    ]
    columns = st.columns(len(plot_candidates))
    for column, plot_path in zip(columns, plot_candidates):
        with column:
            _show_image_if_exists(plot_path.as_posix(), plot_path.name)

    per_sample_rows = _load_json_rows(payload["layout"]["statistics"] / "per_sample_statistics.json", "samples")
    if per_sample_rows:
        st.subheader("Per-sample fits")
        st.dataframe(pd.DataFrame(per_sample_rows), width="stretch", hide_index=True)

    _raw_json_expander("Statistics raw JSON", summary)
    if pooled:
        _raw_json_expander("Pooled fit details", pooled)


def page_sentinel1() -> None:
    _page_intro(
        "Sentinel-1 Local Readiness",
        "Track which local GRD products are actually usable for Bundle A-style metadata and proxy experiments.",
        "Read the summary cards first. Treat the detail table as operational status, not as final scientific evidence.",
    )
    rows = load_sentinel1_manifest_rows(REPO_ROOT)
    if not rows:
        st.info("No Sentinel-1 manifest rows were found.")
        return

    total_rows = len(rows)
    ready_rows = [row for row in rows if str(row.get("prepared_status", "")).lower() == "ready"]
    noise_rows = [row for row in rows if row.get("noise_xml_path")]
    calibration_rows = [row for row in rows if row.get("calibration_xml_path")]
    runnable_rows = [row for row in rows if str(row.get("prepared_status", "")).lower() == "ready" and row.get("image_path")]

    cards = st.columns(5)
    cards[0].metric("Manifest rows", str(total_rows))
    cards[1].metric("Local usable products", str(len(ready_rows)))
    cards[2].metric("With noise XML", str(len(noise_rows)))
    cards[3].metric("With calibration XML", str(len(calibration_rows)))
    cards[4].metric("Runnable in Bundle A", str(len(runnable_rows)))

    st.warning("Current Sentinel-1 runs are overview/proxy-scale checks, not full-scene production-grade filtering.")
    st.info("Recommendation: add 5-10 more maritime GRD scenes so Bundle A has stronger real-product evidence.")

    display_rows = []
    for row in rows:
        prepared_status = str(row.get("prepared_status", "")).lower() or "metadata-only"
        display_rows.append(
            {
                "Product": row.get("product_name", ""),
                "Readiness": prepared_status,
                "Type": row.get("product_type", ""),
                "Mode": row.get("mode", ""),
                "Measurements": row.get("measurement_count", ""),
                "Primary pol.": row.get("primary_polarization", ""),
                "Noise XML": "Yes" if row.get("noise_xml_path") else "No",
                "Calibration XML": "Yes" if row.get("calibration_xml_path") else "No",
                "Local image": shorten_path(row.get("prepared_image_path", "") or row.get("image_path", "")),
                "Notes": row.get("notes", ""),
            }
        )
    st.dataframe(_as_dataframe(display_rows), width="stretch", hide_index=True)

    batch_payload = _load_sentinel1_batch_payload()
    if batch_payload is not None:
        st.subheader("Bundle A Scene Comparison")
        st.caption(
            "Use this section to compare A0/A1/A2/A3 evidence on real local Sentinel-1 GRD scenes. "
            "Start with the scene summary, then drill into one scene and one submethod at a time."
        )
        batch_topline = batch_payload.get("topline_metrics", {})
        scene_rows = batch_payload.get("scene_summary", {}).get("scenes", [])
        comparison_rows = batch_payload.get("submethod_comparison", {}).get("rows", [])
        aggregate_rows = batch_payload.get("submethod_aggregate", {}).get("groups", [])

        comparison_cards = st.columns(5)
        comparison_cards[0].metric("Evaluated scenes", str(batch_topline.get("evaluated_scene_count", 0)))
        comparison_cards[1].metric("Comparison rows", str(batch_topline.get("comparison_row_count", 0)))
        comparison_cards[2].metric("Metadata-rich scenes", str(batch_topline.get("metadata_rich_scene_count", 0)))
        comparison_cards[3].metric("Overview-only scenes", str(batch_topline.get("overview_only_scene_count", 0)))
        comparison_cards[4].metric("Current Sentinel-1 hint", str(batch_topline.get("recommended_submethod_hint", "") or "n/a"))

        maturity_cols = st.columns(5)
        maturity_cols[0].metric("Evidence confidence", str(batch_topline.get("evidence_confidence_level", "n/a")))
        maturity_cols[1].metric("A0 wins", str(batch_topline.get("a0_win_count", 0)))
        maturity_cols[2].metric("A1 wins", str(batch_topline.get("a1_win_count", 0)))
        maturity_cols[3].metric("A2 wins", str(batch_topline.get("a2_win_count", 0)))
        maturity_cols[4].metric("A3 wins", str(batch_topline.get("a3_win_count", 0)))

        recommendation_text = str(batch_topline.get("current_recommendation", "")).strip()
        if recommendation_text:
            st.info(recommendation_text)
        evidence_plan = batch_payload.get("evidence_plan", {})
        if evidence_plan:
            with st.expander("Next Sentinel-1 evidence plan"):
                st.write(
                    f"Recommended additional scenes: `{evidence_plan.get('recommended_next_scene_count', 0)}`"
                )
                underrepresented = evidence_plan.get("underrepresented_regimes", [])
                if underrepresented:
                    st.caption("Underrepresented regimes: " + ", ".join(str(item) for item in underrepresented))
                for item in evidence_plan.get("recommendations", []):
                    st.write("- " + str(item))
        if batch_topline.get("decision_basis"):
            _help_expander("How Sentinel-1 submethods are ranked", str(batch_topline["decision_basis"]))
        for warning in batch_topline.get("warnings", []):
            st.warning(str(warning))

        if scene_rows:
            scene_summary_rows = []
            for row in scene_rows:
                scene_summary_rows.append(
                    {
                        "Scene": row.get("scene_id", ""),
                        "Product": row.get("product_name", ""),
                        "Status": row.get("scene_status", ""),
                        "Regime": row.get("regime_label", ""),
                        "Best": row.get("best_submethod", ""),
                        "Runner-up": row.get("runner_up_submethod", ""),
                        "Confidence": row.get("decision_confidence", row.get("decision_evidence_grade", "")),
                        "Score margin": row.get("decision_score_margin", ""),
                        "Submethods ran": row.get("submethods_ran", ""),
                        "Metadata ready": "Yes" if row.get("metadata_ready_for_a1") else "No",
                        "Overview-only": "Yes" if row.get("overview_only_evaluation") else "No",
                        "Note": row.get("recommendation_summary", row.get("notes", "")),
                    }
                )
            st.dataframe(_as_dataframe(scene_summary_rows), width="stretch", hide_index=True)

            selected_scene_id = st.selectbox(
                "Scene for detailed comparison",
                [str(row.get("scene_id", "")) for row in scene_rows],
                key="sentinel1_batch_scene",
            )
            selected_scene = next((row for row in scene_rows if str(row.get("scene_id", "")) == selected_scene_id), None)
            selected_scene_rows = [row for row in comparison_rows if str(row.get("scene_id", "")) == selected_scene_id]

            if selected_scene is not None:
                detail_left, detail_right = st.columns([1.0, 1.1])
                with detail_left:
                    st.markdown("**Scene interpretation**")
                    st.write(f"Product: `{selected_scene.get('product_name', '')}`")
                    st.write(f"Regime: `{selected_scene.get('regime_label', 'n/a')}`")
                    st.write(f"Best submethod: `{selected_scene.get('best_submethod', 'n/a')}`")
                    st.write(f"Runner-up: `{selected_scene.get('runner_up_submethod', 'n/a')}`")
                    st.write(f"Confidence: `{selected_scene.get('decision_confidence', selected_scene.get('decision_evidence_grade', 'n/a'))}`")
                    st.caption(selected_scene.get("recommendation_why", ""))
                    if selected_scene.get("recommendation_caveats"):
                        st.warning(str(selected_scene.get("recommendation_caveats", "")))
                    if selected_scene.get("decision_metric_bias_warning"):
                        st.info(str(selected_scene["decision_metric_bias_warning"]))
                with detail_right:
                    st.markdown("**Scene comparison table**")
                    st.dataframe(_as_dataframe(_friendly_sentinel1_comparison_rows(selected_scene_rows)), width="stretch", hide_index=True)

                if selected_scene_rows:
                    selected_submethod = st.selectbox(
                        "Visualize submethod",
                        selected_scene_rows,
                        format_func=lambda row: f"{row.get('additive_submethod_used', '')} (requested {row.get('requested_additive_submethod', '')})",
                        key="sentinel1_batch_submethod",
                    )
                    visual_cols = st.columns(3)
                    with visual_cols[0]:
                        _show_image_if_exists(selected_submethod.get("before_panel_path", ""), "Before")
                    with visual_cols[1]:
                        _show_image_if_exists(selected_submethod.get("after_panel_path", ""), "After")
                    with visual_cols[2]:
                        _show_image_if_exists(selected_submethod.get("difference_panel_path", ""), "Difference")
                    st.caption(
                        "Warnings: "
                        + (
                            str(selected_submethod.get("warnings", "")).strip()
                            or "No extra scene-specific warning was recorded."
                        )
                    )

            if aggregate_rows:
                st.markdown("**Aggregate by submethod**")
                st.dataframe(_as_dataframe(aggregate_rows), width="stretch", hide_index=True)

            recommendations_markdown = str(batch_payload.get("scene_recommendations_markdown", "")).strip()
            if recommendations_markdown:
                with st.expander("Scene recommendations markdown"):
                    st.markdown(recommendations_markdown)
    else:
        st.info(
            "No Sentinel-1 batch comparison artifacts were found yet. Run the Sentinel-1 batch evaluator to populate scene-level evidence."
        )

    sentinel_runs = [run for run in _load_runs() if run["dataset"] == "sentinel1"]
    if sentinel_runs:
        st.subheader("Processed Sentinel-1 runs")
        selected_run = st.selectbox("Sentinel-1 run", sentinel_runs, format_func=_run_label)
        payload = _load_run_payload(selected_run)
        if payload is not None:
            snapshot = run_snapshot(payload)
            _run_metric_cards(snapshot)
            st.info(snapshot.get("interpretation", "No interpretation was available for this run."))
            sample_rows = payload["per_sample_metrics"].get("samples", [])
            if sample_rows:
                st.dataframe(_as_dataframe(_friendly_sample_rows(sample_rows)), width="stretch", hide_index=True)


def page_dataset_audit() -> None:
    _page_intro(
        "Dataset Audit",
        "Review dataset readiness, split counts, and the practical role of each local dataset.",
        "Use the main table for availability decisions. Open the expanders only when you need detailed audit notes.",
    )
    audit_payload = load_dataset_audit_snapshot(REPO_ROOT).get("datasets", {})
    if not audit_payload:
        st.info("No audit summary is available.")
        return

    st.subheader("Status definitions")
    st.dataframe(pd.DataFrame(dataset_status_help_rows()), width="stretch", hide_index=True)

    rows = []
    for dataset_name, payload in sorted(audit_payload.items()):
        rows.append(
            {
                "Dataset": dataset_name,
                "Status": payload.get("status", ""),
                "Recommended usage": _dataset_recommendation(dataset_name),
                "Total count": payload.get("total_count", 0),
                "Splits": ", ".join(f"{name}:{count}" for name, count in sorted(payload.get("split_counts", {}).items())),
                "Missing images": payload.get("missing_image_files", 0),
                "Missing annotations": payload.get("missing_annotation_files", 0),
                "Duplicates": payload.get("duplicate_sample_ids", 0),
                "Leakage": payload.get("leakage_by_canonical_id", 0),
            }
        )
    st.dataframe(_as_dataframe(rows), width="stretch", hide_index=True)

    for dataset_name, payload in sorted(audit_payload.items()):
        with st.expander(dataset_name):
            st.caption(_dataset_recommendation(dataset_name))
            st.json(payload)


def page_registry() -> None:
    _page_intro(
        "Dataset Registry",
        "Use the registry as the source of truth for local dataset availability and manifest locations.",
        "The main table is intentionally shortened for readability. Open a dataset expander when you need the full local or manifest path.",
    )
    registry = load_dataset_registry_snapshot(REPO_ROOT)
    rows = []
    for dataset_name, registration in registry.items():
        rows.append(
            {
                "Dataset": dataset_name,
                "Status": registration.status,
                "Samples": registration.sample_count,
                "Local path": shorten_path(registration.local_path or ""),
                "Manifest path": shorten_path(registration.manifest_path or ""),
                "Remote source": shorten_path(registration.remote_source or "", max_chars=55),
            }
        )
    st.dataframe(_as_dataframe(rows), width="stretch", hide_index=True)

    for dataset_name, registration in registry.items():
        with st.expander(dataset_name):
            st.write(f"Status: `{registration.status}`")
            st.write(f"Samples: `{registration.sample_count}`")
            st.write(f"Local path: `{registration.local_path}`")
            st.write(f"Manifest path: `{registration.manifest_path}`")
            if registration.external_path:
                st.write(f"External path: `{registration.external_path}`")
            if registration.remote_source:
                st.write(f"Remote source: `{registration.remote_source}`")
            if registration.notes:
                st.caption(registration.notes)


PAGES = {
    "Start Here": page_start_here,
    "Final Results": page_final_results,
    "About / Glossary": page_about,
    "Overview": page_overview,
    "Bundle Results": page_bundle_results,
    "Visual Comparison": page_visual_comparison,
    "Demo: Try a Scene": page_demo,
    "Downstream Detection": page_downstream_detection,
    "Bundle A Submethods": page_bundle_a_submethods,
    "Statistics": page_statistics,
    "Sentinel-1": page_sentinel1,
    "Dataset Audit": page_dataset_audit,
}
if APP_SURFACE == "private":
    PAGES = {
        "Start Here": page_start_here,
        "Handoff Workspace": page_ai_handoff,
        "Final Results": page_final_results,
        "About / Glossary": page_about,
        "Overview": page_overview,
        "Bundle Results": page_bundle_results,
        "Visual Comparison": page_visual_comparison,
        "Demo: Try a Scene": page_demo,
        "Downstream Detection": page_downstream_detection,
        "Bundle A Submethods": page_bundle_a_submethods,
        "Statistics": page_statistics,
        "Sentinel-1": page_sentinel1,
        "Dataset Audit": page_dataset_audit,
        "Registry": page_registry,
    }


st.sidebar.title("Browse")
st.sidebar.caption(f"Surface: {APP_SURFACE}")
selected_page = st.sidebar.radio("Page", list(PAGES.keys()))
try:
    PAGES[selected_page]()
except Exception as exc:  # pragma: no cover - UI safety
    st.error(f"This page hit an unexpected error: {exc}")
    st.info("The rest of the app and result files are still intact. Use the traceback below only for debugging.")
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
