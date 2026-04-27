from __future__ import annotations

"""Decision heuristics for proxy-only Stage-1 screening.

The scores in this module are intentionally not scientific claims and are not
detector mAP. They turn several imperfect proxy signals into an interpretable
ranking while making the caveats explicit:

- ENL gain can mean useful speckle reduction, but it can also mean oversmoothing.
- Edge loss is penalized so smoothing is not automatically promoted.
- Metadata-driven A1 can be under-credited by overview-scale Sentinel-1 proxies.
- Confidence depends on evidence breadth and whether real labels/full scenes exist.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


SUBMETHOD_ORDER = ("A0", "A1", "A2", "A3")


@dataclass(frozen=True, slots=True)
class DecisionScore:
    """Compact explanation of a proxy decision heuristic."""

    score: float
    confidence: str
    evidence_grade: str
    rationale: str
    flags: list[str]
    components: dict[str, float]
    metric_bias_warning: str = ""

    def as_fields(self, prefix: str = "decision") -> dict[str, Any]:
        return {
            f"{prefix}_score": self.score,
            f"{prefix}_confidence": self.confidence,
            f"{prefix}_evidence_grade": self.evidence_grade,
            f"{prefix}_rationale": self.rationale,
            f"{prefix}_flags": self.flags,
            f"{prefix}_components": self.components,
            f"{prefix}_metric_bias_warning": self.metric_bias_warning,
        }


def safe_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def clipped(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def relative_delta(row: dict[str, Any], delta_key: str, before_key: str, *, floor: float = 1.0) -> float:
    delta = safe_float(row.get(delta_key)) or 0.0
    before = abs(safe_float(row.get(before_key)) or 0.0)
    return float(delta / max(before, floor))


def evidence_grade_for_run(
    *,
    downstream_status: str = "proxy-only",
    dataset_name: str = "",
    bundle_name: str = "",
    sample_count: int = 0,
    overview_only: bool = False,
) -> str:
    """Return a plain-language evidence grade for app/reporting surfaces."""

    if bundle_name == "bundle_c":
        return "feasibility-grade"
    if downstream_status == "proxy-only":
        if dataset_name == "sentinel1":
            return "proxy-only / overview-scale" if overview_only else "proxy-only"
        return "screening-grade / proxy-only" if sample_count >= 8 else "low-confidence / proxy-only"
    return "claim-grade"


def evidence_confidence_from_counts(evaluated_count: int, *, overview_only_count: int = 0) -> str:
    if evaluated_count <= 0:
        return "none"
    if evaluated_count == 1:
        return "very-thin"
    if evaluated_count < 4:
        return "thin"
    if evaluated_count < 8:
        return "thin-but-improving"
    confidence = "developing"
    if overview_only_count >= evaluated_count:
        confidence += " / overview-limited"
    return confidence


def score_proxy_tradeoff(row: dict[str, Any], *, regime: dict[str, Any] | None = None) -> DecisionScore:
    """Score one method row using balanced proxy tradeoffs.

    The formula deliberately weights target/background separability and simple
    threshold behavior more than raw ENL gain. ENL is capped and edge loss is
    penalized so a strongly smoothed image does not win only because it looks
    statistically homogeneous.
    """

    regime = regime or {}
    submethod = str(row.get("additive_submethod_used") or row.get("additive_submethod_code") or "").upper()
    sep_delta = safe_float(row.get("distribution_separability_delta"))
    if sep_delta is None:
        before = safe_float(row.get("distribution_separability_before"))
        after = safe_float(row.get("distribution_separability_after"))
        sep_delta = (after - before) if before is not None and after is not None else 0.0

    f1_delta = safe_float(row.get("threshold_f1_delta"))
    if f1_delta is None:
        before = safe_float(row.get("threshold_f1_before"))
        after = safe_float(row.get("threshold_f1_after"))
        f1_delta = (after - before) if before is not None and after is not None else 0.0

    enl_ratio = relative_delta(row, "proxy_enl_gain", "proxy_enl_before", floor=1.0)
    edge_ratio = relative_delta(row, "edge_sharpness_delta", "edge_sharpness_before", floor=1.0)
    edge_loss = max(0.0, -edge_ratio)

    components = {
        "separability": 0.36 * clipped(float(sep_delta) / 0.06),
        "threshold_f1": 0.26 * clipped(float(f1_delta) / 0.06),
        "enl_gain_capped": 0.14 * clipped(enl_ratio / 0.75),
        "edge_retention": -0.20 * clipped(edge_loss / 0.35, 0.0, 1.0),
    }

    flags: list[str] = []
    if enl_ratio > 0.75 and edge_loss > 0.25:
        components["oversmoothing_penalty"] = -0.10
        flags.append("possible oversmoothing: high ENL gain with notable edge loss")
    else:
        components["oversmoothing_penalty"] = 0.0

    metadata_rich = regime.get("metadata_regime") == "metadata-rich" or bool(row.get("metadata_available"))
    structured_artifact = bool(regime.get("structured_artifact_likely"))
    overview_only = bool(regime.get("overview_only_evaluation") or row.get("overview_fallback_used"))

    if submethod == "A1" and metadata_rich:
        components["regime_alignment"] = 0.045
        if overview_only:
            flags.append("A1 may be under-credited by overview-scale proxy metrics")
    elif submethod == "A2" and not metadata_rich:
        components["regime_alignment"] = 0.035
    elif submethod == "A3":
        components["regime_alignment"] = 0.05 if structured_artifact else -0.045
        if not structured_artifact:
            flags.append("A3 needs visible structured-artifact evidence")
    elif submethod == "A0":
        components["regime_alignment"] = 0.025
    else:
        components["regime_alignment"] = 0.0

    score = float(sum(components.values()))
    if overview_only:
        confidence = "low-confidence / overview-scale"
    elif str(row.get("source_note", "")).lower().find("proxy") >= 0:
        confidence = "screening-grade / proxy-only"
    else:
        confidence = "screening-grade"

    if metadata_rich and submethod != "A1":
        flags.append("metadata exists; compare against A1 before making a final claim")
    if submethod == "A1" and metadata_rich and score < 0:
        flags.append("A1 is metadata-correct but proxy score is weak on this view")

    metric_bias_warning = ""
    if any("under-credited" in flag for flag in flags):
        metric_bias_warning = "Metadata-driven correction can look subtle in overview/proxy metrics; inspect visuals before dismissing A1."
    elif any("oversmoothing" in flag for flag in flags):
        metric_bias_warning = "The score penalized smoothing that improves ENL while eroding edges."

    rationale_parts = [
        f"separability {sep_delta:+.3f}",
        f"threshold F1 {f1_delta:+.3f}",
        f"ENL ratio {enl_ratio:+.3f}",
        f"edge ratio {edge_ratio:+.3f}",
    ]
    if flags:
        rationale_parts.append("; ".join(flags))

    return DecisionScore(
        score=score,
        confidence=confidence,
        evidence_grade="proxy-only / overview-scale" if overview_only else "screening-grade / proxy-only",
        rationale="; ".join(rationale_parts),
        flags=flags,
        components={key: float(value) for key, value in components.items()},
        metric_bias_warning=metric_bias_warning,
    )


def summarize_winner_counts(scene_rows: list[dict[str, Any]]) -> dict[str, int]:
    winners = [str(row.get("best_submethod", "")).upper() for row in scene_rows if row.get("best_submethod")]
    return {code: winners.count(code) for code in SUBMETHOD_ORDER}

