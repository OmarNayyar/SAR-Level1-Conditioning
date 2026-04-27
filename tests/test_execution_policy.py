from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.utils.execution import (
    ExecutionPolicy,
    artifact_manifest_path,
    decide_artifact_action,
    write_artifact_manifest,
)


def _workspace() -> Path:
    root = Path("outputs") / "test-workspaces" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_decide_artifact_action_reuses_matching_manifest() -> None:
    workspace = _workspace()
    artifact_root = workspace / "bundle_a"
    artifact_root.mkdir(parents=True, exist_ok=True)
    required = artifact_root / "metrics" / "run_summary.json"
    required.parent.mkdir(parents=True, exist_ok=True)
    required.write_text("{}", encoding="utf-8")
    identity = {"bundle": "bundle_a", "dataset": "ssdd"}
    write_artifact_manifest(
        artifact_root,
        artifact_kind="bundle_run",
        identity=identity,
        status="complete",
        files={"run_summary": required.as_posix()},
    )

    decision = decide_artifact_action(
        artifact_kind="bundle_run",
        output_root=artifact_root,
        identity=identity,
        required_files=["metrics/run_summary.json"],
        capability="conditioning",
        policy=ExecutionPolicy(reuse_only=True),
    )

    assert artifact_manifest_path(artifact_root).exists()
    assert decision.action == "reuse"
    assert decision.matched_identity is True


def test_decide_artifact_action_blocks_missing_prepare_when_policy_is_reuse_only() -> None:
    workspace = _workspace()
    identity = {"dataset": "ssdd", "variant": "raw"}
    decision = decide_artifact_action(
        artifact_kind="prepared_yolo_dataset",
        output_root=workspace / "prepared" / "ssdd" / "raw",
        identity=identity,
        required_files=["dataset.yaml"],
        capability="prepare",
        policy=ExecutionPolicy(reuse_only=True),
    )

    assert decision.action == "blocked"
    assert "execution policy forbids `prepare` work" in decision.reason
