from __future__ import annotations

"""Shared reuse-first execution policy and lightweight artifact manifests.

This repo now treats expensive work as opt-in. Normal CLI, reporting, and app
flows should prefer reusing existing artifacts and should fail clearly when the
requested artifact has not been generated yet.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from src.datasets.common import read_json, write_json


ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    return value


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(_normalize_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_fingerprint(payload: Any, *, length: int = 12) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()[:length]


def artifact_manifest_path(output_root: Path) -> Path:
    return output_root / ARTIFACT_MANIFEST_FILENAME


def read_artifact_manifest(output_root: Path) -> dict[str, Any]:
    path = artifact_manifest_path(output_root)
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_artifact_manifest(
    output_root: Path,
    *,
    artifact_kind: str,
    identity: dict[str, Any],
    status: str,
    files: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    notes: Iterable[str] | None = None,
) -> Path:
    output_root = output_root.resolve()
    existing = read_artifact_manifest(output_root)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "artifact_kind": artifact_kind,
        "status": status,
        "output_root": output_root.as_posix(),
        "identity": _normalize_value(identity),
        "identity_hash": payload_fingerprint(identity),
        "files": _normalize_value(files or {}),
        "metadata": _normalize_value(metadata or {}),
        "notes": list(notes or []),
        "created_at": existing.get("created_at", timestamp),
        "updated_at": timestamp,
    }
    write_json(artifact_manifest_path(output_root), payload)
    return artifact_manifest_path(output_root)


def write_artifact_index(index_path: Path, rows: list[dict[str, Any]]) -> Path:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(index_path, {"artifacts": [_normalize_value(row) for row in rows]})
    return index_path


def required_files_present(output_root: Path, required_files: Iterable[str | Path]) -> list[str]:
    missing: list[str] = []
    for item in required_files:
        path = Path(item)
        resolved = path if path.is_absolute() else (output_root / path)
        if not resolved.exists():
            missing.append(resolved.as_posix())
    return missing


def artifact_matches_identity(manifest: dict[str, Any], *, artifact_kind: str, identity: dict[str, Any]) -> bool:
    return (
        str(manifest.get("artifact_kind", "")) == artifact_kind
        and str(manifest.get("identity_hash", "")) == payload_fingerprint(identity)
    )


@dataclass(slots=True)
class ExecutionPolicy:
    """Controls whether a caller may prepare, condition, train, or evaluate."""

    reuse_only: bool = True
    allow_prepare: bool = False
    allow_conditioning: bool = False
    allow_train: bool = False
    allow_eval: bool = False
    dry_run: bool = False
    resume: bool = True
    force: bool = False

    def allows(self, capability: Literal["prepare", "conditioning", "train", "eval"]) -> bool:
        if capability == "prepare":
            return self.allow_prepare
        if capability == "conditioning":
            return self.allow_conditioning
        if capability == "train":
            return self.allow_train
        if capability == "eval":
            return self.allow_eval or self.allow_train
        raise ValueError(f"Unsupported execution capability: {capability}")

    @property
    def allows_heavy_work(self) -> bool:
        return any((self.allow_prepare, self.allow_conditioning, self.allow_train, self.allow_eval))


def add_execution_policy_args(
    parser: argparse.ArgumentParser,
    *,
    include_prepare: bool = False,
    include_conditioning: bool = False,
    include_train: bool = False,
    include_eval: bool = False,
) -> None:
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Use cached artifacts only. This is the default behavior when no allow-* flag is provided.",
    )
    if include_prepare:
        parser.add_argument(
            "--allow-prepare",
            action="store_true",
            help="Allow expensive preparation when required artifacts are missing.",
        )
    if include_conditioning:
        parser.add_argument(
            "--allow-conditioning",
            action="store_true",
            help="Allow bundle/image generation when cached conditioning artifacts are missing.",
        )
    if include_train:
        parser.add_argument(
            "--allow-train",
            action="store_true",
            help="Allow detector training when completed metrics are not already cached.",
        )
    if include_eval:
        parser.add_argument(
            "--allow-eval",
            action="store_true",
            help="Allow detector evaluation when evaluation artifacts are missing.",
        )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run or be reused without launching heavy work.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite mismatched artifacts instead of blocking on the existing cache.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Ignore resumable partial caches and require a clean rerun when heavy work is allowed.",
    )


def execution_policy_from_args(args: Any) -> ExecutionPolicy:
    policy = ExecutionPolicy(
        reuse_only=bool(getattr(args, "reuse_only", False)),
        allow_prepare=bool(getattr(args, "allow_prepare", False)),
        allow_conditioning=bool(getattr(args, "allow_conditioning", False)),
        allow_train=bool(getattr(args, "allow_train", False)),
        allow_eval=bool(getattr(args, "allow_eval", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        resume=bool(getattr(args, "resume", True)),
        force=bool(getattr(args, "force", False)),
    )
    if not policy.allows_heavy_work:
        policy.reuse_only = True
    return policy


@dataclass(slots=True)
class ArtifactDecision:
    artifact_kind: str
    output_root: Path
    action: Literal["reuse", "run", "would_run", "blocked"]
    reason: str
    identity_hash: str
    manifest_found: bool
    matched_identity: bool
    missing_files: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "output_root": self.output_root.as_posix(),
            "action": self.action,
            "reason": self.reason,
            "identity_hash": self.identity_hash,
            "manifest_found": self.manifest_found,
            "matched_identity": self.matched_identity,
            "missing_files": self.missing_files,
        }


def decide_artifact_action(
    *,
    artifact_kind: str,
    output_root: Path,
    identity: dict[str, Any],
    required_files: Iterable[str | Path],
    capability: Literal["prepare", "conditioning", "train", "eval"],
    policy: ExecutionPolicy,
    accept_existing_without_manifest: bool = False,
) -> ArtifactDecision:
    output_root = output_root.resolve()
    manifest = read_artifact_manifest(output_root)
    manifest_found = bool(manifest)
    matched_identity = artifact_matches_identity(manifest, artifact_kind=artifact_kind, identity=identity)
    missing_files = required_files_present(output_root, required_files)
    identity_hash = payload_fingerprint(identity)

    if matched_identity and not missing_files and not policy.force and policy.resume:
        return ArtifactDecision(
            artifact_kind=artifact_kind,
            output_root=output_root,
            action="reuse",
            reason="Loading cached artifact because the manifest and required files match the request.",
            identity_hash=identity_hash,
            manifest_found=manifest_found,
            matched_identity=True,
            missing_files=[],
        )

    if accept_existing_without_manifest and not manifest_found and not missing_files and not policy.force and policy.resume:
        return ArtifactDecision(
            artifact_kind=artifact_kind,
            output_root=output_root,
            action="reuse",
            reason=(
                "Loading an existing artifact because the required files are present, although this older artifact does not "
                "yet have a manifest to verify the exact request fingerprint."
            ),
            identity_hash=identity_hash,
            manifest_found=False,
            matched_identity=False,
            missing_files=[],
        )

    if manifest_found and not matched_identity and not policy.force:
        return ArtifactDecision(
            artifact_kind=artifact_kind,
            output_root=output_root,
            action="blocked",
            reason=(
                "An artifact already exists at this path, but its manifest does not match the requested configuration. "
                "Use a different output root or rerun intentionally with --force."
            ),
            identity_hash=identity_hash,
            manifest_found=True,
            matched_identity=False,
            missing_files=missing_files,
        )

    if not policy.allows(capability):
        return ArtifactDecision(
            artifact_kind=artifact_kind,
            output_root=output_root,
            action="blocked",
            reason=(
                f"The requested {artifact_kind} is not available in cache and execution policy forbids `{capability}` work. "
                f"Re-run with --allow-{capability} or use the final sweep command."
            ),
            identity_hash=identity_hash,
            manifest_found=manifest_found,
            matched_identity=matched_identity,
            missing_files=missing_files,
        )

    if policy.dry_run:
        return ArtifactDecision(
            artifact_kind=artifact_kind,
            output_root=output_root,
            action="would_run",
            reason=f"Dry-run: this {artifact_kind} would be generated because cache reuse was not possible.",
            identity_hash=identity_hash,
            manifest_found=manifest_found,
            matched_identity=matched_identity,
            missing_files=missing_files,
        )

    return ArtifactDecision(
        artifact_kind=artifact_kind,
        output_root=output_root,
        action="run",
        reason=f"Generating {artifact_kind} because heavy work is explicitly allowed.",
        identity_hash=identity_hash,
        manifest_found=manifest_found,
        matched_identity=matched_identity,
        missing_files=missing_files,
    )


def describe_policy(policy: ExecutionPolicy) -> dict[str, Any]:
    return {
        "reuse_only": policy.reuse_only,
        "allow_prepare": policy.allow_prepare,
        "allow_conditioning": policy.allow_conditioning,
        "allow_train": policy.allow_train,
        "allow_eval": policy.allow_eval,
        "dry_run": policy.dry_run,
        "resume": policy.resume,
        "force": policy.force,
    }
