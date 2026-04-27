from __future__ import annotations

"""Validate the intended public/private repository surface before export."""

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _manifest_path(surface: str) -> Path:
    name = "public_release_manifest.yaml" if surface == "public" else "private_handoff_manifest.yaml"
    return REPO_ROOT / "manifests" / name


def _load_manifest(surface: str) -> dict[str, Any]:
    path = _manifest_path(surface)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path.as_posix()}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest {path.as_posix()} must contain a mapping.")
    return payload


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _expand_entry(entry: str) -> list[Path]:
    pattern = entry.replace("\\", "/")
    if any(token in pattern for token in ("*", "?", "[")):
        return sorted(REPO_ROOT.glob(pattern))
    path = REPO_ROOT / entry
    return [path] if path.exists() else []


def _is_excluded(rel_path: str, exclude_patterns: list[str]) -> bool:
    for raw_pattern in exclude_patterns:
        pattern = raw_pattern.replace("\\", "/").strip()
        if not pattern:
            continue
        directory_pattern = pattern.rstrip("/")
        if pattern.endswith("/") and (rel_path == directory_pattern or rel_path.startswith(directory_pattern + "/")):
            return True
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        if rel_path.startswith(directory_pattern + "/"):
            return True
    return False


def _collect_files(paths: list[Path], exclude_patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            rel_path = _rel(path)
            if not _is_excluded(rel_path, exclude_patterns):
                files.append(path)
            continue
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            rel_path = _rel(child)
            if _is_excluded(rel_path, exclude_patterns):
                continue
            files.append(child)
    return sorted(set(files), key=lambda item: item.as_posix())


def _text_files(files: list[Path]) -> list[Path]:
    return [path for path in files if path.suffix.lower() in TEXT_SUFFIXES]


def _missing_required(manifest: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for entry in manifest.get("required_paths", []) or []:
        if not _expand_entry(str(entry)):
            missing.append(str(entry))
    return missing


def _included_roots(manifest: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for key in ("required_paths", "optional_paths"):
        for entry in manifest.get(key, []) or []:
            roots.extend(_expand_entry(str(entry)))
    return roots


def _keyword_warnings(files: list[Path], manifest: dict[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    keyword_specs = manifest.get("forbidden_keywords", []) or []
    for path in _text_files(files):
        rel_path = _rel(path)
        if rel_path.startswith("manifests/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for spec in keyword_specs:
            keyword = str(spec.get("keyword", "") if isinstance(spec, dict) else spec)
            if not keyword:
                continue
            if keyword.lower() in text.lower():
                warnings.append(
                    {
                        "path": _rel(path),
                        "keyword": keyword,
                        "reason": str(spec.get("reason", "")) if isinstance(spec, dict) else "",
                    }
                )
    return warnings


def check_surface(surface: str) -> dict[str, Any]:
    manifest = _load_manifest(surface)
    exclude_patterns = [str(item) for item in (manifest.get("exclude_paths", []) or [])]
    missing = _missing_required(manifest)
    included_files = _collect_files(_included_roots(manifest), exclude_patterns)
    warnings = _keyword_warnings(included_files, manifest)
    return {
        "surface": surface,
        "manifest": _manifest_path(surface).relative_to(REPO_ROOT).as_posix(),
        "missing_required_paths": missing,
        "included_file_count": len(included_files),
        "keyword_warning_count": len(warnings),
        "keyword_warnings": warnings,
        "required_checks": manifest.get("required_checks", []) or [],
        "notes": manifest.get("notes", []) or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the intended public/private repository surface before export.")
    parser.add_argument("--surface", choices=["public", "private"], required=True)
    parser.add_argument("--json", action="store_true", help="Print the full check result as JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on keyword warnings as well as missing files.")
    args = parser.parse_args()

    result = check_surface(args.surface)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Surface: {result['surface']}")
        print(f"Manifest: {result['manifest']}")
        print(f"Included files: {result['included_file_count']}")
        print(f"Missing required paths: {len(result['missing_required_paths'])}")
        for item in result["missing_required_paths"]:
            print(f"  MISSING {item}")
        print(f"Keyword warnings: {result['keyword_warning_count']}")
        for warning in result["keyword_warnings"]:
            reason = f" - {warning['reason']}" if warning.get("reason") else ""
            print(f"  WARNING {warning['path']}: {warning['keyword']}{reason}")

    if result["missing_required_paths"] or (args.strict and result["keyword_warnings"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
