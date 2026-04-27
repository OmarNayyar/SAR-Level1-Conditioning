from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from src.stage1.pipeline import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundle_configs_have_expected_shape() -> None:
    for bundle_name in ("bundle_a", "bundle_b", "bundle_c", "bundle_d"):
        config_path = REPO_ROOT / "configs" / f"{bundle_name}.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert payload["bundle"]["name"] == bundle_name
        assert "dataset" in payload
        assert "processing" in payload
        assert "outputs" in payload


def test_bundle_a_defaults_to_ssdd_train_subset() -> None:
    payload = yaml.safe_load((REPO_ROOT / "configs" / "bundle_a.yaml").read_text(encoding="utf-8"))
    assert payload["dataset"]["name"] == "ssdd"
    assert payload["dataset"]["split"] == "train"
    assert int(payload["dataset"]["sample_limit"]) > 0


def test_repo_configs_pass_lightweight_validation() -> None:
    for bundle_name in ("bundle_a", "bundle_b", "bundle_c", "bundle_d", "bundle_a_conservative"):
        load_yaml(REPO_ROOT / "configs" / f"{bundle_name}.yaml", expected_kind="bundle")
    load_yaml(REPO_ROOT / "configs" / "bundles" / "profiles" / "bundle_a_conservative.yaml", expected_kind="bundle")
    for detector_name in ("yolo_smoke", "yolo_medium", "yolo_serious"):
        load_yaml(REPO_ROOT / "configs" / "downstream" / f"{detector_name}.yaml", expected_kind="detection")
    load_yaml(REPO_ROOT / "configs" / "final_sweep.yaml", expected_kind="final_sweep")


def test_bundle_validation_rejects_unknown_top_level_keys() -> None:
    workspace = REPO_ROOT / "outputs" / "test-workspaces" / uuid4().hex
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "invalid_bundle.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "bundle": {"name": "bundle_a"},
                "dataset": {"name": "ssdd"},
                "processing": {"additive": {"submethod": "A0"}, "multiplicative": {"method": "refined_lee"}},
                "outputs": {"root": "results/bundle_a"},
                "typo_section": {"unexpected": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported key"):
        load_yaml(config_path, expected_kind="bundle")
