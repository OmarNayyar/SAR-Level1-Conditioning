from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.common import write_json
from src.datasets.external_detection_adapter import validate_external_detection_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a future external or private ship-detection dataset config.")
    parser.add_argument("--config", required=True, help="Dataset adapter YAML under configs/datasets/ or an absolute path.")
    parser.add_argument("--output", default="outputs/external_dataset_validation/validation_report.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    report = validate_external_detection_dataset(config_path)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()
    write_json(output_path, report.to_dict())
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
