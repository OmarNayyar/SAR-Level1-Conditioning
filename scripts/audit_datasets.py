from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.audit import audit_registered_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit all registered datasets and save machine-readable summaries plus previews.")
    parser.add_argument("--preview-count", type=int, default=4, help="Maximum number of preview images to save per dataset.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "data_audit"),
        help="Audit output directory.",
    )
    parser.add_argument(
        "--docs-summary",
        default=str(REPO_ROOT / "docs" / "data_audit_summary.md"),
        help="Markdown summary path for docs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_registered_datasets(
        REPO_ROOT,
        preview_count=args.preview_count,
        output_root=Path(args.output_dir),
        docs_summary_path=Path(args.docs_summary),
    )


if __name__ == "__main__":
    main()
