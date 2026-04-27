from __future__ import annotations

"""Generate compact public/private project summary artifacts."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reporting import write_surface_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write public-safe and/or private recommendation summaries.")
    parser.add_argument("--surface", choices=["public", "private", "all"], default="all", help="Which summary pack to write.")
    args = parser.parse_args()
    paths = write_surface_artifacts(REPO_ROOT, surface=args.surface)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
