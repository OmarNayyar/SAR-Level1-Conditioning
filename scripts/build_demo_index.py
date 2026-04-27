from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reporting.demo_examples import write_demo_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lean Streamlit demo-example index from existing outputs.")
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()
    path = write_demo_index(REPO_ROOT, max_examples=args.max_examples)
    print(json.dumps({"demo_index": path.resolve().as_posix()}, indent=2))


if __name__ == "__main__":
    main()

