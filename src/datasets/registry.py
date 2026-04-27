from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common import DatasetStatus, write_json


@dataclass(slots=True)
class DatasetRegistration:
    dataset_name: str
    manifest_path: str
    local_path: str = ""
    external_path: str = ""
    remote_source: str = ""
    split_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    status: str = DatasetStatus.MISSING.value
    sample_count: int = 0
    size_bytes: int | None = None


class DatasetRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: dict[str, DatasetRegistration] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        datasets = payload.get("datasets", {})
        for dataset_name, record in datasets.items():
            self._records[dataset_name] = DatasetRegistration(**record)

    def upsert(self, registration: DatasetRegistration) -> None:
        self._records[registration.dataset_name] = registration

    def get(self, dataset_name: str) -> DatasetRegistration | None:
        return self._records.get(dataset_name)

    def save(self) -> None:
        payload = {
            "datasets": {
                name: asdict(record)
                for name, record in sorted(self._records.items(), key=lambda item: item[0])
            }
        }
        write_json(self.path, payload)


def default_registry_path(repo_root: Path) -> Path:
    import os

    override_root = os.getenv("SAR_DATA_LAYOUT_ROOT")
    data_root = Path(override_root).expanduser().resolve() if override_root else repo_root / "data"

    preferred = data_root / "external" / "manifests" / "dataset_registry.json"
    if preferred.parent.exists():
        return preferred
    return data_root / "metadata_indexes" / "dataset_registry.json"
