from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SPLIT_DIRS = {
    "train": ("Noisy", "GTruth"),
    "val": ("Noisy_val", "GTruth_val"),
}


@dataclass(frozen=True, slots=True)
class MendeleyPair:
    split: str
    noisy_path: Path
    reference_path: Path
    pair_id: str

    def to_row(self) -> dict[str, str]:
        return {
            "split": self.split,
            "pair_id": self.pair_id,
            "noisy_path": self.noisy_path.resolve().as_posix(),
            "reference_path": self.reference_path.resolve().as_posix(),
        }


def _image_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.relative_to(root).with_suffix("").as_posix()
        files[key] = path
    return files


def _pair_id(split: str, key: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in key)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return f"{split}_{cleaned.strip('_')}"


def discover_mendeley_pairs(root: str | Path, split: str | None = None) -> list[MendeleyPair]:
    """Discover noisy/reference SAR despeckling pairs by relative filename stem."""

    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Mendeley SAR dataset root was not found: {dataset_root.as_posix()}")

    requested_splits = [split] if split else list(SPLIT_DIRS)
    pairs: list[MendeleyPair] = []
    for split_name in requested_splits:
        if split_name not in SPLIT_DIRS:
            raise ValueError(f"Unsupported Mendeley split {split_name!r}; expected one of {sorted(SPLIT_DIRS)}.")
        noisy_dir_name, reference_dir_name = SPLIT_DIRS[split_name]
        noisy_dir = dataset_root / noisy_dir_name
        reference_dir = dataset_root / reference_dir_name
        if not noisy_dir.exists():
            raise FileNotFoundError(f"Mendeley noisy directory was not found: {noisy_dir.as_posix()}")
        if not reference_dir.exists():
            raise FileNotFoundError(f"Mendeley reference directory was not found: {reference_dir.as_posix()}")

        noisy_files = _image_files(noisy_dir)
        reference_files = _image_files(reference_dir)
        for key in sorted(set(noisy_files).intersection(reference_files)):
            pairs.append(
                MendeleyPair(
                    split=split_name,
                    noisy_path=noisy_files[key],
                    reference_path=reference_files[key],
                    pair_id=_pair_id(split_name, key),
                )
            )
    return pairs

