from __future__ import annotations

"""Optional Ultralytics YOLO runner for real downstream validation.

The rest of the repo does not import Ultralytics at module import time.  That
keeps dataset setup, bundle screening, and the Streamlit app lightweight while
still making a real detector path available when the dependency is installed.
"""

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import payload_fingerprint, read_artifact_manifest, write_artifact_manifest


class MissingDetectorDependency(RuntimeError):
    """Raised when the optional detector backend is not installed."""


@dataclass(slots=True)
class DetectorRunResult:
    status: str
    model: str
    run_dir: str
    metrics: dict[str, Any]
    notes: list[str]


def detector_run_artifact_identity(
    *,
    dataset_yaml: Path,
    variant_name: str,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str | None,
    eval_split: str,
    prepared_identity_hash: str,
) -> dict[str, Any]:
    return {
        "artifact_kind": "detector_run",
        "dataset_yaml": dataset_yaml.resolve().as_posix(),
        "dataset_yaml_mtime": dataset_yaml.stat().st_mtime if dataset_yaml.exists() else None,
        "variant_name": variant_name,
        "model": model,
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "device": device or "",
        "eval_split": eval_split,
        "prepared_identity_hash": prepared_identity_hash,
    }


def load_detector_run_result(run_root: Path) -> DetectorRunResult | None:
    manifest = read_artifact_manifest(run_root)
    summary_path = run_root / "detector_run_result.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    notes = list(payload.get("notes", [])) if isinstance(payload.get("notes"), list) else []
    if manifest and isinstance(manifest.get("notes"), list):
        notes.extend(str(note) for note in manifest.get("notes", []) if str(note))
    return DetectorRunResult(
        status=str(payload.get("status", "completed")),
        model=str(payload.get("model", "")),
        run_dir=str(payload.get("run_dir", "")),
        metrics=dict(payload.get("metrics", {})),
        notes=notes,
    )


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def _extract_metrics(result: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    box = getattr(result, "box", None)
    if box is not None:
        for key, attr in {
            "map": "map",
            "map50": "map50",
            "map75": "map75",
            "precision": "mp",
            "recall": "mr",
        }.items():
            value = getattr(box, attr, None)
            if value is not None:
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    metrics[key] = str(value)
    results_dict = getattr(result, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            normalized_key = str(key).replace("metrics/", "").replace("(B)", "").strip().lower()
            try:
                metrics.setdefault(normalized_key, float(value))
            except (TypeError, ValueError):
                metrics.setdefault(normalized_key, str(value))
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    if isinstance(precision, float) and isinstance(recall, float):
        metrics["f1"] = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return metrics


class _SerialPool:
    """Drop-in serial replacement for Ultralytics label-cache ThreadPool.

    Some locked-down Windows environments deny the IPC pipe creation used by
    `multiprocessing.pool.ThreadPool`, even with Ultralytics `workers=0`.
    Label verification is tiny for our smoke datasets, so serial execution is a
    safer default than failing before any detector metric can be produced.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_SerialPool":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def imap(self, func: Any, iterable: Any) -> Any:
        for item in iterable:
            yield func(item)


def _patch_ultralytics_windows_cache_builder() -> None:
    if os.name != "nt":
        return
    try:
        import ultralytics.data.dataset as dataset_module  # type: ignore
    except Exception:
        return
    dataset_module.ThreadPool = _SerialPool


def _resolve_trained_model_path(
    *,
    trainer_model: Any,
    output_root: Path,
    project_root: Path,
    run_name: str,
) -> tuple[Path | None, Path]:
    """Return the actual Ultralytics `best.pt` path and run directory.

    Ultralytics may redirect runs under a user/config-controlled `Ultralytics/`
    folder even when `project=` is supplied.  On Windows this showed up as an
    uppercase `Ultralytics` directory while our original code expected a
    lowercase `ultralytics` path.  Evaluating the wrong path silently falls
    back to the base model, so we resolve the trained weight path from the
    trainer object first and then scan the output root as a safety net.
    """

    candidate_models: list[Path] = []
    candidate_dirs: list[Path] = []
    trainer = getattr(trainer_model, "trainer", None)
    if trainer is not None:
        best = getattr(trainer, "best", None)
        save_dir = getattr(trainer, "save_dir", None)
        if best:
            candidate_models.append(Path(best))
        if save_dir:
            save_dir_path = Path(save_dir)
            candidate_dirs.append(save_dir_path)
            candidate_models.append(save_dir_path / "weights" / "best.pt")

    expected_dir = project_root / run_name
    candidate_dirs.append(expected_dir)
    candidate_models.append(expected_dir / "weights" / "best.pt")
    candidate_models.append(output_root / "Ultralytics" / run_name / "weights" / "best.pt")

    for model_path in candidate_models:
        if model_path.exists():
            return model_path.resolve(), model_path.parent.parent.resolve()

    # Last-resort scan keeps future Ultralytics path changes from causing an
    # unnoticed fallback to the pretrained COCO model.
    for model_path in output_root.rglob("best.pt"):
        if model_path.parent.name == "weights" and model_path.parent.parent.name == run_name:
            return model_path.resolve(), model_path.parent.parent.resolve()

    run_dir = next((path for path in candidate_dirs if path.exists()), expected_dir)
    return None, run_dir.resolve()


def run_ultralytics_detector(
    *,
    dataset_yaml: Path,
    output_root: Path,
    variant_name: str,
    model: str = "yolov8n.pt",
    epochs: int = 3,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 0,
    device: str | None = None,
    eval_split: str = "test",
    prepared_identity_hash: str = "",
) -> DetectorRunResult:
    """Train a compact YOLO detector and evaluate it on the prepared dataset.

    `workers=0` is the conservative Windows-friendly default.  It avoids
    multiprocessing dataloader/cache edge cases that can look like generic
    `Access is denied` failures on local laptops.
    """

    if not ultralytics_available():
        raise MissingDetectorDependency(
            "Ultralytics is not installed. Install it with `pip install ultralytics` "
            "when you are ready to run the detector baseline."
        )

    # Ultralytics appends its own `Ultralytics/` child under this config root.
    # Point it at the run output root so settings/fonts stay in ignored outputs
    # instead of leaking into the user's AppData or the repository root.
    os.environ.setdefault("YOLO_CONFIG_DIR", output_root.as_posix())
    _patch_ultralytics_windows_cache_builder()

    from ultralytics import YOLO  # type: ignore

    project_root = output_root / "ultralytics"
    run_name = variant_name
    trainer = YOLO(model)
    train_kwargs: dict[str, Any] = {
        "data": dataset_yaml.as_posix(),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "project": project_root.as_posix(),
        "name": run_name,
        "exist_ok": True,
    }
    if device:
        train_kwargs["device"] = device
    trainer.train(**train_kwargs)

    best_model, run_dir = _resolve_trained_model_path(
        trainer_model=trainer,
        output_root=output_root,
        project_root=project_root,
        run_name=run_name,
    )
    evaluator = YOLO(best_model.as_posix() if best_model is not None else model)
    val_kwargs: dict[str, Any] = {
        "data": dataset_yaml.as_posix(),
        "split": eval_split,
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "project": project_root.as_posix(),
        "name": f"{run_name}_eval",
        "exist_ok": True,
    }
    if device:
        val_kwargs["device"] = device
    metrics_result = evaluator.val(**val_kwargs)
    notes = [
        "This is a lightweight detector baseline. Treat metrics as downstream validation evidence for this setup, not as a tuned SOTA claim."
    ]
    if best_model is None:
        notes.append(
            "WARNING: trained best.pt was not found, so evaluation used the base model. Treat this run as invalid for raw-vs-conditioned comparison."
        )
    result = DetectorRunResult(
        status="completed",
        model=best_model.as_posix() if best_model is not None else model,
        run_dir=run_dir.resolve().as_posix(),
        metrics=_extract_metrics(metrics_result),
        notes=notes,
    )
    run_root = Path(result.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "detector_run_result.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": result.status,
                "model": result.model,
                "run_dir": result.run_dir,
                "metrics": result.metrics,
                "notes": result.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_artifact_manifest(
        run_root,
        artifact_kind="detector_run",
        identity=detector_run_artifact_identity(
            dataset_yaml=dataset_yaml,
            variant_name=variant_name,
            model=model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            eval_split=eval_split,
            prepared_identity_hash=prepared_identity_hash
            or payload_fingerprint(
                {
                    "dataset_yaml": dataset_yaml.resolve().as_posix(),
                    "variant_name": variant_name,
                }
            ),
        ),
        status=result.status,
        files={
            "detector_run_result": summary_path.resolve().as_posix(),
            "run_dir": run_root.resolve().as_posix(),
            "trained_model": result.model,
        },
        metadata={"metrics": result.metrics},
        notes=result.notes,
    )
    return result
