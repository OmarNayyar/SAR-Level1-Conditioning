"""Ship-detection downstream validation helpers.

This package is intentionally small and optional.  It prepares SSDD/HRSID
manifests for a YOLO-style detector and runs Ultralytics when that dependency is
installed.  The repo should still work without Ultralytics; in that case the
preparation artifacts are written and the detector status is reported honestly
as `dependency_missing`.
"""

from .yolo_dataset import (
    PreparedYoloDataset,
    load_prepared_yolo_dataset,
    prepare_yolo_dataset,
    prepared_yolo_artifact_identity,
)
from .ultralytics_runner import (
    DetectorRunResult,
    MissingDetectorDependency,
    detector_run_artifact_identity,
    load_detector_run_result,
    run_ultralytics_detector,
    ultralytics_available,
)

__all__ = [
    "DetectorRunResult",
    "MissingDetectorDependency",
    "PreparedYoloDataset",
    "detector_run_artifact_identity",
    "load_detector_run_result",
    "load_prepared_yolo_dataset",
    "prepare_yolo_dataset",
    "prepared_yolo_artifact_identity",
    "run_ultralytics_detector",
    "ultralytics_available",
]
