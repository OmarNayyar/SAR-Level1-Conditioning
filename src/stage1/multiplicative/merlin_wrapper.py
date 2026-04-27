from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.stage1.additive.starlet_complex_denoise import starlet_complex_denoise
from src.stage1.external import run_external_array_command


@dataclass(slots=True)
class MerlinResult:
    output_complex: np.ndarray
    output_intensity: np.ndarray
    backend: str
    fallback_used: bool
    notes: str


def run_merlin_wrapper(
    complex_image: np.ndarray,
    *,
    external: dict[str, Any] | None = None,
    fallback_levels: int = 3,
    fallback_threshold_scale: float = 2.5,
) -> MerlinResult:
    complex_array = np.asarray(complex_image)
    if not np.iscomplexobj(complex_array):
        raise ValueError("MERLIN wrapper expects a complex-valued image.")

    external = external or {}
    command_template = external.get("command")
    cwd = external.get("cwd")
    if isinstance(command_template, list) and command_template:
        stacked = np.stack((np.real(complex_array), np.imag(complex_array)), axis=-1).astype(np.float32)
        external_output = run_external_array_command(
            input_array=stacked,
            command_template=[str(token) for token in command_template],
            cwd=str(cwd) if cwd else None,
        )
        if external_output.ndim == 3 and external_output.shape[-1] == 2:
            output_complex = external_output[..., 0].astype(np.float32) + 1j * external_output[..., 1].astype(np.float32)
        else:
            raise RuntimeError("MERLIN external backend must return a 2-channel real/imag array saved as .npy.")
        output_intensity = np.square(np.abs(output_complex), dtype=np.float32)
        return MerlinResult(
            output_complex=output_complex.astype(np.complex64),
            output_intensity=output_intensity.astype(np.float32),
            backend="external_command",
            fallback_used=False,
            notes="Ran the configured external MERLIN-style backend via a command adapter.",
        )

    fallback = starlet_complex_denoise(
        complex_array,
        levels=int(fallback_levels),
        threshold_scale=float(fallback_threshold_scale),
    )
    return MerlinResult(
        output_complex=fallback.denoised_complex.astype(np.complex64),
        output_intensity=fallback.surrogate_intensity.astype(np.float32),
        backend="local_starlet_fallback",
        fallback_used=True,
        notes=(
            "MERLIN was not configured as an external backend, so Bundle C used the local starlet complex fallback. "
            "This is honest feasibility-mode behavior, not a claim of upstream MERLIN parity."
        ),
    )
