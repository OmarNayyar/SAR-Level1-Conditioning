from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def run_external_array_command(
    *,
    input_array: np.ndarray,
    command_template: list[str],
    cwd: str | None = None,
) -> np.ndarray:
    """Run an external method that reads `{input}` and writes `{output}` `.npy` files."""

    if not command_template:
        raise ValueError("command_template must not be empty.")

    if shutil.which(command_template[0]) is None and not Path(command_template[0]).exists():
        raise FileNotFoundError(f"External executable was not found: {command_template[0]}")

    with tempfile.TemporaryDirectory(prefix="stage1_external_") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / "input.npy"
        output_path = temp_root / "output.npy"
        np.save(input_path, np.asarray(input_array))
        command = [
            token.format(input=input_path.as_posix(), output=output_path.as_posix(), temp_dir=temp_root.as_posix())
            for token in command_template
        ]
        subprocess.run(command, cwd=cwd, check=True)
        if not output_path.exists():
            raise RuntimeError(
                "External command completed without writing the expected output file. "
                "Expected path: "
                f"{output_path.as_posix()}"
            )
        return np.load(output_path)
