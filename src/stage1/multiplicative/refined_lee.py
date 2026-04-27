from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, uniform_filter


def _directional_kernel_bank() -> tuple[list[np.ndarray], list[np.ndarray]]:
    base_rect = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    base_diag = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )

    rect_kernels = []
    diag_kernels = []
    kernel = base_rect.copy()
    for _ in range(4):
        rect_kernels.append(kernel.copy())
        kernel = np.rot90(kernel)

    kernel = base_diag.copy()
    for _ in range(4):
        diag_kernels.append(kernel.copy())
        kernel = np.rot90(kernel)
    return rect_kernels, diag_kernels


def refined_lee_filter(intensity_image: np.ndarray, *, window_size: int = 7) -> np.ndarray:
    """Apply a Refined Lee speckle filter in the intensity domain.

    The implementation follows the standard directional-window approximation:
    local gradients are estimated from 3x3 samples inside a 7x7 neighborhood,
    a directional kernel is selected, and the Lee gain is computed against an
    estimated noise variance term.
    """

    if window_size != 7:
        raise ValueError("This Refined Lee implementation currently expects a 7x7 directional window.")

    image = np.asarray(intensity_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Refined Lee filtering expects a 2D intensity image.")

    image = np.clip(image, 0.0, None)
    eps = 1e-6

    mean_3 = uniform_filter(image, size=3, mode="reflect")
    mean_sq_3 = uniform_filter(image * image, size=3, mode="reflect")
    var_3 = np.maximum(mean_sq_3 - mean_3 * mean_3, 0.0)

    sample_mean = np.stack(
        [
            mean_3[:-6, :-6],
            mean_3[:-6, 3:-3],
            mean_3[:-6, 6:],
            mean_3[3:-3, :-6],
            mean_3[3:-3, 3:-3],
            mean_3[3:-3, 6:],
            mean_3[6:, :-6],
            mean_3[6:, 3:-3],
            mean_3[6:, 6:],
        ],
        axis=0,
    )
    sample_var = np.stack(
        [
            var_3[:-6, :-6],
            var_3[:-6, 3:-3],
            var_3[:-6, 6:],
            var_3[3:-3, :-6],
            var_3[3:-3, 3:-3],
            var_3[3:-3, 6:],
            var_3[6:, :-6],
            var_3[6:, 3:-3],
            var_3[6:, 6:],
        ],
        axis=0,
    )

    gradients = np.stack(
        [
            np.abs(sample_mean[1] - sample_mean[7]),
            np.abs(sample_mean[6] - sample_mean[2]),
            np.abs(sample_mean[3] - sample_mean[5]),
            np.abs(sample_mean[0] - sample_mean[8]),
        ],
        axis=0,
    )
    max_gradient = np.argmax(gradients, axis=0)

    directions = np.zeros_like(max_gradient, dtype=np.int32)
    directions[max_gradient == 0] = 1 + (sample_mean[1][max_gradient == 0] > sample_mean[7][max_gradient == 0])
    directions[max_gradient == 1] = 3 + (sample_mean[6][max_gradient == 1] > sample_mean[2][max_gradient == 1])
    directions[max_gradient == 2] = 5 + (sample_mean[3][max_gradient == 2] > sample_mean[5][max_gradient == 2])
    directions[max_gradient == 3] = 7 + (sample_mean[0][max_gradient == 3] > sample_mean[8][max_gradient == 3])

    sample_noise = np.sort(sample_var / np.maximum(sample_mean * sample_mean, eps), axis=0)[:5]
    sigma_v = np.mean(sample_noise, axis=0)

    rect_kernels, diag_kernels = _directional_kernel_bank()
    directional_mean = np.zeros_like(image, dtype=np.float32)
    directional_var = np.zeros_like(image, dtype=np.float32)

    all_kernels = rect_kernels + diag_kernels
    for direction_index, kernel in enumerate(all_kernels, start=1):
        kernel_norm = kernel / np.sum(kernel)
        dir_mean = convolve(image, kernel_norm, mode="reflect")
        dir_mean_sq = convolve(image * image, kernel_norm, mode="reflect")
        dir_var = np.maximum(dir_mean_sq - dir_mean * dir_mean, 0.0)
        mask = directions == direction_index
        if not np.any(mask):
            continue
        directional_mean[3:-3, 3:-3][mask] = dir_mean[3:-3, 3:-3][mask]
        directional_var[3:-3, 3:-3][mask] = dir_var[3:-3, 3:-3][mask]

    directional_mean[:3, :] = mean_3[:3, :]
    directional_mean[-3:, :] = mean_3[-3:, :]
    directional_mean[:, :3] = mean_3[:, :3]
    directional_mean[:, -3:] = mean_3[:, -3:]
    directional_var[:3, :] = var_3[:3, :]
    directional_var[-3:, :] = var_3[-3:, :]
    directional_var[:, :3] = var_3[:, :3]
    directional_var[:, -3:] = var_3[:, -3:]

    sigma_full = np.full_like(image, float(np.mean(sigma_v)), dtype=np.float32)
    sigma_full[3:-3, 3:-3] = sigma_v.astype(np.float32)

    var_x = np.maximum((directional_var - directional_mean * directional_mean * sigma_full) / (sigma_full + 1.0), 0.0)
    gain = np.where(directional_var > eps, var_x / np.maximum(directional_var, eps), 0.0)
    gain = np.clip(gain, 0.0, 1.0)
    filtered = directional_mean + gain * (image - directional_mean)
    return np.clip(filtered.astype(np.float32), 0.0, None)
