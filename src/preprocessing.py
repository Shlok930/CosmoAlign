"""
Controlled Preprocessing Experiments Module for CosmoAlign Phase 3.

Provides isolated preprocessing functions for scientific lunar imagery:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Percentile Radiometric Contrast Stretching
- Valid-Data Masking for nodata border exclusion
- Controlled Image Resampling / Rescaling
"""

from typing import Tuple
import cv2
import numpy as np


def apply_clahe(
    gray_img: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

    WHY FOR LUNAR IMAGERY:
    Lunar surface imagery acquired under low solar elevation angles contains extreme
    contrast differences between brightly sunlit crater rims and pitch-black shadowed slopes.
    CLAHE enhances local contrast in sub-regions without over-amplifying global noise.

    Args:
        gray_img (np.ndarray): Single-channel uint8 grayscale image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization.

    Returns:
        np.ndarray: Enhanced single-channel uint8 grayscale image.
    """
    if len(gray_img.shape) != 2 or gray_img.dtype != np.uint8:
        raise ValueError("CLAHE requires a single-channel uint8 grayscale image.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray_img)
    return enhanced


def apply_percentile_stretch(
    gray_img: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0
) -> np.ndarray:
    """
    Applies 2%-98% radiometric percentile contrast stretching to single-channel image.
    """
    arr_float = gray_img.astype(np.float32)
    valid_mask = arr_float > 0

    if np.any(valid_mask):
        valid_vals = arr_float[valid_mask]
        vmin = np.percentile(valid_vals, p_low)
        vmax = np.percentile(valid_vals, p_high)
    else:
        vmin, vmax = np.min(arr_float), np.max(arr_float)

    if vmax <= vmin:
        vmax = vmin + 1.0

    clipped = np.clip(arr_float, vmin, vmax)
    normalized = ((clipped - vmin) / (vmax - vmin)) * 255.0
    stretched = np.uint8(normalized)

    if np.any(~valid_mask):
        stretched[~valid_mask] = 0

    return stretched


def create_valid_mask(gray_img: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Generates a binary valid-data mask (255 for valid surface, 0 for nodata background).
    Prevents SIFT from placing artificial keypoints on sharp black image borders.
    """
    mask = np.zeros_like(gray_img, dtype=np.uint8)
    mask[gray_img > threshold] = 255
    return mask


def apply_scaling(gray_img: np.ndarray, scale_factor: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Controlled isotropic image scaling / resampling.

    Args:
        gray_img (np.ndarray): Input grayscale image.
        scale_factor (float): Multiplier for image width and height.

    Returns:
        Tuple[np.ndarray, float]: (Rescaled image array, actual applied scale factor)
    """
    if scale_factor == 1.0 or scale_factor <= 0:
        return gray_img, 1.0

    h, w = gray_img.shape[:2]
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))

    interp = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR
    scaled_img = cv2.resize(gray_img, (new_w, new_h), interpolation=interp)
    return scaled_img, scale_factor
