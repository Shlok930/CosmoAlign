"""
Lunar Dataset Manager Module for CosmoAlign Phase 3.

Handles loading raw scientific TIFF images (Chandrayaan-2 OHRC and LRO NAC),
reading pair metadata JSON files, and generating non-destructive 8-bit display previews.
"""

import os
import json
from typing import Tuple, Dict, Any
import cv2
import numpy as np
import tifffile


def load_lunar_pair(
    pair_id: str = "pair_001",
    data_dir: str = "data"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads authentic scientific lunar image pair TIFFs and associated metadata JSON.

    CRITICAL SCIENTIFIC DISTINCTION:
    The returned raw NumPy arrays preserve exact 16-bit uint16 / 32-bit float scientific values.
    They are NOT modified or converted to 8-bit during loading.

    Args:
        pair_id (str): Identifier folder name under data_dir (e.g. 'pair_001').
        data_dir (str): Base data directory path.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - Raw Source image array (Chandrayaan-2 OHRC)
            - Raw Reference image array (LRO NAC)
            - Metadata dictionary from pair_info.json

    Raises:
        FileNotFoundError: If TIFF files or pair_info.json are missing.
    """
    pair_folder = os.path.join(data_dir, pair_id)
    source_path = os.path.join(pair_folder, "source.tif")
    ref_path = os.path.join(pair_folder, "reference.tif")
    meta_path = os.path.join(pair_folder, "metadata", "pair_info.json")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source lunar scientific TIFF not found at: {source_path}")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference lunar scientific TIFF not found at: {ref_path}")

    # Read raw scientific TIFF arrays preserving full bit-depth
    source_raw = tifffile.imread(source_path)
    ref_raw = tifffile.imread(ref_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return source_raw, ref_raw, metadata


def create_display_visualization(
    scientific_array: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0
) -> np.ndarray:
    """
    Creates a non-destructive 8-bit contrast-stretched preview for visual display.

    Performs 2%-98% percentile linear intensity clipping to make low-contrast
    lunar crater terrain details and shadow boundaries visible without affecting
    the underlying scientific array.

    Args:
        scientific_array (np.ndarray): Original 16-bit uint16 or 32-bit float image array.
        p_low (float): Lower percentile for contrast clipping.
        p_high (float): Upper percentile for contrast clipping.

    Returns:
        np.ndarray: Single-channel uint8 grayscale image array (0-255).
    """
    arr_float = scientific_array.astype(np.float32)

    # Exclude zero background nodata pixels from percentile calculation if present
    valid_mask = arr_float > 0
    if np.any(valid_mask):
        valid_vals = arr_float[valid_mask]
        vmin = np.percentile(valid_vals, p_low)
        vmax = np.percentile(valid_vals, p_high)
    else:
        vmin = np.min(arr_float)
        vmax = np.max(arr_float)

    if vmax <= vmin:
        vmax = vmin + 1.0

    # Clip dynamic range to percentiles and stretch to uint8 (0-255)
    clipped = np.clip(arr_float, vmin, vmax)
    normalized = ((clipped - vmin) / (vmax - vmin)) * 255.0
    display_uint8 = np.uint8(normalized)

    if np.any(~valid_mask):
        display_uint8[~valid_mask] = 0

    return display_uint8
