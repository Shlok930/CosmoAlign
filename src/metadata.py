"""
Scientific Metadata Inspection Module for CosmoAlign Phase 3.

Provides functions to analyze image array statistics (shape, dtype, min, max, mean, std,
nodata count) and compare spatial resolutions/scale ratios.
"""

from typing import Dict, Any, Tuple
import numpy as np


def inspect_image_stats(image_array: np.ndarray, nodata_value: float = 0.0) -> Dict[str, Any]:
    """
    Computes numerical statistics for a scientific image array.

    Args:
        image_array (np.ndarray): Scientific input image array (uint8, uint16, float32, etc.).
        nodata_value (float): Numerical value representing invalid/nodata background pixels.

    Returns:
        Dict[str, Any]: Dictionary containing shape, dtype, channels, min, max, mean, std, nodata_count.
    """
    shape = image_array.shape
    dtype_str = str(image_array.dtype)
    channels = 1 if len(shape) == 2 else shape[2]

    arr_float = image_array.astype(np.float64)
    nodata_mask = (arr_float == nodata_value)
    nodata_count = int(np.sum(nodata_mask))

    valid_pixels = arr_float[~nodata_mask] if np.any(~nodata_mask) else arr_float

    if len(valid_pixels) > 0:
        min_val = float(np.min(valid_pixels))
        max_val = float(np.max(valid_pixels))
        mean_val = float(np.mean(valid_pixels))
        std_val = float(np.std(valid_pixels))
    else:
        min_val, max_val, mean_val, std_val = 0.0, 0.0, 0.0, 0.0

    return {
        "shape": shape,
        "dtype": dtype_str,
        "channels": channels,
        "min": min_val,
        "max": max_val,
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "nodata_count": nodata_count,
        "valid_pixel_ratio": round(float(len(valid_pixels)) / float(image_array.size) * 100.0, 2)
    }


def calculate_scale_ratio(
    source_res_m: float,
    reference_res_m: float
) -> Tuple[float, float]:
    """
    Calculates spatial resolution ratio between Reference GSD and Source GSD.

    Returns:
        Tuple[float, float]: (Scale ratio Ref/Source, Inverse ratio Source/Ref)
    """
    if source_res_m <= 0 or reference_res_m <= 0:
        return 1.0, 1.0

    ratio = reference_res_m / source_res_m
    inv_ratio = source_res_m / reference_res_m
    return ratio, inv_ratio


def format_metadata_report(
    source_stats: Dict[str, Any],
    ref_stats: Dict[str, Any],
    pair_info: Dict[str, Any]
) -> str:
    """Formats a clean console string for Phase 3 Data Inspection Report."""
    lines = [
        "=" * 65,
        " COSMOALIGN PHASE 3 DATA INSPECTION REPORT ",
        "=" * 65,
        "SOURCE IMAGE (Chandrayaan-2 OHRC):",
        f"  * Product ID:   {pair_info.get('source_product', 'N/A')}",
        f"  * Resolution:   {pair_info.get('source_resolution_m_per_px', 'N/A')} m/px GSD",
        f"  * Dimensions:   {source_stats['shape'][1]}x{source_stats['shape'][0]} px ({source_stats['channels']} channel)",
        f"  * Data Type:    {source_stats['dtype']}",
        f"  * Dynamic Range: [{source_stats['min']:.1f} to {source_stats['max']:.1f}]",
        f"  * Intensity Mean ± Std: {source_stats['mean']:.1f} ± {source_stats['std']:.1f}",
        f"  * Valid Pixel Coverage: {source_stats['valid_pixel_ratio']}%",
        "",
        "REFERENCE IMAGE (LRO NAC):",
        f"  * Product ID:   {pair_info.get('reference_product', 'N/A')}",
        f"  * Resolution:   {pair_info.get('reference_resolution_m_per_px', 'N/A')} m/px GSD",
        f"  * Dimensions:   {ref_stats['shape'][1]}x{ref_stats['shape'][0]} px ({ref_stats['channels']} channel)",
        f"  * Data Type:    {ref_stats['dtype']}",
        f"  * Dynamic Range: [{ref_stats['min']:.1f} to {ref_stats['max']:.1f}]",
        f"  * Intensity Mean ± Std: {ref_stats['mean']:.1f} ± {ref_stats['std']:.1f}",
        f"  * Valid Pixel Coverage: {ref_stats['valid_pixel_ratio']}%",
        "",
        "FOOTPRINT & SCALE SUMMARY:",
        f"  * Overlap Region Verified: {pair_info.get('same_region_verified', False)}",
        f"  * Nominal Scale Ratio (Ref/Source): {pair_info.get('nominal_scale_ratio_ref_to_source', 1.0)}x",
        "=" * 65
    ]
    return "\n".join(lines)
