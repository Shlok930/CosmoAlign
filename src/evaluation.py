"""
Scientific Evaluation, Spatial Distribution, and Homography Sanity Module for CosmoAlign Phase 3.

Provides functions to analyze:
- Inlier Spatial Distribution across a 3x3 grid panel (coverage ratio & entropy)
- Homography Transformation Sanity (detecting area collapse, flip, or extreme shear)
- Multi-Percentile Reprojection Errors (RMSE, Median, 95th Percentile, Max Error)
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


def compute_reprojection_stats(
    kp_source: List[cv2.KeyPoint],
    kp_ref: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    H: np.ndarray,
    inliers_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Computes precise pixel reprojection error statistics across inlier correspondences.

    Returns:
        Dict[str, Any]: Dictionary containing rmse, median_px, p95_px, max_px, total_inliers.
    """
    if H is None or matches is None or len(matches) == 0 or inliers_mask is None:
        return {"rmse": None, "median_px": None, "p95_px": None, "max_px": None, "total_inliers": 0}

    mask_flat = inliers_mask.ravel()
    inlier_indices = np.where(mask_flat == 1)[0]

    if len(inlier_indices) == 0:
        return {"rmse": None, "median_px": None, "p95_px": None, "max_px": None, "total_inliers": 0}

    src_pts = np.float32([kp_source[matches[i].queryIdx].pt for i in inlier_indices]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp_ref[matches[i].trainIdx].pt for i in inlier_indices]).reshape(-1, 1, 2)

    try:
        transformed_src = cv2.perspectiveTransform(src_pts, H)
    except Exception:
        return {"rmse": None, "median_px": None, "p95_px": None, "max_px": None, "total_inliers": len(inlier_indices)}

    # Euclidean distance errors in pixels
    diff = transformed_src - ref_pts
    errors = np.sqrt(np.sum(diff ** 2, axis=2)).ravel() # Shape (N,)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_px = float(np.median(errors))
    p95_px = float(np.percentile(errors, 95))
    max_px = float(np.max(errors))

    return {
        "rmse": round(rmse, 4),
        "median_px": round(median_px, 4),
        "p95_px": round(p95_px, 4),
        "max_px": round(max_px, 4),
        "total_inliers": len(inlier_indices)
    }


def analyze_inlier_spatial_distribution(
    kp_source: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inliers_mask: np.ndarray,
    source_shape: Tuple[int, int],
    grid_rows: int = 3,
    grid_cols: int = 3
) -> Dict[str, Any]:
    """
    Evaluates spatial coverage and distribution of RANSAC inliers across a 3x3 grid.
    """
    h, w = source_shape[:2]
    mask_flat = inliers_mask.ravel()
    inlier_indices = np.where(mask_flat == 1)[0]

    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    total_inliers = len(inlier_indices)

    if total_inliers == 0 or h == 0 or w == 0:
        return {
            "grid_counts": grid_counts.tolist(),
            "total_inliers": 0,
            "occupied_cells": 0,
            "total_cells": grid_rows * grid_cols,
            "coverage_ratio": 0.0,
            "spatial_entropy": 0.0,
            "is_single_cluster": True
        }

    cell_h = h / float(grid_rows)
    cell_w = w / float(grid_cols)

    for idx in inlier_indices:
        m = matches[idx]
        pt = kp_source[m.queryIdx].pt
        x, y = pt[0], pt[1]

        c = min(grid_cols - 1, max(0, int(x / cell_w)))
        r = min(grid_rows - 1, max(0, int(y / cell_h)))
        grid_counts[r, c] += 1

    total_cells = grid_rows * grid_cols
    occupied_cells = int(np.sum(grid_counts > 0))
    coverage_ratio = round(occupied_cells / float(total_cells), 4)

    # Compute Normalized Shannon Entropy
    counts_flat = grid_counts.ravel()
    probabilities = counts_flat[counts_flat > 0] / float(total_inliers)
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
    max_possible_entropy = np.log2(total_cells)
    normalized_entropy = round(float(shannon_entropy / max_possible_entropy), 4) if max_possible_entropy > 0 else 0.0

    # Single-cluster check: If occupied_cells <= 1 or 90% of inliers fall in a single cell
    max_single_cell_pct = float(np.max(counts_flat)) / float(total_inliers) * 100.0
    is_single_cluster = (occupied_cells <= 1) or (max_single_cell_pct >= 85.0)

    return {
        "grid_counts": grid_counts.tolist(),
        "total_inliers": total_inliers,
        "occupied_cells": occupied_cells,
        "total_cells": total_cells,
        "coverage_ratio": coverage_ratio,
        "spatial_entropy": normalized_entropy,
        "max_cell_concentration_pct": round(max_single_cell_pct, 1),
        "is_single_cluster": is_single_cluster
    }


def check_homography_sanity(
    H: np.ndarray,
    source_shape: Tuple[int, int],
    reference_shape: Tuple[int, int]
) -> Tuple[bool, str]:
    """
    Performs geometric sanity checks on the estimated 3x3 Homography matrix H.
    """
    if H is None or H.shape != (3, 3):
        return False, "Homography matrix is None or invalid shape."

    det_H = np.linalg.det(H)
    if det_H <= 0:
        return False, f"Degenerate Homography: Determinant is non-positive ({det_H:.6f}), indicating image reflection/flip."

    h_src, w_src = source_shape[:2]
    h_ref, w_ref = reference_shape[:2]
    ref_area = float(w_ref * h_ref)

    src_corners = np.float32([[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]).reshape(-1, 1, 2)
    try:
        transformed_corners = cv2.perspectiveTransform(src_corners, H)
    except Exception as e:
        return False, f"Perspective transformation of corners failed: {e}"

    pts = transformed_corners.reshape(-1, 2)

    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return False, "Transformed corners contain NaN or Infinite values."

    poly_area = float(cv2.contourArea(pts.astype(np.float32)))

    if poly_area <= 0:
        return False, "Transformed polygon area collapsed to zero or self-intersected."

    area_ratio = poly_area / ref_area
    if area_ratio < 0.05:
        return False, f"Transformed area collapsed to {area_ratio*100:.1f}% of reference frame (below 5% threshold)."
    if area_ratio > 10.0:
        return False, f"Transformed area exploded to {area_ratio:.1f}x reference frame (above 10x threshold)."

    return True, f"Homography is geometrically sane (Area ratio: {area_ratio*100:.1f}% of reference frame, det={det_H:.4f})."
