"""Experimental affine geometry utilities for Phase 4 Step 5."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def estimate_affine(
    kp_source: List[cv2.KeyPoint],
    kp_ref: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_reproj_threshold: float = 5.0,
    min_matches: int = 3,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Estimate a source-to-reference affine matrix with RANSAC.

    Args:
        kp_source: Keypoints detected in the source image.
        kp_ref: Keypoints detected in the reference image.
        matches: Good matches shared with the Homography experiment.
        ransac_reproj_threshold: Maximum inlier reprojection error in pixels.
        min_matches: Minimum number of correspondences required by affine geometry.

    Returns:
        A tuple containing the 2x3 affine matrix, the RANSAC inlier mask, and
        metrics for the supplied matches. The matrix and mask are ``None`` when
        OpenCV cannot estimate a model.

    Raises:
        ValueError: If fewer than three matched correspondences are supplied.
    """
    match_count = len(matches)
    if match_count < min_matches:
        raise ValueError(
            f"Affine estimation failed: Insufficient good matches ({match_count} found, "
            f"minimum required is {min_matches})."
        )

    source_points = np.float32([kp_source[match.queryIdx].pt for match in matches])
    reference_points = np.float32([kp_ref[match.trainIdx].pt for match in matches])

    affine_matrix, mask = cv2.estimateAffine2D(
        source_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
    )

    if affine_matrix is None or mask is None:
        metrics = {
            "good_matches_count": match_count,
            "inlier_count": 0,
            "outlier_count": match_count,
            "inlier_ratio": 0.0,
            "rmse": None,
        }
        return None, None, metrics

    inliers_mask = mask.ravel()
    inlier_count = int(np.sum(inliers_mask))
    outlier_count = match_count - inlier_count
    inlier_ratio = (inlier_count / match_count) * 100.0 if match_count else 0.0
    rmse = compute_affine_reprojection_rmse(
        source_points,
        reference_points,
        affine_matrix,
        inliers_mask,
    )
    metrics = {
        "good_matches_count": match_count,
        "inlier_count": inlier_count,
        "outlier_count": outlier_count,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
    }
    return affine_matrix, mask, metrics


def compute_affine_reprojection_rmse(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    affine_matrix: np.ndarray,
    inliers_mask: np.ndarray,
) -> Optional[float]:
    """Compute pixel RMSE for the inlier points under an affine matrix.

    The point arrays may be shaped as ``(N, 2)`` or ``(N, 1, 2)``. Only points
    marked as inliers are included in the returned error.
    """
    mask = np.asarray(inliers_mask).ravel()
    inlier_indices = np.where(mask == 1)[0]
    if len(inlier_indices) == 0:
        return None

    source_inliers = np.asarray(source_points, dtype=np.float32)[inlier_indices]
    reference_inliers = np.asarray(reference_points, dtype=np.float32)[inlier_indices]

    for name, points in (("source_points", source_inliers), ("reference_points", reference_inliers)):
        if points.ndim == 2 and points.shape[1] == 2:
            points = points.reshape(-1, 1, 2)
        elif points.ndim == 3 and points.shape[1:] == (1, 2):
            points = points
        else:
            raise ValueError(
                f"{name} must have shape (N, 2) or (N, 1, 2); received {points.shape}."
            )
        if name == "source_points":
            source_inliers = points
        else:
            reference_inliers = points

    transformed_points = cv2.transform(source_inliers, affine_matrix)
    differences = transformed_points - reference_inliers
    squared_errors = np.sum(differences ** 2, axis=-1)
    return float(np.sqrt(np.mean(squared_errors)))


def warp_source_image_affine(
    source_img: np.ndarray,
    affine_matrix: np.ndarray,
    reference_shape: Tuple[int, ...],
) -> np.ndarray:
    """Warp a source image into the reference frame using an affine matrix.

    Args:
        source_img: Source image array, grayscale or multi-channel.
        affine_matrix: 2x3 source-to-reference affine matrix.
        reference_shape: Reference image shape; its first two dimensions are
            interpreted as height and width.

    Returns:
        The linearly interpolated warped image with reference dimensions.
    """
    reference_height, reference_width = reference_shape[:2]
    return cv2.warpAffine(
        source_img,
        affine_matrix,
        (reference_width, reference_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
