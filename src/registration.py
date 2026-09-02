"""
Geometric Registration, RANSAC Homography, and Warping Module for CosmoAlign.

Estimates the 3x3 homography matrix using RANSAC, geometrically warps the source
image into alignment with the reference image, computes registration evaluation metrics,
and builds verification overlay visualizations.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


def estimate_homography(
    kp_source: List[cv2.KeyPoint],
    kp_ref: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_reproj_threshold: float = 5.0,
    min_matches: int = 4
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Estimates a 3x3 Homography matrix using RANSAC algorithm.

    CONCEPTUAL EXPLANATION:
    - Homography (H): A 3x3 transformation matrix mapping 2D homogeneous coordinates from 
      the source frame to the reference frame under planar or distant perspective assumptions.
    - RANSAC (Random Sample Consensus): Iteratively selects minimal subsets (4 random point pairs),
      fits candidate homography matrices, and counts how many overall point matches agree (inliers) 
      within `ransac_reproj_threshold`. It returns the homography supported by the maximum inliers.
    - Inliers vs Outliers: Inliers are geometrically valid correspondences consistent with the transformation.
      Outliers are incorrect matches filtered out during estimation.

    HOMOGRAPHY DIRECTION CONVENTION:
      SOURCE_POINT (x, y) -> H -> REFERENCE_POINT (x', y')

    Args:
        kp_source (List[cv2.KeyPoint]): Keypoints from source image.
        kp_ref (List[cv2.KeyPoint]): Keypoints from reference image.
        matches (List[cv2.DMatch]): Filtered good matches passing Lowe's ratio test.
        ransac_reproj_threshold (float): Maximum allowed reprojection error in pixels for inlier classification.
        min_matches (int): Minimum required good matches (minimum 4 required mathematically).

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
            - Homography Matrix H (3x3 float64 array), or None if failed.
            - Inliers mask (N, 1 binary array where 1 = inlier, 0 = outlier), or None if failed.
            - Metrics dictionary containing:
              'good_matches_count', 'inlier_count', 'outlier_count', 'inlier_ratio', 'rmse'

    Raises:
        ValueError: If there are fewer matches than `min_matches`.
    """
    num_matches = len(matches)
    if num_matches < min_matches:
        raise ValueError(
            f"Registration failed: Insufficient good matches ({num_matches} found, "
            f"minimum required is {min_matches})."
        )

    # Extract source and reference 2D point arrays (N, 1, 2)
    # QueryIdx refers to source keypoints; TrainIdx refers to reference keypoints
    src_pts = np.float32([kp_source[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute Homography via RANSAC: Source -> Reference
    H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, ransac_reproj_threshold)

    if H is None or mask is None:
        metrics = {
            "good_matches_count": num_matches,
            "inlier_count": 0,
            "outlier_count": num_matches,
            "inlier_ratio": 0.0,
            "rmse": None
        }
        return None, None, metrics

    inliers_mask = mask.ravel()
    inlier_count = int(np.sum(inliers_mask))
    outlier_count = num_matches - inlier_count
    inlier_ratio = (inlier_count / num_matches) * 100.0 if num_matches > 0 else 0.0

    # Calculate Root Mean Square Error (RMSE) of inlier reprojections
    rmse = compute_reprojection_rmse(src_pts, ref_pts, H, inliers_mask)

    metrics = {
        "good_matches_count": num_matches,
        "inlier_count": inlier_count,
        "outlier_count": outlier_count,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse
    }

    return H, mask, metrics


def compute_reprojection_rmse(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    H: np.ndarray,
    inliers_mask: np.ndarray
) -> Optional[float]:
    """
    Computes the Root Mean Square Error (RMSE) of the reprojected inlier points.

    RMSE measures the average pixel distance between transformed source inlier points 
    and their target reference positions:
        RMSE = sqrt( (1 / N) * sum( || ref_i - H * src_i ||^2 ) )
    """
    inlier_indices = np.where(inliers_mask == 1)[0]
    if len(inlier_indices) == 0:
        return None

    inlier_src = src_pts[inlier_indices] # Shape (N, 1, 2)
    inlier_ref = ref_pts[inlier_indices] # Shape (N, 1, 2)

    # Perspective transformation of inlier source points using H
    transformed_src = cv2.perspectiveTransform(inlier_src, H)

    # Pixel offset error
    diff = transformed_src - inlier_ref
    squared_errors = np.sum(diff ** 2, axis=2) # Squared Euclidean distances
    mean_squared_error = np.mean(squared_errors)
    rmse = float(np.sqrt(mean_squared_error))

    return rmse


def warp_source_image(
    source_img: np.ndarray,
    H: np.ndarray,
    reference_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Geometrically transforms the source image using the estimated 3x3 homography matrix.

    Args:
        source_img (np.ndarray): Original source image (BGR or Grayscale).
        H (np.ndarray): 3x3 Homography matrix mapping Source -> Reference.
        reference_shape (Tuple[int, int]): Height and Width (h, w) of target reference frame.

    Returns:
        np.ndarray: Registered (warped) source image matching reference frame dimensions.
    """
    h_ref, w_ref = reference_shape[:2]
    registered_img = cv2.warpPerspective(
        source_img,
        H,
        (w_ref, h_ref),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return registered_img


def create_overlay_blend(
    registered_img: np.ndarray,
    reference_img: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Creates a visual verification overlay by blending the registered source image 
    and the reference image.

    Args:
        registered_img (np.ndarray): Warped source image.
        reference_img (np.ndarray): Reference image.
        alpha (float): Blend weight for registered source image (0.0 to 1.0).

    Returns:
        np.ndarray: Blended BGR output image.
    """
    if registered_img.shape != reference_img.shape:
        if len(registered_img.shape) == 2 and len(reference_img.shape) == 3:
            registered_img = cv2.cvtColor(registered_img, cv2.COLOR_GRAY2BGR)
        elif len(registered_img.shape) == 3 and len(reference_img.shape) == 2:
            reference_img = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)

    beta = 1.0 - alpha
    overlay = cv2.addWeighted(registered_img, alpha, reference_img, beta, 0)
    return overlay
