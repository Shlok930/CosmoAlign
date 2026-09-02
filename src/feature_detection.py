"""
Feature Detection Module for CosmoAlign.

Extracts Scale-Invariant Feature Transform (SIFT) keypoints and descriptors
from grayscale images.
"""

from typing import List, Tuple, Dict, Any
import cv2
import numpy as np


def extract_sift_features(
    gray_image: np.ndarray,
    nfeatures: int = 0,
    contrastThreshold: float = 0.04,
    edgeThreshold: float = 10.0,
    sigma: float = 1.6
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detects keypoints and computes 128-dimensional SIFT descriptors for a grayscale image.

    CONCEPTUAL EXPLANATION:
    - Keypoints: Specific 2D spatial locations (x, y) in the image that are distinctive 
      (corners, blobs, scale-space extrema) along with their scale (size) and orientation (angle).
    - Descriptors: 128-dimensional numerical vectors describing the local image patch/gradient 
      around each keypoint. They allow comparing features across images independently of scale/rotation.

    Args:
        gray_image (np.ndarray): Single-channel uint8 grayscale image.
        nfeatures (int): Max number of keypoints to retain (0 = all keypoints).
        contrastThreshold (float): Contrast threshold to filter weak features in low-contrast regions.
        edgeThreshold (float): Threshold to filter out edge-like features.
        sigma (float): Gaussian blur sigma for the scale space base layer.

    Returns:
        Tuple[List[cv2.KeyPoint], np.ndarray]: 
            - List of OpenCV KeyPoint objects.
            - 2D NumPy array of shape (N, 128) containing float32 descriptors.

    Raises:
        ValueError: If the input is not a valid 2D grayscale image.
        RuntimeError: If SIFT fails to extract any keypoints or descriptors.
    """
    if len(gray_image.shape) != 2:
        raise ValueError("Input image to SIFT feature extraction must be single-channel grayscale.")

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )

    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    if keypoints is None or len(keypoints) == 0 or descriptors is None:
        raise RuntimeError(
            "SIFT feature detection failed: No keypoints or descriptors detected in the image. "
            "The image may lack sufficient contrast or visual texture."
        )

    return keypoints, descriptors


def get_keypoint_debug_info(keypoints: List[cv2.KeyPoint], num_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Extracts numerical metadata from keypoint objects for inspection and debugging.

    Returns a list of dicts with keys: ['id', 'x', 'y', 'size', 'angle', 'response', 'octave']
    """
    sample_info = []
    sample_kps = keypoints[:num_samples] if len(keypoints) >= num_samples else keypoints

    for idx, kp in enumerate(sample_kps):
        sample_info.append({
            "id": idx + 1,
            "x": round(kp.pt[0], 2),
            "y": round(kp.pt[1], 2),
            "size": round(kp.size, 2),
            "angle": round(kp.angle, 2),
            "response": round(kp.response, 6),
            "octave": kp.octave
        })

    return sample_info
