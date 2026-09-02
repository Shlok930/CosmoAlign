"""
Configuration parameters for CosmoAlign Image Registration Engine.
Centralized default settings to avoid magic numbers across the codebase.
"""

# Lowe's Ratio Test threshold for filtering ambiguous KNN descriptor matches.
# A lower ratio (e.g. 0.70) is stricter; a higher ratio (e.g. 0.80) retains more matches.
DEFAULT_LOWE_RATIO_THRESHOLD: float = 0.75

# RANSAC maximum reprojection error allowed in pixels to consider a point match an inlier.
DEFAULT_RANSAC_REPROJ_THRESHOLD: float = 5.0

# Minimum number of Lowe ratio filtered good matches required to attempt homography estimation.
# Mathematically 4 non-collinear point pairs are required for an 8-DOF 3x3 homography matrix.
MIN_GOOD_MATCHES_REQUIRED: int = 4

# Minimum number of geometrically consistent RANSAC inliers required to consider registration valid.
MIN_INLIERS_REQUIRED: int = 4

# Default debug mode flag (prints keypoint metadata tables and detailed metrics)
DEFAULT_DEBUG_MODE: bool = False
