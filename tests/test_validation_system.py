"""
Synthetic False-Positive Validation Test Suite for CosmoAlign.

Tests the Multi-Gate Scientific Validation Engine against synthetic negative failure cases:
- Test A: Random uncorrelated image pairs -> Expect FAILED verdict
- Test B: Inverted/reflection matrix (det(H) <= 0) -> Expect GEOMETRY_VALIDATION FAIL
- Test C: Clustered inliers in a single cell -> Expect SPATIAL_VALIDATION FAIL / is_single_cluster True
- Test D: Degenerate polygon collapse (< 5% area) -> Expect GEOMETRY_VALIDATION FAIL
"""

import unittest
import os
import sys
import numpy as np
import cv2

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from validation import ScientificValidator
from evaluation import check_homography_sanity, analyze_inlier_spatial_distribution


class TestValidationSystem(unittest.TestCase):

    def setUp(self):
        self.validator = ScientificValidator(min_inliers=10, min_ratio_pct=20.0, max_rmse_px=10.0)

    def test_inverted_homography_rejection(self):
        """Tests that a homography matrix with negative determinant (reflection/flip) is rejected."""
        # Reflection matrix across Y axis
        H_inverted = np.array([
            [-1.0,  0.0, 100.0],
            [ 0.0,  1.0,  50.0],
            [ 0.0,  0.0,   1.0]
        ], dtype=np.float64)

        is_sane, msg = check_homography_sanity(H_inverted, (1000, 1200), (500, 600))
        self.assertFalse(is_sane)
        self.assertIn("non-positive", msg.lower())

    def test_degenerate_bounding_box_collapse(self):
        """Tests that a homography collapsing the image into a tiny line/point is rejected."""
        # Scale down matrix by 0.001x -> 0.0001% area
        H_collapse = np.array([
            [0.001, 0.0,   10.0],
            [0.0,   0.001, 10.0],
            [0.0,   0.0,    1.0]
        ], dtype=np.float64)

        is_sane, msg = check_homography_sanity(H_collapse, (1000, 1200), (500, 600))
        self.assertFalse(is_sane)
        self.assertIn("collapsed", msg.lower())

    def test_single_cluster_inlier_rejection(self):
        """Tests that inliers concentrated in a single grid cell trigger single_cluster flag."""
        # Create keypoints all concentrated at top-left corner (x=10, y=10)
        kp_src = [cv2.KeyPoint(10.0 + i*0.1, 10.0 + i*0.1, 5.0) for i in range(20)]
        matches = [cv2.DMatch(i, i, 1.0) for i in range(20)]
        inliers_mask = np.ones((20, 1), dtype=np.uint8)

        spatial_eval = analyze_inlier_spatial_distribution(kp_src, matches, inliers_mask, (1000, 1200))
        
        self.assertTrue(spatial_eval["is_single_cluster"])
        self.assertEqual(spatial_eval["occupied_cells"], 1)
        self.assertLessEqual(spatial_eval["coverage_ratio"], 0.15)

    def test_random_image_pair_validation(self):
        """Tests validation engine on random noise data -> Expects FAILED verdict."""
        # Fake metadata & stats for random failure case
        pair_info = {"pair_id": "test_random", "source_sha256": "UNKNOWN", "reference_sha256": "UNKNOWN"}
        src_stats = {"shape": (100, 100), "dtype": "uint8", "valid_pixel_ratio": 100.0}
        ref_stats = {"shape": (100, 100), "dtype": "uint8", "valid_pixel_ratio": 100.0}

        report = self.validator.validate_registration(
            pair_id="test_random",
            source_path="non_existent.tif",
            ref_path="non_existent.tif",
            pair_info=pair_info,
            source_stats=src_stats,
            ref_stats=ref_stats,
            kp_source=[],
            kp_ref=[],
            desc_source=None,
            desc_ref=None,
            raw_matches=[],
            good_matches=[],
            H=None,
            inliers_mask=None,
            ransac_metrics={"inlier_count": 0, "inlier_ratio": 0.0, "rmse": None}
        )

        self.assertEqual(report["final_verdict"], "FAILED")
        self.assertEqual(report["confidence_level"], "LOW")
        self.assertIn("RANSAC_VALIDATION", report["gates"])
        self.assertEqual(report["gates"]["RANSAC_VALIDATION"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
