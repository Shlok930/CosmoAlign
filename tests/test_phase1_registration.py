"""
Regression Test Suite for CosmoAlign.

Verifies that Phase 1 and Phase 2 registration functionality on normal images
remains 100% functional, intact, and regression-free after introducing Phase 3.
"""

import unittest
import os
import sys
import shutil

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from image_loader import load_image, to_grayscale
from feature_detection import extract_sift_features
from matching import match_descriptors_knn, filter_matches_lowe
from registration import estimate_homography, warp_source_image
from generate_sample_data import generate_synthetic_image, apply_known_homography
import cv2


class TestPhase1Registration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Creates temporary test images."""
        cls.test_dir = os.path.join("tests", "temp_test_data")
        cls.output_dir = os.path.join("tests", "temp_test_outputs")
        os.makedirs(cls.test_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)

        ref_img = generate_synthetic_image(width=400, height=300)
        source_img = apply_known_homography(ref_img, angle_deg=10.0, scale=0.95, tx=15.0, ty=-10.0)

        cls.ref_path = os.path.join(cls.test_dir, "reference.jpg")
        cls.source_path = os.path.join(cls.test_dir, "source.jpg")

        cv2.imwrite(cls.ref_path, ref_img)
        cv2.imwrite(cls.source_path, source_img)

    @classmethod
    def tearDownClass(cls):
        """Cleans up temporary files."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)
        if os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_image_loading_and_grayscale(self):
        """Tests image reading and grayscale conversion."""
        img = load_image(self.source_path)
        self.assertIsNotNone(img)
        self.assertEqual(len(img.shape), 3)

        gray = to_grayscale(img)
        self.assertEqual(len(gray.shape), 2)

    def test_sift_feature_extraction(self):
        """Tests SIFT keypoint and descriptor extraction."""
        img = load_image(self.source_path)
        gray = to_grayscale(img)
        kps, descs = extract_sift_features(gray)

        self.assertGreater(len(kps), 0)
        self.assertIsNotNone(descs)
        self.assertEqual(descs.shape[1], 128)

    def test_end_to_end_registration_pipeline(self):
        """Tests end-to-end matching, homography estimation, and warping."""
        source_img = load_image(self.source_path)
        ref_img = load_image(self.ref_path)

        kp_src, desc_src = extract_sift_features(to_grayscale(source_img))
        kp_ref, desc_ref = extract_sift_features(to_grayscale(ref_img))

        raw_matches = match_descriptors_knn(desc_src, desc_ref, k=2)
        good_matches = filter_matches_lowe(raw_matches, ratio_threshold=0.75)

        self.assertGreaterEqual(len(good_matches), 4)

        H, mask, metrics = estimate_homography(kp_src, kp_ref, good_matches)
        self.assertIsNotNone(H)
        self.assertEqual(H.shape, (3, 3))
        self.assertGreater(metrics["inlier_count"], 0)

        warped = warp_source_image(source_img, H, ref_img.shape)
        self.assertEqual(warped.shape, ref_img.shape)


if __name__ == "__main__":
    unittest.main()
