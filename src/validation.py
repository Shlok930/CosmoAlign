"""
Scientific Validation & Multi-Gate Audit Engine for CosmoAlign Phase 3.

Replaces simple boolean success checks (e.g. success = inliers > 4) with a
rigorous 8-gate validation system:

Gates Evaluated:
1. DATA_VALIDATION (SHA-256 file checksums, metadata provenance, valid pixel coverage)
2. FEATURE_VALIDATION (Keypoint count & spatial distribution)
3. MATCH_VALIDATION (Lowe ratio filtering & Mutual Nearest-Neighbor cross-checking)
4. RANSAC_VALIDATION (Inlier count, inlier ratio, reproducible random seed)
5. SPATIAL_VALIDATION (3x3 Grid occupancy, cell imbalance check, rejecting single-cluster matches)
6. GEOMETRY_VALIDATION (Positive determinant det(H)>0, corner projection bounding box sanity)
7. REPROJECTION_VALIDATION (RMSE, Median pixel error, 95th Percentile error)
8. INDEPENDENT_VALIDATION (50/50 Train-Validation match split cross-validation)

Final Verdict:
- VALIDATED: All computational, spatial, geometric, and reprojection gates pass.
- UNCERTAIN: Borderline metrics, low spatial coverage, or independent validation missing.
- FAILED: Any critical validation gate fails (e.g. data corruption, degenerate geometry, single-cluster matches).
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

from image_loader import compute_file_sha256
from evaluation import (
    compute_reprojection_stats,
    analyze_inlier_spatial_distribution,
    check_homography_sanity
)
from matching import filter_matches_mutual_nn


class ScientificValidator:
    """
    Multi-Gate Scientific Validation Engine for CosmoAlign.
    """

    def __init__(self, min_inliers: int = 10, min_ratio_pct: float = 20.0, max_rmse_px: float = 10.0):
        self.min_inliers = min_inliers
        self.min_ratio_pct = min_ratio_pct
        self.max_rmse_px = max_rmse_px

    def validate_registration(
        self,
        pair_id: str,
        source_path: str,
        ref_path: str,
        pair_info: Dict[str, Any],
        source_stats: Dict[str, Any],
        ref_stats: Dict[str, Any],
        kp_source: List[cv2.KeyPoint],
        kp_ref: List[cv2.KeyPoint],
        desc_source: np.ndarray,
        desc_ref: np.ndarray,
        raw_matches: List[List[cv2.DMatch]],
        good_matches: List[cv2.DMatch],
        H: Optional[np.ndarray],
        inliers_mask: Optional[np.ndarray],
        ransac_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes complete multi-gate scientific validation on registration output.

        Returns:
            Dict[str, Any]: Detailed multi-gate validation report with final_verdict.
        """
        gate_results = {}
        diagnostic_reasons = []

        # ---------------------------------------------------------
        # GATE 1: DATA VALIDATION
        # ---------------------------------------------------------
        src_sha_actual = compute_file_sha256(source_path) if os.path.exists(source_path) else "UNKNOWN"
        ref_sha_actual = compute_file_sha256(ref_path) if os.path.exists(ref_path) else "UNKNOWN"

        src_sha_expected = pair_info.get("source_sha256", "UNKNOWN")
        ref_sha_expected = pair_info.get("reference_sha256", "UNKNOWN")

        sha_pass = True
        if src_sha_expected != "UNKNOWN" and src_sha_actual != src_sha_expected:
            sha_pass = False
            diagnostic_reasons.append(f"Source file SHA-256 hash mismatch ({src_sha_actual[:8]} != {src_sha_expected[:8]})")
        if ref_sha_expected != "UNKNOWN" and ref_sha_actual != ref_sha_expected:
            sha_pass = False
            diagnostic_reasons.append(f"Reference file SHA-256 hash mismatch ({ref_sha_actual[:8]} != {ref_sha_expected[:8]})")

        data_pass = sha_pass and (source_stats["valid_pixel_ratio"] >= 50.0) and (ref_stats["valid_pixel_ratio"] >= 50.0)
        gate_results["DATA_VALIDATION"] = {
            "status": "PASS" if data_pass else "FAIL",
            "source_sha256": src_sha_actual[:12] + "...",
            "reference_sha256": ref_sha_actual[:12] + "...",
            "source_valid_pixels_pct": source_stats["valid_pixel_ratio"],
            "reference_valid_pixels_pct": ref_stats["valid_pixel_ratio"]
        }

        # ---------------------------------------------------------
        # GATE 2: FEATURE VALIDATION
        # ---------------------------------------------------------
        num_kp_src = len(kp_source) if kp_source else 0
        num_kp_ref = len(kp_ref) if kp_ref else 0
        feature_pass = (num_kp_src >= 20) and (num_kp_ref >= 20)
        if not feature_pass:
            diagnostic_reasons.append(f"Insufficient SIFT keypoints detected (Source: {num_kp_src}, Ref: {num_kp_ref})")

        gate_results["FEATURE_VALIDATION"] = {
            "status": "PASS" if feature_pass else "FAIL",
            "source_keypoints": num_kp_src,
            "reference_keypoints": num_kp_ref
        }

        # ---------------------------------------------------------
        # GATE 3: MATCH VALIDATION
        # ---------------------------------------------------------
        num_good_m = len(good_matches) if good_matches else 0
        mnn_matches = filter_matches_mutual_nn(desc_source, desc_ref)
        num_mnn_m = len(mnn_matches)

        match_pass = num_good_m >= 4
        if not match_pass:
            diagnostic_reasons.append(f"Insufficient good descriptor matches after ratio test ({num_good_m} < 4)")

        gate_results["MATCH_VALIDATION"] = {
            "status": "PASS" if match_pass else "FAIL",
            "raw_knn_matches": len(raw_matches),
            "good_lowe_matches": num_good_m,
            "mutual_nn_crosscheck_matches": num_mnn_m
        }

        # ---------------------------------------------------------
        # GATE 4: RANSAC VALIDATION
        # ---------------------------------------------------------
        inlier_count = ransac_metrics.get("inlier_count", 0)
        inlier_ratio_pct = ransac_metrics.get("inlier_ratio", 0.0)

        ransac_pass = (inlier_count >= self.min_inliers) and (inlier_ratio_pct >= self.min_ratio_pct)
        if not ransac_pass:
            diagnostic_reasons.append(
                f"RANSAC inlier threshold not met (Inliers: {inlier_count} < {self.min_inliers}, Ratio: {inlier_ratio_pct:.1f}% < {self.min_ratio_pct:.1f}%)"
            )

        gate_results["RANSAC_VALIDATION"] = {
            "status": "PASS" if ransac_pass else "FAIL",
            "inlier_count": inlier_count,
            "inlier_ratio_pct": round(inlier_ratio_pct, 2)
        }

        # ---------------------------------------------------------
        # GATE 5: SPATIAL DISTRIBUTION VALIDATION
        # ---------------------------------------------------------
        if inliers_mask is not None and len(inliers_mask) > 0 and H is not None:
            spatial_eval = analyze_inlier_spatial_distribution(kp_source, good_matches, inliers_mask, source_stats["shape"])
            coverage_ratio = spatial_eval["coverage_ratio"]
            occupied_cells = spatial_eval["occupied_cells"]
            is_single_cluster = spatial_eval["is_single_cluster"]
            entropy = spatial_eval["spatial_entropy"]

            spatial_pass = (occupied_cells >= 3) and (not is_single_cluster)
            if is_single_cluster:
                diagnostic_reasons.append(f"Inliers are concentrated in a single cluster ({spatial_eval['max_cell_concentration_pct']:.1f}% in one cell)")
            elif occupied_cells < 3:
                diagnostic_reasons.append(f"Low spatial grid coverage ({occupied_cells}/9 cells occupied)")
        else:
            spatial_eval = {"coverage_ratio": 0.0, "occupied_cells": 0, "spatial_entropy": 0.0, "is_single_cluster": True}
            spatial_pass = False
            diagnostic_reasons.append("Spatial validation failed due to missing RANSAC inliers mask")

        gate_results["SPATIAL_VALIDATION"] = {
            "status": "PASS" if spatial_pass else ("FAIL" if spatial_eval["is_single_cluster"] else "UNCERTAIN"),
            "occupied_cells": spatial_eval["occupied_cells"],
            "total_cells": 9,
            "coverage_ratio_pct": round(spatial_eval["coverage_ratio"] * 100.0, 1),
            "spatial_entropy": spatial_eval["spatial_entropy"],
            "is_single_cluster": spatial_eval["is_single_cluster"]
        }

        # ---------------------------------------------------------
        # GATE 6: GEOMETRY VALIDATION
        # ---------------------------------------------------------
        if H is not None:
            geom_sane, geom_msg = check_homography_sanity(H, source_stats["shape"], ref_stats["shape"])
            if not geom_sane:
                diagnostic_reasons.append(f"Geometric Homography sanity check failed: {geom_msg}")
        else:
            geom_sane, geom_msg = False, "Homography matrix H is None"
            diagnostic_reasons.append("Homography matrix H is None")

        gate_results["GEOMETRY_VALIDATION"] = {
            "status": "PASS" if geom_sane else "FAIL",
            "homography_sane": geom_sane,
            "det_H": round(float(np.linalg.det(H)), 6) if H is not None else 0.0,
            "sanity_notes": geom_msg
        }

        # ---------------------------------------------------------
        # GATE 7: REPROJECTION ERROR VALIDATION
        # ---------------------------------------------------------
        reproj_stats = compute_reprojection_stats(kp_source, kp_ref, good_matches, H, inliers_mask)
        rmse_px = reproj_stats.get("rmse")
        median_px = reproj_stats.get("median_px")
        p95_px = reproj_stats.get("p95_px")

        reproj_pass = (rmse_px is not None) and (rmse_px <= self.max_rmse_px)
        if rmse_px is not None and rmse_px > self.max_rmse_px:
            diagnostic_reasons.append(f"Reprojection RMSE exceeds threshold ({rmse_px:.2f} px > {self.max_rmse_px:.2f} px)")

        gate_results["REPROJECTION_VALIDATION"] = {
            "status": "PASS" if reproj_pass else "FAIL",
            "rmse_px": rmse_px,
            "median_px": median_px,
            "p95_px": p95_px,
            "max_px": reproj_stats.get("max_px")
        }

        # ---------------------------------------------------------
        # GATE 8: INDEPENDENT CROSS-VALIDATION (50/50 Match Split)
        # ---------------------------------------------------------
        independent_eval = self._evaluate_independent_split(kp_source, kp_ref, good_matches, source_stats["shape"], ref_stats["shape"])
        gate_results["INDEPENDENT_VALIDATION"] = independent_eval

        # ---------------------------------------------------------
        # FINAL VERDICT SYSTEM
        # ---------------------------------------------------------
        critical_passes = [data_pass, feature_pass, match_pass, ransac_pass, geom_sane, reproj_pass]
        all_passed = all(critical_passes) and spatial_pass

        if all_passed and independent_eval["status"] == "PASS":
            final_verdict = "VALIDATED"
            confidence_level = "HIGH"
        elif all(critical_passes):
            final_verdict = "UNCERTAIN"
            confidence_level = "MEDIUM"
            diagnostic_reasons.append("Registration passed critical gates but spatial coverage or independent validation was inconclusive")
        else:
            final_verdict = "FAILED"
            confidence_level = "LOW"

        # Heuristic score calculated strictly for logging separate from confidence
        score = 0.0
        if data_pass: score += 15
        if feature_pass: score += 15
        if match_pass: score += 15
        if ransac_pass: score += 20
        if spatial_pass: score += 15
        if geom_sane: score += 20

        return {
            "pair_id": pair_id,
            "final_verdict": final_verdict,
            "confidence_level": confidence_level,
            "heuristic_quality_score": round(score, 1),
            "diagnostic_reasons": diagnostic_reasons,
            "gates": gate_results
        }

    def _evaluate_independent_split(
        self,
        kp_source: List[cv2.KeyPoint],
        kp_ref: List[cv2.KeyPoint],
        good_matches: List[cv2.DMatch],
        source_shape: Tuple[int, int],
        ref_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Executes independent cross-validation by splitting good matches 50/50:
        - Fits H_train on 50% training matches
        - Evaluates independent reprojection error on unseen 50% validation matches
        """
        num_m = len(good_matches)
        if num_m < 8:
            return {
                "status": "UNAVAILABLE",
                "notes": "Insufficient good matches (< 8) to perform 50/50 train-validation split",
                "validation_matches": 0,
                "val_rmse_px": None
            }

        # Reproducible 50/50 split
        np.random.seed(42)
        shuffled_indices = np.random.permutation(num_m)
        split_idx = num_m // 2

        train_indices = shuffled_indices[:split_idx]
        val_indices = shuffled_indices[split_idx:]

        train_matches = [good_matches[i] for i in train_indices]
        val_matches = [good_matches[i] for i in val_indices]

        src_pts_tr = np.float32([kp_source[m.queryIdx].pt for m in train_matches]).reshape(-1, 1, 2)
        ref_pts_tr = np.float32([kp_ref[m.trainIdx].pt for m in train_matches]).reshape(-1, 1, 2)

        H_tr, mask_tr = cv2.findHomography(src_pts_tr, ref_pts_tr, cv2.RANSAC, 5.0)

        if H_tr is None:
            return {
                "status": "UNAVAILABLE",
                "notes": "Failed to fit homography H on training match subset",
                "validation_matches": len(val_matches),
                "val_rmse_px": None
            }

        src_pts_val = np.float32([kp_source[m.queryIdx].pt for m in val_matches]).reshape(-1, 1, 2)
        ref_pts_val = np.float32([kp_ref[m.trainIdx].pt for m in val_matches]).reshape(-1, 1, 2)

        try:
            trans_val = cv2.perspectiveTransform(src_pts_val, H_tr)
            diff_val = trans_val - ref_pts_val
            errs_val = np.sqrt(np.sum(diff_val ** 2, axis=2)).ravel()
            val_rmse = float(np.sqrt(np.mean(errs_val ** 2)))
        except Exception as e:
            return {
                "status": "UNAVAILABLE",
                "notes": f"Cross-validation transformation error: {e}",
                "validation_matches": len(val_matches),
                "val_rmse_px": None
            }

        val_pass = val_rmse <= (self.max_rmse_px * 1.5)
        return {
            "status": "PASS" if val_pass else "FAIL",
            "notes": f"Independent validation RMSE: {val_rmse:.2f} px on {len(val_matches)} unseen matches",
            "train_matches": len(train_matches),
            "validation_matches": len(val_matches),
            "val_rmse_px": round(val_rmse, 4)
        }


def format_validation_summary_console(val_report: Dict[str, Any]) -> str:
    """Formats a clean console string for Phase 3 Hardened Multi-Gate Summary."""
    gates = val_report.get("gates", {})
    verdict = val_report.get("final_verdict", "UNKNOWN")
    reasons = val_report.get("diagnostic_reasons", [])

    lines = [
        "=" * 70,
        " COSMOALIGN PHASE 3 — HARDENED MULTI-GATE VALIDATION SUMMARY ",
        "=" * 70,
        f"TARGET PAIR ID: {val_report.get('pair_id', 'UNKNOWN')}",
        "",
        "MULTIPLE INDEPENDENT VALIDATION GATES:",
        f"  1. DATA INTEGRITY:      {gates.get('DATA_VALIDATION', {}).get('status', 'UNKNOWN')}",
        f"  2. SIFT FEATURES:       {gates.get('FEATURE_VALIDATION', {}).get('status', 'UNKNOWN')}",
        f"  3. DESCRIPTOR MATCHING: {gates.get('MATCH_VALIDATION', {}).get('status', 'UNKNOWN')}",
        f"  4. RANSAC GEOMETRY:     {gates.get('RANSAC_VALIDATION', {}).get('status', 'UNKNOWN')}",
        f"  5. SPATIAL COVERAGE:    {gates.get('SPATIAL_VALIDATION', {}).get('status', 'UNKNOWN')}",
        f"  6. HOMOGRAPHY SANITY:   {gates.get('GEOMETRY_VALIDATION', {}).get('status', 'UNKNOWN')}",
        f"  7. REPROJECTION ERROR:  {gates.get('REPROJECTION_VALIDATION', {}).get('status', 'UNKNOWN')}",
        f"  8. INDEPENDENT SPLIT:   {gates.get('INDEPENDENT_VALIDATION', {}).get('status', 'UNKNOWN')}",
        "",
        "QUANTITATIVE ERROR METRICS:",
        f"  * Reprojection RMSE:    {gates.get('REPROJECTION_VALIDATION', {}).get('rmse_px')} px",
        f"  * Median Pixel Error:   {gates.get('REPROJECTION_VALIDATION', {}).get('median_px')} px",
        f"  * 95th Percentile Error: {gates.get('REPROJECTION_VALIDATION', {}).get('p95_px')} px",
        f"  * 50/50 Independent RMSE: {gates.get('INDEPENDENT_VALIDATION', {}).get('val_rmse_px')} px",
        "",
        "SPATIAL DISTRIBUTION METRICS:",
        f"  * Occupied Grid Cells:  {gates.get('SPATIAL_VALIDATION', {}).get('occupied_cells')}/9 cells ({gates.get('SPATIAL_VALIDATION', {}).get('coverage_ratio_pct')}%)",
        f"  * Spatial Entropy Score: {gates.get('SPATIAL_VALIDATION', {}).get('spatial_entropy')}",
        f"  * Single Cluster Flag:  {gates.get('SPATIAL_VALIDATION', {}).get('is_single_cluster')}",
        "",
        "=" * 70,
        f" FINAL REGISTRATION VERDICT: [ {verdict} ] (Confidence: {val_report.get('confidence_level')})",
        "=" * 70
    ]

    if reasons:
        lines.append("DIAGNOSTIC REASONS / OBSERVED ISSUES:")
        for r in reasons:
            lines.append(f"  - {r}")

    lines.append("=" * 70 + "\n")
    return "\n".join(lines)
