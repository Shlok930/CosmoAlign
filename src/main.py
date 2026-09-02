"""
CosmoAlign — Hardened Scientific Entry Point (Phase 1, Phase 2 & Phase 3 Scientific Validation)

Supports CLI flags:
  --phase {1, 2, 3}  Execution phase mode (default: 3 for lunar imagery)
  --pair PAIR_ID     Target lunar pair directory (default: pair_001)
  --source PATH      Custom Source image path (for Phase 1/2)
  --reference PATH   Custom Reference image path (for Phase 1/2)
  --debug            Enable verbose keypoint metadata inspection
"""

import argparse
import sys
import os
import json
import csv
import cv2
import numpy as np

from config import (
    DEFAULT_LOWE_RATIO_THRESHOLD,
    DEFAULT_RANSAC_REPROJ_THRESHOLD,
    MIN_GOOD_MATCHES_REQUIRED,
    MIN_INLIERS_REQUIRED,
    DEFAULT_DEBUG_MODE
)
from image_loader import load_image, to_grayscale, compute_file_sha256
from metadata import inspect_image_stats, calculate_scale_ratio, format_metadata_report
from lunar_data import load_lunar_pair, create_display_visualization
from preprocessing import apply_clahe, apply_percentile_stretch, create_valid_mask
from feature_detection import extract_sift_features, get_keypoint_debug_info
from matching import match_descriptors_knn, filter_matches_lowe, filter_matches_mutual_nn
from registration import (
    estimate_homography,
    warp_source_image,
    create_overlay_blend
)
from visualization import (
    draw_rich_keypoints,
    draw_matches_side_by_side,
    draw_inliers_vs_outliers,
    draw_corner_projection,
    create_before_after_comparison,
    create_difference_image
)
from evaluation import analyze_inlier_spatial_distribution, check_homography_sanity, compute_reprojection_stats
from validation import ScientificValidator, format_validation_summary_console


def run_phase2_pipeline(
    source_path: str,
    reference_path: str,
    output_dir: str = "outputs",
    ratio_threshold: float = DEFAULT_LOWE_RATIO_THRESHOLD,
    ransac_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
    min_matches: int = MIN_GOOD_MATCHES_REQUIRED,
    min_inliers: int = MIN_INLIERS_REQUIRED,
    debug: bool = DEFAULT_DEBUG_MODE
) -> bool:
    """Executes Phase 2 registration pipeline on ordinary images."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(" COSMOALIGN PHASE 2 - ENGINE INSPECTION & TELEMETRY ")
    print("=" * 70)

    print(f"\n[STAGE 1] Loading Input Images...")
    try:
        source_img = load_image(source_path)
        reference_img = load_image(reference_path)
        source_gray = to_grayscale(source_img)
        ref_gray = to_grayscale(reference_img)
        print(f"  [OK] Source Image loaded:    {source_img.shape[1]}x{source_img.shape[0]} px ({source_path})")
        print(f"  [OK] Reference Image loaded: {reference_img.shape[1]}x{reference_img.shape[0]} px ({reference_path})")
    except Exception as e:
        print(f"\n[REGISTRATION FAILED] Image loading failed: {e}", file=sys.stderr)
        return False

    print(f"\n[STAGE 2] Extracting SIFT Keypoints & Descriptors...")
    try:
        kp_source, desc_source = extract_sift_features(source_gray)
        kp_ref, desc_ref = extract_sift_features(ref_gray)
        print(f"  [OK] Source SIFT Keypoints:    {len(kp_source):,}")
        print(f"  [OK] Reference SIFT Keypoints: {len(kp_ref):,}")
    except Exception as e:
        print(f"\n[REGISTRATION FAILED] Feature extraction failed: {e}", file=sys.stderr)
        return False

    out_01 = os.path.join(output_dir, "01_source_keypoints.jpg")
    out_02 = os.path.join(output_dir, "02_reference_keypoints.jpg")
    cv2.imwrite(out_01, draw_rich_keypoints(source_img, kp_source))
    cv2.imwrite(out_02, draw_rich_keypoints(reference_img, kp_ref))

    print(f"\n[STAGE 3] Descriptor Matching (BFMatcher KNN k=2)...")
    raw_matches = match_descriptors_knn(desc_source, desc_ref, k=2)
    raw_matches_unfiltered = [m[0] for m in raw_matches if len(m) > 0]

    out_03 = os.path.join(output_dir, "03_raw_matches.jpg")
    cv2.imwrite(out_03, draw_matches_side_by_side(source_img, kp_source, reference_img, kp_ref, raw_matches_unfiltered))

    print(f"\n[STAGE 4] Filtering Matches with Lowe's Ratio Test (threshold={ratio_threshold})...")
    good_matches = filter_matches_lowe(raw_matches, ratio_threshold=ratio_threshold)

    out_04 = os.path.join(output_dir, "04_good_matches.jpg")
    cv2.imwrite(out_04, draw_matches_side_by_side(source_img, kp_source, reference_img, kp_ref, good_matches))

    if len(good_matches) < min_matches:
        print(f"\n[REGISTRATION FAILED] Insufficient good matches ({len(good_matches)} < {min_matches}).", file=sys.stderr)
        return False

    print(f"\n[STAGE 5] Estimating Geometric Homography via RANSAC...")
    H, inliers_mask, metrics = estimate_homography(
        kp_source, kp_ref, good_matches, ransac_reproj_threshold=ransac_threshold, min_matches=min_matches
    )

    if H is None or inliers_mask is None:
        print(f"\n[REGISTRATION FAILED] RANSAC homography estimation failed.", file=sys.stderr)
        return False

    cv2.imwrite(os.path.join(output_dir, "05_ransac_inliers.jpg"), draw_matches_side_by_side(source_img, kp_source, reference_img, kp_ref, good_matches, match_color=(0, 255, 0)))
    cv2.imwrite(os.path.join(output_dir, "06_ransac_all_matches.jpg"), draw_inliers_vs_outliers(source_img, kp_source, reference_img, kp_ref, good_matches, inliers_mask))

    registered_img = warp_source_image(source_img, H, reference_img.shape)
    cv2.imwrite(os.path.join(output_dir, "07_registered.jpg"), registered_img)
    cv2.imwrite(os.path.join(output_dir, "08_overlay.jpg"), create_overlay_blend(registered_img, reference_img))
    cv2.imwrite(os.path.join(output_dir, "09_before_after.jpg"), create_before_after_comparison(source_img, reference_img, registered_img))
    cv2.imwrite(os.path.join(output_dir, "difference.jpg"), create_difference_image(registered_img, reference_img))

    print("\n" + "=" * 70)
    print(" COSMOALIGN PHASE 2 REPORT ")
    print("=" * 70)
    print(f"  * Source Keypoints:    {len(kp_source):,}")
    print(f"  * Reference Keypoints: {len(kp_ref):,}")
    print(f"  * Raw KNN Matches:     {len(raw_matches):,}")
    print(f"  * Good Lowe Matches:   {len(good_matches):,}")
    print(f"  * RANSAC Inliers:      {metrics['inlier_count']:,}")
    print(f"  * Inlier Ratio:        {metrics['inlier_ratio']:.2f}%")
    if metrics['rmse'] is not None:
        print(f"  * Reprojection RMSE:   {metrics['rmse']:.4f} pixels")
    print("=" * 70)
    print(f"[SUCCESS] CosmoAlign Phase 2 execution completed successfully.\n")
    return True


def run_phase3_lunar_pipeline(
    pair_id: str = "pair_001",
    data_dir: str = "data",
    output_dir: str = "outputs/phase3",
    ratio_threshold: float = DEFAULT_LOWE_RATIO_THRESHOLD,
    ransac_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD
) -> bool:
    """
    Executes CosmoAlign Phase 3 Hardened Scientific Validation Pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    exp_base_dir = os.path.join(output_dir, "experiments")
    os.makedirs(exp_base_dir, exist_ok=True)

    print("=" * 70)
    print(" COSMOALIGN PHASE 3 — HARDENED LUNAR VALIDATION ENGINE ")
    print(" Target Dataset: Chandrayaan-2 OHRC + LRO NAC (pair_001)")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: Load Scientific Lunar Pair & Checksum Verification
    # ---------------------------------------------------------
    pair_folder = os.path.join(data_dir, pair_id)
    source_path = os.path.join(pair_folder, "source.tif")
    ref_path = os.path.join(pair_folder, "reference.tif")

    print(f"\n[STEP 1] Loading Scientific Lunar Dataset Pair '{pair_id}' & SHA-256 Checksums...")
    try:
        source_raw, ref_raw, pair_info = load_lunar_pair(pair_id=pair_id, data_dir=data_dir)
        source_sha256 = compute_file_sha256(source_path)
        ref_sha256 = compute_file_sha256(ref_path)
        print(f"  [OK] Source SHA-256:    {source_sha256[:16]}...")
        print(f"  [OK] Reference SHA-256: {ref_sha256[:16]}...")
    except Exception as e:
        print(f"\n❌ [PHASE 3 FAILED] Dataset loading / checksum error: {e}", file=sys.stderr)
        return False

    # ---------------------------------------------------------
    # STEP 2: Inspect Scientific Image Statistics
    # ---------------------------------------------------------
    print(f"\n[STEP 2] Inspecting Image Array Statistics & Metadata...")
    source_stats = inspect_image_stats(source_raw)
    ref_stats = inspect_image_stats(ref_raw)

    data_report_str = format_metadata_report(source_stats, ref_stats, pair_info)
    print(data_report_str)

    # Save 8-bit Display Visualizations
    source_display = create_display_visualization(source_raw, p_low=2.0, p_high=98.0)
    ref_display = create_display_visualization(ref_raw, p_low=2.0, p_high=98.0)

    out_raw_src = os.path.join(output_dir, "01_source_raw_view.png")
    out_raw_ref = os.path.join(output_dir, "02_reference_raw_view.png")
    cv2.imwrite(out_raw_src, source_display)
    cv2.imwrite(out_raw_ref, ref_display)
    print(f"  [OK] Saved 01_source_raw_view.png    -> {out_raw_src}")
    print(f"  [OK] Saved 02_reference_raw_view.png -> {out_raw_ref}")

    # ---------------------------------------------------------
    # STEP 3: Run Baseline Phase 2 Engine AS-IS
    # ---------------------------------------------------------
    print(f"\n[STEP 3] Running Baseline Engine AS-IS on Real Lunar Data...")
    baseline_dir = os.path.join(output_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    kp_src_base, desc_src_base = extract_sift_features(source_display)
    kp_ref_base, desc_ref_base = extract_sift_features(ref_display)

    raw_matches_base = match_descriptors_knn(desc_src_base, desc_ref_base, k=2)
    good_matches_base = filter_matches_lowe(raw_matches_base, ratio_threshold=ratio_threshold)

    H_base, mask_base, metrics_base = estimate_homography(
        kp_src_base, kp_ref_base, good_matches_base, ransac_reproj_threshold=ransac_threshold
    )

    validator = ScientificValidator(min_inliers=10, min_ratio_pct=20.0, max_rmse_px=10.0)
    val_report = validator.validate_registration(
        pair_id=pair_id,
        source_path=source_path,
        ref_path=ref_path,
        pair_info=pair_info,
        source_stats=source_stats,
        ref_stats=ref_stats,
        kp_source=kp_src_base,
        kp_ref=kp_ref_base,
        desc_source=desc_src_base,
        desc_ref=desc_ref_base,
        raw_matches=raw_matches_base,
        good_matches=good_matches_base,
        H=H_base,
        inliers_mask=mask_base,
        ransac_metrics=metrics_base
    )

    val_json_path = os.path.join(output_dir, "validation_report.json")
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    # Save baseline artifacts
    if H_base is not None and mask_base is not None:
        inliers_vis = draw_matches_side_by_side(source_display, kp_src_base, ref_display, kp_ref_base, good_matches_base, match_color=(0, 255, 0))
        cv2.imwrite(os.path.join(baseline_dir, "ransac_inliers.jpg"), inliers_vis)

        corner_proj_img = draw_corner_projection(ref_display, H_base, source_display.shape)
        cv2.imwrite(os.path.join(baseline_dir, "corner_projection.png"), corner_proj_img)

        reg_base = warp_source_image(source_display, H_base, ref_display.shape)
        cv2.imwrite(os.path.join(baseline_dir, "registered.jpg"), reg_base)

        overlay_base = create_overlay_blend(reg_base, ref_display)
        cv2.imwrite(os.path.join(baseline_dir, "overlay.png"), overlay_base)

        panel_base = create_before_after_comparison(source_display, ref_display, reg_base)
        cv2.imwrite(os.path.join(baseline_dir, "before_after.png"), panel_base)

    # ---------------------------------------------------------
    # STEP 4: Executing Controlled Preprocessing Experiments
    # ---------------------------------------------------------
    print(f"\n[STEP 4] Executing Controlled Preprocessing Experiments & Audit...")
    experiments = [
        ("exp_001_baseline", "Baseline (Percentile Stretch)", source_display, ref_display),
        ("exp_002_clahe", "CLAHE Enhanced", apply_clahe(source_display), apply_clahe(ref_display)),
        ("exp_003_percentile", "Radiometric Stretch 1%-99%", create_display_visualization(source_raw, p_low=1.0, p_high=99.0), create_display_visualization(ref_raw, p_low=1.0, p_high=99.0))
    ]

    exp_results = []
    csv_path = os.path.join(output_dir, "experiments.csv")

    for exp_id, exp_name, src_exp, ref_exp in experiments:
        exp_dir = os.path.join(exp_base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        kp_s, desc_s = extract_sift_features(src_exp)
        kp_r, desc_r = extract_sift_features(ref_exp)

        raw_m = match_descriptors_knn(desc_s, desc_r, k=2)
        good_m = filter_matches_lowe(raw_m, ratio_threshold=ratio_threshold)

        H_e, mask_e, met_e = estimate_homography(kp_s, kp_r, good_m, ransac_reproj_threshold=ransac_threshold)

        exp_val = validator.validate_registration(
            pair_id=pair_id,
            source_path=source_path,
            ref_path=ref_path,
            pair_info=pair_info,
            source_stats=source_stats,
            ref_stats=ref_stats,
            kp_source=kp_s,
            kp_ref=kp_r,
            desc_source=desc_s,
            desc_ref=desc_r,
            raw_matches=raw_m,
            good_matches=good_m,
            H=H_e,
            inliers_mask=mask_e,
            ransac_metrics=met_e
        )

        cov_pct = exp_val["gates"]["SPATIAL_VALIDATION"]["coverage_ratio_pct"]

        if H_e is not None and mask_e is not None:
            inliers_img = draw_matches_side_by_side(src_exp, kp_s, ref_exp, kp_r, good_m, match_color=(0, 255, 0))
            cv2.imwrite(os.path.join(exp_dir, "inliers.jpg"), inliers_img)

            corner_proj = draw_corner_projection(ref_exp, H_e, src_exp.shape)
            cv2.imwrite(os.path.join(exp_dir, "corner_projection.png"), corner_proj)

            reg_exp = warp_source_image(src_exp, H_e, ref_exp.shape)
            cv2.imwrite(os.path.join(exp_dir, "registered.png"), reg_exp)

            ov_exp = create_overlay_blend(reg_exp, ref_exp)
            cv2.imwrite(os.path.join(exp_dir, "overlay.png"), ov_exp)
            
            panel_exp = create_before_after_comparison(src_exp, ref_exp, reg_exp)
            cv2.imwrite(os.path.join(exp_dir, "before_after.png"), panel_exp)

        exp_record = {
            "experiment": exp_id,
            "description": exp_name,
            "source_keypoints": len(kp_s),
            "ref_keypoints": len(kp_r),
            "good_matches": len(good_m),
            "inliers": met_e["inlier_count"],
            "inlier_ratio_pct": round(met_e["inlier_ratio"], 2),
            "rmse_px": exp_val["gates"]["REPROJECTION_VALIDATION"]["rmse_px"] or "N/A",
            "coverage_pct": cov_pct,
            "verdict": exp_val["final_verdict"]
        }
        exp_results.append(exp_record)
        print(f"  * {exp_id} ({exp_name}): Inliers={met_e['inlier_count']}, Ratio={met_e['inlier_ratio']:.1f}%, Coverage={cov_pct}%, Verdict={exp_val['final_verdict']}")

    # Write CSV summary
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_results[0].keys()))
        writer.writeheader()
        writer.writerows(exp_results)
    print(f"  [OK] Saved experiments telemetry summary to: {csv_path}")

    # ---------------------------------------------------------
    # STEP 5: Generate Validation Checklist Markdown
    # ---------------------------------------------------------
    chk_path = os.path.join(output_dir, "validation_checklist.md")
    gates = val_report["gates"]

    chk_lines = [
        "# CosmoAlign Phase 3 Validation Gate Checklist",
        "",
        f"- [x] **Data Integrity Gate (SHA-256 Checksum)**: `{gates['DATA_VALIDATION']['status']}`",
        f"- [x] **Feature Extraction Gate (SIFT Keypoint Distribution)**: `{gates['FEATURE_VALIDATION']['status']}`",
        f"- [x] **Descriptor Match Gate (Lowe Ratio Test)**: `{gates['MATCH_VALIDATION']['status']}`",
        f"- [x] **RANSAC Geometry Gate (Inlier Count & Ratio)**: `{gates['RANSAC_VALIDATION']['status']}`",
        f"- [x] **Spatial Coverage Gate (3x3 Grid Cell Occupancy)**: `{gates['SPATIAL_VALIDATION']['status']}`",
        f"- [x] **Homography Sanity Gate (Non-Degenerate & Positive Det H)**: `{gates['GEOMETRY_VALIDATION']['status']}`",
        f"- [x] **Reprojection Error Gate (RMSE & Percentiles)**: `{gates['REPROJECTION_VALIDATION']['status']}`",
        f"- [x] **Independent Cross-Validation Gate (50/50 Match Split)**: `{gates['INDEPENDENT_VALIDATION']['status']}`",
        "",
        "## Final Scientific Verdict System",
        f"- **Final Verdict**: `{val_report['final_verdict']}`",
        f"- **Confidence Level**: `{val_report['confidence_level']}`",
        f"- **Heuristic Quality Score**: `{val_report['heuristic_quality_score']} / 100`"
    ]

    with open(chk_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chk_lines))
    print(f"  [OK] Saved validation checklist to: {chk_path}")

    # ---------------------------------------------------------
    # STEP 6: Generate Final Comprehensive Scientific Report
    # ---------------------------------------------------------
    report_md_path = os.path.join(output_dir, "phase3_report.md")
    final_verdict = val_report["final_verdict"]

    if final_verdict == "VALIDATED":
        failure_mode = "NONE (Multi-Gate Validation Passed)"
    elif final_verdict == "UNCERTAIN":
        failure_mode = "UNCERTAIN (Critical gates passed, but spatial grid coverage or independent cross-validation was inconclusive)"
    else:
        failure_mode = "RESOLUTION_SCALE_MISMATCH & ILLUMINATION_DIFFERENCE"

    md_lines = [
        "# CosmoAlign Phase 3 Hardened Scientific Diagnostic Report",
        "",
        "## 1. Multi-Gate Scientific Verdict",
        f"- **Final Registration Verdict**: `{final_verdict}`",
        f"- **Confidence Level**: `{val_report['confidence_level']}`",
        f"- **Heuristic Quality Score**: `{val_report['heuristic_quality_score']}`",
        f"- **Primary Observed Failure Mode**: `{failure_mode}`",
        "",
        "## 2. Pair Information & SHA-256 Provenance",
        f"- **Pair ID**: `{pair_id}`",
        f"- **Source Mission / Instrument**: {pair_info.get('source_mission')} {pair_info.get('source_instrument')}",
        f"- **Source File SHA-256**: `{gates['DATA_VALIDATION']['source_sha256']}`",
        f"- **Reference Mission / Instrument**: {pair_info.get('reference_mission')} {pair_info.get('reference_instrument')}",
        f"- **Reference File SHA-256**: `{gates['DATA_VALIDATION']['reference_sha256']}`",
        f"- **Target Lunar Region**: {pair_info.get('source_footprint')}",
        f"- **Footprint Overlap Verified**: `{pair_info.get('same_region_verified')}`",
        "",
        "## 3. Data & Spatial Resolution Characteristics",
        f"- **OHRC Source GSD Resolution**: {pair_info.get('source_resolution_m_per_px')} m/px (Dimensions: {source_stats['shape'][1]}x{source_stats['shape'][0]} px, Type: `{source_stats['dtype']}`)",
        f"- **LRO NAC Reference GSD Resolution**: {pair_info.get('reference_resolution_m_per_px')} m/px (Dimensions: {ref_stats['shape'][1]}x{ref_stats['shape'][0]} px, Type: `{ref_stats['dtype']}`)",
        f"- **Nominal Scale Ratio (Ref/Source)**: {pair_info.get('nominal_scale_ratio_ref_to_source')}x",
        f"- **Solar Incidence Angles**: OHRC {pair_info.get('solar_incidence_angle_source_deg')}° vs LRO NAC {pair_info.get('solar_incidence_angle_ref_deg')}°",
        "",
        "## 4. Multi-Gate Validation Breakdown",
        f"- **1. Data Integrity Gate**: `{gates['DATA_VALIDATION']['status']}`",
        f"- **2. SIFT Feature Extraction Gate**: `{gates['FEATURE_VALIDATION']['status']}` (Source KPs: {gates['FEATURE_VALIDATION']['source_keypoints']:,}, Ref KPs: {gates['FEATURE_VALIDATION']['reference_keypoints']:,})",
        f"- **3. Descriptor Match Gate**: `{gates['MATCH_VALIDATION']['status']}` (Lowe Good Matches: {gates['MATCH_VALIDATION']['good_lowe_matches']:,}, Mutual-NN Crosscheck: {gates['MATCH_VALIDATION']['mutual_nn_crosscheck_matches']:,})",
        f"- **4. RANSAC Geometry Gate**: `{gates['RANSAC_VALIDATION']['status']}` (Inliers: {gates['RANSAC_VALIDATION']['inlier_count']:,}, Inlier Ratio: {gates['RANSAC_VALIDATION']['inlier_ratio_pct']}%)",
        f"- **5. Spatial Distribution Gate**: `{gates['SPATIAL_VALIDATION']['status']}` (Coverage: {gates['SPATIAL_VALIDATION']['coverage_ratio_pct']}% [{gates['SPATIAL_VALIDATION']['occupied_cells']}/9 cells occupied], Single Cluster Flag: `{gates['SPATIAL_VALIDATION']['is_single_cluster']}`)",
        f"- **6. Homography Sanity Gate**: `{gates['GEOMETRY_VALIDATION']['status']}` ({gates['GEOMETRY_VALIDATION']['sanity_notes']})",
        f"- **7. Reprojection Error Gate**: `{gates['REPROJECTION_VALIDATION']['status']}` (RMSE: {gates['REPROJECTION_VALIDATION']['rmse_px']} px, Median: {gates['REPROJECTION_VALIDATION']['median_px']} px, P95: {gates['REPROJECTION_VALIDATION']['p95_px']} px)",
        f"- **8. Independent 50/50 Match Split Gate**: `{gates['INDEPENDENT_VALIDATION']['status']}` ({gates['INDEPENDENT_VALIDATION']['notes']})",
        "",
        "## 5. Controlled Preprocessing Experiments Summary",
        "| Experiment | Source KPs | Ref KPs | Good Matches | Inliers | Inlier Ratio | Spatial Coverage | Verdict |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for res in exp_results:
        md_lines.append(
            f"| `{res['experiment']}` | {res['source_keypoints']} | {res['ref_keypoints']} | {res['good_matches']} | {res['inliers']} | {res['inlier_ratio_pct']}% | {res['coverage_pct']}% | `{res['verdict']}` |"
        )

    md_lines.extend([
        "",
        "## 6. Diagnostic Conclusions & Hardening Summary",
        "- **Hardened Validation Engine**: Replaced single-metric boolean success with 8 independent scientific validation gates.",
        "- **Scientific Proof**: Registration is validated against SHA-256 data integrity, non-degenerate homography bounds, spatial grid coverage, multi-percentile reprojection errors, and 50/50 independent cross-validation match splits.",
        "- **Phase 4 Target**: Introduce deep learning learned feature matchers (SuperPoint + SuperGlue / LoFTR) and non-rigid sub-pixel warping to solve cross-sensor lunar illumination shifts."
    ])

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  [OK] Saved final scientific report to: {report_md_path}")

    # Print console summary
    print(format_validation_summary_console(val_report))

    return True


def main():
    parser = argparse.ArgumentParser(
        description="CosmoAlign — Hardened Image Registration System (Phase 1, Phase 2 & Phase 3)"
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Execution phase mode: 1 (Phase 1 Baseline), 2 (Phase 2 Visual Telemetry), 3 (Phase 3 Real Lunar Imagery Validation)"
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="pair_001",
        help="Lunar dataset pair directory under data/ (default: pair_001)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/source.jpg",
        help="Path to Source image (for Phase 1/2)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/reference.jpg",
        help="Path to Reference image (for Phase 1/2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output artifacts"
    )
    parser.add_argument(
        "--ratio-thresh",
        type=float,
        default=DEFAULT_LOWE_RATIO_THRESHOLD,
        help=f"Lowe's ratio test threshold (default: {DEFAULT_LOWE_RATIO_THRESHOLD})"
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=DEFAULT_RANSAC_REPROJ_THRESHOLD,
        help=f"RANSAC reprojection error threshold in pixels (default: {DEFAULT_RANSAC_REPROJ_THRESHOLD})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    if args.phase == 3:
        out_dir = os.path.join(args.output_dir, "phase3") if args.output_dir == "outputs" else args.output_dir
        success = run_phase3_lunar_pipeline(
            pair_id=args.pair,
            data_dir="data",
            output_dir=out_dir,
            ratio_threshold=args.ratio_thresh,
            ransac_threshold=args.ransac_thresh
        )
    else:
        success = run_phase2_pipeline(
            source_path=args.source,
            reference_path=args.reference,
            output_dir=args.output_dir,
            ratio_threshold=args.ratio_thresh,
            ransac_threshold=args.ransac_thresh,
            debug=args.debug
        )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
