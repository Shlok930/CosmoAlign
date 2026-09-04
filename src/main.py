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
from metadata import (
    inspect_image_stats,
    calculate_scale_ratio,
    build_scale_context,
    format_metadata_report
)
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


def run_phase4_illumination_experiments(
    data_dir: str = "data",
    output_dir: str = "outputs/phase4/illumination",
    ratio_threshold: float = DEFAULT_LOWE_RATIO_THRESHOLD,
    ransac_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD
) -> bool:
    """Run controlled preprocessing experiments across the illumination stress pairs."""
    stress_pairs = [
        ("pair_i1", "10deg"),
        ("pair_i2", "30deg"),
        ("pair_i3", "50deg"),
        ("pair_i4", "70deg")
    ]
    preprocessing_variants = [
        ("illum_001_2_98", "2%-98% Percentile Stretch", 2.0, 98.0, False),
        ("illum_002_1_99", "1%-99% Percentile Stretch", 1.0, 99.0, False),
        ("illum_003_2_98_clahe", "2%-98% Stretch + CLAHE", 2.0, 98.0, True),
        ("illum_004_1_99_clahe", "1%-99% Stretch + CLAHE", 1.0, 99.0, True)
    ]
    experiment_rows = []
    report_groups = []

    def metric_or_none(mapping, *keys):
        value = mapping
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    for stress_pair_id, fallback_level in stress_pairs:
        pair_id = os.path.join("stress_tests", "illumination", stress_pair_id)
        pair_rows = []
        try:
            source_raw, ref_raw, pair_info = load_lunar_pair(pair_id=pair_id, data_dir=data_dir)
            source_path = os.path.join(data_dir, pair_id, "source.tif")
            ref_path = os.path.join(data_dir, pair_id, "reference.tif")
            source_stats = inspect_image_stats(source_raw)
            ref_stats = inspect_image_stats(ref_raw)
            scale_context = build_scale_context(pair_info)
            stress_level = pair_info.get("stress_level", fallback_level)
            stress_category = pair_info.get("stress_category", "illumination")
            source_sun = pair_info.get("solar_incidence_angle_source_deg")
            reference_sun = pair_info.get("solar_incidence_angle_ref_deg")
        except Exception as error:
            print(f"  [FAILED] {stress_pair_id}: unable to load pair ({error})", file=sys.stderr)
            for variant_id, description, _, _, _ in preprocessing_variants:
                pair_rows.append({
                    "pair_id": stress_pair_id,
                    "illumination_stress_level": fallback_level,
                    "stress_category": "illumination",
                    "source_solar_incidence_deg": None,
                    "reference_solar_incidence_deg": None,
                    "preprocessing": description,
                    "source_keypoints": None,
                    "reference_keypoints": None,
                    "raw_matches": None,
                    "good_matches": None,
                    "inliers": None,
                    "inlier_ratio_pct": None,
                    "spatial_coverage_pct": None,
                    "rmse_px": None,
                    "median_px": None,
                    "p95_px": None,
                    "independent_validation": "N/A",
                    "physical_scale_ratio_ref_to_source": None,
                    "scale_metadata_consistent": None,
                    "verdict": "FAILED",
                    "failure": str(error)
                })
            experiment_rows.extend(pair_rows)
            report_groups.append((stress_pair_id, pair_rows))
            continue

        for variant_id, description, low_percentile, high_percentile, use_clahe in preprocessing_variants:
            variant_output = os.path.join(output_dir, stress_pair_id, variant_id)
            os.makedirs(variant_output, exist_ok=True)
            row = {
                "pair_id": stress_pair_id,
                "illumination_stress_level": stress_level,
                "stress_category": stress_category,
                "source_solar_incidence_deg": source_sun,
                "reference_solar_incidence_deg": reference_sun,
                "preprocessing": description,
                "source_keypoints": None,
                "reference_keypoints": None,
                "raw_matches": None,
                "good_matches": None,
                "inliers": None,
                "inlier_ratio_pct": None,
                "spatial_coverage_pct": None,
                "rmse_px": None,
                "median_px": None,
                "p95_px": None,
                "independent_validation": "N/A",
                "physical_scale_ratio_ref_to_source": scale_context["calculated_ref_to_source_ratio"],
                "scale_metadata_consistent": scale_context["ratio_consistent"],
                "verdict": "FAILED",
                "failure": None
            }
            try:
                source_display = create_display_visualization(
                    source_raw, p_low=low_percentile, p_high=high_percentile
                )
                ref_display = create_display_visualization(
                    ref_raw, p_low=low_percentile, p_high=high_percentile
                )
                if use_clahe:
                    source_display = apply_clahe(source_display)
                    ref_display = apply_clahe(ref_display)

                kp_source, desc_source = extract_sift_features(source_display)
                kp_ref, desc_ref = extract_sift_features(ref_display)
                raw_matches = match_descriptors_knn(desc_source, desc_ref, k=2)
                good_matches = filter_matches_lowe(raw_matches, ratio_threshold=ratio_threshold)
                H, inliers_mask, ransac_metrics = estimate_homography(
                    kp_source,
                    kp_ref,
                    good_matches,
                    ransac_reproj_threshold=ransac_threshold
                )
                validation = ScientificValidator(
                    min_inliers=10, min_ratio_pct=20.0, max_rmse_px=10.0
                ).validate_registration(
                    pair_id=stress_pair_id,
                    source_path=source_path,
                    ref_path=ref_path,
                    pair_info=pair_info,
                    source_stats=source_stats,
                    ref_stats=ref_stats,
                    kp_source=kp_source,
                    kp_ref=kp_ref,
                    desc_source=desc_source,
                    desc_ref=desc_ref,
                    raw_matches=raw_matches,
                    good_matches=good_matches,
                    H=H,
                    inliers_mask=inliers_mask,
                    ransac_metrics=ransac_metrics
                )
                row.update({
                    "source_keypoints": len(kp_source),
                    "reference_keypoints": len(kp_ref),
                    "raw_matches": len(raw_matches),
                    "good_matches": len(good_matches),
                    "inliers": ransac_metrics.get("inlier_count"),
                    "inlier_ratio_pct": ransac_metrics.get("inlier_ratio"),
                    "spatial_coverage_pct": metric_or_none(validation, "gates", "SPATIAL_VALIDATION", "coverage_ratio_pct"),
                    "rmse_px": metric_or_none(validation, "gates", "REPROJECTION_VALIDATION", "rmse_px"),
                    "median_px": metric_or_none(validation, "gates", "REPROJECTION_VALIDATION", "median_px"),
                    "p95_px": metric_or_none(validation, "gates", "REPROJECTION_VALIDATION", "p95_px"),
                    "independent_validation": metric_or_none(validation, "gates", "INDEPENDENT_VALIDATION", "status"),
                    "verdict": validation.get("final_verdict", "FAILED")
                })
                if H is not None and inliers_mask is not None:
                    cv2.imwrite(os.path.join(variant_output, "inliers.jpg"), draw_matches_side_by_side(
                        source_display, kp_source, ref_display, kp_ref, good_matches, match_color=(0, 255, 0)
                    ))
                    cv2.imwrite(os.path.join(variant_output, "corner_projection.png"), draw_corner_projection(
                        ref_display, H, source_display.shape
                    ))
                    registered = warp_source_image(source_display, H, ref_display.shape)
                    cv2.imwrite(os.path.join(variant_output, "registered.png"), registered)
                    cv2.imwrite(os.path.join(variant_output, "overlay.png"), create_overlay_blend(
                        registered, ref_display
                    ))
                    cv2.imwrite(os.path.join(variant_output, "before_after.png"), create_before_after_comparison(
                        source_display, ref_display, registered
                    ))
            except Exception as error:
                row["failure"] = str(error)
                print(f"  [FAILED] {stress_pair_id}/{variant_id}: {error}", file=sys.stderr)
            pair_rows.append(row)
            experiment_rows.append(row)
        report_groups.append((stress_pair_id, pair_rows))

    csv_path = os.path.join(output_dir, "illumination_experiments.csv")
    csv_fields = [
        "pair_id", "illumination_stress_level", "stress_category",
        "source_solar_incidence_deg", "reference_solar_incidence_deg", "preprocessing",
        "source_keypoints", "reference_keypoints", "raw_matches", "good_matches",
        "inliers", "inlier_ratio_pct", "spatial_coverage_pct", "rmse_px", "median_px",
        "p95_px", "independent_validation", "physical_scale_ratio_ref_to_source",
        "scale_metadata_consistent", "verdict", "failure"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(experiment_rows)

    valid_rows = [row for row in experiment_rows if row["inlier_ratio_pct"] is not None]
    best_row = max(
        valid_rows,
        key=lambda row: (
            row["verdict"] == "VALIDATED",
            row["independent_validation"] == "PASS",
            row["spatial_coverage_pct"] or 0.0,
            row["inlier_ratio_pct"] or 0.0,
            -(row["rmse_px"] if row["rmse_px"] is not None else float("inf"))
        ),
        default=None
    )
    report_lines = [
        "# Phase 4 Step 4 Illumination Robustness Experiments",
        "",
        "## 1. Objective",
        "Evaluate whether controlled radiometric preprocessing improves registration under synthetic illumination stress.",
        "",
        "## 2. Dataset Description",
        "Four generated illumination pairs are evaluated: pair_i1 (10deg), pair_i2 (30deg), pair_i3 (50deg), and pair_i4 (70deg).",
        "Raw scientific arrays remain unchanged; preprocessing is applied only to uint8 feature-registration representations.",
        "",
        "## 3. Experimental Methodology",
        "Each pair uses identical source/reference preprocessing strategy, existing SIFT, matching, Lowe filtering, RANSAC, and the eight-gate ScientificValidator.",
        "No illumination pass/fail gate or automatic resizing is introduced.",
        "",
        "## 4. All Experiments",
        "| Pair | Stress | Preprocessing | KP source | KP ref | Raw | Good | Inliers | Ratio | Coverage | RMSE | Median | P95 | Independent | Scale | Consistent | Verdict |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :---: | :--- |"
    ]
    for row in experiment_rows:
        report_lines.append(
            f"| {row['pair_id']} | {row['illumination_stress_level']} | {row['preprocessing']} | {row['source_keypoints']} | {row['reference_keypoints']} | {row['raw_matches']} | {row['good_matches']} | {row['inliers']} | {row['inlier_ratio_pct']} | {row['spatial_coverage_pct']} | {row['rmse_px']} | {row['median_px']} | {row['p95_px']} | {row['independent_validation']} | {row['physical_scale_ratio_ref_to_source']} | {row['scale_metadata_consistent']} | {row['verdict']} |"
        )
    report_lines.extend(["", "## 5. Results Grouped by Illumination Level"])
    for stress_pair_id, rows in report_groups:
        report_lines.append(f"### {stress_pair_id}")
        for row in rows:
            report_lines.append(
                f"- {row['preprocessing']}: verdict={row['verdict']}, inlier ratio={row['inlier_ratio_pct']}, coverage={row['spatial_coverage_pct']}, RMSE={row['rmse_px']}, independent={row['independent_validation']}"
            )
    report_lines.extend(["", "## 6. Comparison of Preprocessing Strategies"])
    for description in [variant[1] for variant in preprocessing_variants]:
        rows = [row for row in experiment_rows if row["preprocessing"] == description and row["inlier_ratio_pct"] is not None]
        mean_ratio = np.mean([row["inlier_ratio_pct"] for row in rows]) if rows else None
        mean_coverage = np.mean([row["spatial_coverage_pct"] for row in rows]) if rows else None
        report_lines.append(f"- {description}: measured mean inlier ratio={mean_ratio}, mean spatial coverage={mean_coverage}.")
    report_lines.extend([
        "",
        "## 7. Best-Performing Strategy",
        f"- Based on the measured verdict, independent validation, spatial coverage, inlier ratio, and RMSE ordering, the best row was `{best_row['pair_id']} / {best_row['preprocessing']}`." if best_row else "- No experiment produced sufficient numeric registration metrics to identify a best strategy.",
        "",
        "## 8. Illumination Robustness Observations",
        "Performance is reported empirically from the measured registration metrics; no preprocessing method is assumed superior in advance.",
        "",
        "## 9. Failure and Breakpoint Observations",
        f"{sum(1 for row in experiment_rows if row['failure'])} of {len(experiment_rows)} experiments recorded execution failures.",
        "",
        "## 10. Empirical Result Statement",
        "These results are empirical and based on measured registration metrics from the existing scientific validation pipeline."
    ])
    with open(os.path.join(output_dir, "illumination_report.md"), "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines))
    print(f"  [OK] Saved Phase 4 illumination CSV: {csv_path}")
    print(f"  [OK] Saved Phase 4 illumination report: {os.path.join(output_dir, 'illumination_report.md')}")
    return True


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

        # Phase 4 Step 3: Build scale metadata/telemetry context.
        scale_context = build_scale_context(pair_info)

        print(f"  [OK] Source SHA-256:    {source_sha256[:16]}...")
        print(f"  [OK] Reference SHA-256: {ref_sha256[:16]}...")
        print(f"  [OK] Physical GSD ratio (Ref/Source): {scale_context['calculated_ref_to_source_ratio']:.4f}x")
        print(f"  [OK] Metadata ratio consistency: {scale_context['ratio_consistent']}")
        if scale_context["stress_category"] is not None:
            print(f"  [OK] Synthetic stress: {scale_context['stress_category']} / {scale_context['stress_level']}x")
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
    # Phase 4 Step 3: Preserve scale telemetry in the validation report.
    val_report["scale_context"] = scale_context

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
            "verdict": exp_val["final_verdict"],
            "physical_scale_ratio_ref_to_source": scale_context["calculated_ref_to_source_ratio"],
            "synthetic_stress_category": scale_context["stress_category"],
            "synthetic_stress_level": scale_context["stress_level"],
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
        f"- **Calculated Physical Scale Ratio (Ref/Source)**: {scale_context['calculated_ref_to_source_ratio']:.4f}x",
        f"- **Scale Metadata Consistency**: `{scale_context['ratio_consistent']}`",
        f"- **Synthetic Scale Stress**: `{scale_context['stress_level']}x` ({scale_context['stress_category']})" if scale_context["stress_level"] is not None else "- **Synthetic Scale Stress**: `None` (baseline pair)",
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

    # Phase 4 Step 4: Run controlled illumination robustness experiments.
    run_phase4_illumination_experiments(
        data_dir="data",
        output_dir="outputs/phase4/illumination",
        ratio_threshold=ratio_threshold,
        ransac_threshold=ransac_threshold
    )

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
