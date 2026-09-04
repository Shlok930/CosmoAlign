"""Phase 4 Step 5.2: compare Homography and Affine under viewpoint stress."""

import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .evaluation import analyze_inlier_spatial_distribution, check_homography_sanity, compute_reprojection_stats
from .feature_detection import extract_sift_features
from .lunar_data import create_display_visualization, load_lunar_pair
from .matching import filter_matches_lowe, match_descriptors_knn
from .metadata import build_scale_context, inspect_image_stats
from .phase4_geometry import compute_affine_reprojection_rmse, estimate_affine, warp_source_image_affine
from .registration import create_overlay_blend, estimate_homography, warp_source_image
from .visualization import (
    create_before_after_comparison,
    draw_corner_projection,
    draw_inliers_vs_outliers,
    draw_matches_side_by_side,
)


VIEWPOINT_PAIRS = (
    ("pair_v1", "5deg"),
    ("pair_v2", "15deg"),
    ("pair_v3", "30deg"),
    ("pair_v4", "45deg"),
)
RANSAC_REPROJECTION_THRESHOLD = 5.0


def _empty_row(pair_id: str, stress_level: Any, model: str) -> Dict[str, Any]:
    return {
        "pair_id": pair_id,
        "viewpoint_stress_level": stress_level,
        "model": model,
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
        "geometry_sanity": None,
        "independent_validation": None,
        "final_verdict": None,
        "failure": None,
    }


def _affine_reprojection_stats(
    kp_source: List[cv2.KeyPoint],
    kp_ref: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    affine_matrix: np.ndarray,
    inliers_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Calculate affine inlier error percentiles using cv2.transform."""
    mask = np.asarray(inliers_mask).ravel()
    inlier_indices = np.where(mask == 1)[0]
    if len(inlier_indices) == 0:
        return {"rmse": None, "median_px": None, "p95_px": None}

    source_points = np.float32([kp_source[matches[i].queryIdx].pt for i in inlier_indices]).reshape(-1, 1, 2)
    reference_points = np.float32([kp_ref[matches[i].trainIdx].pt for i in inlier_indices]).reshape(-1, 1, 2)
    transformed_points = cv2.transform(source_points, affine_matrix)
    errors = np.linalg.norm(transformed_points - reference_points, axis=2).ravel()
    rmse = compute_affine_reprojection_rmse(source_points, reference_points, affine_matrix, np.ones(len(inlier_indices), dtype=np.uint8))
    return {
        "rmse": rmse,
        "median_px": float(np.median(errors)),
        "p95_px": float(np.percentile(errors, 95)),
    }


def _affine_geometry_sanity(
    affine_matrix: Optional[np.ndarray],
    source_shape: Tuple[int, ...],
    reference_shape: Tuple[int, ...],
) -> Tuple[bool, str]:
    """Perform local sanity checks appropriate for a 2x3 affine matrix."""
    if affine_matrix is None or affine_matrix.shape != (2, 3):
        return False, "Affine matrix is None or has invalid shape."
    if not np.isfinite(affine_matrix).all():
        return False, "Affine matrix contains non-finite values."

    linear = affine_matrix[:, :2]
    determinant = float(np.linalg.det(linear))
    if abs(determinant) < 1e-8:
        return False, "Affine linear component is degenerate."

    source_height, source_width = source_shape[:2]
    reference_height, reference_width = reference_shape[:2]
    corners = np.float32([[0, 0], [source_width, 0], [source_width, source_height], [0, source_height]])
    projected = cv2.transform(corners.reshape(-1, 1, 2), affine_matrix).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return False, "Transformed source corners contain non-finite values."

    area = abs(float(cv2.contourArea(projected.astype(np.float32))))
    reference_area = float(reference_width * reference_height)
    if area <= 0:
        return False, "Transformed source corner polygon has non-positive area."
    if area < reference_area * 0.05 or area > reference_area * 10.0:
        return False, "Affine transformed area is outside the non-pathological range."
    return True, f"Affine geometry is sane (area ratio: {area / reference_area:.3f}, determinant: {determinant:.6f})."


def _run_homography_validation(
    pair_id: str,
    source_path: str,
    reference_path: str,
    pair_info: Dict[str, Any],
    source_stats: Dict[str, Any],
    reference_stats: Dict[str, Any],
    kp_source: List[cv2.KeyPoint],
    kp_ref: List[cv2.KeyPoint],
    desc_source: np.ndarray,
    desc_ref: np.ndarray,
    raw_matches: List[List[cv2.DMatch]],
    good_matches: List[cv2.DMatch],
    homography: Optional[np.ndarray],
    inliers_mask: Optional[np.ndarray],
    metrics: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Use the existing validator only for the Homography model."""
    try:
        src_dir = str(Path(__file__).resolve().parent)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from validation import ScientificValidator

        report = ScientificValidator().validate_registration(
            pair_id=pair_id,
            source_path=source_path,
            ref_path=reference_path,
            pair_info=pair_info,
            source_stats=source_stats,
            ref_stats=reference_stats,
            kp_source=kp_source,
            kp_ref=kp_ref,
            desc_source=desc_source,
            desc_ref=desc_ref,
            raw_matches=raw_matches,
            good_matches=good_matches,
            H=homography,
            inliers_mask=inliers_mask,
            ransac_metrics=metrics,
        )
        independent = report["gates"]["INDEPENDENT_VALIDATION"].get("status")
        return independent, report.get("final_verdict")
    except Exception as error:
        return "ERROR", f"Validator error: {error}"


def _populate_model_row(
    row: Dict[str, Any],
    model_output: Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]],
    kp_source: List[cv2.KeyPoint],
    kp_ref: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    source_shape: Tuple[int, ...],
    reference_shape: Tuple[int, ...],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    matrix, mask, metrics = model_output
    row["inliers"] = metrics.get("inlier_count", 0)
    row["inlier_ratio_pct"] = metrics.get("inlier_ratio")
    if matrix is None or mask is None:
        row["geometry_sanity"] = "FAIL"
        row["failure"] = "Model estimation returned no matrix or inlier mask."
        return matrix, mask

    spatial = analyze_inlier_spatial_distribution(kp_source, good_matches, mask, source_shape)
    row["spatial_coverage_pct"] = round(spatial["coverage_ratio"] * 100.0, 1)
    return matrix, mask


def _write_artifacts(
    output_dir: Path,
    model: str,
    source_display: np.ndarray,
    reference_display: np.ndarray,
    kp_source: List[cv2.KeyPoint],
    kp_ref: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    inliers_mask: np.ndarray,
    matrix: np.ndarray,
) -> None:
    """Write the small set of model comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{model.lower()}_matches.png"), draw_inliers_vs_outliers(
        source_display, kp_source, reference_display, kp_ref, good_matches, inliers_mask
    ))
    if model == "Homography":
        cv2.imwrite(str(output_dir / "corner_projection.png"), draw_corner_projection(
            reference_display, matrix, source_display.shape
        ))
        registered = warp_source_image(source_display, matrix, reference_display.shape)
    else:
        registered = warp_source_image_affine(source_display, matrix, reference_display.shape)
    cv2.imwrite(str(output_dir / "registered.png"), registered)
    cv2.imwrite(str(output_dir / "overlay.png"), create_overlay_blend(registered, reference_display))
    cv2.imwrite(str(output_dir / "before_after.png"), create_before_after_comparison(
        source_display, reference_display, registered
    ))


def run_viewpoint_geometry_experiment(
    data_dir: str = "data",
    output_dir: str = "outputs/phase4/geometry",
    ratio_threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """Compare Homography and Affine using shared SIFT matches for four pairs.

    The function refuses to overwrite an existing output directory. It returns
    the eight CSV-compatible result rows and writes the CSV and Markdown report.
    """
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing geometry output: {root}")
    root.mkdir(parents=True)
    rows: List[Dict[str, Any]] = []

    for pair_id, expected_level in VIEWPOINT_PAIRS:
        nested_pair_id = f"stress_tests/viewpoint/{pair_id}"
        source_raw, reference_raw, pair_info = load_lunar_pair(nested_pair_id, data_dir=data_dir)
        source_path = os.path.join(data_dir, nested_pair_id, "source.tif")
        reference_path = os.path.join(data_dir, nested_pair_id, "reference.tif")
        source_stats = inspect_image_stats(source_raw)
        reference_stats = inspect_image_stats(reference_raw)
        scale_context = build_scale_context(pair_info)
        stress_level = pair_info.get("stress_level", expected_level)

        source_display = create_display_visualization(source_raw, p_low=2.0, p_high=98.0)
        reference_display = create_display_visualization(reference_raw, p_low=2.0, p_high=98.0)
        kp_source, desc_source = extract_sift_features(source_display)
        kp_ref, desc_ref = extract_sift_features(reference_display)
        raw_matches = match_descriptors_knn(desc_source, desc_ref, k=2)
        good_matches = filter_matches_lowe(raw_matches, ratio_threshold=ratio_threshold)

        common = {
            "source_keypoints": len(kp_source),
            "reference_keypoints": len(kp_ref),
            "raw_matches": len(raw_matches),
            "good_matches": len(good_matches),
        }
        model_jobs = [
            ("Homography", lambda: estimate_homography(
                kp_source, kp_ref, good_matches,
                ransac_reproj_threshold=RANSAC_REPROJECTION_THRESHOLD,
            )),
            ("Affine", lambda: estimate_affine(
                kp_source, kp_ref, good_matches,
                ransac_reproj_threshold=RANSAC_REPROJECTION_THRESHOLD,
            )),
        ]

        for model, estimate in model_jobs:
            row = _empty_row(pair_id, stress_level, model)
            row.update(common)
            try:
                model_output = estimate()
                matrix, mask = _populate_model_row(
                    row, model_output, kp_source, kp_ref, good_matches,
                    source_raw.shape, reference_raw.shape,
                )
                metrics = model_output[2]
                if matrix is not None and mask is not None:
                    if model == "Homography":
                        sane, sanity_message = check_homography_sanity(matrix, source_raw.shape, reference_raw.shape)
                        reprojection = compute_reprojection_stats(
                            kp_source, kp_ref, good_matches, matrix, mask
                        )
                        independent, verdict = _run_homography_validation(
                            nested_pair_id, source_path, reference_path, pair_info,
                            source_stats, reference_stats, kp_source, kp_ref,
                            desc_source, desc_ref, raw_matches, good_matches,
                            matrix, mask, metrics,
                        )
                    else:
                        sane, sanity_message = _affine_geometry_sanity(matrix, source_raw.shape, reference_raw.shape)
                        reprojection = _affine_reprojection_stats(
                            kp_source, kp_ref, good_matches, matrix, mask
                        )
                        independent, verdict = "NOT_APPLICABLE", "NOT_APPLICABLE"
                    row["geometry_sanity"] = "PASS" if sane else f"FAIL: {sanity_message}"
                    row["independent_validation"] = independent
                    row["final_verdict"] = verdict
                    row["rmse_px"] = reprojection.get("rmse")
                    row["median_px"] = reprojection.get("median_px")
                    row["p95_px"] = reprojection.get("p95_px")
                    if sane:
                        _write_artifacts(
                            root / pair_id / model.lower(), model,
                            source_display, reference_display, kp_source, kp_ref,
                            good_matches, mask, matrix,
                        )
                elif row["independent_validation"] is None:
                    row["independent_validation"] = "NOT_AVAILABLE"
                    row["final_verdict"] = "FAILED"
            except Exception as error:
                row["failure"] = str(error)
                row["independent_validation"] = "NOT_AVAILABLE"
                row["final_verdict"] = "FAILED"
            row["physical_scale_ratio_ref_to_source"] = scale_context["calculated_ref_to_source_ratio"]
            row["scale_metadata_consistent"] = scale_context["ratio_consistent"]
            rows.append(row)

    csv_fields = list(rows[0].keys())
    with (root / "geometry_experiments.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_report(root / "geometry_report.md", rows)
    return rows


def _write_report(report_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write an evidence-based comparison report from measured rows."""
    lines = [
        "# Phase 4 Step 5.2: Homography vs Affine Viewpoint Experiment",
        "",
        "## 1. Objective",
        "Compare the existing Homography model with the experimental Affine model under controlled viewpoint stress.",
        "",
        "## 2. Experimental Controls",
        "The same source/reference image, SIFT configuration, extracted keypoints, and Lowe-filtered good matches were used for both models. Both estimators used the same 5.0-pixel RANSAC threshold; only the geometric model changed.",
        "",
        "## 3. Results",
        "| Pair | Level | Model | Good matches | Inliers | Ratio % | Coverage % | RMSE px | Median px | P95 px | Geometry | Independent | Verdict | Failure |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        values = [
            row["pair_id"], row["viewpoint_stress_level"], row["model"], row["good_matches"],
            row["inliers"], row["inlier_ratio_pct"], row["spatial_coverage_pct"], row["rmse_px"],
            row["median_px"], row["p95_px"], row["geometry_sanity"], row["independent_validation"],
            row["final_verdict"], row["failure"],
        ]
        lines.append("| " + " | ".join(str(value) if value is not None else "N/A" for value in values) + " |")

    lines.extend([
        "",
        "## 4. Per-Level Analysis",
    ])
    for pair_id, level in VIEWPOINT_PAIRS:
        pair_rows = [row for row in rows if row["pair_id"] == pair_id]
        lines.append(f"### {level} ({pair_id})")
        for row in pair_rows:
            lines.append(
                f"- {row['model']}: inliers={row['inliers']}, inlier ratio={row['inlier_ratio_pct']}%, "
                f"coverage={row['spatial_coverage_pct']}%, RMSE={row['rmse_px']} px, "
                f"independent validation={row['independent_validation']}, verdict={row['final_verdict']}."
            )

    lines.extend([
        "",
        "## 5. Model Comparison",
        "The measurements above are the basis for comparing inlier reliability, inlier ratio, reprojection errors, spatial coverage, and high-stress failures. Physical scale metadata is reported separately from viewpoint stress.",
        "",
        "## 6. Conclusion",
        "No model is declared superior a priori. The conclusion is empirical and must be drawn from the measured inlier ratio, spatial coverage, RMSE, independent validation, and final verdict in this report.",
        "",
        "Affine independent validation is marked NOT_APPLICABLE because the existing ScientificValidator performs Homography-specific independent validation and was not modified for this experiment.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
