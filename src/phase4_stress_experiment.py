"""Phase 4 Step 6 categorized evaluation for scale and combined stress."""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from .evaluation import analyze_inlier_spatial_distribution, check_homography_sanity, compute_reprojection_stats
from .feature_detection import extract_sift_features
from .lunar_data import create_display_visualization, load_lunar_pair
from .matching import filter_matches_lowe, match_descriptors_knn
from .metadata import build_scale_context, inspect_image_stats
from .registration import create_overlay_blend, estimate_homography, warp_source_image
from .phase4_reporting import point_transform_error
from .visualization import create_before_after_comparison, draw_corner_projection, draw_inliers_vs_outliers
from .validation import ScientificValidator


INPUT_CATEGORIES = ("scale", "combined")
CSV_FIELDS = [
    "category", "pair_id", "stress_level", "nominal_scale_ratio_ref_to_source",
    "illumination_difference_deg", "viewpoint_rotation_deg", "source_keypoints",
    "reference_keypoints", "raw_matches", "good_matches", "inliers", "inlier_ratio_pct",
    "spatial_coverage_pct", "rmse_px", "median_px", "p95_px", "geometry_sanity",
    "independent_validation", "final_verdict", "failure", "physical_scale_ratio_ref_to_source",
    "scale_metadata_consistent", "ground_truth_comparison_status", "transform_error_mean_px",
    "transform_error_max_px",
]


def _row(category: str, pair_id: str) -> Dict[str, Any]:
    return {field: None for field in CSV_FIELDS} | {"category": category, "pair_id": pair_id}


def _run_pair(category: str, pair_id: str, data_dir: str, output_root: Path) -> Dict[str, Any]:
    row = _row(category, pair_id)
    pair_id_for_loader = f"stress_tests/{category}/{pair_id}"
    source_path = Path(data_dir) / pair_id_for_loader / "source.tif"
    reference_path = Path(data_dir) / pair_id_for_loader / "reference.tif"
    try:
        source_raw, reference_raw, pair_info = load_lunar_pair(pair_id_for_loader, data_dir=data_dir)
        row.update({
            "stress_level": pair_info.get("stress_level"),
            "nominal_scale_ratio_ref_to_source": pair_info.get("nominal_scale_ratio_ref_to_source"),
            "illumination_difference_deg": pair_info.get("illumination_difference_deg"),
            "viewpoint_rotation_deg": pair_info.get("viewpoint_rotation_deg"),
        })
        source_stats = inspect_image_stats(source_raw)
        reference_stats = inspect_image_stats(reference_raw)
        scale_context = build_scale_context(pair_info)
        row["physical_scale_ratio_ref_to_source"] = scale_context["calculated_ref_to_source_ratio"]
        row["scale_metadata_consistent"] = scale_context["ratio_consistent"]
        source_display = create_display_visualization(source_raw, p_low=2.0, p_high=98.0)
        reference_display = create_display_visualization(reference_raw, p_low=2.0, p_high=98.0)
        kp_source, desc_source = extract_sift_features(source_display)
        kp_ref, desc_ref = extract_sift_features(reference_display)
        raw_matches = match_descriptors_knn(desc_source, desc_ref, k=2)
        good_matches = filter_matches_lowe(raw_matches)
        row.update({
            "source_keypoints": len(kp_source), "reference_keypoints": len(kp_ref),
            "raw_matches": len(raw_matches), "good_matches": len(good_matches),
        })
        homography, mask, metrics = estimate_homography(
            kp_source, kp_ref, good_matches, ransac_reproj_threshold=5.0
        )
        row.update({"inliers": metrics.get("inlier_count"), "inlier_ratio_pct": metrics.get("inlier_ratio")})
        if homography is None or mask is None:
            row.update({"geometry_sanity": "FAIL", "final_verdict": "FAILED", "failure": "Homography estimation returned no model."})
            return row

        spatial = analyze_inlier_spatial_distribution(kp_source, good_matches, mask, source_raw.shape)
        reprojection = compute_reprojection_stats(kp_source, kp_ref, good_matches, homography, mask)
        sane, sanity_message = check_homography_sanity(homography, source_raw.shape, reference_raw.shape)
        row.update({
            "spatial_coverage_pct": round(spatial["coverage_ratio"] * 100.0, 1),
            "rmse_px": reprojection.get("rmse"), "median_px": reprojection.get("median_px"),
            "p95_px": reprojection.get("p95_px"),
            "geometry_sanity": "PASS" if sane else f"FAIL: {sanity_message}",
        })
        ground_truth = np.asarray(pair_info.get("ground_truth_transform", {}).get("source_to_reference_homography"), dtype=float)
        comparison = point_transform_error(homography, ground_truth, source_raw.shape)
        row.update({
            "ground_truth_comparison_status": comparison["status"],
            "transform_error_mean_px": comparison["mean_px"],
            "transform_error_max_px": comparison["max_px"],
        })
        validator = ScientificValidator()
        validation = validator.validate_registration(
            pair_id=pair_id_for_loader, source_path=str(source_path), ref_path=str(reference_path),
            pair_info=pair_info, source_stats=source_stats, ref_stats=reference_stats,
            kp_source=kp_source, kp_ref=kp_ref, desc_source=desc_source, desc_ref=desc_ref,
            raw_matches=raw_matches, good_matches=good_matches, H=homography,
            inliers_mask=mask, ransac_metrics=metrics,
        )
        row["independent_validation"] = validation["gates"]["INDEPENDENT_VALIDATION"].get("status")
        row["final_verdict"] = validation.get("final_verdict")
        if sane:
            artifact_dir = output_root / category / pair_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(artifact_dir / "inliers.png"), draw_inliers_vs_outliers(
                source_display, kp_source, reference_display, kp_ref, good_matches, mask
            ))
            cv2.imwrite(str(artifact_dir / "corner_projection.png"), draw_corner_projection(
                reference_display, homography, source_display.shape
            ))
            registered = warp_source_image(source_display, homography, reference_display.shape)
            cv2.imwrite(str(artifact_dir / "registered.png"), registered)
            cv2.imwrite(str(artifact_dir / "overlay.png"), create_overlay_blend(registered, reference_display))
            cv2.imwrite(str(artifact_dir / "before_after.png"), create_before_after_comparison(
                source_display, reference_display, registered
            ))
    except Exception as error:
        row["failure"] = str(error)
        row["final_verdict"] = "FAILED"
    artifact_dir = output_root / category / pair_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    if row.get("failure") or str(row.get("geometry_sanity", "")).startswith("FAIL"):
        failure = {
            "pair_id": pair_id,
            "stress_category": category,
            "stress_level": row.get("stress_level"),
            "failure_stage": "geometry_or_validation",
            "failure_reason": row.get("failure") or row.get("geometry_sanity"),
            "good_matches": row.get("good_matches"),
            "inliers": row.get("inliers"),
            "inlier_ratio_pct": row.get("inlier_ratio_pct"),
            "rmse_px": row.get("rmse_px"),
            "spatial_coverage_pct": row.get("spatial_coverage_pct"),
            "geometry_status": row.get("geometry_sanity"),
            "visualization_status": "unavailable" if row.get("geometry_sanity") != "PASS" else "available",
        }
        (artifact_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
    return row


def _write_report(path: Path, rows: List[Dict[str, Any]], reused: Dict[str, List[Dict[str, Any]]]) -> None:
    lines = [
        "# Phase 4 Step 6: Categorized Stress Evaluation", "",
        "## 1. Objective",
        "Evaluate Homography registration robustness across controlled scale and combined stress categories.", "",
        "## 2. Experimental Scope",
        "Scale and Combined were newly evaluated by this script. Illumination and Viewpoint were reused from their completed CSV outputs; neither experiment was rerun.", "",
        "## 3. Scale Results", "",
        _table([row for row in rows if row["category"] == "scale"], include_combined=False), "",
        "## 4. Combined Results", "",
        _table([row for row in rows if row["category"] == "combined"], include_combined=True), "",
        "## 5. Cross-Category Summary", "",
    ]
    summary_groups = [
        ("Scale", [r for r in rows if r["category"] == "scale"]),
        ("Combined", [r for r in rows if r["category"] == "combined"]),
    ] + list(reused.items())
    for category, category_rows in summary_groups:
        lines.append(f"### {category}")
        numeric_ratio = [float(r["inlier_ratio_pct"]) for r in category_rows if r.get("inlier_ratio_pct") is not None]
        numeric_rmse = [float(r["rmse_px"]) for r in category_rows if r.get("rmse_px") is not None]
        coverage = [float(r["spatial_coverage_pct"]) for r in category_rows if r.get("spatial_coverage_pct") is not None]
        lines.append(f"- Rows: {len(category_rows)}; mean/min inlier ratio: {np.mean(numeric_ratio) if numeric_ratio else 'N/A'} / {min(numeric_ratio) if numeric_ratio else 'N/A'}%; mean/max RMSE: {np.mean(numeric_rmse) if numeric_rmse else 'N/A'} / {max(numeric_rmse) if numeric_rmse else 'N/A'} px; mean coverage: {np.mean(coverage) if coverage else 'N/A'}%.")
    lines.extend([
        "", "## 6. Stress Degradation",
        "Measured Scale and Combined rows should be compared by their recorded stress parameters; no arbitrary robustness boundary is inferred.", "",
        "## 7. Validator Caveat",
        "The existing independent-validation gate can report failures when other registration metrics are strong. This is preserved exactly and reserved for Phase 4 Step 7.", "",
        "## 8. Conclusion",
        "This report is an empirical summary of measured registration metrics. It does not declare the system robust merely because evaluation completed.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(rows: List[Dict[str, Any]], include_combined: bool) -> str:
    columns = ["pair_id", "stress_level", "nominal_scale_ratio_ref_to_source"]
    if include_combined:
        columns += ["illumination_difference_deg", "viewpoint_rotation_deg"]
    columns += ["good_matches", "inliers", "inlier_ratio_pct", "spatial_coverage_pct", "rmse_px", "median_px", "p95_px", "geometry_sanity", "independent_validation", "final_verdict"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(str(row.get(column)) if row.get(column) is not None else "N/A" for column in columns) + " |" for row in rows]
    return "\n".join(lines)


def run_categorized_stress_evaluation(
    data_dir: str = "data",
    output_dir: str = "outputs/phase4/stress",
    reuse_illumination_csv: str = "outputs/phase4/illumination/illumination_experiments.csv",
    reuse_geometry_csv: str = "outputs/phase4/geometry_v2/geometry_experiments.csv",
) -> List[Dict[str, Any]]:
    """Evaluate Scale and Combined pairs and write a unified Step 6 report."""
    output_root = Path(output_dir)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing stress output: {output_root}")
    output_root.mkdir(parents=True)
    rows = []
    for category in INPUT_CATEGORIES:
        pair_root = Path(data_dir) / "stress_tests" / category
        for pair_dir in sorted(pair_root.iterdir()):
            if pair_dir.is_dir():
                rows.append(_run_pair(category, pair_dir.name, data_dir, output_root))

    reused: Dict[str, List[Dict[str, Any]]] = {"Illumination": [], "Viewpoint": []}
    illumination_path = Path(reuse_illumination_csv)
    geometry_path = Path(reuse_geometry_csv)
    if illumination_path.exists():
        with illumination_path.open(newline="", encoding="utf-8") as csv_file:
            reused["Illumination"] = list(csv.DictReader(csv_file))
    if geometry_path.exists():
        with geometry_path.open(newline="", encoding="utf-8") as csv_file:
            reused["Viewpoint"] = [row for row in csv.DictReader(csv_file) if row.get("model") == "Homography"]

    with (output_root / "stress_experiments.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_report(output_root / "stress_report.md", rows, reused)
    return rows
