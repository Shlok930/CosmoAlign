"""Phase 4 scale stress versus in-memory scale-normalized evaluation."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .evaluation import analyze_inlier_spatial_distribution, check_homography_sanity, compute_reprojection_stats
from .feature_detection import extract_sift_features
from .lunar_data import create_display_visualization, load_lunar_pair
from .matching import filter_matches_lowe, match_descriptors_knn
from .metadata import build_scale_context
from .preprocessing import apply_scaling
from .registration import estimate_homography
from .phase4_reporting import point_transform_error


PAIR_ROOT = "stress_tests/scale"
PAIR_IDS = ("pair_s1", "pair_s2", "pair_s3", "pair_s4")
RANSAC_THRESHOLD = 5.0
CSV_FIELDS = [
    "pair_id", "stress_level", "scale_ratio", "native_good_matches", "native_inliers",
    "native_inlier_ratio_pct", "native_rmse_px", "native_spatial_coverage_pct",
    "native_geometry_status", "normalized_good_matches", "normalized_inliers",
    "normalized_inlier_ratio_pct", "normalized_rmse_px", "normalized_spatial_coverage_pct",
    "normalized_geometry_status", "improvement_inlier_ratio_pct", "improvement_rmse_px",
    "native_ground_truth_error_px", "normalized_ground_truth_error_px", "failure_reason",
]


def _metrics(source: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    keypoints_source, descriptors_source = extract_sift_features(source)
    keypoints_reference, descriptors_reference = extract_sift_features(reference)
    raw_matches = match_descriptors_knn(descriptors_source, descriptors_reference, k=2)
    good_matches = filter_matches_lowe(raw_matches)
    homography, mask, ransac = estimate_homography(
        keypoints_source, keypoints_reference, good_matches,
        ransac_reproj_threshold=RANSAC_THRESHOLD,
    )
    result = {
        "good_matches": len(good_matches),
        "inliers": ransac.get("inlier_count"),
        "inlier_ratio_pct": ransac.get("inlier_ratio"),
        "rmse_px": None,
        "spatial_coverage_pct": None,
        "geometry_status": "FAIL",
        "homography": None,
    }
    if homography is not None and mask is not None:
        reprojection = compute_reprojection_stats(keypoints_source, keypoints_reference, good_matches, homography, mask)
        spatial = analyze_inlier_spatial_distribution(keypoints_source, good_matches, mask, source.shape)
        sane, _ = check_homography_sanity(homography, source.shape, reference.shape)
        result.update({
            "rmse_px": reprojection.get("rmse"),
            "spatial_coverage_pct": round(spatial["coverage_ratio"] * 100.0, 1),
            "geometry_status": "PASS" if sane else "FAIL",
            "homography": homography,
        })
    return result


def _failure_artifact(path: Path, row: Dict[str, Any], reason: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    (path / "failure.json").write_text(json.dumps({"failure_stage": "evaluation", "failure_reason": reason, **row}, indent=2) + "\n", encoding="utf-8")


def run_scale_normalization_experiment(
    data_dir: str = "data",
    output_dir: str = "outputs/phase4/scale_normalization",
) -> List[Dict[str, Any]]:
    """Compare native and metadata-driven in-memory scale normalization."""
    output_root = Path(output_dir)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing scale-normalization output: {output_root}")
    output_root.mkdir(parents=True)
    rows: List[Dict[str, Any]] = []
    for pair_id in PAIR_IDS:
        row = {field: None for field in CSV_FIELDS}
        row["pair_id"] = pair_id
        try:
            loader_id = f"{PAIR_ROOT}/{pair_id}"
            source_raw, reference_raw, pair_info = load_lunar_pair(loader_id, data_dir=data_dir)
            ratio = float(pair_info["nominal_scale_ratio_ref_to_source"])
            row["stress_level"] = pair_info.get("stress_level")
            row["scale_ratio"] = ratio
            source_display = create_display_visualization(source_raw, p_low=2.0, p_high=98.0)
            reference_display = create_display_visualization(reference_raw, p_low=2.0, p_high=98.0)
            native = _metrics(source_display, reference_display)
            normalized_source, applied_factor = apply_scaling(source_display, 1.0 / ratio)
            normalized = _metrics(normalized_source, reference_display)
            ground_truth = np.asarray(pair_info.get("ground_truth_transform", {}).get("source_to_reference_homography"), dtype=float)
            native_comparison = point_transform_error(native["homography"], ground_truth, source_raw.shape)
            normalized_homography = normalized["homography"]
            normalized_common = normalized_homography @ np.array([[1.0 / ratio, 0.0, 0.0], [0.0, 1.0 / ratio, 0.0], [0.0, 0.0, 1.0]]) if normalized_homography is not None else None
            normalized_comparison = point_transform_error(normalized_common, ground_truth, source_raw.shape)
            row.update({"native_good_matches": native["good_matches"], "native_inliers": native["inliers"], "native_inlier_ratio_pct": native["inlier_ratio_pct"], "native_rmse_px": native["rmse_px"], "native_spatial_coverage_pct": native["spatial_coverage_pct"], "native_geometry_status": native["geometry_status"], "normalized_good_matches": normalized["good_matches"], "normalized_inliers": normalized["inliers"], "normalized_inlier_ratio_pct": normalized["inlier_ratio_pct"], "normalized_rmse_px": normalized["rmse_px"], "normalized_spatial_coverage_pct": normalized["spatial_coverage_pct"], "normalized_geometry_status": normalized["geometry_status"], "improvement_inlier_ratio_pct": normalized["inlier_ratio_pct"] - native["inlier_ratio_pct"], "improvement_rmse_px": (native["rmse_px"] - normalized["rmse_px"]) if native["rmse_px"] is not None and normalized["rmse_px"] is not None else None})
            row["native_ground_truth_error_px"] = native_comparison["mean_px"]
            row["normalized_ground_truth_error_px"] = normalized_comparison["mean_px"]
            pair_output = output_root / pair_id
            (pair_output / "normalization.json").parent.mkdir(parents=True, exist_ok=True)
            (pair_output / "normalization.json").write_text(json.dumps({"scale_ratio": ratio, "applied_source_scale_factor": applied_factor}, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            row["failure_reason"] = str(error)
            _failure_artifact(output_root / pair_id, row, str(error))
        rows.append(row)
    with (output_root / "scale_normalization_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_report(output_root / "scale_normalization_report.md", rows)
    return rows


def _write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = ["# Phase 4 Scale Normalization Experiment", "", "Native resolution is compared with in-memory source resampling by the metadata-derived inverse reference/source GSD ratio. Raw TIFF files are never changed.", "", "| Pair | Level | Ratio | Native good/inliers/ratio/RMSE | Normalized good/inliers/ratio/RMSE | Ratio improvement | RMSE improvement | Geometry | Failure |", "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |"]
    for row in rows:
        lines.append("| {} | {} | {} | {}/{}/{}/{} | {}/{}/{}/{} | {} | {} | {}/{} | {} |".format(row["pair_id"], row["stress_level"], row["scale_ratio"], row["native_good_matches"], row["native_inliers"], row["native_inlier_ratio_pct"], row["native_rmse_px"], row["normalized_good_matches"], row["normalized_inliers"], row["normalized_inlier_ratio_pct"], row["normalized_rmse_px"], row["improvement_inlier_ratio_pct"], row["improvement_rmse_px"], row["native_geometry_status"], row["normalized_geometry_status"], row["failure_reason"] or "N/A"))
    lines += ["", "Positive improvement values indicate higher normalized inlier ratio or lower normalized RMSE. No improvement is assumed; values are measured."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
