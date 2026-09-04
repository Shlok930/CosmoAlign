"""Standalone Phase 4 illumination preprocessing experiment."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2

from .feature_detection import extract_sift_features
from .lunar_data import create_display_visualization, load_lunar_pair
from .matching import filter_matches_lowe, match_descriptors_knn
from .metadata import build_scale_context
from .registration import estimate_homography
from .evaluation import analyze_inlier_spatial_distribution, check_homography_sanity, compute_reprojection_stats
from .preprocessing import apply_clahe


PAIR_IDS = ("pair_i1", "pair_i2", "pair_i3", "pair_i4")
VARIANTS = (
    ("illum_001_2_98", "2%-98% Percentile Stretch", 2.0, 98.0, False),
    ("illum_002_1_99", "1%-99% Percentile Stretch", 1.0, 99.0, False),
    ("illum_003_2_98_clahe", "2%-98% Stretch + CLAHE", 2.0, 98.0, True),
    ("illum_004_1_99_clahe", "1%-99% Stretch + CLAHE", 1.0, 99.0, True),
)
CSV_FIELDS = ["pair_id", "stress_level", "preprocessing", "good_matches", "inliers", "inlier_ratio_pct", "spatial_coverage_pct", "rmse_px", "geometry_status", "failure"]


def _evaluate(source, reference) -> Dict[str, Any]:
    source_keypoints, source_descriptors = extract_sift_features(source)
    reference_keypoints, reference_descriptors = extract_sift_features(reference)
    raw_matches = match_descriptors_knn(source_descriptors, reference_descriptors, k=2)
    good_matches = filter_matches_lowe(raw_matches)
    homography, mask, metrics = estimate_homography(source_keypoints, reference_keypoints, good_matches, ransac_reproj_threshold=5.0)
    result = {"good_matches": len(good_matches), "inliers": metrics.get("inlier_count"), "inlier_ratio_pct": metrics.get("inlier_ratio"), "spatial_coverage_pct": None, "rmse_px": None, "geometry_status": "FAIL"}
    if homography is not None and mask is not None:
        spatial = analyze_inlier_spatial_distribution(source_keypoints, good_matches, mask, source.shape)
        errors = compute_reprojection_stats(source_keypoints, reference_keypoints, good_matches, homography, mask)
        sane, _ = check_homography_sanity(homography, source.shape, reference.shape)
        result.update({"spatial_coverage_pct": round(spatial["coverage_ratio"] * 100.0, 1), "rmse_px": errors.get("rmse"), "geometry_status": "PASS" if sane else "FAIL"})
    return result


def run_illumination_experiment(data_dir: str = "data", output_dir: str = "outputs/phase4/illumination_v2") -> List[Dict[str, Any]]:
    """Run the four existing preprocessing strategies without invoking Phase 3."""
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing illumination output: {root}")
    root.mkdir(parents=True)
    rows = []
    for pair_id in PAIR_IDS:
        loader_id = f"stress_tests/illumination/{pair_id}"
        pair_info = None
        for variant_id, description, low, high, clahe in VARIANTS:
            row = {field: None for field in CSV_FIELDS} | {"pair_id": pair_id, "preprocessing": description}
            try:
                source_raw, reference_raw, pair_info = load_lunar_pair(loader_id, data_dir=data_dir)
                row["stress_level"] = pair_info.get("stress_level")
                source = create_display_visualization(source_raw, p_low=low, p_high=high)
                reference = create_display_visualization(reference_raw, p_low=low, p_high=high)
                if clahe:
                    source, reference = apply_clahe(source), apply_clahe(reference)
                result = _evaluate(source, reference)
                row.update(result)
                if result["geometry_status"] == "FAIL":
                    (root / pair_id / variant_id).mkdir(parents=True, exist_ok=True)
                    (root / pair_id / variant_id / "failure.json").write_text(json.dumps({"failure_stage": "geometry", **row}, indent=2) + "\n", encoding="utf-8")
            except Exception as error:
                row["failure"] = str(error)
                (root / pair_id / variant_id).mkdir(parents=True, exist_ok=True)
                (root / pair_id / variant_id / "failure.json").write_text(json.dumps({"failure_stage": "evaluation", "failure_reason": str(error), **row}, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
    with (root / "illumination_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_report(root / "illumination_report.md", rows)
    return rows


def _write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = ["# Standalone Phase 4 Illumination Experiment", "", "This run is independent of run_phase3_lunar_pipeline(). Each preprocessing strategy is applied independently to source and reference display images.", "", "| Pair | Level | Preprocessing | Good | Inliers | Ratio | Coverage | RMSE | Geometry | Failure |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(*[row[field] if row[field] is not None else "N/A" for field in ["pair_id", "stress_level", "preprocessing", "good_matches", "inliers", "inlier_ratio_pct", "spatial_coverage_pct", "rmse_px", "geometry_status", "failure"]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
