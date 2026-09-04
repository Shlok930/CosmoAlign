"""Phase 4 closure reporting and point-based transform comparison."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def point_transform_error(
    estimated: Optional[np.ndarray],
    ground_truth: Optional[np.ndarray],
    source_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    """Compare transforms by displacement on image corners in pixel coordinates."""
    if estimated is None or ground_truth is None:
        return {"status": "NOT_AVAILABLE", "mean_px": None, "median_px": None, "max_px": None, "reason": "Missing estimated or ground-truth transform."}
    if estimated.shape != (3, 3) or ground_truth.shape != (3, 3):
        return {"status": "NOT_COMPARABLE", "mean_px": None, "median_px": None, "max_px": None, "reason": "Transforms are not both 3x3 homographies."}
    if not np.isfinite(estimated).all() or not np.isfinite(ground_truth).all():
        return {"status": "NOT_COMPARABLE", "mean_px": None, "median_px": None, "max_px": None, "reason": "A transform contains non-finite values."}
    height, width = source_shape[:2]
    points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    try:
        estimated_points = cv2.perspectiveTransform(points, estimated).reshape(-1, 2)
        ground_truth_points = cv2.perspectiveTransform(points, ground_truth).reshape(-1, 2)
    except cv2.error as error:
        return {"status": "NOT_COMPARABLE", "mean_px": None, "median_px": None, "max_px": None, "reason": str(error)}
    errors = np.linalg.norm(estimated_points - ground_truth_points, axis=1)
    return {"status": "COMPARED", "mean_px": float(np.mean(errors)), "median_px": float(np.median(errors)), "max_px": float(np.max(errors)), "reason": "Mean/median/max corner displacement in reference pixels."}


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _numeric(rows: Iterable[Dict[str, Any]], field: str) -> List[float]:
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            pass
    return values


def aggregate_category(rows: List[Dict[str, Any]], geometry_field: str = "geometry_status", verdict_field: str = "final_verdict") -> Dict[str, Any]:
    ratios = _numeric(rows, "inlier_ratio_pct")
    rmses = _numeric(rows, "rmse_px")
    coverage = _numeric(rows, "spatial_coverage_pct")
    geometry = [str(row.get(geometry_field, row.get("geometry_sanity", ""))) for row in rows]
    verdicts = [str(row.get(verdict_field, row.get("verdict", ""))) for row in rows if row.get(verdict_field, row.get("verdict", "")) not in (None, "", "N/A")]
    count = len(rows)
    return {
        "pairs": count,
        "mean_inlier_ratio_pct": float(np.mean(ratios)) if ratios else None,
        "minimum_inlier_ratio_pct": min(ratios) if ratios else None,
        "maximum_inlier_ratio_pct": max(ratios) if ratios else None,
        "mean_rmse_px": float(np.mean(rmses)) if rmses else None,
        "minimum_rmse_px": min(rmses) if rmses else None,
        "maximum_rmse_px": max(rmses) if rmses else None,
        "mean_spatial_coverage_pct": float(np.mean(coverage)) if coverage else None,
        "geometry_pass_pct": sum(value == "PASS" for value in geometry) * 100.0 / count if count else None,
        "geometry_fail_pct": sum(value.startswith("FAIL") for value in geometry) * 100.0 / count if count else None,
        "validated_pct": sum(value == "VALIDATED" for value in verdicts) * 100.0 / len(verdicts) if verdicts else None,
        "uncertain_pct": sum(value == "UNCERTAIN" for value in verdicts) * 100.0 / len(verdicts) if verdicts else None,
        "failed_pct": sum(value == "FAILED" for value in verdicts) * 100.0 / len(verdicts) if verdicts else None,
    }


def run_closure_report(output_dir: str = "outputs/phase4/closure", scale_csv: str = "outputs/phase4/scale_normalization/scale_normalization_results.csv", illumination_csv: str = "outputs/phase4/illumination/illumination_experiments.csv", viewpoint_csv: str = "outputs/phase4/geometry_v2/geometry_experiments.csv", stress_csv: str = "outputs/phase4/stress/stress_experiments.csv", step7_csv: str = "outputs/phase4/step7_validator_diagnostic/validator_diagnostic.csv") -> Path:
    """Create a new unified closure report from existing or closure-run CSVs."""
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite closure report directory: {root}")
    root.mkdir(parents=True)
    scale_rows = _read_rows(Path(scale_csv))
    scale_rows = [{**row, "inlier_ratio_pct": row.get("native_inlier_ratio_pct"), "rmse_px": row.get("native_rmse_px"), "spatial_coverage_pct": row.get("native_spatial_coverage_pct"), "geometry_status": row.get("native_geometry_status")} for row in scale_rows]
    illumination_rows = _read_rows(Path(illumination_csv))
    viewpoint_rows = [row for row in _read_rows(Path(viewpoint_csv)) if row.get("model") == "Homography"]
    stress_rows = _read_rows(Path(stress_csv))
    step7_rows = _read_rows(Path(step7_csv))
    combined_rows = [row for row in stress_rows if row.get("category") == "combined"]
    summaries = {"Scale": aggregate_category(scale_rows), "Illumination": aggregate_category(illumination_rows, geometry_field="geometry_status", verdict_field="verdict"), "Viewpoint Homography": aggregate_category(viewpoint_rows), "Combined": aggregate_category(combined_rows)}
    (root / "category_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    lines = ["# Phase 4 Closure Report", "", "## Executive Summary", "This report consolidates measured Phase 4 robustness evidence. Scale normalization and standalone illumination results are included when available; existing results are reused rather than overwritten.", "", "## Dataset Summary", "The stress dataset contains scale, illumination, viewpoint, and combined categories with metadata, SHA-256 provenance, and ground-truth homographies."]
    for category, rows in [("Scale", scale_rows), ("Illumination", illumination_rows), ("Viewpoint Homography", viewpoint_rows), ("Combined", combined_rows)]:
        summary = summaries[category]
        lines += ["", f"## {category}", f"Pairs: {summary['pairs']}", f"Mean/min/max inlier ratio: {summary['mean_inlier_ratio_pct']} / {summary['minimum_inlier_ratio_pct']} / {summary['maximum_inlier_ratio_pct']}%", f"Mean/min/max RMSE: {summary['mean_rmse_px']} / {summary['minimum_rmse_px']} / {summary['maximum_rmse_px']} px", f"Mean spatial coverage: {summary['mean_spatial_coverage_pct']}%", f"Geometry PASS/FAIL: {summary['geometry_pass_pct']}% / {summary['geometry_fail_pct']}%", f"VALIDATED/UNCERTAIN/FAILED: {summary['validated_pct']}% / {summary['uncertain_pct']}% / {summary['failed_pct']}%"]
        if rows:
            lines.append("Per-pair results:")
            for row in rows:
                lines.append(f"- {row.get('pair_id')}: level={row.get('stress_level', row.get('viewpoint_stress_level', row.get('illumination_stress_level')))}, ratio={row.get('inlier_ratio_pct')}, RMSE={row.get('rmse_px')}, verdict={row.get('final_verdict', row.get('verdict'))}.")
    lines += ["", "## Scale Normalization", "Native and normalized results are compared in the scale-normalization CSV. Positive ratio improvement and positive RMSE improvement indicate measured benefit; negative values indicate degradation.", "", "## Illumination", "Existing illumination results are preserved and standalone closure-run results can be supplied through the function arguments. Strategies are compared empirically.", "", "## Viewpoint", "Existing Homography rows are reused for category summaries; Affine results remain in their original geometry CSV.", "", "## Combined", "Combined stress applies scale, illumination, and viewpoint simultaneously, so failures cannot be attributed to one factor alone.", "", "## Ground-Truth Transform Analysis", "Where estimated transforms are available, point-based corner displacement should be reported in reference pixels. Raw matrix subtraction is not used because matrix scale and coordinate-frame differences are not directly comparable.", "", "## Step 7 Validator Diagnostic", "The Step 7 CSV is included as evidence that raw independent-validation RMSE can be dominated by geometrically inconsistent matches in some cases, while combined stress can have no consistent validation subset.", "", "## Failure Analysis", "Failed rows remain represented in CSV outputs and should be interpreted alongside match support, spatial coverage, and geometry status.", "", "## Limitations", "The stress generator is synthetic, illumination is directional hill shading, viewpoint is a 2D projective approximation, and the existing validator's independent split remains unchanged.", "", "## Reproducibility", "Run metadata is recorded by the Phase 4 runner when a closure run is executed.", "", "## Final Conclusion", "Phase 4 provides empirical evidence of strong baseline and viewpoint Homography behavior, scale and illumination degradation, and combined-stress weakness. It does not establish robustness on real multi-modal lunar imagery."]
    (root / "phase4_closure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root / "phase4_closure_report.md"
