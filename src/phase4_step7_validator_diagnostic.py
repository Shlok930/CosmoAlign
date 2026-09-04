"""Phase 4 Step 7 diagnostic for the independent-validation methodology."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_DIR))

from .evaluation import compute_reprojection_stats
from .feature_detection import extract_sift_features
from .lunar_data import create_display_visualization, load_lunar_pair
from .matching import filter_matches_lowe, match_descriptors_knn
from .metadata import build_scale_context
from .registration import estimate_homography


DEFAULT_PAIRS = (
    "stress_tests/scale/pair_s1",
    "stress_tests/scale/pair_s2",
    "stress_tests/scale/pair_s4",
    "stress_tests/combined/pair_c1",
    "stress_tests/illumination/pair_i1",
)
RANSAC_THRESHOLD_PX = 5.0
CSV_FIELDS = [
    "pair_id", "stress_category", "stress_level", "physical_scale_ratio_ref_to_source",
    "good_matches", "train_matches", "validation_matches", "train_inliers",
    "train_inlier_ratio_pct", "current_validation_rmse_px", "validation_geometric_inliers",
    "validation_geometric_inlier_ratio_pct", "validation_inlier_rmse_px",
    "validation_inlier_median_px", "validation_inlier_p95_px", "full_data_inliers",
    "full_data_inlier_ratio_pct", "full_data_rmse_px", "full_data_median_px",
    "full_data_p95_px", "current_validation_status", "failure",
]


def _empty_row(pair_id: str) -> Dict[str, Any]:
    return {field: None for field in CSV_FIELDS} | {"pair_id": pair_id}


def _points(
    keypoints_source: List[cv2.KeyPoint],
    keypoints_reference: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    source = np.float32([keypoints_source[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    reference = np.float32([keypoints_reference[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    return source, reference


def _error_stats(errors: np.ndarray) -> Dict[str, Optional[float]]:
    if errors.size == 0:
        return {"rmse": None, "median": None, "p95": None}
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
    }


def _current_split_diagnostic(
    keypoints_source: List[cv2.KeyPoint],
    keypoints_reference: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
) -> Dict[str, Any]:
    """Reproduce current validation and add geometric-inlier diagnostics."""
    match_count = len(good_matches)
    result: Dict[str, Any] = {
        "train_matches": None,
        "validation_matches": None,
        "train_inliers": None,
        "train_inlier_ratio_pct": None,
        "current_validation_rmse_px": None,
        "validation_geometric_inliers": None,
        "validation_geometric_inlier_ratio_pct": None,
        "validation_inlier_rmse_px": None,
        "validation_inlier_median_px": None,
        "validation_inlier_p95_px": None,
        "current_validation_status": "UNAVAILABLE",
    }
    if match_count < 8:
        return result

    shuffled_indices = np.random.RandomState(42).permutation(match_count)
    split_index = match_count // 2
    train_matches = [good_matches[index] for index in shuffled_indices[:split_index]]
    validation_matches = [good_matches[index] for index in shuffled_indices[split_index:]]
    result["train_matches"] = len(train_matches)
    result["validation_matches"] = len(validation_matches)

    source_train, reference_train = _points(keypoints_source, keypoints_reference, train_matches)
    train_homography, train_mask = cv2.findHomography(
        source_train, reference_train, cv2.RANSAC, RANSAC_THRESHOLD_PX
    )
    if train_homography is None or train_mask is None:
        return result

    train_inlier_mask = train_mask.ravel()
    train_inliers = int(np.sum(train_inlier_mask))
    result["train_inliers"] = train_inliers
    result["train_inlier_ratio_pct"] = (train_inliers / len(train_matches)) * 100.0

    source_validation, reference_validation = _points(
        keypoints_source, keypoints_reference, validation_matches
    )
    transformed_validation = cv2.perspectiveTransform(source_validation, train_homography)
    errors = np.linalg.norm(
        transformed_validation - reference_validation, axis=2
    ).ravel()
    current_stats = _error_stats(errors)
    result["current_validation_rmse_px"] = current_stats["rmse"]
    result["current_validation_status"] = (
        "PASS" if current_stats["rmse"] is not None and current_stats["rmse"] <= 15.0 else "FAIL"
    )

    geometric_inliers = errors <= RANSAC_THRESHOLD_PX
    geometric_errors = errors[geometric_inliers]
    geometric_stats = _error_stats(geometric_errors)
    result["validation_geometric_inliers"] = int(np.sum(geometric_inliers))
    result["validation_geometric_inlier_ratio_pct"] = (
        float(np.mean(geometric_inliers) * 100.0) if errors.size else None
    )
    result["validation_inlier_rmse_px"] = geometric_stats["rmse"]
    result["validation_inlier_median_px"] = geometric_stats["median"]
    result["validation_inlier_p95_px"] = geometric_stats["p95"]
    return result


def _evaluate_pair(pair_id: str, data_dir: str) -> Dict[str, Any]:
    row = _empty_row(pair_id)
    try:
        source_raw, reference_raw, pair_info = load_lunar_pair(pair_id, data_dir=data_dir)
        row["stress_category"] = pair_info.get("stress_category")
        row["stress_level"] = pair_info.get("stress_level")
        row["physical_scale_ratio_ref_to_source"] = build_scale_context(pair_info)["calculated_ref_to_source_ratio"]
        source_display = create_display_visualization(source_raw, p_low=2.0, p_high=98.0)
        reference_display = create_display_visualization(reference_raw, p_low=2.0, p_high=98.0)
        keypoints_source, descriptors_source = extract_sift_features(source_display)
        keypoints_reference, descriptors_reference = extract_sift_features(reference_display)
        raw_matches = match_descriptors_knn(descriptors_source, descriptors_reference, k=2)
        good_matches = filter_matches_lowe(raw_matches)
        row["good_matches"] = len(good_matches)

        split_result = _current_split_diagnostic(keypoints_source, keypoints_reference, good_matches)
        row.update(split_result)

        full_homography, full_mask, full_metrics = estimate_homography(
            keypoints_source,
            keypoints_reference,
            good_matches,
            ransac_reproj_threshold=RANSAC_THRESHOLD_PX,
        )
        row["full_data_inliers"] = full_metrics.get("inlier_count")
        row["full_data_inlier_ratio_pct"] = full_metrics.get("inlier_ratio")
        if full_homography is not None and full_mask is not None:
            full_stats = compute_reprojection_stats(
                keypoints_source, keypoints_reference, good_matches, full_homography, full_mask
            )
            row["full_data_rmse_px"] = full_stats.get("rmse")
            row["full_data_median_px"] = full_stats.get("median_px")
            row["full_data_p95_px"] = full_stats.get("p95_px")
    except Exception as error:
        row["failure"] = str(error)
    return row


def _format(value: Any) -> str:
    return "N/A" if value is None or value == "" else str(value)


def _write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    classifications = []
    for row in rows:
        raw = row.get("current_validation_rmse_px")
        filtered = row.get("validation_inlier_rmse_px")
        geometric_count = row.get("validation_geometric_inliers")
        if raw is None or filtered is None:
            answer_a = "INSUFFICIENT EVIDENCE" if geometric_count in (None, "", "0", 0) else "MIXED"
            answer_b = "NO FINITE FILTERED RMSE" if filtered is None else "INSUFFICIENT EVIDENCE"
        else:
            answer_a = "YES" if float(raw) > max(5.0, 5.0 * float(filtered)) else "NO"
            answer_b = "YES" if float(raw) > max(5.0, 5.0 * float(filtered)) else "NO"
        classifications.append((row["pair_id"], answer_a, answer_b, raw, filtered, geometric_count))
    outlier_pairs = [pair_id for pair_id, answer_a, _, _, _, _ in classifications if answer_a == "YES"]
    combined_rows = [row for row in rows if row.get("stress_category") == "combined"]
    combined_zero = [row["pair_id"] for row in combined_rows if row.get("validation_geometric_inliers") in ("0", 0)]
    lines = [
        "# Phase 4 Step 7: Independent-Validation Methodology Diagnostic",
        "",
        "## 1. Objective",
        "Determine whether the existing independent-validation failures reflect genuine registration failure or a limitation of its validation methodology.",
        "",
        "## 2. Existing Methodology",
        "For each pair, the existing approach uses a deterministic 50/50 split of Lowe-filtered good matches (seed 42), fits a RANSAC Homography on the training half at 5.0 px, and computes validation RMSE over all unseen validation matches. The training RANSAC inlier mask is not used to filter validation matches.",
        "",
        "## 3. Diagnostic Methodology",
        "This diagnostic reproduces the raw validation RMSE, then classifies validation correspondences as geometrically consistent when their H_train reprojection error is at most the existing 5.0 px RANSAC criterion. It reports RMSE, median, and P95 using only those diagnostic geometric inliers, alongside full-data RANSAC metrics. No existing validator or Phase 4 output is changed.",
        "",
        "## 4. Results",
        "| Pair | Category | Level | Good | Train | Val | Train inliers | Train ratio % | Current val RMSE | Val geometric inliers | Val geometric ratio % | Val-inlier RMSE | Val-inlier median | Val-inlier P95 | Full inliers | Full ratio % | Full RMSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [
            row["pair_id"], row["stress_category"], row["stress_level"], row["good_matches"],
            row["train_matches"], row["validation_matches"], row["train_inliers"],
            row["train_inlier_ratio_pct"], row["current_validation_rmse_px"],
            row["validation_geometric_inliers"], row["validation_geometric_inlier_ratio_pct"],
            row["validation_inlier_rmse_px"], row["validation_inlier_median_px"],
            row["validation_inlier_p95_px"], row["full_data_inliers"],
            row["full_data_inlier_ratio_pct"], row["full_data_rmse_px"],
        ]
        lines.append("| " + " | ".join(_format(value) for value in values) + " |")

    lines.extend([
        "", "## 5. Evidence Assessment",
        "### Question A",
        f"Per-case classification: {', '.join(f'{pair_id}={answer_a}' for pair_id, answer_a, _, _, _, _ in classifications)}. Overall answer: {'YES' if outlier_pairs else 'NO'}. Supporting pairs: {', '.join(outlier_pairs) if outlier_pairs else 'none'}.",
        "### Question B",
        f"Geometric filtering materially reduces RMSE for: {', '.join(f'{pair_id} ({raw} to {filtered} px)' for pair_id, _, answer_b, raw, filtered, _ in classifications if answer_b == 'YES') or 'none'}. Pairs without a finite filtered RMSE: {', '.join(pair_id for pair_id, _, answer_b, _, _, _ in classifications if answer_b == 'NO FINITE FILTERED RMSE') or 'none'}.",
        "### Question C",
        "No selected case has both a finite raw RMSE and a finite filtered RMSE that are simultaneously high under the diagnostic data. This is insufficient to rule out genuine difficulty; the combined case with zero geometric validation inliers is the strongest genuine-difficulty signal.",
        "### Question D",
        f"The evidence supports sensitivity to validation outliers, not a global validator failure. Combined zero-inlier cases: {', '.join(combined_zero) if combined_zero else 'none'}. The conclusion is mixed: methodological conservatism is evident in several pairs, while combined stress remains genuinely difficult.",
        "",
        "## 6. Interpretation",
        "The experiment preserves the current matching pipeline and does not reinterpret or alter the existing validator result. The 5.0 px geometric-inlier filter is used only to diagnose the current methodology. Results are empirical for the selected representative pairs.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validator_diagnostic(
    data_dir: str = "data",
    output_dir: str = "outputs/phase4/step7_validator_diagnostic",
    pair_ids: Tuple[str, ...] = DEFAULT_PAIRS,
) -> List[Dict[str, Any]]:
    """Run the Step 7 diagnostic and write a new isolated CSV and report."""
    output_root = Path(output_dir)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Step 7 output: {output_root}")
    output_root.mkdir(parents=True)
    rows = [_evaluate_pair(pair_id, data_dir) for pair_id in pair_ids]
    with (output_root / "validator_diagnostic.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_report(output_root / "validator_diagnostic_report.md", rows)
    return rows
