# Phase 4 Step 6: Categorized Stress Evaluation

## 1. Objective
Evaluate Homography registration robustness across controlled scale and combined stress categories.

## 2. Experimental Scope
Scale and Combined were newly evaluated by this script. Illumination and Viewpoint were reused from their completed CSV outputs; neither experiment was rerun.

## 3. Scale Results

| pair_id | stress_level | nominal_scale_ratio_ref_to_source | good_matches | inliers | inlier_ratio_pct | spatial_coverage_pct | rmse_px | median_px | p95_px | geometry_sanity | independent_validation | final_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pair_s1 | 1x | 1.0 | 21416 | 21416 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | PASS | PASS | VALIDATED |
| pair_s2 | 2x | 2.0 | 57 | 32 | 56.14035087719298 | 100.0 | 1.1372 | 0.829 | 2.1617 | PASS | FAIL | UNCERTAIN |
| pair_s3 | 4x | 4.0 | 42 | 20 | 47.61904761904761 | 66.7 | 1.5722 | 0.8268 | 2.0206 | PASS | FAIL | UNCERTAIN |
| pair_s4 | 8x | 8.0 | 26 | 9 | 34.61538461538461 | 55.6 | 11.3837 | 3.7766 | 21.5321 | FAIL: Transformed area collapsed to 0.0% of reference frame (below 5% threshold). | FAIL | FAILED |

## 4. Combined Results

| pair_id | stress_level | nominal_scale_ratio_ref_to_source | illumination_difference_deg | viewpoint_rotation_deg | good_matches | inliers | inlier_ratio_pct | spatial_coverage_pct | rmse_px | median_px | p95_px | geometry_sanity | independent_validation | final_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pair_c1 | scale_2x_illumination_30deg_viewpoint_15deg | 2.0 | 30.0 | 15.0 | 20 | 7 | 35.0 | 66.7 | 1.9208 | 1.8091 | 3.034 | FAIL: Degenerate Homography: Determinant is non-positive (-0.003748), indicating image reflection/flip. | FAIL | FAILED |
| pair_c2 | scale_4x_illumination_50deg_viewpoint_30deg | 4.0 | 50.0 | 30.0 | 20 | 7 | 35.0 | 55.6 | 21.406 | 0.7652 | 41.4616 | FAIL: Transformed area collapsed to 0.0% of reference frame (below 5% threshold). | FAIL | FAILED |
| pair_c3 | scale_8x_illumination_30deg_viewpoint_45deg | 8.0 | 30.0 | 45.0 | 20 | 7 | 35.0 | 55.6 | 12.5893 | 3.934 | 23.6477 | FAIL: Transformed area collapsed to 0.8% of reference frame (below 5% threshold). | FAIL | FAILED |
| pair_c4 | scale_2x_illumination_70deg_viewpoint_30deg | 2.0 | 70.0 | 30.0 | 25 | 6 | 24.0 | 44.4 | 3.0582 | 2.3206 | 5.1441 | FAIL: Degenerate Homography: Determinant is non-positive (-0.000003), indicating image reflection/flip. | FAIL | FAILED |

## 5. Cross-Category Summary

### Scale
- Rows: 4; mean/min inlier ratio: 59.593695777906305 / 34.61538461538461%; mean/max RMSE: 3.523275 / 11.3837 px; mean coverage: 80.575%.
### Combined
- Rows: 4; mean/min inlier ratio: 32.25 / 24.0%; mean/max RMSE: 9.743575 / 21.406 px; mean coverage: 55.575%.
### Illumination
- Rows: 16; mean/min inlier ratio: 39.53592821208562 / 13.513513513513514%; mean/max RMSE: 10.197375000000001 / 69.8257 px; mean coverage: 53.443749999999994%.
### Viewpoint
- Rows: 4; mean/min inlier ratio: 97.41875575153098 / 91.94029850746269%; mean/max RMSE: 0.5636 / 0.8912 px; mean coverage: 100.0%.

## 6. Stress Degradation
Measured Scale and Combined rows should be compared by their recorded stress parameters; no arbitrary robustness boundary is inferred.

## 7. Validator Caveat
The existing independent-validation gate can report failures when other registration metrics are strong. This is preserved exactly and reserved for Phase 4 Step 7.

## 8. Conclusion
This report is an empirical summary of measured registration metrics. It does not declare the system robust merely because evaluation completed.
