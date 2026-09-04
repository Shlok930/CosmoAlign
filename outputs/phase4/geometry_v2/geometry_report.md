# Phase 4 Step 5.2: Homography vs Affine Viewpoint Experiment

## 1. Objective
Compare the existing Homography model with the experimental Affine model under controlled viewpoint stress.

## 2. Experimental Controls
The same source/reference image, SIFT configuration, extracted keypoints, and Lowe-filtered good matches were used for both models. Both estimators used the same 5.0-pixel RANSAC threshold; only the geometric model changed.

## 3. Results
| Pair | Level | Model | Good matches | Inliers | Ratio % | Coverage % | RMSE px | Median px | P95 px | Geometry | Independent | Verdict | Failure |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| pair_v1 | 5deg | Homography | 3762 | 3746 | 99.57469431153642 | 100.0 | 0.3924 | 0.1984 | 0.6756 | PASS | FAIL | UNCERTAIN | N/A |
| pair_v1 | 5deg | Affine | 3762 | 3745 | 99.54811270600744 | 100.0 | 1.2921476364135742 | 0.9711976647377014 | 2.385835409164428 | PASS | NOT_APPLICABLE | NOT_APPLICABLE | N/A |
| pair_v2 | 15deg | Homography | 2911 | 2890 | 99.27859841978702 | 100.0 | 0.4331 | 0.214 | 0.6906 | PASS | FAIL | UNCERTAIN | N/A |
| pair_v2 | 15deg | Affine | 2911 | 2055 | 70.5942974922707 | 100.0 | 2.459411144256592 | 2.0261600017547607 | 4.388223886489866 | PASS | NOT_APPLICABLE | NOT_APPLICABLE | N/A |
| pair_v3 | 30deg | Homography | 1341 | 1326 | 98.88143176733782 | 100.0 | 0.5377 | 0.2817 | 0.9184 | PASS | FAIL | UNCERTAIN | N/A |
| pair_v3 | 30deg | Affine | 1341 | 622 | 46.38329604772558 | 55.6 | 2.8346030712127686 | 2.3177876472473145 | 4.987007260322569 | PASS | NOT_APPLICABLE | NOT_APPLICABLE | N/A |
| pair_v4 | 45deg | Homography | 335 | 308 | 91.94029850746269 | 100.0 | 0.8912 | 0.3776 | 1.8794 | PASS | FAIL | UNCERTAIN | N/A |
| pair_v4 | 45deg | Affine | 335 | 158 | 47.16417910447761 | 44.4 | 2.3632235527038574 | 1.7419607639312744 | 4.293482136726381 | PASS | NOT_APPLICABLE | NOT_APPLICABLE | N/A |

## 4. Per-Level Analysis
### 5deg (pair_v1)
- Homography: inliers=3746, inlier ratio=99.57469431153642%, coverage=100.0%, RMSE=0.3924 px, independent validation=FAIL, verdict=UNCERTAIN.
- Affine: inliers=3745, inlier ratio=99.54811270600744%, coverage=100.0%, RMSE=1.2921476364135742 px, independent validation=NOT_APPLICABLE, verdict=NOT_APPLICABLE.
### 15deg (pair_v2)
- Homography: inliers=2890, inlier ratio=99.27859841978702%, coverage=100.0%, RMSE=0.4331 px, independent validation=FAIL, verdict=UNCERTAIN.
- Affine: inliers=2055, inlier ratio=70.5942974922707%, coverage=100.0%, RMSE=2.459411144256592 px, independent validation=NOT_APPLICABLE, verdict=NOT_APPLICABLE.
### 30deg (pair_v3)
- Homography: inliers=1326, inlier ratio=98.88143176733782%, coverage=100.0%, RMSE=0.5377 px, independent validation=FAIL, verdict=UNCERTAIN.
- Affine: inliers=622, inlier ratio=46.38329604772558%, coverage=55.6%, RMSE=2.8346030712127686 px, independent validation=NOT_APPLICABLE, verdict=NOT_APPLICABLE.
### 45deg (pair_v4)
- Homography: inliers=308, inlier ratio=91.94029850746269%, coverage=100.0%, RMSE=0.8912 px, independent validation=FAIL, verdict=UNCERTAIN.
- Affine: inliers=158, inlier ratio=47.16417910447761%, coverage=44.4%, RMSE=2.3632235527038574 px, independent validation=NOT_APPLICABLE, verdict=NOT_APPLICABLE.

## 5. Model Comparison
The measurements above are the basis for comparing inlier reliability, inlier ratio, reprojection errors, spatial coverage, and high-stress failures. Physical scale metadata is reported separately from viewpoint stress.

## 6. Conclusion
No model is declared superior a priori. The conclusion is empirical and must be drawn from the measured inlier ratio, spatial coverage, RMSE, independent validation, and final verdict in this report.

Affine independent validation is marked NOT_APPLICABLE because the existing ScientificValidator performs Homography-specific independent validation and was not modified for this experiment.
