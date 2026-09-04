# Phase 4 Step 7: Independent-Validation Methodology Diagnostic

## 1. Objective
Determine whether the existing independent-validation failures reflect genuine registration failure or a limitation of its validation methodology.

## 2. Existing Methodology
For each pair, the existing approach uses a deterministic 50/50 split of Lowe-filtered good matches (seed 42), fits a RANSAC Homography on the training half at 5.0 px, and computes validation RMSE over all unseen validation matches. The training RANSAC inlier mask is not used to filter validation matches.

## 3. Diagnostic Methodology
This diagnostic reproduces the raw validation RMSE, then classifies validation correspondences as geometrically consistent when their H_train reprojection error is at most the existing 5.0 px RANSAC criterion. It reports RMSE, median, and P95 using only those diagnostic geometric inliers, alongside full-data RANSAC metrics. No existing validator or Phase 4 output is changed.

## 4. Results
| Pair | Category | Level | Good | Train | Val | Train inliers | Train ratio % | Current val RMSE | Val geometric inliers | Val geometric ratio % | Val-inlier RMSE | Val-inlier median | Val-inlier P95 | Full inliers | Full ratio % | Full RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stress_tests/scale/pair_s1 | scale | 1x | 21416 | 10708 | 10708 | 10708 | 100.0 | 0.0 | 10708 | 100.0 | 0.0 | 0.0 | 0.0 | 21416 | 100.0 | 0.0 |
| stress_tests/scale/pair_s2 | scale | 2x | 57 | 28 | 29 | 18 | 64.28571428571429 | 201.05284118652344 | 14 | 48.275862068965516 | 1.3259402513504028 | 0.9229110479354858 | 2.452890187501907 | 32 | 56.14035087719298 | 1.1372 |
| stress_tests/scale/pair_s4 | scale | 8x | 26 | 13 | 13 | 7 | 53.84615384615385 | 43.68666076660156 | 3 | 23.076923076923077 | 0.20155580341815948 | 0.04966023936867714 | 0.3130749404430389 | 9 | 34.61538461538461 | 11.3837 |
| stress_tests/combined/pair_c1 | combined | scale_2x_illumination_30deg_viewpoint_15deg | 20 | 10 | 10 | 5 | 50.0 | 247.2830047607422 | 0 | 0.0 | N/A | N/A | N/A | 7 | 35.0 | 1.9208 |
| stress_tests/illumination/pair_i1 | illumination | 10deg | 2579 | 1289 | 1290 | 1275 | 98.91388673390225 | 47.94349670410156 | 1278 | 99.06976744186046 | 0.4385019540786743 | 0.21789652109146118 | 0.7326262056827515 | 2553 | 98.99185730903451 | 0.4042 |

## 5. Evidence Assessment
### Question A
Per-case classification: stress_tests/scale/pair_s1=NO, stress_tests/scale/pair_s2=YES, stress_tests/scale/pair_s4=YES, stress_tests/combined/pair_c1=INSUFFICIENT EVIDENCE, stress_tests/illumination/pair_i1=YES. Overall answer: YES. Supporting pairs: stress_tests/scale/pair_s2, stress_tests/scale/pair_s4, stress_tests/illumination/pair_i1.
### Question B
Geometric filtering materially reduces RMSE for: stress_tests/scale/pair_s2 (201.05284118652344 to 1.3259402513504028 px), stress_tests/scale/pair_s4 (43.68666076660156 to 0.20155580341815948 px), stress_tests/illumination/pair_i1 (47.94349670410156 to 0.4385019540786743 px). Pairs without a finite filtered RMSE: stress_tests/combined/pair_c1.
### Question C
No selected case has both a finite raw RMSE and a finite filtered RMSE that are simultaneously high under the diagnostic data. This is insufficient to rule out genuine difficulty; the combined case with zero geometric validation inliers is the strongest genuine-difficulty signal.
### Question D
The evidence supports sensitivity to validation outliers, not a global validator failure. Combined zero-inlier cases: stress_tests/combined/pair_c1. The conclusion is mixed: methodological conservatism is evident in several pairs, while combined stress remains genuinely difficult.

## 6. Interpretation
The experiment preserves the current matching pipeline and does not reinterpret or alter the existing validator result. The 5.0 px geometric-inlier filter is used only to diagnose the current methodology. Results are empirical for the selected representative pairs.
