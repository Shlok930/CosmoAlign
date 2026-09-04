# Phase 4 Step 4 Illumination Robustness Experiments

## 1. Objective
Evaluate whether controlled radiometric preprocessing improves registration under synthetic illumination stress.

## 2. Dataset Description
Four generated illumination pairs are evaluated: pair_i1 (10deg), pair_i2 (30deg), pair_i3 (50deg), and pair_i4 (70deg).
Raw scientific arrays remain unchanged; preprocessing is applied only to uint8 feature-registration representations.

## 3. Experimental Methodology
Each pair uses identical source/reference preprocessing strategy, existing SIFT, matching, Lowe filtering, RANSAC, and the eight-gate ScientificValidator.
No illumination pass/fail gate or automatic resizing is introduced.

## 4. All Experiments
| Pair | Stress | Preprocessing | KP source | KP ref | Raw | Good | Inliers | Ratio | Coverage | RMSE | Median | P95 | Independent | Scale | Consistent | Verdict |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :---: | :--- |
| pair_i1 | 10deg | 2%-98% Percentile Stretch | 21416 | 21487 | 21416 | 2579 | 2553 | 98.99185730903451 | 100.0 | 0.4042 | 0.2181 | 0.6916 | FAIL | 1.0 | True | UNCERTAIN |
| pair_i1 | 10deg | 1%-99% Percentile Stretch | 21332 | 21382 | 21332 | 2617 | 2591 | 99.00649598777225 | 100.0 | 0.4043 | 0.2195 | 0.6937 | FAIL | 1.0 | True | UNCERTAIN |
| pair_i1 | 10deg | 2%-98% Stretch + CLAHE | 20740 | 20844 | 20740 | 2559 | 2530 | 98.86674482219617 | 100.0 | 0.4187 | 0.2131 | 0.7218 | FAIL | 1.0 | True | UNCERTAIN |
| pair_i1 | 10deg | 1%-99% Stretch + CLAHE | 20727 | 20786 | 20727 | 2577 | 2549 | 98.91346526969345 | 100.0 | 0.4101 | 0.2145 | 0.7033 | FAIL | 1.0 | True | UNCERTAIN |
| pair_i2 | 30deg | 2%-98% Percentile Stretch | 21416 | 21808 | 21416 | 24 | 5 | 20.833333333333336 | 33.3 | 0.0 | 0.0 | 0.0 | FAIL | 1.0 | True | FAILED |
| pair_i2 | 30deg | 1%-99% Percentile Stretch | 21332 | 21738 | 21332 | 19 | 5 | 26.31578947368421 | 33.3 | 0.0 | 0.0 | 0.0 | FAIL | 1.0 | True | FAILED |
| pair_i2 | 30deg | 2%-98% Stretch + CLAHE | 20740 | 21186 | 20740 | 25 | 5 | 20.0 | 33.3 | 0.9945 | 0.0646 | 1.5854 | FAIL | 1.0 | True | FAILED |
| pair_i2 | 30deg | 1%-99% Stretch + CLAHE | 20727 | 21187 | 20727 | 23 | 5 | 21.73913043478261 | 44.4 | 69.8257 | 35.18 | 114.357 | FAIL | 1.0 | True | FAILED |
| pair_i3 | 50deg | 2%-98% Percentile Stretch | 21416 | 21191 | 21416 | 30 | 6 | 20.0 | 44.4 | 6.5811 | 1.4864 | 12.5087 | FAIL | 1.0 | True | FAILED |
| pair_i3 | 50deg | 1%-99% Percentile Stretch | 21332 | 21067 | 21332 | 31 | 6 | 19.35483870967742 | 44.4 | 7.8455 | 1.2138 | 15.0604 | FAIL | 1.0 | True | FAILED |
| pair_i3 | 50deg | 2%-98% Stretch + CLAHE | 20740 | 20657 | 20740 | 36 | 7 | 19.444444444444446 | 44.4 | 45.896 | 5.5936 | 81.9922 | FAIL | 1.0 | True | FAILED |
| pair_i3 | 50deg | 1%-99% Stretch + CLAHE | 20727 | 20460 | 20727 | 37 | 5 | 13.513513513513514 | 33.3 | 0.9883 | 0.078 | 1.5617 | FAIL | 1.0 | True | FAILED |
| pair_i4 | 70deg | 2%-98% Percentile Stretch | 21416 | 21568 | 21416 | 24 | 5 | 20.833333333333336 | 44.4 | 0.6173 | 0.2029 | 1.0125 | FAIL | 1.0 | True | FAILED |
| pair_i4 | 70deg | 1%-99% Percentile Stretch | 21332 | 21467 | 21332 | 30 | 5 | 16.666666666666664 | 33.3 | 10.3028 | 10.2906 | 14.1413 | FAIL | 1.0 | True | FAILED |
| pair_i4 | 70deg | 2%-98% Stretch + CLAHE | 20740 | 20989 | 20740 | 28 | 6 | 21.428571428571427 | 33.3 | 18.4695 | 4.5279 | 32.16 | FAIL | 1.0 | True | FAILED |
| pair_i4 | 70deg | 1%-99% Stretch + CLAHE | 20727 | 20894 | 20727 | 36 | 6 | 16.666666666666664 | 33.3 | 0.0 | 0.0 | 0.0 | FAIL | 1.0 | True | FAILED |

## 5. Results Grouped by Illumination Level
### pair_i1
- 2%-98% Percentile Stretch: verdict=UNCERTAIN, inlier ratio=98.99185730903451, coverage=100.0, RMSE=0.4042, independent=FAIL
- 1%-99% Percentile Stretch: verdict=UNCERTAIN, inlier ratio=99.00649598777225, coverage=100.0, RMSE=0.4043, independent=FAIL
- 2%-98% Stretch + CLAHE: verdict=UNCERTAIN, inlier ratio=98.86674482219617, coverage=100.0, RMSE=0.4187, independent=FAIL
- 1%-99% Stretch + CLAHE: verdict=UNCERTAIN, inlier ratio=98.91346526969345, coverage=100.0, RMSE=0.4101, independent=FAIL
### pair_i2
- 2%-98% Percentile Stretch: verdict=FAILED, inlier ratio=20.833333333333336, coverage=33.3, RMSE=0.0, independent=FAIL
- 1%-99% Percentile Stretch: verdict=FAILED, inlier ratio=26.31578947368421, coverage=33.3, RMSE=0.0, independent=FAIL
- 2%-98% Stretch + CLAHE: verdict=FAILED, inlier ratio=20.0, coverage=33.3, RMSE=0.9945, independent=FAIL
- 1%-99% Stretch + CLAHE: verdict=FAILED, inlier ratio=21.73913043478261, coverage=44.4, RMSE=69.8257, independent=FAIL
### pair_i3
- 2%-98% Percentile Stretch: verdict=FAILED, inlier ratio=20.0, coverage=44.4, RMSE=6.5811, independent=FAIL
- 1%-99% Percentile Stretch: verdict=FAILED, inlier ratio=19.35483870967742, coverage=44.4, RMSE=7.8455, independent=FAIL
- 2%-98% Stretch + CLAHE: verdict=FAILED, inlier ratio=19.444444444444446, coverage=44.4, RMSE=45.896, independent=FAIL
- 1%-99% Stretch + CLAHE: verdict=FAILED, inlier ratio=13.513513513513514, coverage=33.3, RMSE=0.9883, independent=FAIL
### pair_i4
- 2%-98% Percentile Stretch: verdict=FAILED, inlier ratio=20.833333333333336, coverage=44.4, RMSE=0.6173, independent=FAIL
- 1%-99% Percentile Stretch: verdict=FAILED, inlier ratio=16.666666666666664, coverage=33.3, RMSE=10.3028, independent=FAIL
- 2%-98% Stretch + CLAHE: verdict=FAILED, inlier ratio=21.428571428571427, coverage=33.3, RMSE=18.4695, independent=FAIL
- 1%-99% Stretch + CLAHE: verdict=FAILED, inlier ratio=16.666666666666664, coverage=33.3, RMSE=0.0, independent=FAIL

## 6. Comparison of Preprocessing Strategies
- 2%-98% Percentile Stretch: measured mean inlier ratio=40.1646309939253, mean spatial coverage=55.525000000000006.
- 1%-99% Percentile Stretch: measured mean inlier ratio=40.33594770945013, mean spatial coverage=52.75.
- 2%-98% Stretch + CLAHE: measured mean inlier ratio=39.93494017380301, mean spatial coverage=52.75.
- 1%-99% Stretch + CLAHE: measured mean inlier ratio=37.70819397116406, mean spatial coverage=52.75.

## 7. Best-Performing Strategy
- Based on the measured verdict, independent validation, spatial coverage, inlier ratio, and RMSE ordering, the best row was `pair_i1 / 1%-99% Percentile Stretch`.

## 8. Illumination Robustness Observations
Performance is reported empirically from the measured registration metrics; no preprocessing method is assumed superior in advance.

## 9. Failure and Breakpoint Observations
0 of 16 experiments recorded execution failures.

## 10. Empirical Result Statement
These results are empirical and based on measured registration metrics from the existing scientific validation pipeline.