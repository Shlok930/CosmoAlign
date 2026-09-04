# Standalone Phase 4 Illumination Experiment

This run is independent of run_phase3_lunar_pipeline(). Each preprocessing strategy is applied independently to source and reference display images.

| Pair | Level | Preprocessing | Good | Inliers | Ratio | Coverage | RMSE | Geometry | Failure |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| pair_i1 | 10deg | 2%-98% Percentile Stretch | 2579 | 2553 | 98.99185730903451 | 100.0 | 0.4042 | PASS | N/A |
| pair_i1 | 10deg | 1%-99% Percentile Stretch | 2617 | 2591 | 99.00649598777225 | 100.0 | 0.4043 | PASS | N/A |
| pair_i1 | 10deg | 2%-98% Stretch + CLAHE | 2559 | 2530 | 98.86674482219617 | 100.0 | 0.4187 | PASS | N/A |
| pair_i1 | 10deg | 1%-99% Stretch + CLAHE | 2577 | 2549 | 98.91346526969345 | 100.0 | 0.4101 | PASS | N/A |
| pair_i2 | 30deg | 2%-98% Percentile Stretch | 24 | 5 | 20.833333333333336 | 33.3 | 0.0 | PASS | N/A |
| pair_i2 | 30deg | 1%-99% Percentile Stretch | 19 | 5 | 26.31578947368421 | 33.3 | 0.0 | FAIL | N/A |
| pair_i2 | 30deg | 2%-98% Stretch + CLAHE | 25 | 5 | 20.0 | 33.3 | 0.9945 | FAIL | N/A |
| pair_i2 | 30deg | 1%-99% Stretch + CLAHE | 23 | 5 | 21.73913043478261 | 44.4 | 69.8257 | FAIL | N/A |
| pair_i3 | 50deg | 2%-98% Percentile Stretch | 30 | 6 | 20.0 | 44.4 | 6.5811 | FAIL | N/A |
| pair_i3 | 50deg | 1%-99% Percentile Stretch | 31 | 6 | 19.35483870967742 | 44.4 | 7.8455 | FAIL | N/A |
| pair_i3 | 50deg | 2%-98% Stretch + CLAHE | 36 | 7 | 19.444444444444446 | 44.4 | 45.896 | FAIL | N/A |
| pair_i3 | 50deg | 1%-99% Stretch + CLAHE | 37 | 5 | 13.513513513513514 | 33.3 | 0.9883 | FAIL | N/A |
| pair_i4 | 70deg | 2%-98% Percentile Stretch | 24 | 5 | 20.833333333333336 | 44.4 | 0.6173 | PASS | N/A |
| pair_i4 | 70deg | 1%-99% Percentile Stretch | 30 | 5 | 16.666666666666664 | 33.3 | 10.3028 | FAIL | N/A |
| pair_i4 | 70deg | 2%-98% Stretch + CLAHE | 28 | 6 | 21.428571428571427 | 33.3 | 18.4695 | FAIL | N/A |
| pair_i4 | 70deg | 1%-99% Stretch + CLAHE | 36 | 6 | 16.666666666666664 | 33.3 | 0.0 | FAIL | N/A |
