# Phase 4 Scale Normalization Experiment

Native resolution is compared with in-memory source resampling by the metadata-derived inverse reference/source GSD ratio. Raw TIFF files are never changed.

| Pair | Level | Ratio | Native good/inliers/ratio/RMSE | Normalized good/inliers/ratio/RMSE | Ratio improvement | RMSE improvement | Geometry | Failure |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| pair_s1 | 1x | 1.0 | 21416/21416/100.0/0.0 | 21416/21416/100.0/0.0 | 0.0 | 0.0 | PASS/PASS | N/A |
| pair_s2 | 2x | 2.0 | 57/32/56.14035087719298/1.1372 | 48/31/64.58333333333334/1.2228 | 8.442982456140363 | -0.08560000000000012 | PASS/PASS | N/A |
| pair_s3 | 4x | 4.0 | 42/20/47.61904761904761/1.5722 | 27/14/51.85185185185185/1.6878 | 4.232804232804234 | -0.11559999999999993 | PASS/PASS | N/A |
| pair_s4 | 8x | 8.0 | 26/9/34.61538461538461/11.3837 | 12/7/58.333333333333336/0.6568 | 23.717948717948723 | 10.726899999999999 | FAIL/PASS | N/A |

Positive improvement values indicate higher normalized inlier ratio or lower normalized RMSE. No improvement is assumed; values are measured.
