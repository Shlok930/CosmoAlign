# Phase 4 Closure Report

## Executive Summary
This report consolidates measured Phase 4 robustness evidence. Scale normalization and standalone illumination results are included when available; existing results are reused rather than overwritten.

## Dataset Summary
The stress dataset contains scale, illumination, viewpoint, and combined categories with metadata, SHA-256 provenance, and ground-truth homographies.

## Scale
Pairs: 4
Mean/min/max inlier ratio: 59.593695777906305 / 34.61538461538461 / 100.0%
Mean/min/max RMSE: 3.523275 / 0.0 / 11.3837 px
Mean spatial coverage: 80.575%
Geometry PASS/FAIL: 75.0% / 25.0%
VALIDATED/UNCERTAIN/FAILED: None% / None% / None%
Per-pair results:
- pair_s1: level=1x, ratio=100.0, RMSE=0.0, verdict=None.
- pair_s2: level=2x, ratio=56.14035087719298, RMSE=1.1372, verdict=None.
- pair_s3: level=4x, ratio=47.61904761904761, RMSE=1.5722, verdict=None.
- pair_s4: level=8x, ratio=34.61538461538461, RMSE=11.3837, verdict=None.

## Illumination
Pairs: 16
Mean/min/max inlier ratio: 39.53592821208562 / 13.513513513513514 / 99.00649598777225%
Mean/min/max RMSE: 10.197375000000001 / 0.0 / 69.8257 px
Mean spatial coverage: 53.443749999999994%
Geometry PASS/FAIL: 37.5% / 62.5%
VALIDATED/UNCERTAIN/FAILED: None% / None% / None%
Per-pair results:
- pair_i1: level=10deg, ratio=98.99185730903451, RMSE=0.4042, verdict=None.
- pair_i1: level=10deg, ratio=99.00649598777225, RMSE=0.4043, verdict=None.
- pair_i1: level=10deg, ratio=98.86674482219617, RMSE=0.4187, verdict=None.
- pair_i1: level=10deg, ratio=98.91346526969345, RMSE=0.4101, verdict=None.
- pair_i2: level=30deg, ratio=20.833333333333336, RMSE=0.0, verdict=None.
- pair_i2: level=30deg, ratio=26.31578947368421, RMSE=0.0, verdict=None.
- pair_i2: level=30deg, ratio=20.0, RMSE=0.9945, verdict=None.
- pair_i2: level=30deg, ratio=21.73913043478261, RMSE=69.8257, verdict=None.
- pair_i3: level=50deg, ratio=20.0, RMSE=6.5811, verdict=None.
- pair_i3: level=50deg, ratio=19.35483870967742, RMSE=7.8455, verdict=None.
- pair_i3: level=50deg, ratio=19.444444444444446, RMSE=45.896, verdict=None.
- pair_i3: level=50deg, ratio=13.513513513513514, RMSE=0.9883, verdict=None.
- pair_i4: level=70deg, ratio=20.833333333333336, RMSE=0.6173, verdict=None.
- pair_i4: level=70deg, ratio=16.666666666666664, RMSE=10.3028, verdict=None.
- pair_i4: level=70deg, ratio=21.428571428571427, RMSE=18.4695, verdict=None.
- pair_i4: level=70deg, ratio=16.666666666666664, RMSE=0.0, verdict=None.

## Viewpoint Homography
Pairs: 4
Mean/min/max inlier ratio: 97.41875575153098 / 91.94029850746269 / 99.57469431153642%
Mean/min/max RMSE: 0.5636 / 0.3924 / 0.8912 px
Mean spatial coverage: 100.0%
Geometry PASS/FAIL: 100.0% / 0.0%
VALIDATED/UNCERTAIN/FAILED: 0.0% / 100.0% / 0.0%
Per-pair results:
- pair_v1: level=5deg, ratio=99.57469431153642, RMSE=0.3924, verdict=UNCERTAIN.
- pair_v2: level=15deg, ratio=99.27859841978702, RMSE=0.4331, verdict=UNCERTAIN.
- pair_v3: level=30deg, ratio=98.88143176733782, RMSE=0.5377, verdict=UNCERTAIN.
- pair_v4: level=45deg, ratio=91.94029850746269, RMSE=0.8912, verdict=UNCERTAIN.

## Combined
Pairs: 4
Mean/min/max inlier ratio: 32.25 / 24.0 / 35.0%
Mean/min/max RMSE: 9.743575 / 1.9208 / 21.406 px
Mean spatial coverage: 55.575%
Geometry PASS/FAIL: 0.0% / 100.0%
VALIDATED/UNCERTAIN/FAILED: 0.0% / 0.0% / 100.0%
Per-pair results:
- pair_c1: level=scale_2x_illumination_30deg_viewpoint_15deg, ratio=35.0, RMSE=1.9208, verdict=FAILED.
- pair_c2: level=scale_4x_illumination_50deg_viewpoint_30deg, ratio=35.0, RMSE=21.406, verdict=FAILED.
- pair_c3: level=scale_8x_illumination_30deg_viewpoint_45deg, ratio=35.0, RMSE=12.5893, verdict=FAILED.
- pair_c4: level=scale_2x_illumination_70deg_viewpoint_30deg, ratio=24.0, RMSE=3.0582, verdict=FAILED.

## Scale Normalization
Native and normalized results are compared in the scale-normalization CSV. Positive ratio improvement and positive RMSE improvement indicate measured benefit; negative values indicate degradation.

## Illumination
Existing illumination results are preserved and standalone closure-run results can be supplied through the function arguments. Strategies are compared empirically.

## Viewpoint
Existing Homography rows are reused for category summaries; Affine results remain in their original geometry CSV.

## Combined
Combined stress applies scale, illumination, and viewpoint simultaneously, so failures cannot be attributed to one factor alone.

## Ground-Truth Transform Analysis
Where estimated transforms are available, point-based corner displacement should be reported in reference pixels. Raw matrix subtraction is not used because matrix scale and coordinate-frame differences are not directly comparable.

## Step 7 Validator Diagnostic
The Step 7 CSV is included as evidence that raw independent-validation RMSE can be dominated by geometrically inconsistent matches in some cases, while combined stress can have no consistent validation subset.

## Failure Analysis
Failed rows remain represented in CSV outputs and should be interpreted alongside match support, spatial coverage, and geometry status.

## Limitations
The stress generator is synthetic, illumination is directional hill shading, viewpoint is a 2D projective approximation, and the existing validator's independent split remains unchanged.

## Reproducibility
Run metadata is recorded by the Phase 4 runner when a closure run is executed.

## Final Conclusion
Phase 4 provides empirical evidence of strong baseline and viewpoint Homography behavior, scale and illumination degradation, and combined-stress weakness. It does not establish robustness on real multi-modal lunar imagery.
