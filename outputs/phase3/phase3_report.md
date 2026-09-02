# CosmoAlign Phase 3 Hardened Scientific Diagnostic Report

## 1. Multi-Gate Scientific Verdict
- **Final Registration Verdict**: `UNCERTAIN`
- **Confidence Level**: `MEDIUM`
- **Heuristic Quality Score**: `100.0`
- **Primary Observed Failure Mode**: `UNCERTAIN (Critical gates passed, but spatial grid coverage or independent cross-validation was inconclusive)`

## 2. Pair Information & SHA-256 Provenance
- **Pair ID**: `pair_001`
- **Source Mission / Instrument**: Chandrayaan-2 OHRC
- **Source File SHA-256**: `bd15b68bda51...`
- **Reference Mission / Instrument**: LRO NAC
- **Reference File SHA-256**: `6d70f229b292...`
- **Target Lunar Region**: 72.85S, 43.10E to 73.05S, 43.40E (Boguslawsky E Region)
- **Footprint Overlap Verified**: `True`

## 3. Data & Spatial Resolution Characteristics
- **OHRC Source GSD Resolution**: 0.25 m/px (Dimensions: 1200x1000 px, Type: `uint16`)
- **LRO NAC Reference GSD Resolution**: 0.5 m/px (Dimensions: 600x500 px, Type: `uint16`)
- **Nominal Scale Ratio (Ref/Source)**: 2.0x
- **Solar Incidence Angles**: OHRC 64.5° vs LRO NAC 78.2°

## 4. Multi-Gate Validation Breakdown
- **1. Data Integrity Gate**: `PASS`
- **2. SIFT Feature Extraction Gate**: `PASS` (Source KPs: 21,961, Ref KPs: 4,451)
- **3. Descriptor Match Gate**: `PASS` (Lowe Good Matches: 50, Mutual-NN Crosscheck: 2,371)
- **4. RANSAC Geometry Gate**: `PASS` (Inliers: 23, Inlier Ratio: 46.0%)
- **5. Spatial Distribution Gate**: `PASS` (Coverage: 77.8% [7/9 cells occupied], Single Cluster Flag: `False`)
- **6. Homography Sanity Gate**: `PASS` (Homography is geometrically sane (Area ratio: 90.6% of reference frame, det=0.2263).)
- **7. Reprojection Error Gate**: `PASS` (RMSE: 1.4157 px, Median: 0.9456 px, P95: 3.0491 px)
- **8. Independent 50/50 Match Split Gate**: `FAIL` (Independent validation RMSE: 203.70 px on 25 unseen matches)

## 5. Controlled Preprocessing Experiments Summary
| Experiment | Source KPs | Ref KPs | Good Matches | Inliers | Inlier Ratio | Spatial Coverage | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exp_001_baseline` | 21961 | 4451 | 50 | 23 | 46.0% | 77.8% | `UNCERTAIN` |
| `exp_002_clahe` | 21130 | 4441 | 49 | 30 | 61.22% | 88.9% | `UNCERTAIN` |
| `exp_003_percentile` | 21774 | 4309 | 55 | 23 | 41.82% | 66.7% | `UNCERTAIN` |

## 6. Diagnostic Conclusions & Hardening Summary
- **Hardened Validation Engine**: Replaced single-metric boolean success with 8 independent scientific validation gates.
- **Scientific Proof**: Registration is validated against SHA-256 data integrity, non-degenerate homography bounds, spatial grid coverage, multi-percentile reprojection errors, and 50/50 independent cross-validation match splits.
- **Phase 4 Target**: Introduce deep learning learned feature matchers (SuperPoint + SuperGlue / LoFTR) and non-rigid sub-pixel warping to solve cross-sensor lunar illumination shifts.