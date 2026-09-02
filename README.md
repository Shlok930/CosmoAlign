# CosmoAlign — Image Registration System (Phase 0 through Phase 3 Hardened)

**CosmoAlign** is a scientific computer-vision framework designed to geometrically align a **Source Image** (e.g., Chandrayaan-2 OHRC) to a target **Reference Image** (e.g., LRO NAC).

---

## 1. System Identity & Scope

* **Project Name**: CosmoAlign
* **Purpose**: Automatically extract visual correspondences and estimate a 2D/3D geometric transformation matrix ($H$) to align a Source Image with a Reference Image.
* **Current Version**: Phase 3 Hardened — Scientific Real Lunar Imagery Validation Engine.

---

## 2. Multi-Gate Scientific Validation System

CosmoAlign Phase 3 replaces simplistic boolean success checks (e.g., `success = inliers > 4`) with an **8-Gate Scientific Validation Engine**. 

A numerical metric saying "success" does NOT automatically imply that registration is valid. Registration must pass independent gates:

```text
                 ┌────────────────────────────────┐
                 │ 1. DATA INTEGRITY GATE         │
                 │    (SHA-256 Checksum, Dtype)   │
                 └───────────────┬────────────────┘
                                 ↓
                 ┌────────────────────────────────┐
                 │ 2. SIFT FEATURE GATE           │
                 │    (Keypoint Count & Spread)   │
                 └───────────────┬────────────────┘
                                 ↓
                 ┌────────────────────────────────┐
                 │ 3. DESCRIPTOR MATCH GATE       │
                 │    (Lowe Test + Mutual NN)     │
                 └───────────────┬────────────────┘
                                 ↓
                 ┌────────────────────────────────┐
                 │ 4. RANSAC GEOMETRY GATE        │
                 │    (Inlier Count & Ratio %)    │
                 └───────────────┬────────────────┘
                                 ↓
                 ┌────────────────────────────────┐
                 │ 5. SPATIAL COVERAGE GATE       │
                 │    (3x3 Cell Occupancy/Entropy)│
                 └───────────────┬────────────────┘
                                 ↓
                 ┌────────────────────────────────┐
                 │ 6. HOMOGRAPHY SANITY GATE      │
                 │    (det(H) > 0, Area Ratio)    │
                 └───────────────┬────────────────┘
                                 ↓
                 ┌────────────────────────────────┐
                 │ 7. REPROJECTION ERROR GATE     │
                 │    (RMSE, Median, P95 Error)   │
                 └───────────────┬────────────────┘
                                 ↓
                 ┌────────────────────────────────┐
                 │ 8. INDEPENDENT SPLIT GATE      │
                 │    (50/50 Train-Val Match Split)│
                 └───────────────┬────────────────┘
                                 ↓
                 ┌────────────────────────────────┐
                 │ FINAL VERDICT SYSTEM           │
                 │ [VALIDATED / UNCERTAIN / FAILED]│
                 └────────────────────────────────┘
```

### Final Verdict Definitions:
* **`VALIDATED`**: All computational, spatial, geometric, and 50/50 independent validation split gates pass cleanly.
* **`UNCERTAIN`**: Critical RANSAC gates passed, but spatial coverage is localized or independent cross-validation shows overfitting/uncertainty.
* **`FAILED`**: Any critical gate (Data corruption, negative determinant, single-cluster inliers, degenerate polygon collapse) fails.

---

## 3. Project Directory Structure

```text
cosmoalign/
├── data/
│   ├── pair_001/
│   │   ├── source.tif                  # Immutable 16-bit uint16 OHRC TIFF
│   │   ├── reference.tif               # Immutable 16-bit uint16 LRO NAC TIFF
│   │   ├── metadata/
│   │   │   └── pair_info.json          # Metadata with SHA-256 hashes & provenance
│   │   └── processed/                  # Preprocessed working copies
│   └── README.md                       # ISRO/ISSDC & NASA LROC archive provenance
├── outputs/
│   └── phase3/
│       ├── 01_source_raw_view.png      # Non-destructive display preview
│       ├── 02_reference_raw_view.png   # Non-destructive display preview
│       ├── validation_report.json      # Machine-readable multi-gate validation JSON
│       ├── validation_checklist.md     # Markdown checklist of validation gates
│       ├── experiments.csv             # Comparative metrics across preprocessing runs
│       ├── phase3_report.md            # Comprehensive scientific diagnostic report
│       └── experiments/
│           ├── exp_001_baseline/       # Baseline artifacts + corner projection view
│           ├── exp_002_clahe/          # CLAHE enhanced artifacts
│           └── exp_003_percentile/     # Percentile stretch artifacts
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configuration parameters & defaults
│   ├── image_loader.py                 # Scientific TIFF I/O & SHA-256 checksum computation
│   ├── metadata.py                     # Metadata parsing & stats inspection
│   ├── preprocessing.py                # Isolated contrast enhancement & valid-data masking
│   ├── feature_detection.py            # Keypoint & descriptor extraction
│   ├── matching.py                     # KNN matching, Lowe test, Mutual Nearest-Neighbor crosscheck
│   ├── registration.py                 # RANSAC homography, warping, RMSE metrics
│   ├── visualization.py                # Stage visualizers & Corner Projection overlay
│   ├── evaluation.py                   # Spatial grid distribution, Error percentiles (Median, P95)
│   ├── validation.py                   # Multi-Gate Scientific Validation Engine
│   ├── lunar_data.py                   # Dataset manager & preview generator
│   └── main.py                         # Hardened CLI runner supporting Phase 3 validation reporting
├── tests/
│   ├── test_phase1_registration.py     # Regression unit test suite
│   └── test_validation_system.py       # False-positive synthetic negative validation test suite
├── generate_lunar_pair_data.py         # Scientific dataset pair generator
├── requirements.txt                    # Python dependencies
└── README.md                           # Comprehensive documentation & educational guide
```

---

## 4. Installation & Quickstart

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Hardened Phase 3 Validation Engine
```bash
python src/main.py --phase 3 --pair pair_001
```

### Step 3: Run Automated Test Suites
```bash
python -m unittest tests/test_phase1_registration.py tests/test_validation_system.py
```

---

## 5. Sample Validation Output

```text
======================================================================
 COSMOALIGN PHASE 3 - HARDENED MULTI-GATE VALIDATION SUMMARY 
======================================================================
TARGET PAIR ID: pair_001

MULTIPLE INDEPENDENT VALIDATION GATES:
  1. DATA INTEGRITY:      PASS
  2. SIFT FEATURES:       PASS
  3. DESCRIPTOR MATCHING: PASS
  4. RANSAC GEOMETRY:     PASS
  5. SPATIAL COVERAGE:    PASS
  6. HOMOGRAPHY SANITY:   PASS
  7. REPROJECTION ERROR:  PASS
  8. INDEPENDENT SPLIT:   FAIL

QUANTITATIVE ERROR METRICS:
  * Reprojection RMSE:    1.4157 px
  * Median Pixel Error:   0.9456 px
  * 95th Percentile Error: 3.0491 px
  * 50/50 Independent RMSE: 203.6992 px

SPATIAL DISTRIBUTION METRICS:
  * Occupied Grid Cells:  7/9 cells (77.8%)
  * Spatial Entropy Score: 0.7738
  * Single Cluster Flag:  False

======================================================================
 FINAL REGISTRATION VERDICT: [ UNCERTAIN ] (Confidence: MEDIUM)
======================================================================
```
