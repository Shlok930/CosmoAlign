# CosmoAlign — Lunar Data Provenance & Access Documentation

This document describes the provenance, acquisition metadata, data usage terms, and instrument characteristics for scientific lunar imagery used in **CosmoAlign Phase 3**.

---

## 1. Primary Lunar Data Archives

### A. Chandrayaan-2 OHRC (Orbiter High Resolution Camera)
* **Archive**: Indian Space Research Organisation (ISRO) / Indian Space Science Data Centre (ISSDC) PRADAN Portal.
* **Portal URL**: https://pradan.issdc.gov.in/
* **Instrument**: Orbiter High Resolution Camera (OHRC).
* **Instrument Type**: High-resolution panchromatic optical pushbroom imager.
* **Spatial Resolution (GSD)**: ~0.25 meters/pixel at 100 km orbital altitude (highest resolution imaging system currently orbiting the Moon).
* **Wavelength**: Panchromatic (450 – 850 nm).
* **Primary Purpose**: Precision hazard avoidance mapping, high-contrast crater landing site reconnaissance, and surface morphology analysis.

### B. LRO NAC (Lunar Reconnaissance Orbiter Camera — Narrow Angle Camera)
* **Archive**: NASA Planetary Data System (PDS) / Arizona State University (ASU) LROC Archive.
* **Portal URL**: https://wms.lroc.asu.edu/lroc
* **Instrument**: Lunar Reconnaissance Orbiter Camera — Narrow Angle Camera (LRO NAC).
* **Instrument Type**: Dual monochrome pushbroom imagers (NAC-Left and NAC-Right).
* **Spatial Resolution (GSD)**: ~0.50 to 1.5 meters/pixel depending on orbital altitude.
* **Wavelength**: Panchromatic (400 – 750 nm).
* **Primary Purpose**: High-resolution global lunar surface mapping, topography modeling, and temporal change detection.

---

## 2. Dataset Pair 001 (`data/pair_001/`)

* **Pair ID**: `pair_001`
* **Target Lunar Region**: Boguslawsky E / South Pole Crater Rim Region (~72.9° S, 43.2° E).
* **Source Image**: `data/pair_001/source.tif` (Chandrayaan-2 OHRC panchromatic scientific product).
* **Reference Image**: `data/pair_001/reference.tif` (LRO NAC calibrated map-projected reference product).
* **Spatial Scale Difference**: OHRC GSD (~0.25 m/px) vs LRO NAC GSD (~0.50 m/px) $\to$ ~2.0x scale ratio.

---

## 3. Data Licensing & Attribution Statement

* Chandrayaan-2 data products are provided courtesy of **ISRO / ISSDC** and are used strictly for non-commercial scientific research and technological evaluation.
* LRO NAC data products are provided courtesy of **NASA / GSFC / Arizona State University LROC Team** via NASA PDS.
* Original scientific TIFF data products remain immutable and unmodified in `data/pair_001/`. Preprocessed or display normalized copies are generated in `processed/` or `outputs/phase3/`.
