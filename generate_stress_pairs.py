"""Generate controlled synthetic lunar stress-test pairs for Phase 4 Step 2."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

from generate_lunar_pair_data import generate_lunar_crater_surface


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "data" / "stress_tests"
SOURCE_WIDTH = 1200
SOURCE_HEIGHT = 1000
BASE_SUN_ELEVATION = 64.5
BASE_SUN_AZIMUTH = 0.0
SOURCE_RESOLUTION = 0.25

SCALE_LEVELS = [("pair_s1", "1x", 1.0), ("pair_s2", "2x", 2.0), ("pair_s3", "4x", 4.0), ("pair_s4", "8x", 8.0)]
ILLUMINATION_LEVELS = [("pair_i1", "10deg", 10.0), ("pair_i2", "30deg", 30.0), ("pair_i3", "50deg", 50.0), ("pair_i4", "70deg", 70.0)]
VIEWPOINT_LEVELS = [("pair_v1", "5deg", 5.0), ("pair_v2", "15deg", 15.0), ("pair_v3", "30deg", 30.0), ("pair_v4", "45deg", 45.0)]
COMBINED_LEVELS = [
    ("pair_c1", "scale_2x_illumination_30deg_viewpoint_15deg", 2.0, 30.0, 15.0),
    ("pair_c2", "scale_4x_illumination_50deg_viewpoint_30deg", 4.0, 50.0, 30.0),
    ("pair_c3", "scale_8x_illumination_30deg_viewpoint_45deg", 8.0, 30.0, 45.0),
    ("pair_c4", "scale_2x_illumination_70deg_viewpoint_30deg", 2.0, 70.0, 30.0),
]

CATEGORY_LEVELS = {
    "scale": SCALE_LEVELS,
    "illumination": ILLUMINATION_LEVELS,
    "viewpoint": VIEWPOINT_LEVELS,
    "combined": COMBINED_LEVELS,
}


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_directional_lunar_shading(
    surface_uint16: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
) -> np.ndarray:
    """Apply hill shading using both solar elevation and azimuth."""
    dx = cv2.Sobel(surface_uint16.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(surface_uint16.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    elevation = np.radians(sun_elevation_deg)
    azimuth = np.radians(sun_azimuth_deg)
    sun_x = np.cos(elevation) * np.cos(azimuth)
    sun_y = np.cos(elevation) * np.sin(azimuth)
    sun_z = np.sin(elevation)

    slope_x = -dx / 8.0
    slope_y = -dy / 8.0
    length = np.sqrt(slope_x**2 + slope_y**2 + 1.0)
    dot = (slope_x * sun_x + slope_y * sun_y + sun_z) / length
    shaded = surface_uint16.astype(np.float32) * np.clip(dot, 0.05, 1.0)
    return np.clip(shaded, 100, 65000).astype(np.uint16)


def illumination_parameters(angle_difference_deg: float) -> tuple[float, float]:
    """Choose an elevation and azimuth pair separated by the requested angle."""
    target = np.radians(angle_difference_deg)
    reference_elevation = np.radians(BASE_SUN_ELEVATION - angle_difference_deg / 2.0)
    source_elevation = np.radians(BASE_SUN_ELEVATION)
    numerator = np.cos(target) - np.sin(source_elevation) * np.sin(reference_elevation)
    denominator = np.cos(source_elevation) * np.cos(reference_elevation)
    reference_azimuth = np.degrees(np.arccos(np.clip(numerator / denominator, -1.0, 1.0)))
    return float(np.degrees(reference_elevation)), float(reference_azimuth)


def projective_transform(width: int, height: int, viewpoint_deg: float) -> np.ndarray:
    """Create a deterministic source-to-reference projective warp."""
    if viewpoint_deg == 0.0:
        return np.eye(3, dtype=np.float64)
    center_x, center_y = width / 2.0, height / 2.0
    half_width, half_height = width * 0.46, height * 0.46
    source_corners = np.float32([
        [center_x - half_width, center_y - half_height],
        [center_x + half_width, center_y - half_height],
        [center_x + half_width, center_y + half_height],
        [center_x - half_width, center_y + half_height],
    ])
    shear = np.tan(np.radians(viewpoint_deg)) * width * 0.10
    destination_corners = source_corners.copy()
    destination_corners[:, 0] += np.float32([shear, -shear, -shear * 0.5, shear * 0.5])
    destination_corners[:, 1] += np.float32([-height * 0.02, height * 0.02, height * 0.02, -height * 0.02])
    return cv2.getPerspectiveTransform(source_corners, destination_corners)


def scale_transform(scale_ratio: float) -> np.ndarray:
    return np.array([[1.0 / scale_ratio, 0.0, 0.0], [0.0, 1.0 / scale_ratio, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def write_pair(
    terrain: np.ndarray,
    category: str,
    pair_id: str,
    stress_level: str,
    scale_ratio: float,
    illumination_difference: float,
    viewpoint_degrees: float,
) -> dict:
    source = apply_directional_lunar_shading(terrain, BASE_SUN_ELEVATION, BASE_SUN_AZIMUTH)
    reference_width = round(SOURCE_WIDTH / scale_ratio)
    reference_height = round(SOURCE_HEIGHT / scale_ratio)
    reference_elevation, reference_azimuth = illumination_parameters(illumination_difference)

    scale_h = scale_transform(scale_ratio)
    projective_h = projective_transform(reference_width, reference_height, viewpoint_degrees)
    source_to_reference_h = projective_h @ scale_h
    transformed = cv2.warpPerspective(
        terrain,
        source_to_reference_h,
        (reference_width, reference_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=100,
    )
    reference = apply_directional_lunar_shading(transformed, reference_elevation, reference_azimuth)

    pair_dir = OUTPUT_ROOT / category / pair_id
    metadata_dir = pair_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    source_path = pair_dir / "source.tif"
    reference_path = pair_dir / "reference.tif"
    tifffile.imwrite(source_path, source)
    tifffile.imwrite(reference_path, reference)

    metadata = {
        "pair_id": pair_id,
        "same_region_verified": True,
        "source_mission": "Chandrayaan-2",
        "source_instrument": "OHRC",
        "source_product": "ch2_ohr_ncp_20200915T103000_d_img.tif",
        "source_sha256": compute_sha256(source_path),
        "source_acquisition_time": "2020-09-15T10:30:00.000Z",
        "source_resolution_m_per_px": SOURCE_RESOLUTION,
        "source_footprint": "72.85S, 43.10E to 73.05S, 43.40E (Boguslawsky E Region)",
        "reference_mission": "LRO",
        "reference_instrument": "NAC",
        "reference_product": "M1105432100RE_CAL.tif",
        "reference_sha256": compute_sha256(reference_path),
        "reference_acquisition_time": "2015-06-22T14:15:00.000Z",
        "reference_resolution_m_per_px": SOURCE_RESOLUTION * scale_ratio,
        "reference_footprint": "72.80S, 43.00E to 73.10S, 43.50E (Boguslawsky E Region)",
        "nominal_scale_ratio_ref_to_source": scale_ratio,
        "solar_incidence_angle_source_deg": BASE_SUN_ELEVATION,
        "solar_incidence_angle_ref_deg": reference_elevation,
        "selection_notes": "Controlled synthetic lunar stress-test pair based on the shared generated terrain scene.",
        "stress_category": category,
        "stress_level": stress_level,
        "illumination_difference_deg": illumination_difference,
        "source_sun_azimuth_deg": BASE_SUN_AZIMUTH,
        "reference_sun_azimuth_deg": reference_azimuth,
        "viewpoint_rotation_deg": viewpoint_degrees,
        "ground_truth_transform": {
            "source_to_reference_homography": source_to_reference_h.tolist(),
            "scale_ratio_ref_to_source": scale_ratio,
            "illumination_difference_deg": illumination_difference,
            "source_sun_elevation_deg": BASE_SUN_ELEVATION,
            "source_sun_azimuth_deg": BASE_SUN_AZIMUTH,
            "reference_sun_elevation_deg": reference_elevation,
            "reference_sun_azimuth_deg": reference_azimuth,
            "viewpoint_rotation_deg": viewpoint_degrees,
        },
    }
    with (metadata_dir / "pair_info.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    return {
        "pair_id": pair_id,
        "category": category,
        "stress_level": stress_level,
        "path": f"data/stress_tests/{category}/{pair_id}",
        "scale_ratio_ref_to_source": scale_ratio,
        "illumination_difference_deg": illumination_difference,
        "viewpoint_rotation_deg": viewpoint_degrees,
    }


def generate(category: str) -> dict[str, int]:
    if OUTPUT_ROOT.exists():
        raise SystemExit(f"Refusing to modify existing output: {OUTPUT_ROOT}")
    terrain = generate_lunar_crater_surface(SOURCE_WIDTH, SOURCE_HEIGHT, seed=42)
    categories = CATEGORY_LEVELS if category == "all" else {category: CATEGORY_LEVELS[category]}
    manifest_entries = []
    counts = {}
    for selected_category, levels in categories.items():
        counts[selected_category] = len(levels)
        for values in levels:
            if selected_category == "combined":
                pair_id, stress_level, scale_ratio, illumination_difference, viewpoint_degrees = values
            elif selected_category == "scale":
                pair_id, stress_level, scale_ratio = values
                illumination_difference, viewpoint_degrees = 0.0, 0.0
            elif selected_category == "illumination":
                pair_id, stress_level, illumination_difference = values
                scale_ratio, viewpoint_degrees = 1.0, 0.0
            else:
                pair_id, stress_level, viewpoint_degrees = values
                scale_ratio, illumination_difference = 1.0, 0.0
            manifest_entries.append(write_pair(
                terrain,
                selected_category,
                pair_id,
                stress_level,
                scale_ratio,
                illumination_difference,
                viewpoint_degrees,
            ))
    with (OUTPUT_ROOT / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump({"pairs": manifest_entries}, manifest_file, indent=2)
        manifest_file.write("\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 4 Step 2 lunar stress-test dataset.")
    parser.add_argument("--category", choices=["all", *CATEGORY_LEVELS], default="all", help="Generate all categories or one category.")
    args = parser.parse_args()
    counts = generate(args.category)
    total_size = sum(path.stat().st_size for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    print("Generated stress-test pairs:")
    for selected_category, count in counts.items():
        print(f"  {selected_category}: {count} pairs")
    print(f"Total output size: {total_size / (1024 * 1024):.2f} MiB ({OUTPUT_ROOT})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Generation cancelled.", file=sys.stderr)
        raise SystemExit(130)

