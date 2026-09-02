"""
Sample Data Generator for CosmoAlign Testing.

Creates a high-contrast synthetic reference image with rich visual features
(shapes, text, grids, textures) and applies a known perspective transformation
(rotation, scaling, translation, perspective tilt) to create the source image.
"""

import os
import cv2
import numpy as np


def generate_synthetic_image(width: int = 800, height: int = 600) -> np.ndarray:
    """Creates a high-detail synthetic image with varied textures and geometric features."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background gradient
    x = np.linspace(30, 220, width, dtype=np.uint8)
    y = np.linspace(30, 220, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    img[:, :, 0] = xx
    img[:, :, 1] = yy
    img[:, :, 2] = (xx // 2 + yy // 2)

    # Add grid lines
    grid_size = 50
    for i in range(0, width, grid_size):
        cv2.line(img, (i, 0), (i, height), (80, 80, 80), 1)
    for j in range(0, height, grid_size):
        cv2.line(img, (0, j), (width, j), (80, 80, 80), 1)

    # Draw geometric shapes (corners, circles, polygons)
    cv2.rectangle(img, (100, 100), (300, 250), (255, 200, 50), -1)
    cv2.circle(img, (500, 200), 80, (50, 250, 150), -1)
    cv2.circle(img, (250, 450), 70, (200, 50, 250), -1)
    
    pts = np.array([[600, 350], [720, 480], [520, 520]], np.int32)
    cv2.fillPoly(img, [pts], (250, 250, 50))

    # Add high-contrast feature dots / crosshairs
    np.random.seed(42)
    for _ in range(120):
        rx = np.random.randint(20, width - 20)
        ry = np.random.randint(20, height - 20)
        radius = np.random.randint(3, 12)
        color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        cv2.circle(img, (rx, ry), radius, color, -1)
        cv2.drawMarker(img, (rx, ry), (255, 255, 255), cv2.MARKER_CROSS, 8, 1)

    # Add distinct text labels (excellent for SIFT corners)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "COSMOALIGN REF", (120, 180), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "FEATURE REGISTRATION", (420, 210), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "LUNAR CRATER BASELINE", (150, 460), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def apply_known_homography(
    img: np.ndarray,
    angle_deg: float = 12.0,
    scale: float = 0.92,
    tx: float = 40.0,
    ty: float = -25.0,
    perspective_skew: float = 0.00015
) -> np.ndarray:
    """Applies rotation, scale, translation, and subtle perspective transformation."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    # 2D similarity transform matrix
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty

    # Expand to 3x3 homography matrix
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M_rot

    # Add mild perspective skew
    H[2, 0] = perspective_skew
    H[2, 1] = -perspective_skew * 0.8

    warped_img = cv2.warpPerspective(
        img,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(15, 15, 15)
    )
    return warped_img


def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    print("Generating synthetic reference image...")
    ref_img = generate_synthetic_image(width=900, height=700)
    ref_path = os.path.join(data_dir, "reference.jpg")
    cv2.imwrite(ref_path, ref_img)
    print(f"  [OK] Saved reference image to: {ref_path}")

    print("Applying known homography to generate source image...")
    source_img = apply_known_homography(ref_img)
    source_path = os.path.join(data_dir, "source.jpg")
    cv2.imwrite(source_path, source_img)
    print(f"  [OK] Saved source image to:    {source_path}")

    print("\n[SUCCESS] Sample dataset generated successfully in 'data/' folder.")


if __name__ == "__main__":
    main()
