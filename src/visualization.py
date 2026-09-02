"""
Visualization Module for CosmoAlign Phase 2 & Phase 3.

Provides dedicated stage-by-stage visual inspection utilities including:
- Rich SIFT Keypoints (location, scale/size, orientation angle)
- Raw Matches, Good Matches, and Inlier/Outlier side-by-side drawings
- Corner Projection View (projecting Source bounding box over Reference frame)
- 2x2 Panel Before vs After Registration comparison
- Absolute Intensity Difference maps
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


def draw_rich_keypoints(
    image: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draws SIFT keypoints showing location, scale (circle size), and orientation angle.
    """
    vis_img = cv2.drawKeypoints(
        image,
        keypoints,
        outImage=None,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return vis_img


def draw_matches_side_by_side(
    img_source: np.ndarray,
    kp_source: List[cv2.KeyPoint],
    img_ref: np.ndarray,
    kp_ref: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    match_color: Tuple[int, int, int] = (0, 255, 255),
    max_matches_to_draw: Optional[int] = 150
) -> np.ndarray:
    """
    Draws matches side-by-side connecting source keypoints to reference keypoints.
    """
    matches_to_draw = matches
    if max_matches_to_draw and len(matches) > max_matches_to_draw:
        matches_to_draw = matches[:max_matches_to_draw]

    vis_img = cv2.drawMatches(
        img_source,
        kp_source,
        img_ref,
        kp_ref,
        matches_to_draw,
        outImg=None,
        matchColor=match_color,
        singlePointColor=(100, 100, 100),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis_img


def draw_inliers_vs_outliers(
    img_source: np.ndarray,
    kp_source: List[cv2.KeyPoint],
    img_ref: np.ndarray,
    kp_ref: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    inliers_mask: np.ndarray,
    max_matches_to_draw: Optional[int] = 200
) -> np.ndarray:
    """
    Draws matches side-by-side highlighting Inliers in GREEN and Outliers in RED.
    """
    mask_flat = inliers_mask.ravel().tolist()

    matches_to_draw = good_matches
    mask_to_use = mask_flat

    if max_matches_to_draw and len(good_matches) > max_matches_to_draw:
        matches_to_draw = good_matches[:max_matches_to_draw]
        mask_to_use = mask_flat[:max_matches_to_draw]

    outlier_mask = [1 - val for val in mask_to_use]
    outlier_vis = cv2.drawMatches(
        img_source,
        kp_source,
        img_ref,
        kp_ref,
        matches_to_draw,
        outImg=None,
        matchColor=(0, 0, 255), # Red for outliers
        matchesMask=outlier_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    full_vis = cv2.drawMatches(
        img_source,
        kp_source,
        img_ref,
        kp_ref,
        matches_to_draw,
        outImg=outlier_vis,
        matchColor=(0, 255, 0), # Green for inliers
        matchesMask=mask_to_use,
        flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return full_vis


def draw_corner_projection(
    reference_img: np.ndarray,
    H: np.ndarray,
    source_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Renders the projected 4 corners of the Source Image polygon onto the Reference target image.

    Makes geometric flips, collapse, or extreme perspective distortion visually obvious.
    """
    if len(reference_img.shape) == 2:
        vis_img = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = reference_img.copy()

    if H is None or H.shape != (3, 3):
        cv2.putText(vis_img, "CORNER PROJECTION FAILED (H is None)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis_img

    h_src, w_src = source_shape[:2]
    src_corners = np.float32([[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]).reshape(-1, 1, 2)

    try:
        proj_corners = cv2.perspectiveTransform(src_corners, H).reshape(-1, 2)
    except Exception as e:
        cv2.putText(vis_img, f"CORNER PROJECTION ERROR: {e}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return vis_img

    pts_int = np.int32(proj_corners)

    # Draw polygon border connecting projected corners
    cv2.polylines(vis_img, [pts_int], isClosed=True, color=(255, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    # Mark & label the 4 corner points
    labels = ["C1 (Top-Left)", "C2 (Top-Right)", "C3 (Bot-Right)", "C4 (Bot-Left)"]
    colors = [(0, 255, 255), (0, 200, 255), (0, 255, 0), (255, 100, 0)]

    for idx, (pt, label, col) in enumerate(zip(pts_int, labels, colors)):
        cv2.circle(vis_img, (pt[0], pt[1]), 7, col, -1)
        cv2.putText(vis_img, label, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_img, label, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    cv2.putText(vis_img, "PROJECTED SOURCE BOUNDING BOX OVER REFERENCE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return vis_img


def create_before_after_comparison(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    registered_img: np.ndarray
) -> np.ndarray:
    """
    Creates a 2x2 side-by-side panel comparing BEFORE vs AFTER registration.
    """
    h_ref, w_ref = reference_img.shape[:2]

    source_resized = cv2.resize(source_img, (w_ref, h_ref))
    reg_resized = cv2.resize(registered_img, (w_ref, h_ref))

    if len(source_resized.shape) == 2:
        source_resized = cv2.cvtColor(source_resized, cv2.COLOR_GRAY2BGR)
    if len(reference_img.shape) == 2:
        reference_img_bgr = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)
    else:
        reference_img_bgr = reference_img.copy()
    if len(reg_resized.shape) == 2:
        reg_resized = cv2.cvtColor(reg_resized, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)

    top_left = source_resized.copy()
    top_right = reference_img_bgr.copy()
    cv2.putText(top_left, "BEFORE: Source (Moving)", (20, 40), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(top_right, "BEFORE: Reference Target", (20, 40), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    bot_left = reg_resized.copy()
    bot_right = reference_img_bgr.copy()
    cv2.putText(bot_left, "AFTER: Registered Source (Warped)", (20, 40), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
    cv2.putText(bot_right, "AFTER: Reference Target", (20, 40), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

    top_row = np.hstack([top_left, top_right])
    bot_row = np.hstack([bot_left, bot_right])

    panel = np.vstack([top_row, bot_row])
    return panel


def create_difference_image(
    registered_img: np.ndarray,
    reference_img: np.ndarray
) -> np.ndarray:
    """
    Calculates normalized absolute intensity difference between registered source and reference.
    """
    if len(registered_img.shape) == 3:
        reg_gray = cv2.cvtColor(registered_img, cv2.COLOR_BGR2GRAY)
    else:
        reg_gray = registered_img.copy()

    if len(reference_img.shape) == 3:
        ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = reference_img.copy()

    diff = cv2.absdiff(reg_gray, ref_gray)
    border_mask = (reg_gray > 0).astype(np.uint8)
    diff = diff * border_mask

    diff_norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    heatmap[border_mask == 0] = [0, 0, 0]

    return heatmap
