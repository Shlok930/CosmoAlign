"""
Descriptor Matching and Match Filtering Module for CosmoAlign.

Provides:
- KNN descriptor matching (k=2) with Euclidean (L2) distance
- Lowe's Ratio Test filtering for ambiguous matches
- Mutual Nearest-Neighbor (MNN) bidirectional cross-check filtering
- Side-by-side match visualization drawing functions
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


def match_descriptors_knn(
    desc_source: np.ndarray,
    desc_ref: np.ndarray,
    k: int = 2
) -> List[List[cv2.DMatch]]:
    """
    Finds k-nearest neighbors in the reference descriptors for each source descriptor.
    """
    if desc_source is None or desc_ref is None or len(desc_source) == 0 or len(desc_ref) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desc_source, desc_ref, k=k)
    return raw_matches


def filter_matches_lowe(
    raw_matches: List[List[cv2.DMatch]],
    ratio_threshold: float = 0.75
) -> List[cv2.DMatch]:
    """
    Filters raw KNN matches using Lowe's Ratio Test (d1 / d2 < ratio_threshold).
    """
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) >= 2:
            m, n = match_pair[0], match_pair[1]
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    return good_matches


def filter_matches_mutual_nn(
    desc_source: np.ndarray,
    desc_ref: np.ndarray
) -> List[cv2.DMatch]:
    """
    Filters descriptor matches using Mutual Nearest-Neighbor (MNN) cross-checking.

    A match between source descriptor i and reference descriptor j is retained ONLY if:
    1. j is the closest reference descriptor to i (Source -> Ref), AND
    2. i is the closest source descriptor to j (Ref -> Source).

    Helps eliminate one-directional false correspondences in repetitive terrain.
    """
    if desc_source is None or desc_ref is None or len(desc_source) == 0 or len(desc_ref) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Forward matching: Source -> Ref (k=1)
    fwd_matches = bf.match(desc_source, desc_ref)
    
    # Backward matching: Ref -> Source (k=1)
    bwd_matches = bf.match(desc_ref, desc_source)

    # Map reference index -> best source index
    bwd_dict = {m.queryIdx: m.trainIdx for m in bwd_matches}

    mutual_matches = []
    for m in fwd_matches:
        src_idx = m.queryIdx
        ref_idx = m.trainIdx
        # Check bidirectional consistency
        if bwd_dict.get(ref_idx) == src_idx:
            mutual_matches.append(m)

    return mutual_matches


def draw_matches_visualization(
    img_source: np.ndarray,
    kp_source: List[cv2.KeyPoint],
    img_ref: np.ndarray,
    kp_ref: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inliers_mask: Optional[np.ndarray] = None,
    max_matches_to_draw: Optional[int] = 100
) -> np.ndarray:
    """
    Generates side-by-side visualization of matched points.
    """
    matches_to_draw = matches
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

    if inliers_mask is not None:
        match_mask = inliers_mask.ravel().tolist()
        if max_matches_to_draw and len(matches) > max_matches_to_draw:
            matches_to_draw = matches[:max_matches_to_draw]
            match_mask = match_mask[:max_matches_to_draw]

        vis_img = cv2.drawMatches(
            img_source,
            kp_source,
            img_ref,
            kp_ref,
            matches_to_draw,
            None,
            matchColor=(0, 255, 0),       # Green for inliers
            singlePointColor=(0, 0, 255), # Red for outliers
            matchesMask=match_mask,
            flags=flags
        )
    else:
        if max_matches_to_draw and len(matches) > max_matches_to_draw:
            matches_to_draw = matches[:max_matches_to_draw]

        vis_img = cv2.drawMatches(
            img_source,
            kp_source,
            img_ref,
            kp_ref,
            matches_to_draw,
            None,
            matchColor=(0, 255, 255),
            flags=flags
        )

    return vis_img
