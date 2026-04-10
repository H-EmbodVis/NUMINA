# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 NUMINA Authors. All rights reserved.
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.utils.extmath import randomized_svd

from .config import NuminaConfig



def _cluster_self_attention(
    sa_map: np.ndarray,
    H: int,
    W: int,
    config: NuminaConfig,
) -> np.ndarray:

    mean = sa_map.mean(axis=0, keepdims=True)
    centred = sa_map - mean

    try:
        U, S, Vt = randomized_svd(centred, n_components=3, random_state=0)
        features = U * S  # [H*W, 3]
    except Exception:
        features = centred[:, :3]

    # Normalise features to [0, 1]
    for c in range(features.shape[1]):
        col = features[:, c]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin > 1e-8:
            features[:, c] = (col - cmin) / (cmax - cmin)

    # Add normalised spatial coordinates to encourage contiguous regions
    coords = np.stack(np.meshgrid(
        np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij'
    ), axis=-1).reshape(-1, 2)  # [H*W, 2]

    features_with_coords = np.concatenate([features, coords * 0.5], axis=1)

    bandwidth = config.meanshift_bandwidth
    if bandwidth is None:
        bandwidth = estimate_bandwidth(
            features_with_coords,
            quantile=config.meanshift_quantile,
            n_samples=min(1000, len(features_with_coords)),
        )
        if bandwidth <= 0:
            bandwidth = 0.5

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=1)
    labels = ms.fit_predict(features_with_coords)

    return labels


def _build_focus_mask(
    ca_map_2d: np.ndarray,
    config: NuminaConfig,
) -> np.ndarray:

    H, W = ca_map_2d.shape
    peak_val = ca_map_2d.max()
    threshold = config.cross_attn_peak_ratio * peak_val

    above = ca_map_2d >= threshold
    coords = np.argwhere(above)  # [N, 2]

    if len(coords) == 0:
        return np.zeros((H, W), dtype=bool)

    db = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples)
    cluster_labels = db.fit_predict(coords)

    focus_mask = np.zeros((H, W), dtype=bool)
    for idx, label in enumerate(cluster_labels):
        if label >= 0:
            r, c = coords[idx]
            focus_mask[r, c] = True

    return focus_mask


def construct_layout_single_frame(
    sa_map: np.ndarray,
    ca_map_2d: np.ndarray,
    H: int,
    W: int,
    config: NuminaConfig,
) -> Tuple[np.ndarray, int]:

    cluster_labels = _cluster_self_attention(sa_map, H, W, config)
    cluster_labels_2d = cluster_labels.reshape(H, W)

    # Step 2: Focus mask from cross-attention
    focus_mask = _build_focus_mask(ca_map_2d, config)

    # Step 3: Filter proposals by semantic overlap >= tau
    unique_clusters = sorted(set(cluster_labels.tolist()))
    layout = np.zeros((H, W), dtype=np.int32)
    instance_id = 0

    for cl in unique_clusters:
        region_mask = (cluster_labels_2d == cl)
        region_area = region_mask.sum()
        if region_area == 0:
            continue

        overlap = (region_mask & focus_mask).sum()
        overlap_ratio = overlap / region_area

        if overlap_ratio >= config.semantic_overlap_tau:
            instance_id += 1
            layout[region_mask] = instance_id

    return layout, instance_id


def construct_layouts(
    head_selection_result: Dict,
    targets: Dict,
    num_frames: int,
    H: int,
    W: int,
    config: NuminaConfig,
) -> Dict[str, Dict]:

    all_layouts = {}

    for noun, target in targets.items():
        frames_layouts = []
        frames_counts = []

        for f in range(num_frames):
            # Get selected maps for this frame
            _, sa_map = head_selection_result['self_attn'][f]
            _, ca_map_2d = head_selection_result['cross_attn'][noun][f]

            layout, count = construct_layout_single_frame(
                sa_map, ca_map_2d, H, W, config
            )
            frames_layouts.append(layout)
            frames_counts.append(count)

        all_layouts[noun] = {
            'layouts': frames_layouts,
            'counts': frames_counts,
            'target_count': target.target_count,
        }

    return all_layouts


def _get_regions(layout: np.ndarray) -> List[Tuple[int, np.ndarray]]:

    regions = []
    for inst_id in sorted(set(layout.flat) - {0}):
        if inst_id <= 0:
            continue
        mask = (layout == inst_id)
        regions.append((inst_id, mask))
    regions.sort(key=lambda x: x[1].sum())
    return regions


def _remove_smallest(layout: np.ndarray, label_removed: int) -> np.ndarray:
    """Remove the smallest region, marking its pixels as label_removed."""
    regions = _get_regions(layout)
    if not regions:
        return layout
    inst_id, mask = regions[0]
    layout = layout.copy()
    layout[mask] = label_removed
    return layout


def _create_circle_template(radius: int) -> np.ndarray:
    """Create a boolean circle template of given radius."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y) <= radius * radius
    return mask


def _compute_layout_center(layout: np.ndarray) -> Tuple[float, float]:
    """Compute the geometric center of all foreground pixels."""
    fg = layout > 0
    if not fg.any():
        H, W = layout.shape
        return H / 2.0, W / 2.0
    coords = np.argwhere(fg)
    return coords[:, 0].mean(), coords[:, 1].mean()


def _placement_cost(
    template: np.ndarray,
    cy: int,
    cx: int,
    layout: np.ndarray,
    center_y: float,
    center_x: float,
    prev_cy: Optional[float],
    prev_cx: Optional[float],
    config: NuminaConfig,
) -> float:

    H, W = layout.shape
    th, tw = template.shape
    # Template bounding box
    y0 = cy - th // 2
    x0 = cx - tw // 2
    y1 = y0 + th
    x1 = x0 + tw

    # Out of bounds -> infinite cost
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return float('inf')

    # C_o: overlap with existing layout (any non-background, non-removed)
    region = layout[y0:y1, x0:x1]
    overlap_mask = template & ((region > 0) | (region >= config.LABEL_ADDED_BASE))
    C_o = float(overlap_mask.sum())

    # C_c: distance from layout center
    C_c = (cy - center_y) ** 2 + (cx - center_x) ** 2

    # C_t: temporal consistency (distance from previous frame position)
    C_t = 0.0
    if prev_cy is not None and prev_cx is not None:
        C_t = (cy - prev_cy) ** 2 + (cx - prev_cx) ** 2

    return C_o + config.lambda_center * C_c + config.lambda_temporal * C_t


def _add_instance(
    layout: np.ndarray,
    new_label: int,
    prev_center: Optional[Tuple[float, float]],
    config: NuminaConfig,
) -> Tuple[np.ndarray, Tuple[float, float]]:

    H, W = layout.shape
    layout = layout.copy()

    # Build template
    regions = _get_regions(layout)
    if regions:
        # Copy smallest existing region as template
        _, smallest_mask = regions[0]
        ys, xs = np.where(smallest_mask)
        ty_min, ty_max = ys.min(), ys.max()
        tx_min, tx_max = xs.min(), xs.max()
        template = smallest_mask[ty_min:ty_max + 1, tx_min:tx_max + 1]
    else:
        template = _create_circle_template(config.circle_radius)

    center_y, center_x = _compute_layout_center(layout)
    prev_cy = prev_center[0] if prev_center else None
    prev_cx = prev_center[1] if prev_center else None

    # Grid search for optimal placement
    step = config.placement_grid_step
    th, tw = template.shape
    best_cost = float('inf')
    best_cy, best_cx = H // 2, W // 2

    for cy in range(th // 2, H - th // 2 + 1, step):
        for cx in range(tw // 2, W - tw // 2 + 1, step):
            cost = _placement_cost(
                template, cy, cx, layout,
                center_y, center_x,
                prev_cy, prev_cx,
                config,
            )
            if cost < best_cost:
                best_cost = cost
                best_cy, best_cx = cy, cx

    # Place template
    y0 = best_cy - th // 2
    x0 = best_cx - tw // 2
    for dy in range(th):
        for dx in range(tw):
            if template[dy, dx]:
                layout[y0 + dy, x0 + dx] = new_label

    return layout, (float(best_cy), float(best_cx))


def refine_layout_single_frame(
    layout: np.ndarray,
    detected_count: int,
    target_count: int,
    frame_idx: int,
    prev_added_centers: Optional[List[Tuple[float, float]]],
    config: NuminaConfig,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:

    layout = layout.copy()
    current_count = detected_count
    added_centers = []

    # Object removal: remove smallest instances
    while current_count > target_count:
        layout = _remove_smallest(layout, config.LABEL_REMOVED)
        current_count -= 1

    # Object addition
    add_label = config.LABEL_ADDED_BASE
    add_idx = 0
    while current_count < target_count:
        prev_center = None
        if prev_added_centers is not None and add_idx < len(prev_added_centers):
            prev_center = prev_added_centers[add_idx]

        layout, placed_center = _add_instance(
            layout, add_label, prev_center, config
        )
        added_centers.append(placed_center)
        add_label += 1
        add_idx += 1
        current_count += 1

    return layout, added_centers


def refine_all_layouts(
    layout_data: Dict[str, Dict],
    config: NuminaConfig,
) -> Dict[str, List[np.ndarray]]:

    refined = {}

    for noun, data in layout_data.items():
        layouts = data['layouts']
        counts = data['counts']
        target = data['target_count']
        num_frames = len(layouts)

        refined_frames = []
        prev_added_centers = None

        for f in range(num_frames):
            ref_layout, added_centers = refine_layout_single_frame(
                layouts[f], counts[f], target, f,
                prev_added_centers, config,
            )
            refined_frames.append(ref_layout)
            prev_added_centers = added_centers if added_centers else None

        refined[noun] = refined_frames

    return refined
