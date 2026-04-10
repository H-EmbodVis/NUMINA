# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 NUMINA Authors. All rights reserved.
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F_torch
from sklearn.utils.extmath import randomized_svd

from .config import NuminaConfig

logger = logging.getLogger(__name__)

# Pre-built Sobel kernels (created once, moved to device on first use)
_SOBEL_X = torch.tensor(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
).reshape(1, 1, 3, 3)
_SOBEL_Y = torch.tensor(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
).reshape(1, 1, 3, 3)


def score_sa_head_gpu(
    attn_frame: torch.Tensor,
    H: int,
    W: int,
    config: NuminaConfig,
) -> float:

    global _SOBEL_X, _SOBEL_Y
    device = attn_frame.device

    # --- PCA to grayscale ---
    # Centre rows
    mean = attn_frame.mean(dim=0, keepdim=True)
    centred = attn_frame - mean

    # Truncated SVD: torch.svd_lowrank with q=3 is O(n² * q)
    try:
        U, S, V = torch.svd_lowrank(centred, q=3)
        components = U * S.unsqueeze(0)  # [H*W, 3]
    except Exception:
        components = centred[:, :3]

    # Normalise each channel to [0, 1]
    for c in range(min(3, components.shape[1])):
        col = components[:, c]
        cmin = col.min()
        cmax = col.max()
        rng = cmax - cmin
        if rng > 1e-8:
            components[:, c] = (col - cmin) / rng
        else:
            components[:, c] = 0.5

    # Pad if fewer than 3 components
    if components.shape[1] < 3:
        pad = torch.full(
            (components.shape[0], 3 - components.shape[1]),
            0.5, device=device,
        )
        components = torch.cat([components, pad], dim=1)

    # RGB -> grayscale
    gray = (0.299 * components[:, 0]
            + 0.587 * components[:, 1]
            + 0.114 * components[:, 2])  # [H*W]

    gray_2d = gray.reshape(1, 1, H, W)  # for conv2d

    # --- S1: foreground-background separation = std dev ---
    s1 = gray.std().item()

    # --- S2: structural richness (block variance) ---
    bg = config.sa_block_grid
    bh = max(1, H // bg)
    bw = max(1, W // bg)
    # Use avg_pool2d to compute block means, then derive sums
    gray_for_pool = gray.reshape(1, 1, H, W)
    # Crop to be divisible by block size
    H_crop = (H // bh) * bh
    W_crop = (W // bw) * bw
    cropped = gray_for_pool[:, :, :H_crop, :W_crop]
    # Pool to get block means, then multiply by block area for sums
    block_means = F_torch.avg_pool2d(cropped, kernel_size=(bh, bw), stride=(bh, bw))
    block_sums = block_means * (bh * bw)
    s2 = block_sums.var().item()

    # --- S3: edge clarity (Sobel gradient magnitude) ---
    if _SOBEL_X.device != device:
        _SOBEL_X = _SOBEL_X.to(device)
        _SOBEL_Y = _SOBEL_Y.to(device)
    sx = F_torch.conv2d(gray_2d, _SOBEL_X, padding=1)
    sy = F_torch.conv2d(gray_2d, _SOBEL_Y, padding=1)
    mag = torch.sqrt(sx * sx + sy * sy)
    s3 = mag.mean().item()

    score = config.sa_alpha * s1 + config.sa_beta * s2 + config.sa_gamma * s3
    return score


def pca_to_grayscale_cpu(attn_map: np.ndarray) -> np.ndarray:

    mean = attn_map.mean(axis=0, keepdims=True)
    centred = attn_map - mean

    try:
        U, S, Vt = randomized_svd(centred, n_components=3, random_state=0)
        components = U * S
    except Exception:
        components = centred[:, :3]

    for c in range(min(3, components.shape[1])):
        col = components[:, c]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin > 1e-8:
            components[:, c] = (col - cmin) / (cmax - cmin)
        else:
            components[:, c] = 0.5

    if components.shape[1] < 3:
        pad = np.full((components.shape[0], 3 - components.shape[1]), 0.5)
        components = np.concatenate([components, pad], axis=1)

    grayscale = (
        0.299 * components[:, 0]
        + 0.587 * components[:, 1]
        + 0.114 * components[:, 2]
    )
    return grayscale
