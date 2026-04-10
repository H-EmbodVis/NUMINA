# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 NUMINA Authors. All rights reserved.
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import NuminaConfig


@dataclass
class FrameModulationInfo:
    """Modulation instructions for a single frame and single noun category."""
    removed_pixels: np.ndarray      # [N_rem] int, 0-based into H*W
    added_pixels: np.ndarray        # [N_add] int
    has_reference: bool = False
    reference_pixels: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )


@dataclass
class NounModulationInfo:
    """Modulation instructions for one noun across all frames."""
    noun: str
    token_indices: List[int]
    frames: List[FrameModulationInfo]


@dataclass
class ModulationData:
    """Complete modulation data for the regeneration phase."""
    nouns: List[NounModulationInfo]
    config: NuminaConfig
    num_frames: int = 0
    H: int = 0
    W: int = 0


def build_modulation_data(
    refined_layouts: Dict[str, List[np.ndarray]],
    targets: Dict,
    num_frames: int,
    H: int,
    W: int,
    config: NuminaConfig,
) -> ModulationData:

    nouns_info = []

    for noun, frame_layouts in refined_layouts.items():
        target = targets[noun]
        token_indices = target.token_indices

        frames_info = []
        for f in range(num_frames):
            layout = frame_layouts[f]
            flat = layout.flatten()

            removed_pixels = np.where(flat == config.LABEL_REMOVED)[0]
            added_pixels = np.where(flat >= config.LABEL_ADDED_BASE)[0]

            # Find smallest existing region as reference
            existing_pixels = np.where((flat >= 1) & (flat <= 99))[0]
            has_reference = len(existing_pixels) > 0

            reference_pixels = np.array([], dtype=np.int64)
            if has_reference and len(added_pixels) > 0:
                region_ids = sorted(set(flat[(flat >= 1) & (flat <= 99)].tolist()))
                if region_ids:
                    smallest_size = float('inf')
                    smallest_pixels = None
                    for rid in region_ids:
                        rpx = np.where(flat == rid)[0]
                        if len(rpx) < smallest_size:
                            smallest_size = len(rpx)
                            smallest_pixels = rpx
                    if smallest_pixels is not None:
                        reference_pixels = smallest_pixels

            frames_info.append(FrameModulationInfo(
                removed_pixels=removed_pixels,
                added_pixels=added_pixels,
                has_reference=has_reference,
                reference_pixels=reference_pixels,
            ))

        nouns_info.append(NounModulationInfo(
            noun=noun,
            token_indices=token_indices,
            frames=frames_info,
        ))

    return ModulationData(
        nouns=nouns_info, config=config,
        num_frames=num_frames, H=H, W=W,
    )


def build_cross_attention_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    modulation_data: ModulationData,
    step_index: int,
) -> Optional[torch.Tensor]:

    config = modulation_data.config
    delta = config.delta(step_index)

    if delta <= config.modulation_min_delta:
        return None

    device = q.device
    B, N, L_q, D = q.shape
    L_k = k.shape[2]
    tokens_per_frame = modulation_data.H * modulation_data.W
    scale = 1.0 / (D ** 0.5)

    # Allocate bias tensor — zeros mean no modification
    bias = torch.zeros(1, 1, L_q, L_k, dtype=torch.float32, device=device)

    for noun_info in modulation_data.nouns:
        tok_idx = torch.tensor(
            noun_info.token_indices, dtype=torch.long, device=device
        )
        n_tok = tok_idx.shape[0]

        for f, frame_info in enumerate(noun_info.frames):
            offset = f * tokens_per_frame

            # --- Suppression: removed regions ---
            if len(frame_info.removed_pixels) > 0:
                rem_idx = torch.from_numpy(
                    frame_info.removed_pixels.astype(np.int64) + offset
                ).to(device)
                suppress_val = -config.suppress_scale * delta
                # bias[0, 0, rem, tok] += suppress_val
                bias[0, 0, rem_idx.unsqueeze(1), tok_idx.unsqueeze(0)] += suppress_val

            # --- Addition: no reference (circle template) ---
            if (len(frame_info.added_pixels) > 0
                    and not frame_info.has_reference):
                add_idx = torch.from_numpy(
                    frame_info.added_pixels.astype(np.int64) + offset
                ).to(device)
                boost_val = config.circle_bias_k * delta
                bias[0, 0, add_idx.unsqueeze(1), tok_idx.unsqueeze(0)] += boost_val

            # --- Addition: with reference ---
            if (len(frame_info.added_pixels) > 0
                    and frame_info.has_reference
                    and len(frame_info.reference_pixels) > 0):
                ref_idx = torch.from_numpy(
                    frame_info.reference_pixels.astype(np.int64) + offset
                ).to(device)
                add_idx = torch.from_numpy(
                    frame_info.added_pixels.astype(np.int64) + offset
                ).to(device)

                # Partial matmul to get mean reference score.
                # q: [B, N, L_q, D],  k: [B, N, L_k, D]
                # ref_q: [B, N, n_ref, D],  tok_k: [B, N, n_tok, D]
                with torch.no_grad():
                    ref_q = q[:, :, ref_idx, :]     # [B, N, n_ref, D]
                    tok_k = k[:, :, tok_idx, :]     # [B, N, n_tok, D]

                    # [B, N, n_ref, n_tok] — these are the UNSCALED scores.
                    # SDPA internally computes QK^T * scale, so the actual
                    # pre-softmax value at position (i,j) = q[i]·k[j] * scale.
                    ref_scores = torch.matmul(ref_q, tok_k.transpose(-2, -1))
                    ref_scores_scaled = ref_scores * scale
                    # Mean across batch, heads, ref pixels, tok tokens → scalar
                    mean_ref = ref_scores_scaled.mean().item()

                    # Now compute the added pixels' scores to cancel them out.
                    add_q = q[:, :, add_idx, :]     # [B, N, n_add, D]
                    # [B, N, n_add, n_tok]
                    add_scores = torch.matmul(add_q, tok_k.transpose(-2, -1))
                    add_scores_scaled = add_scores * scale
                    # Average across batch and heads → [n_add, n_tok]
                    add_scores_mean = add_scores_scaled.mean(dim=(0, 1))

                # bias = target - original_score
                # SDPA will compute: original_score + bias = target
                target = mean_ref * delta
                # add_scores_mean: [n_add, n_tok]
                bias_values = target - add_scores_mean  # [n_add, n_tok]

                # Write into bias tensor
                bias[0, 0, add_idx.unsqueeze(1), tok_idx.unsqueeze(0)] = (
                    bias_values.float()
                )

    return bias


def has_any_modulation(modulation_data: ModulationData) -> bool:
    """Check whether any actual changes exist (any removed or added pixels)."""
    for noun_info in modulation_data.nouns:
        for frame_info in noun_info.frames:
            if len(frame_info.removed_pixels) > 0 or len(frame_info.added_pixels) > 0:
                return True
    return False
