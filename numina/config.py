# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 NUMINA Authors. All rights reserved.

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NuminaConfig:
    """
    Complete NUMINA configuration.

    Attributes are grouped by pipeline phase:
      - General: shared across phases
      - Phase 1: Numerical misalignment identification
      - Phase 2: Numerically aligned video generation
    """
    # Total denoising steps (must match the sampling_steps used in generation)
    total_steps: int = 50

    # Temporary directory for attention extraction artifacts
    temp_dir: str = "/tmp/numina"

    reference_step: int = 20
    reference_layer: int = 15

    sa_alpha: float = 1.0
    sa_beta: float = 1.0
    sa_gamma: float = 0.1
    sa_block_grid: int = 8

    meanshift_bandwidth: Optional[float] = None
    meanshift_quantile: float = 0.2

    dbscan_eps: float = 3.0
    dbscan_min_samples: int = 5

    cross_attn_peak_ratio: float = 0.05

    semantic_overlap_tau: float = 0.2

    circle_radius: int = 5

    lambda_center: float = 1.0
    lambda_temporal: float = 8.0

    # Grid search step size (pixels) for placement candidates
    placement_grid_step: int = 2

    # --- Layout label conventions ---
    # 0          : background
    # 1 .. 99    : existing detected instances
    # -1         : removed regions (marked for suppression)
    # 100+       : added regions (marked for boosting)
    LABEL_BACKGROUND: int = 0
    LABEL_REMOVED: int = -1
    LABEL_ADDED_BASE: int = 100

    # --- Attention modulation ---
    # Modulation intensity function: delta(t) = -3 * tanh(t - 25) + 4
    # where t is the denoising step index (0-based).
    modulation_center: float = 25.0
    modulation_amplitude: float = 3.0
    modulation_offset: float = 4.0
    modulation_min_delta: float = 0.25

    suppress_scale: float = 10.0

    circle_bias_k: float = 0.8

    # Enable EasyCache for Phase 1 (skip redundant denoising steps)
    easycache_enabled: bool = True

    # Tolerance threshold τ: accumulated error must stay below this
    # to keep reusing the cached transformation vector.
    # Wan2.1 recommended: 5% (0.05).  Higher = more skipping = faster
    easycache_tau: float = 0.05

    # Number of initial warm-up steps R where full computation is
    # mandatory (the transformation rate is unstable early on).
    # Wan2.1 recommended: 10
    easycache_warmup: int = 10


    # How often to call torch.cuda.empty_cache() during head-by-head
    # processing (every N heads).  Lower = more cache clearing = slower
    # but less peak VRAM.
    cache_clear_interval: int = 12

    def delta(self, step_index: int) -> float:
        t = float(step_index)
        return ( -self.modulation_amplitude * math.tanh(
            t - self.modulation_center
        ) + self.modulation_offset ) * self.modulation_min_delta

    # Modulation is only applied when delta(t) > 1
    # Below this, the model routes to FlashAttention entirely (zero overhead).
    def should_modulate(self, step_index: int) -> bool:
        """Whether modulation should be active at this step."""
        return self.delta(step_index) > self.modulation_min_delta


@dataclass
class NuminaTarget:

    noun: str
    target_count: int
    token_indices: List[int] = field(default_factory=list)


@dataclass
class NuminaInput:

    prompt: str
    targets: Dict[str, NuminaTarget] = field(default_factory=dict)
    seed: int = 1
    config: NuminaConfig = field(default_factory=NuminaConfig)

    @classmethod
    def from_noun_counts(
        cls,
        prompt: str,
        noun_counts: Dict[str, int],
        seed: int = 1,
        **config_overrides,
    ) -> "NuminaInput":

        cfg = NuminaConfig(**config_overrides) if config_overrides else NuminaConfig()
        targets = {
            noun: NuminaTarget(noun=noun, target_count=count)
            for noun, count in noun_counts.items()
        }
        return cls(prompt=prompt, targets=targets, seed=seed, config=cfg)
