# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 NUMINA Authors. All rights reserved.
"""
NUMINA: Training-free numerical alignment for text-to-video diffusion models.

Usage:
    from numina import NuminaConfig, NuminaInput

    numina_input = NuminaInput.from_noun_counts(
        prompt="Three cats chasing two dogs",
        noun_counts={"cats": 3, "dogs": 2},
        seed=42,
    )
"""

from .config import NuminaConfig, NuminaInput, NuminaTarget

__all__ = [
    "NuminaConfig",
    "NuminaInput",
    "NuminaTarget",
]
