# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 NUMINA Authors. All rights reserved.
"""
Token mapper: resolve user-provided noun strings to their token indices
in the T5-encoded prompt.

Wan2.1 uses the google/umt5-xxl SentencePiece tokenizer.  A single noun
may correspond to multiple subword tokens (e.g. "excavators" -> ["▁ex",
"cav", "ators"]).  This module finds the exact positions of each noun's
subword sequence within the full prompt's token ID list.

Usage:
    from numina.token_mapper import map_noun_tokens

    # tokenizer is the HuggingfaceTokenizer instance from T5EncoderModel
    numina_input = map_noun_tokens(numina_input, tokenizer)
    # Now numina_input.targets["cats"].token_indices == [5] (example)
"""

import logging
from typing import List, Optional, Sequence

from .config import NuminaInput

logger = logging.getLogger(__name__)


def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """
    Find the starting index of `needle` in `haystack`.

    Returns the index of the first occurrence, or None if not found.
    """
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return None


def _find_all_subsequences(haystack: List[int], needle: List[int]) -> List[int]:
    """
    Find ALL starting indices of `needle` in `haystack`.
    """
    n, m = len(haystack), len(needle)
    results = []
    if m == 0 or m > n:
        return results
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            results.append(i)
    return results


def map_noun_tokens(
    numina_input: NuminaInput,
    tokenizer,
) -> NuminaInput:

    prompt = numina_input.prompt

    # --- Step 1: Tokenize full prompt exactly as Wan2.1 does ---
    # The HuggingfaceTokenizer applies cleaning then calls the underlying
    # AutoTokenizer with add_special_tokens.  We replicate this.
    if hasattr(tokenizer, '_clean') and tokenizer.clean:
        cleaned_prompt = tokenizer._clean(prompt)
    else:
        cleaned_prompt = prompt

    underlying = tokenizer.tokenizer  # the AutoTokenizer
    prompt_ids: List[int] = underlying.encode(
        cleaned_prompt, add_special_tokens=True
    )

    logger.info(f"[NUMINA TokenMapper] Prompt: '{cleaned_prompt}'")
    logger.info(f"[NUMINA TokenMapper] Prompt token IDs ({len(prompt_ids)}): {prompt_ids}")

    # --- Step 2: For each noun, find its token subsequence ---
    for noun, target in numina_input.targets.items():
        noun_indices = _resolve_noun(underlying, prompt_ids, noun, cleaned_prompt)

        if noun_indices is None:
            raise ValueError(
                f"[NUMINA TokenMapper] Could not find noun '{noun}' in the "
                f"tokenized prompt. Prompt tokens: {prompt_ids}, "
                f"Decoded: {[underlying.decode([tid]) for tid in prompt_ids]}"
            )

        target.token_indices = noun_indices
        decoded = [underlying.decode([prompt_ids[i]]) for i in noun_indices]
        logger.info(
            f"[NUMINA TokenMapper] Noun '{noun}' -> indices {noun_indices} "
            f"(tokens: {decoded})"
        )

    return numina_input


def _resolve_noun(
    underlying_tokenizer,
    prompt_ids: List[int],
    noun: str,
    cleaned_prompt: str,
) -> Optional[List[int]]:

    # Strategy 1: space-prefixed (most common case — noun appears mid-sentence)
    noun_ids_spaced = underlying_tokenizer.encode(
        " " + noun, add_special_tokens=False
    )
    if noun_ids_spaced:
        start = _find_subsequence(prompt_ids, noun_ids_spaced)
        if start is not None:
            return list(range(start, start + len(noun_ids_spaced)))

    # Strategy 2: no space prefix (sentence-initial or after special chars)
    noun_ids_bare = underlying_tokenizer.encode(
        noun, add_special_tokens=False
    )
    if noun_ids_bare:
        start = _find_subsequence(prompt_ids, noun_ids_bare)
        if start is not None:
            return list(range(start, start + len(noun_ids_bare)))

    # Strategy 3: character-level alignment fallback.
    # Decode each token, find where the noun text appears in the
    # concatenated decoded strings, then map back to token indices.
    result = _char_alignment_fallback(underlying_tokenizer, prompt_ids, noun)
    if result is not None:
        return result

    return None


def _char_alignment_fallback(
    tokenizer,
    prompt_ids: List[int],
    noun: str,
) -> Optional[List[int]]:
    """
    Fallback: decode tokens one-by-one, build a character-to-token-index
    map, find the noun substring, and return the covering token indices.
    """
    # Build decoded text with character -> token index mapping
    decoded_parts = []
    char_to_token = []

    for idx, tid in enumerate(prompt_ids):
        part = tokenizer.decode([tid])
        decoded_parts.append(part)
        char_to_token.extend([idx] * len(part))

    full_decoded = "".join(decoded_parts)

    # Search for noun (case-insensitive) in the decoded text
    lower_decoded = full_decoded.lower()
    lower_noun = noun.lower()

    pos = lower_decoded.find(lower_noun)
    if pos == -1:
        # Also try with leading space
        pos = lower_decoded.find(" " + lower_noun)
        if pos != -1:
            pos += 1  # skip the space

    if pos == -1:
        return None

    # Collect the unique token indices that span this character range
    token_indices_set = set()
    for char_idx in range(pos, pos + len(noun)):
        if char_idx < len(char_to_token):
            token_indices_set.add(char_to_token[char_idx])

    if not token_indices_set:
        return None

    # Return sorted indices
    return sorted(token_indices_set)
