# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 NUMINA Authors. All rights reserved.
import math

import torch
import torch.nn.functional as F_torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'numina_self_attention_extract',
    'numina_cross_attention_extract',
    'numina_cross_attention_modulate',
]



def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out



USE_MEAN = True

def numina_self_attention_extract(
    q,
    k,
    v,
    real_video_len,
    num_frames,
    tokens_per_frame,
    storage_dict,
    numina_config=None,
    cache_clear_interval=4,
):

    from numina.head_selection import score_sa_head_gpu

    B, L, N, D = q.shape
    out_dtype = q.dtype
    device = q.device
    scale = 1.0 / math.sqrt(D)

    # Stay in native dtype (bf16) for matmuls — only promote small slices
    # for SVD scoring.  This halves VRAM vs the old float32 approach.
    q_t = q.transpose(1, 2)       # [B, N, L, D] in native dtype
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    output = torch.zeros(B, N, L, D, dtype=out_dtype, device=device)

    # Retrieve H, W from storage_dict metadata (set by model.py)
    H = storage_dict.get('_H', None)
    W = storage_dict.get('_W', None)
    can_score = (numina_config is not None and H is not None and W is not None)

    if 'self_attn' not in storage_dict:
        storage_dict['self_attn'] = {}

    # Frame-outer loop: compare all heads per frame to find best
    for f in range(num_frames):
        fs = f * tokens_per_frame
        fe = fs + tokens_per_frame

        best_head = -1
        best_score = -float('inf')
        best_map_cpu = None

        for h in range(N):
            q_f = q_t[:, h:h+1, fs:fe, :]   # [B, 1, tpf, D]
            k_f = k_t[:, h:h+1, fs:fe, :]
            v_f = v_t[:, h:h+1, fs:fe, :]

            # Pre-softmax: [B, 1, tpf, tpf]
            scores_f = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
            attn_weights = F_torch.softmax(scores_f, dim=-1)

            # Compute output for this head and frame
            out_f = torch.matmul(attn_weights, v_f)
            output[:, h:h+1, fs:fe, :] = out_f

            # Score on GPU and track best
            if can_score:
                # Promote only the small [tpf, tpf] slice to float32 for SVD
                attn_2d = attn_weights[0, 0].float()  # [tpf, tpf]
                sc = score_sa_head_gpu(attn_2d, H, W, numina_config)
                if sc > best_score:
                    best_score = sc
                    best_head = h
                    # Offload only if this is the new winner
                    best_map_cpu = attn_2d.detach().cpu().numpy()

            del q_f, k_f, v_f, scores_f, attn_weights, out_f

        # Store winning head for this frame
        if best_map_cpu is not None:
            storage_dict['self_attn'][f] = (best_head, best_map_cpu)

        if (f + 1) % 4 == 0:
            torch.cuda.empty_cache()

    output = output.transpose(1, 2).contiguous().to(out_dtype)
    del q_t, k_t, v_t
    torch.cuda.empty_cache()

    return output


def numina_cross_attention_extract(
    q,
    k,
    v,
    real_video_len,
    storage_dict,
    token_indices_per_noun=None,
    num_frames=0,
    tokens_per_frame=0,
    cache_clear_interval=4,
):

    B, L_q, N, D = q.shape
    out_dtype = q.dtype
    device = q.device
    scale = 1.0 / math.sqrt(D)

    q_t = q.transpose(1, 2)       # [B, N, L_q, D]
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    output = torch.zeros(B, N, L_q, D, dtype=out_dtype, device=device)

    H = storage_dict.get('_H', None)
    W = storage_dict.get('_W', None)
    can_select = (
        token_indices_per_noun is not None
        and num_frames > 0
        and tokens_per_frame > 0
        and H is not None
        and W is not None
    )

    # Initialise trackers based on mode
    accum = {}        # for mean mode
    best_peak = {}    # for single-head mode
    if can_select:
        if 'cross_attn' not in storage_dict:
            storage_dict['cross_attn'] = {}
        for noun in token_indices_per_noun:
            if noun not in storage_dict['cross_attn']:
                storage_dict['cross_attn'][noun] = {}
            if USE_MEAN:
                accum[noun] = {}
                for f in range(num_frames):
                    accum[noun][f] = torch.zeros(
                        tokens_per_frame, dtype=torch.float32, device=device
                    )
            else:
                for f in range(num_frames):
                    best_peak[(noun, f)] = -float('inf')

    for h in range(N):
        q_h = q_t[:, h:h+1, :, :]
        k_h = k_t[:, h:h+1, :, :]
        v_h = v_t[:, h:h+1, :, :]

        # Full cross-attention: [B, 1, L_q, L_k]
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
        # Softmax in float32 for precision (per-token values ~1/512)
        attn_weights_f32 = F_torch.softmax(scores.float(), dim=-1)

        # Compute output in native dtype
        out_h = torch.matmul(attn_weights_f32.to(out_dtype), v_h)
        output[:, h:h+1, :, :] = out_h

        # Accumulate per-noun per-frame response across heads
        if can_select:
            attn_real = attn_weights_f32[0, 0, :real_video_len, :]  # [rvl, L_k]

            for noun, tok_indices in token_indices_per_noun.items():
                tok_idx = torch.tensor(tok_indices, dtype=torch.long, device=device)

                for f in range(num_frames):
                    fs = f * tokens_per_frame
                    fe = fs + tokens_per_frame
                    frame_ca = attn_real[fs:fe, :]             # [tpf, L_k]
                    noun_resp = frame_ca[:, tok_idx].mean(dim=1)  # [tpf]

                    if USE_MEAN:
                        accum[noun][f] += noun_resp
                    else:
                        peak = noun_resp.max().item()
                        if peak > best_peak[(noun, f)]:
                            best_peak[(noun, f)] = peak
                            storage_dict['cross_attn'][noun][f] = (
                                h,
                                noun_resp.detach().cpu().numpy().reshape(H, W),
                            )

        del q_h, k_h, v_h, scores, attn_weights_f32, out_h
        if (h + 1) % cache_clear_interval == 0:
            torch.cuda.empty_cache()

    # Finalise mean mode: divide by N, offload
    if can_select and USE_MEAN:
        for noun in token_indices_per_noun:
            for f in range(num_frames):
                mean_map = (accum[noun][f] / N).cpu().numpy().reshape(H, W)
                storage_dict['cross_attn'][noun][f] = (-1, mean_map)
        del accum
        torch.cuda.empty_cache()

    output = output.transpose(1, 2).contiguous().to(out_dtype)
    del q_t, k_t, v_t
    torch.cuda.empty_cache()

    return output


def numina_cross_attention_modulate(
    q,
    k,
    v,
    modulation_data,
    step_index,
    cache_clear_interval=4,
):

    from numina.modulation import build_cross_attention_bias

    B, L_q, N, D = q.shape
    out_dtype = q.dtype

    # SDPA expects [B, N, L, D]
    q_t = q.transpose(1, 2)  # [B, N, L_q, D]
    k_t = k.transpose(1, 2)  # [B, N, L_k, D]
    v_t = v.transpose(1, 2)  # [B, N, L_k, D]

    # Build bias tensor: [1, 1, L_q, L_k] or None
    bias = build_cross_attention_bias(q_t, k_t, modulation_data, step_index)

    # Single fused SDPA call — processes all heads in one kernel
    # SDPA computes: softmax(QK^T / sqrt(D) + bias) @ V
    output = F_torch.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=bias,
    )

    # Back to [B, L, N, D]
    output = output.transpose(1, 2).contiguous()

    if bias is not None:
        del bias

    return output.to(out_dtype)
