# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2026 NUMINA Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    # ====================================================================
    # Original generate() — UNCHANGED
    # ====================================================================

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    # ====================================================================
    # NUMINA: Two-phase generation with numerical alignment
    # ====================================================================

    def generate_numina(
        self,
        numina_input,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        offload_model=True,
    ):

        from numina.config import NuminaConfig
        from numina.layout import construct_layouts, refine_all_layouts
        from numina.modulation import (
            ModulationData,
            build_modulation_data,
            has_any_modulation,
        )

        cfg = numina_input.config
        input_prompt = numina_input.prompt
        seed = numina_input.seed

        # Ensure sampling_steps matches config
        assert sampling_steps == cfg.total_steps, (
            f"sampling_steps ({sampling_steps}) must match "
            f"NuminaConfig.total_steps ({cfg.total_steps})"
        )

        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.patch_size[1] * self.patch_size[2])
            * target_shape[1]
            / self.sp_size
        ) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # --- Encode text ---
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # --- Compute latent spatial dimensions after patching ---
        # target_shape = (C, F_lat, H_lat, W_lat) where F/H/W_lat are
        # VAE-compressed.  After patch_embedding with patch_size (1,2,2):
        #   F_patch = F_lat / 1 = F_lat
        #   H_patch = H_lat / 2
        #   W_patch = W_lat / 2
        num_frames = target_shape[1]  # F_lat = F_patch (patch_t = 1)
        H_patch = target_shape[2] // self.patch_size[1]
        W_patch = target_shape[3] // self.patch_size[2]
        tokens_per_frame = H_patch * W_patch

        logger.info(
            f"[NUMINA] Latent: F_lat={target_shape[1]}, H_lat={target_shape[2]}, "
            f"W_lat={target_shape[3]} -> patches: F={num_frames}, "
            f"H={H_patch}, W={W_patch}, tpf={tokens_per_frame}"
        )

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            # --- Create scheduler ---
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            logger.info("[NUMINA] === Phase 1: Pre-generation ===")

            # Generate noise (deterministic with seed)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)
            noise = torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )

            # Enable extraction on the model (with inline head selection)
            storage_dict = {}
            token_indices_per_noun = {
                noun: target.token_indices
                for noun, target in numina_input.targets.items()
            }

            self.model.to(self.device)
            self.model.numina_enable_extraction(
                reference_layer=cfg.reference_layer,
                reference_step=cfg.reference_step,
                storage_dict=storage_dict,
                numina_config=cfg,
                token_indices_per_noun=token_indices_per_noun,
                grid_info=(num_frames, H_patch, W_patch),
                cache_interval=cfg.cache_clear_interval,
            )

            # Pre-generate up to and including reference_step
            latents = [noise.clone()]

            # Release CUDA reserved pool from T5 encoder forward pass
            torch.cuda.empty_cache()

            # --- EasyCache state ---
            ec_enabled = cfg.easycache_enabled
            ec_tau = cfg.easycache_tau
            ec_warmup = cfg.easycache_warmup
            ec_k = None                   # transformation rate
            ec_delta_cond = None          # cached Δ for cond pass
            ec_delta_uncond = None        # cached Δ for uncond pass
            ec_accumulated_error = 0.0    # E_t
            ec_prev_raw_input = None      # raw input at last full compute
            ec_prev_output_cond = None    # cond output at last full compute
            ec_skipped = 0
            ec_computed = 0

            total_phase1_steps = cfg.reference_step + 1

            for step_idx, t in enumerate(tqdm(
                timesteps[:total_phase1_steps],
                desc="[NUMINA Phase 1] Pre-generation",
            )):
                self.model.numina_set_step(step_idx)
                timestep = torch.stack([t])

                # Raw input for this step (before model forward)
                raw_input = latents[0]

                # --- EasyCache: decide skip or compute ---
                is_reference_step = (step_idx == cfg.reference_step)
                is_warmup = (step_idx < ec_warmup)
                is_last = (step_idx == total_phase1_steps - 1)
                must_compute = (
                    is_reference_step
                    or is_warmup
                    or is_last
                    or not ec_enabled
                    or ec_delta_cond is None
                    or ec_accumulated_error >= ec_tau
                )

                if must_compute:
                    # --- Full compute ---
                    latent_model_input = latents

                    if is_reference_step:
                        # Extraction on cond pass only
                        self.model.numina_set_active_for_call(True)
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]

                    if is_reference_step:
                        # Uncond pass: no extraction (FlashAttention)
                        self.model.numina_set_active_for_call(False)
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0]

                    if is_reference_step:
                        # Restore default
                        self.model.numina_set_active_for_call(True)
                        # Release CUDA allocator high-water mark from extraction
                        torch.cuda.empty_cache()

                    # Update EasyCache state
                    if ec_prev_output_cond is not None and ec_prev_raw_input is not None:
                        # Compute transformation rate k
                        output_change = (noise_pred_cond - ec_prev_output_cond
                                         ).flatten().abs().mean()
                        input_change = (raw_input - ec_prev_raw_input
                                        ).flatten().abs().mean()
                        if input_change > 1e-10:
                            ec_k = (output_change / input_change).item()

                    # Cache transformation vectors: Δ = output - raw_input
                    ec_delta_cond = noise_pred_cond - raw_input
                    ec_delta_uncond = noise_pred_uncond - raw_input
                    ec_prev_raw_input = raw_input.clone()
                    ec_prev_output_cond = noise_pred_cond.clone()
                    ec_accumulated_error = 0.0
                    ec_computed += 1

                else:
                    # --- Skip: approximate from cached transformation ---
                    noise_pred_cond = raw_input + ec_delta_cond
                    noise_pred_uncond = raw_input + ec_delta_uncond

                    # Accumulate estimated error
                    if ec_k is not None and ec_prev_raw_input is not None:
                        input_change = (raw_input - ec_prev_raw_input
                                        ).flatten().abs().mean().item()
                        output_norm = ec_prev_output_cond.flatten().abs().mean().item()
                        if output_norm > 1e-10:
                            epsilon = ec_k * input_change / output_norm
                            ec_accumulated_error += epsilon

                    ec_skipped += 1

                # CFG combination + scheduler step
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            # Disable extraction
            self.model.numina_disable()

            # Free EasyCache state tensors and release CUDA pool
            del ec_delta_cond, ec_delta_uncond
            del ec_prev_raw_input, ec_prev_output_cond
            gc.collect()
            torch.cuda.empty_cache()

            if ec_enabled:
                logger.info(
                    f"[NUMINA] Phase 1 EasyCache: {ec_computed} computed, "
                    f"{ec_skipped} skipped out of {total_phase1_steps} steps "
                    f"({ec_skipped/total_phase1_steps*100:.0f}% skipped)"
                )

            # --- Layout construction (uses pre-selected maps directly) ---
            layout_data = construct_layouts(
                head_selection_result=storage_dict,
                targets=numina_input.targets,
                num_frames=num_frames,
                H=H_patch,
                W=W_patch,
                config=cfg,
            )

            # --- Layout refinement ---
            refined_layouts = refine_all_layouts(layout_data, cfg)

            # --- Build modulation data ---
            modulation_data = build_modulation_data(
                refined_layouts=refined_layouts,
                targets=numina_input.targets,
                num_frames=num_frames,
                H=H_patch,
                W=W_patch,
                config=cfg,
            )

            # Free extraction data
            del storage_dict, layout_data

            # Free Phase 1 tensors that are no longer needed
            del noise, latents, seed_g
            del sample_scheduler
            gc.collect()
            torch.cuda.empty_cache()

            # Check if any modulation is needed
            if not has_any_modulation(modulation_data):
                logger.info(
                    "[NUMINA] All counts already correct — no modulation needed. "
                    "Running standard generation."
                )
                # Fall through to Phase 2 without modulation enabled
                do_modulation = False
            else:
                do_modulation = True


            logger.info("[NUMINA] === Phase 2: Guided regeneration ===")

            # Re-create scheduler (reset internal state)
            if sample_solver == 'unipc':
                sample_scheduler_2 = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler_2.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps_2 = sample_scheduler_2.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler_2 = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas_2 = get_sampling_sigmas(sampling_steps, shift)
                timesteps_2, _ = retrieve_timesteps(
                    sample_scheduler_2,
                    device=self.device,
                    sigmas=sampling_sigmas_2)

            # Reset to same noise (same seed)
            seed_g_2 = torch.Generator(device=self.device)
            seed_g_2.manual_seed(seed)
            noise_2 = torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g_2,
            )

            # Enable modulation
            if do_modulation:
                self.model.numina_enable_modulation(
                    modulation_data=modulation_data,
                    cache_interval=cfg.cache_clear_interval,
                )

            # Full denoising loop
            latents = [noise_2]

            for step_idx, t in enumerate(tqdm(
                timesteps_2,
                desc="[NUMINA Phase 2] Guided generation",
            )):
                # Set current step (controls delta(t) intensity)
                if do_modulation:
                    self.model.numina_set_step(step_idx)

                latent_model_input = latents
                timestep = torch.stack([t])

                self.model.to(self.device)

                # Modulate only the conditional (positive prompt) pass.
                # The unconditional (negative prompt) pass uses FlashAttention
                # normally — modulating it wastes compute and hurts CFG.
                if do_modulation:
                    self.model.numina_set_active_for_call(True)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]

                if do_modulation:
                    self.model.numina_set_active_for_call(False)
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler_2.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g_2)[0]
                latents = [temp_x0.squeeze(0)]

            # Disable NUMINA
            self.model.numina_disable()

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise_2, latents
        del sample_scheduler_2, modulation_data
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        logger.info("[NUMINA] Generation complete.")
        return videos[0] if self.rank == 0 else None
