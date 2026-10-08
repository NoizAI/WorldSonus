"""Incremental C3 generation with persistent visual, AR, and codec state."""

import torch

from worldsonus.codec import denormalize_latents


class StreamingGenerator:
    """One model/session, three DINO frames in and one 100 ms audio chunk out.

    The stream keeps the previous visual grid, previous audio chunk and bounded
    Ring-KV state. Future video frames are never read by push(). A maximum
    duration is supplied only to allocate rotary positions and prompt masks.
    """

    @torch.inference_mode()
    def __init__(self, model, *, num_chunks, batch_size=1, seed=5031, steps=15, **prompt):
        if num_chunks < 1 or batch_size < 1:
            raise ValueError("Stream length and batch size must be positive")
        if getattr(model, "_active_stream", None) is not None:
            raise RuntimeError("Model already has an active stream")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.num_chunks = num_chunks
        if steps < 1:
            raise ValueError("Denoising step count must be positive")
        self.steps = steps
        self.batch_size = batch_size
        self.index = 0
        self.closed = False
        self.previous_grid = None
        self.previous_audio = None
        self.position = torch.zeros(1, device=self.device, dtype=torch.int)
        self.devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=self.devices):
            torch.manual_seed(seed)
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.device) if self.devices else None
        model._inference_guidance_method = "cfg"
        model.transformer.set_context_extension(
            "swa", window_size=model.transformer.causal_window_size
        )
        with torch.device(self.device):
            model.transformer.setup_caches(batch_size * 3, num_chunks * 2, self.dtype)
        self.set_prompt(**prompt)
        model._active_stream = self

    @torch.inference_mode()
    def set_prompt(
        self,
        prompt_embedding=None,
        prompt_mask=None,
        prompt_start_audio_chunks=None,
        prompt_duration_audio_chunks=None,
    ):
        """Replace prompt K/V at the next chunk boundary; retain audio AR history."""
        if self.closed:
            raise RuntimeError("Stream is closed")
        move = lambda x: None if x is None else x.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16),
        ):
            self.text_condition = self.model.transformer.prepare_text_condition(
                move(prompt_embedding),
                move(prompt_mask),
                batch_size=self.batch_size,
                device=self.device,
                dtype=self.dtype,
                cfg_mode="video_prompt",
                apply_dropout=False,
                prompt_start_chunks=move(prompt_start_audio_chunks),
                prompt_duration_chunks=move(prompt_duration_audio_chunks),
                query_token_count=self.num_chunks * 2,
                query_tokens_per_chunk=2,
            )

    def _encode_video(self, video):
        model = self.model
        grid = model.video_input_adapter(video)
        previous = self.previous_grid
        if previous is None:
            previous = torch.zeros_like(grid[:, :1])
        shifted = torch.cat((previous, grid[:, :-1]), dim=1)
        self.previous_grid = grid[:, -1:].clone()
        features = torch.cat((grid, grid - shifted), dim=-1)
        features = model.video_proj_3d(
            features.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d)
        )
        batch, channels, frames, height, width = features.shape
        spatial = features.permute(0, 2, 3, 4, 1).reshape(batch * frames, height * width, channels)
        tokens = model.aggregate_transformer(
            spatial,
            model.aggregated_tokens.unsqueeze(0).expand(batch * frames, -1, -1),
            spatial_shape=(height, width),
        ).reshape(batch, frames, channels)
        return model._encode_video_chunks_with_slots(tokens)

    @torch.inference_mode()
    def push(self, video):
        if self.closed or self.index >= self.num_chunks:
            raise RuntimeError("Stream is closed or its requested duration is complete")
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or video.shape[:2] != (self.batch_size, 3) or video.shape[-1] != 384:
            raise ValueError("Expected [B,3,H,W,384] DINO features")
        video = video.to(device=self.device, dtype=self.dtype)
        if self.previous_grid is not None and video.shape[2:4] != self.previous_grid.shape[2:4]:
            raise ValueError("Video grid size cannot change within a stream")
        with (
            torch.random.fork_rng(devices=self.devices),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.device.type == "cuda"
                and self.dtype in (torch.float16, torch.bfloat16),
            ),
        ):
            torch.set_rng_state(self.cpu_rng)
            if self.devices:
                torch.cuda.set_rng_state(self.cuda_rng, self.device)
            summary, local = self._encode_video(video)
            kwargs = dict(
                chunk_index=0,
                input_pos=self.position,
                trans_cfg_scale=3.0,
                prompt_cfg_scale=3.0,
                inference_noise=0.1,
                text_condition=self.text_condition,
            )
            if self.previous_audio is None:
                audio_token, video_token = self.model._sample_bos_video_summary_group(
                    summary, **kwargs
                )
            else:
                audio_token, video_token, _ = self.model._sample_audio_video_summary_group(
                    self.previous_audio, summary, **kwargs
                )
            condition = torch.cat((audio_token, video_token), dim=-1)
            if condition.ndim == 3:
                condition = condition[:, 0]
            latent = self.model.diffusion_sampling(
                condition,
                local_cond=local[:, 0],
                device=self.device,
                dh_dino_cfg=8.0,
                num_steps=self.steps,
                solver="euler",
                flow_head_guidance_method="apg",
                flow_head_apg_eta=0.0,
                flow_head_apg_momentum=0.25,
                flow_head_apg_norm_threshold=0.12421,
            )
            self.cpu_rng = torch.get_rng_state()
            if self.devices:
                self.cuda_rng = torch.cuda.get_rng_state(self.device)
        self.previous_audio = latent.clone()
        self.index += 1
        # Retain native sampler precision in history; expose float32 to callers.
        return latent.float()

    def close(self):
        if getattr(self.model, "_active_stream", None) is self:
            self.model._active_stream = None
        self.previous_grid = self.previous_audio = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def stream_audio(model, decoder, video_chunks, *, num_chunks, mean, std, seed=5031, **prompt):
    """Yield [B,2,4800] waveform chunks while consuming three-frame inputs lazily."""
    with StreamingGenerator(model, num_chunks=num_chunks, seed=seed, **prompt) as generator:
        with decoder.stream() as waveform:
            count = 0
            for video in video_chunks:
                latent = generator.push(video)
                yield waveform.push(denormalize_latents(latent, mean, std).transpose(1, 2))
                count += 1
            if count != num_chunks:
                raise ValueError(f"Video ended after {count} chunks; expected {num_chunks}")
