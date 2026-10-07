from __future__ import annotations
from tqdm import tqdm
import math
import os
import torch
import torch.nn as nn
from worldsonus.flow import FlowSampler
from typing import Callable, List, Optional
from worldsonus.transformer import (
    AggregateTransformer,
    MainTransformer,
    TextCondition,
    transformer_cfg_branch_count,
)
from worldsonus.video import VideoInputAdapter, resolve_video_input_spec
from einops import rearrange
from worldsonus.video_delta import concat_video_with_delta


def _combine_transformer_guidance(
    branches: torch.Tensor,
    *,
    trans_cfg_scale: float,
    prompt_cfg_scale: float,
    cfg_mode: str,
    method: str = "cfg",
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    apg_video_norm_threshold: Optional[float] = None,
    apg_prompt_norm_threshold: Optional[float] = None,
    guidance_update_observer: Optional[Callable[[torch.Tensor], None]] = None,
) -> torch.Tensor:
    """Combine ordered Transformer condition branches.

    Single-axis modes deliberately retain the historical arithmetic order so
    existing video-only ``(s_v, 1)`` and prompt-only ``(1, s_t)`` sampling are
    numerically unchanged.  The joint mode is the symmetric additive
    three-branch rule, ordered as full, null-video, null-prompt::

        full + (s_v - 1) * (full - null_video)
             + (s_t - 1) * (full - null_prompt)

    A joint-null fourth branch is intentionally not used.
    """
    branch_count = transformer_cfg_branch_count(cfg_mode)
    if branches.shape[0] % branch_count:
        raise ValueError(
            f"Transformer CFG output batch is not divisible by its branch count: batch={branches.shape[0]} branches={branch_count}"
        )
    method = str(method).strip().lower()
    if method not in {"cfg", "apg"}:
        raise ValueError(f"unsupported Transformer guidance method: {method}")
    if not 0.0 <= float(apg_eta) <= 1.0:
        raise ValueError("APG eta must lie in [0,1]")
    for name, value in (
        ("APG norm threshold", apg_norm_threshold),
        ("video APG norm threshold", apg_video_norm_threshold),
        ("prompt APG norm threshold", apg_prompt_norm_threshold),
    ):
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    if cfg_mode == "none":
        return branches
    split = torch.chunk(branches, branch_count, dim=0)
    full = split[0]
    axis_updates: dict[str, torch.Tensor]
    if cfg_mode == "video":
        null_video = split[1]
        axis_updates = {"video": (float(trans_cfg_scale) - 1.0) * (full - null_video)}
    elif cfg_mode == "prompt":
        null_prompt = split[1]
        axis_updates = {"prompt": (float(prompt_cfg_scale) - 1.0) * (full - null_prompt)}
    elif cfg_mode == "video_prompt":
        (null_video, null_prompt) = (split[1], split[2])
        axis_updates = {
            "video": (float(trans_cfg_scale) - 1.0) * (full - null_video),
            "prompt": (float(prompt_cfg_scale) - 1.0) * (full - null_prompt),
        }
    else:
        raise ValueError(f"unsupported Transformer CFG mode: {cfg_mode}")
    update = sum(axis_updates.values())
    if guidance_update_observer is not None:
        guidance_update_observer(update)
    if method == "cfg":
        return full + update
    axis_thresholds = {
        "video": float(apg_norm_threshold)
        if apg_video_norm_threshold is None
        else float(apg_video_norm_threshold),
        "prompt": float(apg_norm_threshold)
        if apg_prompt_norm_threshold is None
        else float(apg_prompt_norm_threshold),
    }
    if float(apg_eta) == 1.0 and all((axis_thresholds[axis] == 0.0 for axis in axis_updates)):
        return full + update
    dimensions = tuple(range(1, full.ndim))
    reference64 = torch.nn.functional.normalize(full.double(), dim=dimensions)
    projected_updates = []
    for axis, axis_update in axis_updates.items():
        radius = axis_thresholds[axis]
        if radius > 0.0:
            update_norm = torch.linalg.vector_norm(
                axis_update.float(), dim=dimensions, keepdim=True
            ).clamp_min(1e-12)
            scale = torch.clamp(radius / update_norm, max=1.0).to(dtype=axis_update.dtype)
            axis_update = axis_update * scale
        update64 = axis_update.double()
        parallel64 = (update64 * reference64).sum(dim=dimensions, keepdim=True) * reference64
        orthogonal64 = update64 - parallel64
        projected_updates.append((orthogonal64 + float(apg_eta) * parallel64).to(dtype=full.dtype))
    return full + sum(projected_updates)


class TemporalSummaryEncoder(nn.Module):
    """Bidirectionally summarize the fixed slots inside one causal chunk."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 512)
        self.summary_token = nn.Parameter(torch.randn(1, 1, 512) / math.sqrt(512))
        self.position = nn.Parameter(torch.zeros(1, 3, 512))
        layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output_norm = nn.LayerNorm(512)

    def forward_with_slots(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if slots.ndim != 3 or slots.shape[1] != 3:
            raise ValueError(f"expected [B,{3},C] chunk slots, got {tuple(slots.shape)}")
        slots = self.input_proj(slots) + self.position
        summary = self.summary_token.expand(slots.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([summary, slots], dim=1))
        encoded = self.output_norm(encoded)
        return (encoded[:, 0], encoded[:, 1:])

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        (summary, _) = self.forward_with_slots(slots)
        return summary


class WorldSonus(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_pipeline = True
        self.video_input_spec = resolve_video_input_spec(
            name="bottleneck128_delta",
            input_dim=384,
            no_subtract=False,
            center_path=None,
            bottleneck_dim=128,
            raw_rgb_delta_segment_frames=0,
        )
        self.audio_proj = nn.Sequential(nn.Linear(512, 1024), nn.SiLU(), nn.Linear(1024, 1024))
        temporal_encoder_kwargs = {
            "model_dim": 512,
            "chunk_size": 3,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": 0.1,
        }

        def build_temporal_encoder(
            input_dim: int, *, direct_singleton_output: bool = False
        ) -> nn.Module:
            if direct_singleton_output:
                return TemporalSummaryEncoder(input_dim=input_dim)
            return TemporalSummaryEncoder(input_dim=input_dim)

        self.video_temporal_encoder = build_temporal_encoder(512)
        self.audio_temporal_encoder = build_temporal_encoder(64, direct_singleton_output=False)
        self.video_proj = nn.Sequential(nn.Linear(512, 1024), nn.SiLU(), nn.Linear(1024, 1024))
        self.video_proj_3d = nn.Sequential(
            nn.Conv3d(in_channels=768, out_channels=512, kernel_size=(1, 1, 1), bias=True),
            nn.SiLU(),
            nn.Conv3d(
                in_channels=512,
                out_channels=512,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
        )
        with torch.random.fork_rng(devices=[]):
            self.video_proj_3d[0] = nn.Conv3d(
                in_channels=256, out_channels=512, kernel_size=(1, 1, 1), bias=True
            )
        self.aggregate_transformer = AggregateTransformer()
        self.aggregated_tokens = nn.Parameter(0.044194173824159216 * torch.randn(1, 512))
        self.transformer = MainTransformer()
        self.diffloss = FlowSampler()
        with torch.random.fork_rng(devices=[]):
            self.video_input_adapter = VideoInputAdapter()
            self.video_input_adapter.reset_projection_parameters()
        self.initialize_weights()

    def initialize_weights(self):
        self.video_proj.apply(self._init_linear)
        if hasattr(self, "video_proj_3d"):
            self.video_proj_3d.apply(self._init_linear)
        self.audio_proj.apply(self._init_linear)
        if hasattr(self, "keyframe_video_projection"):
            self.keyframe_video_projection.apply(self._init_linear)
        if hasattr(self, "prepooled_video_projection"):
            self.prepooled_video_projection.apply(self._init_linear)

    def _init_linear(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.weight.data.normal_(mean=0.0, std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def _prepare_video_input(self, video: torch.Tensor) -> torch.Tensor:
        """Apply the configured input-only transform, then optional delta."""
        video = self.video_input_adapter(video)
        video = concat_video_with_delta(video, lag=1)
        return video

    def _resolve_video_present(
        self,
        video_present: Optional[torch.Tensor | bool],
        *,
        batch_size: int,
        device: torch.device,
        has_video_input: bool,
    ) -> torch.Tensor:
        """Normalize the public no-video contract to a device boolean vector."""
        resolved = self.transformer._normalize_video_present(
            video_present, batch_size=batch_size, device=device, default=has_video_input
        )
        if not has_video_input:
            return torch.zeros_like(resolved)
        return resolved

    @staticmethod
    def _mask_missing_video_rows(
        value: Optional[torch.Tensor], video_present: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if value.shape[0] != video_present.shape[0]:
            raise ValueError("video condition and video_present batch sizes differ")
        row_mask = video_present.reshape(video_present.shape[0], *(1,) * (value.ndim - 1))
        return torch.where(row_mask, value, torch.zeros_like(value))

    def _empty_video_conditions(
        self, *, batch_size: int, num_chunks: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Create shape-correct pre-projection null inputs for prompt-only rows."""
        summaries = torch.zeros(batch_size, num_chunks * 1, 512, device=device, dtype=dtype)
        local = None
        local = torch.zeros(batch_size, num_chunks, 3, 512, device=device, dtype=dtype)
        return (summaries, local)

    def _encode_video_chunks_with_slots(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode video chunks and flatten their K summaries in time order."""
        (B, T, C) = tokens.shape
        if T % 3:
            raise ValueError("token length must be divisible by chunk_size")
        num_chunks = T // 3
        chunks = tokens.reshape(B * num_chunks, 3, C)
        (summaries, local_slots) = self.video_temporal_encoder.forward_with_slots(chunks)
        if summaries.ndim == 2:
            summaries = summaries.unsqueeze(1)
        if summaries.ndim != 3 or summaries.shape[1] != 1:
            raise ValueError(
                f"video summary encoder returned an unexpected number of tokens: expected {1}, got {tuple(summaries.shape)}"
            )
        return (summaries.reshape(B, num_chunks * 1, -1), local_slots.reshape(B, num_chunks, 3, -1))

    def _resolve_transformer_guidance(
        self, *, trans_cfg_scale: float, prompt_cfg_scale: float
    ) -> tuple[float, float, str]:
        """Resolve independent video/text CFG scales and their branch layout."""
        trans_cfg_scale = float(trans_cfg_scale)
        prompt_cfg_scale = float(prompt_cfg_scale)
        if trans_cfg_scale < 1.0 or prompt_cfg_scale < 1.0:
            raise ValueError("Transformer and prompt CFG scales must be at least 1")
        if trans_cfg_scale > 1.0 and prompt_cfg_scale > 1.0:
            return (trans_cfg_scale, prompt_cfg_scale, "video_prompt")
        if prompt_cfg_scale > 1.0:
            return (trans_cfg_scale, prompt_cfg_scale, "prompt")
        if trans_cfg_scale > 1.0:
            return (trans_cfg_scale, prompt_cfg_scale, "video")
        return (trans_cfg_scale, prompt_cfg_scale, "none")

    @staticmethod
    def _guidance_mode(
        *, trans_cfg_scale: float, prompt_cfg_scale: float, text_condition: Optional[TextCondition]
    ) -> str:
        if text_condition is not None:
            return text_condition.cfg_mode
        if prompt_cfg_scale > 1.0:
            raise ValueError("prompt CFG requires a prepared text condition")
        return "video" if trans_cfg_scale > 1.0 else "none"

    def _combine_inference_transformer_guidance(
        self,
        branches: torch.Tensor,
        *,
        trans_cfg_scale: float,
        prompt_cfg_scale: float,
        cfg_mode: str,
    ) -> torch.Tensor:
        return _combine_transformer_guidance(
            branches,
            trans_cfg_scale=trans_cfg_scale,
            prompt_cfg_scale=prompt_cfg_scale,
            cfg_mode=cfg_mode,
            method=getattr(self, "_inference_guidance_method", "cfg"),
            apg_eta=getattr(self, "_inference_apg_eta", 1.0),
            apg_norm_threshold=getattr(self, "_inference_apg_norm_threshold", 0.0),
            apg_video_norm_threshold=getattr(self, "_inference_apg_video_norm_threshold", None),
            apg_prompt_norm_threshold=getattr(self, "_inference_apg_prompt_norm_threshold", None),
            guidance_update_observer=self._record_transformer_guidance_update
            if getattr(self, "_record_transformer_guidance_norms", False)
            else None,
        )

    @torch.no_grad()
    def _record_transformer_guidance_update(self, update: torch.Tensor) -> None:
        dimensions = tuple(range(1, update.ndim))
        values = torch.linalg.vector_norm(update.float(), dim=dimensions).detach()
        self._transformer_guidance_norm_trace.append(values)

    def _sample_projected_transformer_token(
        self,
        input_token: torch.Tensor,
        type_id: int,
        input_pos: torch.Tensor,
        trans_cfg_scale: float,
        prompt_cfg_scale: float = 1.0,
        text_condition: Optional[TextCondition] = None,
        video_present: Optional[torch.Tensor | bool] = None,
    ) -> torch.Tensor:
        if input_token.ndim == 3:
            if input_token.shape[1] != 1:
                raise ValueError("Transformer sampling accepts exactly one projected token")
            input_token = input_token[:, 0]
        if input_token.ndim != 2:
            raise ValueError(f"expected projected token [B,C], got {tuple(input_token.shape)}")
        cfg_mode = self._guidance_mode(
            trans_cfg_scale=trans_cfg_scale,
            prompt_cfg_scale=prompt_cfg_scale,
            text_condition=text_condition,
        )
        cfg_active_scale = max(float(trans_cfg_scale), float(prompt_cfg_scale))
        if cfg_active_scale > 1.0:
            z = self.transformer.inference(
                input_token,
                type_id,
                input_pos,
                trans_cfg_scale=cfg_active_scale,
                text_condition=text_condition,
                video_present=video_present,
            )
            z = self._combine_inference_transformer_guidance(
                z,
                trans_cfg_scale=trans_cfg_scale,
                prompt_cfg_scale=prompt_cfg_scale,
                cfg_mode=cfg_mode,
            )
        else:
            z = self.transformer.inference(
                input_token,
                type_id,
                input_pos,
                text_condition=text_condition,
                video_present=video_present,
            )
        return z

    def _can_use_paired_ar_inference(self) -> bool:
        window_size = self.transformer.sliding_window
        return bool(getattr(self, "paired_ar_inference", True)) and (
            window_size is None or int(window_size) >= 2
        )

    def _configure_inference_context(self, context_mode: str, *, token_count: int) -> None:
        """Configure bounded inference without slowing in-window requests.

        A request that fits inside the configured deployment window cannot
        evict anything, so the standard fixed-size cache keeps the paired-AR
        fast path. Ring KV is used once a request grows beyond that window.
        """
        train_token_count = int(self.transformer.seq_len)
        context_window_tokens = int(self.transformer.causal_window_size)
        if context_mode in ("pi", "ntk"):
            scale = max(1.0, float(token_count) / float(train_token_count))
            self.transformer.set_context_extension(context_mode, factor=scale)
        elif context_mode in ("swa", "sliding"):
            if token_count <= context_window_tokens:
                self.transformer.set_context_extension("none")
            else:
                self.transformer.set_context_extension("swa", window_size=context_window_tokens)
        elif context_mode == "none":
            self.transformer.set_context_extension("none")
        else:
            raise ValueError(f"unsupported inference context mode: {context_mode}")

    def _sample_projected_transformer_pair(
        self,
        audio_token: torch.Tensor,
        video_token: torch.Tensor,
        input_pos: torch.Tensor,
        trans_cfg_scale: float,
        *,
        prompt_cfg_scale: float = 1.0,
        audio_cfg_expanded: bool = False,
        text_condition: Optional[TextCondition] = None,
        video_present: Optional[torch.Tensor | bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one causal audio/video pair with one Transformer invocation."""
        if input_pos.ndim == 0:
            input_pos = input_pos.view(1)
        if input_pos.ndim != 1 or input_pos.shape[0] != 1:
            raise ValueError(
                f"paired AR start position must have shape [1], got {tuple(input_pos.shape)}"
            )
        pair_pos = input_pos + torch.arange(2, device=input_pos.device, dtype=input_pos.dtype)
        cfg_mode = self._guidance_mode(
            trans_cfg_scale=trans_cfg_scale,
            prompt_cfg_scale=prompt_cfg_scale,
            text_condition=text_condition,
        )
        cfg_active_scale = max(float(trans_cfg_scale), float(prompt_cfg_scale))
        hidden = self.transformer(
            video=video_token,
            audio=audio_token,
            input_pos=pair_pos,
            inference_pair=True,
            trans_cfg_scale=cfg_active_scale,
            audio_cfg_expanded=audio_cfg_expanded,
            text_condition=text_condition,
            semantic_feature=None,
            video_present=video_present,
        )
        if cfg_active_scale > 1.0:
            hidden = self._combine_inference_transformer_guidance(
                hidden,
                trans_cfg_scale=trans_cfg_scale,
                prompt_cfg_scale=prompt_cfg_scale,
                cfg_mode=cfg_mode,
            )
        return (hidden[:, 0:1], hidden[:, 1:2])

    def _encode_audio_summary_group(
        self, input_token: torch.Tensor, inference_noise: float
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        audio_slots = None
        if input_token.ndim != 3 or input_token.shape[1] != 3:
            raise ValueError(f"expected audio chunk [B,{3},C], got {tuple(input_token.shape)}")
        (input_token, audio_slots) = self.audio_temporal_encoder.forward_with_slots(input_token)
        if input_token.ndim == 2:
            input_token = input_token.unsqueeze(1)
        if input_token.ndim != 3 or input_token.shape[1] != 1:
            raise ValueError(
                f"audio temporal encoder returned an unexpected number of summaries: expected {1}, got {tuple(input_token.shape)}"
            )
        input_token = self.audio_proj(input_token)
        if float(inference_noise) != 0.0:
            noise = torch.randn_like(input_token)
            input_token = inference_noise * noise + (1 - inference_noise) * input_token
        return (input_token, audio_slots)

    def transformer_sampling(
        self,
        input_token: torch.Tensor,
        type_id: int,
        input_pos: torch.Tensor,
        trans_cfg_scale: float = 1.0,
        prompt_cfg_scale: float = 1.0,
        inference_noise: float = 0.2,
        return_audio_slots: bool = False,
        text_condition: Optional[TextCondition] = None,
        video_present: Optional[torch.Tensor | bool] = None,
    ):
        audio_slots = None
        if type_id == 0:
            if return_audio_slots:
                raise ValueError("audio slots can only be returned for audio tokens")
            input_token = self.video_proj(input_token)
        elif type_id == 1:
            (input_token, audio_slots) = self._encode_audio_summary_group(
                input_token, inference_noise
            )
            if input_token.shape[1] != 1:
                raise ValueError("multi-summary audio must be sampled with its paired video group")
            input_token = input_token[:, 0]
        else:
            raise ValueError(f"invalid token type id: {type_id}")
        z = self._sample_projected_transformer_token(
            input_token,
            type_id,
            input_pos,
            trans_cfg_scale,
            prompt_cfg_scale,
            text_condition=text_condition,
            video_present=video_present,
        )
        if return_audio_slots:
            return (z, audio_slots)
        return z

    def bos_sampling(
        self,
        input_token: torch.Tensor,
        type_id: int,
        input_pos: torch.Tensor,
        trans_cfg_scale: float = 1.0,
        prompt_cfg_scale: float = 1.0,
        inference_noise: float = 0.2,
        text_condition: Optional[TextCondition] = None,
    ):
        assert type_id == 1
        input_token = input_token
        if float(inference_noise) != 0.0:
            noise = torch.randn_like(input_token)
            input_token = inference_noise * noise + (1 - inference_noise) * input_token
        cfg_mode = self._guidance_mode(
            trans_cfg_scale=trans_cfg_scale,
            prompt_cfg_scale=prompt_cfg_scale,
            text_condition=text_condition,
        )
        cfg_active_scale = max(float(trans_cfg_scale), float(prompt_cfg_scale))
        z = self.transformer.inference(
            input_token,
            type_id,
            input_pos,
            trans_cfg_scale=cfg_active_scale,
            input_cfg_expanded=cfg_active_scale > 1.0,
            text_condition=text_condition,
        )
        if cfg_active_scale > 1.0:
            z = self._combine_inference_transformer_guidance(
                z,
                trans_cfg_scale=trans_cfg_scale,
                prompt_cfg_scale=prompt_cfg_scale,
                cfg_mode=cfg_mode,
            )
        return z

    def diffusion_sampling(
        self,
        z: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        cfg: float = 1.0,
        dh_y_cfg: float = 1.0,
        dh_dino_cfg: float = 1.0,
        dh_prompt_cfg: float = 1.0,
        initial_noise: Optional[torch.Tensor] = None,
        device: torch.device = "cuda",
        **edm_kwargs,
    ):
        return self.diffloss.sample(
            z,
            local_cond=local_cond,
            cfg=cfg,
            dh_y_cfg=dh_y_cfg,
            dh_dino_cfg=dh_dino_cfg,
            dh_prompt_cfg=dh_prompt_cfg,
            initial_noise=initial_noise,
            device=device,
            **edm_kwargs,
        ).clone()

    def _sample_audio_video_summary_group(
        self,
        audio_chunk: torch.Tensor,
        cond_combined: torch.Tensor,
        chunk_index: int,
        input_pos: torch.Tensor,
        *,
        trans_cfg_scale: float,
        prompt_cfg_scale: float = 1.0,
        inference_noise: float,
        text_condition: Optional[TextCondition] = None,
        video_present: Optional[torch.Tensor | bool] = None,
        force_audio_null: bool = False,
        audio_history_null_mask: Optional[torch.Tensor] = None,
        preencoded_audio_summaries: Optional[torch.Tensor] = None,
        preencoded_audio_slots: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Feed K previous-audio and K current-video summaries as A,V pairs."""
        batch = cond_combined.shape[0]
        if force_audio_null and audio_history_null_mask is not None:
            raise ValueError("force_audio_null and audio_history_null_mask are mutually exclusive")
        if force_audio_null and preencoded_audio_summaries is not None:
            raise ValueError(
                "force_audio_null and preencoded audio summaries are mutually exclusive"
            )
        if force_audio_null:
            if audio_chunk.shape[0] != batch:
                raise ValueError("history-null audio/video batches must match")
            audio_slots = None
            audio_summaries = self.transformer.initial_audio_tokens(batch)
            if float(inference_noise) != 0.0:
                noise = torch.randn_like(audio_summaries)
                audio_summaries = (
                    inference_noise * noise + (1.0 - inference_noise) * audio_summaries
                )
        elif preencoded_audio_summaries is not None:
            audio_summaries = preencoded_audio_summaries
            audio_slots = preencoded_audio_slots
            if audio_summaries.ndim != 3:
                raise ValueError("preencoded audio summaries must have shape [B,K,C]")
            if audio_summaries.shape[0] != batch:
                raise ValueError("preencoded audio/video summary batches must match")
            if audio_summaries.shape[1] != 1:
                raise ValueError("preencoded audio summaries have the wrong slot count")
        else:
            (audio_summaries, audio_slots) = self._encode_audio_summary_group(
                audio_chunk, inference_noise
            )
        if audio_history_null_mask is not None:
            audio_history_null_mask = torch.as_tensor(
                audio_history_null_mask, device=audio_summaries.device, dtype=torch.bool
            )
            if tuple(audio_history_null_mask.shape) != (batch,):
                raise ValueError("audio_history_null_mask must have one value per batch row")
            learned_null = self.transformer.initial_audio_tokens(batch).to(
                device=audio_summaries.device, dtype=audio_summaries.dtype
            )
            audio_summaries = torch.where(
                audio_history_null_mask[:, None, None], learned_null, audio_summaries
            )
        start = int(chunk_index) * 1
        stop = start + 1
        if stop > cond_combined.shape[1]:
            raise IndexError(
                f"video summary group {chunk_index} exceeds {cond_combined.shape[1]} tokens"
            )
        audio_out = None
        video_out = None
        slot_audio_outputs: list[torch.Tensor] = []
        slot_video_outputs: list[torch.Tensor] = []
        for summary_index, token_index in enumerate(range(start, stop)):
            input_token = cond_combined[:, token_index].view(batch, 1, -1)
            if self._can_use_paired_ar_inference():
                video_token = self.video_proj(input_token)
                (audio_out, video_out) = self._sample_projected_transformer_pair(
                    audio_summaries[:, summary_index],
                    video_token,
                    input_pos,
                    trans_cfg_scale,
                    prompt_cfg_scale=prompt_cfg_scale,
                    text_condition=text_condition,
                    video_present=video_present,
                )
                input_pos += 2
            else:
                audio_out = self._sample_projected_transformer_token(
                    audio_summaries[:, summary_index],
                    type_id=1,
                    input_pos=input_pos,
                    trans_cfg_scale=trans_cfg_scale,
                    prompt_cfg_scale=prompt_cfg_scale,
                    text_condition=text_condition,
                )
                input_pos += 1
                video_out = self.transformer_sampling(
                    input_token,
                    type_id=0,
                    input_pos=input_pos,
                    trans_cfg_scale=trans_cfg_scale,
                    prompt_cfg_scale=prompt_cfg_scale,
                    inference_noise=inference_noise,
                    text_condition=text_condition,
                    video_present=video_present,
                )
                input_pos += 1
        if audio_out is None or video_out is None:
            raise RuntimeError("audio/video summary group must contain at least one pair")
        return (audio_out, video_out, audio_slots)

    def _sample_bos_video_summary_group(
        self,
        cond_combined: torch.Tensor,
        chunk_index: int,
        input_pos: torch.Tensor,
        *,
        trans_cfg_scale: float,
        prompt_cfg_scale: float = 1.0,
        inference_noise: float,
        text_condition: Optional[TextCondition] = None,
        video_present: Optional[torch.Tensor | bool] = None,
        return_audio_summaries: bool = False,
        preencoded_audio_summaries: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Feed K learned audio-start tokens paired with the first K videos."""
        batch = cond_combined.shape[0]
        cfg_mode = self._guidance_mode(
            trans_cfg_scale=trans_cfg_scale,
            prompt_cfg_scale=prompt_cfg_scale,
            text_condition=text_condition,
        )
        cfg_branches = transformer_cfg_branch_count(cfg_mode)
        cache_batch = batch * cfg_branches
        audio_starts = (
            self.transformer.initial_audio_tokens(cache_batch)
            if preencoded_audio_summaries is None
            else preencoded_audio_summaries
        )
        if audio_starts.ndim != 3 or tuple(audio_starts.shape[:2]) != (cache_batch, 1):
            raise ValueError(f"preencoded BOS summaries must have shape [{cache_batch},{1},C]")
        start = int(chunk_index) * 1
        stop = start + 1
        if stop > cond_combined.shape[1]:
            raise IndexError(
                f"video summary group {chunk_index} exceeds {cond_combined.shape[1]} tokens"
            )
        audio_out = None
        video_out = None
        slot_audio_outputs: list[torch.Tensor] = []
        slot_video_outputs: list[torch.Tensor] = []
        committed_audio_inputs: list[torch.Tensor] = []
        for summary_index, token_index in enumerate(range(start, stop)):
            input_token = cond_combined[:, token_index].view(batch, 1, -1)
            if self._can_use_paired_ar_inference():
                audio_token = audio_starts[:, summary_index]
                if preencoded_audio_summaries is None and float(inference_noise) != 0.0:
                    noise = torch.randn_like(audio_token)
                    audio_token = inference_noise * noise + (1 - inference_noise) * audio_token
                committed_audio_inputs.append(audio_token[:, None])
                video_token = self.video_proj(input_token)
                (audio_out, video_out) = self._sample_projected_transformer_pair(
                    audio_token,
                    video_token,
                    input_pos,
                    trans_cfg_scale,
                    prompt_cfg_scale=prompt_cfg_scale,
                    audio_cfg_expanded=cfg_branches > 1,
                    text_condition=text_condition,
                    video_present=video_present,
                )
                input_pos += 2
            else:
                if return_audio_summaries or preencoded_audio_summaries is not None:
                    raise RuntimeError(
                        "cache rebase requires paired or chunk-bidirectional AR inference"
                    )
                audio_out = self.bos_sampling(
                    audio_starts[:, summary_index],
                    type_id=1,
                    input_pos=input_pos,
                    trans_cfg_scale=trans_cfg_scale,
                    prompt_cfg_scale=prompt_cfg_scale,
                    inference_noise=inference_noise,
                    text_condition=text_condition,
                )
                input_pos += 1
                video_out = self.transformer_sampling(
                    input_token,
                    type_id=0,
                    input_pos=input_pos,
                    trans_cfg_scale=trans_cfg_scale,
                    prompt_cfg_scale=prompt_cfg_scale,
                    inference_noise=inference_noise,
                    text_condition=text_condition,
                    video_present=video_present,
                )
                input_pos += 1
        if audio_out is None or video_out is None:
            raise RuntimeError("BOS/video summary group must contain at least one pair")
        if return_audio_summaries:
            return (audio_out, video_out, torch.cat(committed_audio_inputs, dim=1).detach().clone())
        return (audio_out, video_out)

    def decode_n_tokens(
        self,
        cond_combined: torch.Tensor,
        local_video: Optional[torch.Tensor],
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_chunks: int,
        diff_cfg_scale: float = 1.0,
        trans_cfg_scale: float = 1.0,
        prompt_cfg_scale: float = 1.0,
        inference_noise: float = 0.2,
        text_condition: Optional[TextCondition] = None,
        dh_y_cfg_scale: float = 1.0,
        dh_dino_cfg_scale: float = 1.0,
        audio_valid_mask: Optional[torch.Tensor] = None,
        video_present: Optional[torch.Tensor | bool] = None,
        initial_noise_by_chunk: Optional[torch.Tensor] = None,
        start_chunk_index: int = 0,
        **edm_kwargs,
    ):
        new_audio_tokens = []
        history_queue: List[torch.Tensor] = []
        bs = cur_token.size(0)
        if not torch.is_tensor(input_pos):
            input_pos = torch.tensor([int(input_pos)], device=cur_token.device, dtype=torch.int)
        else:
            input_pos = input_pos.to(device=cur_token.device, dtype=torch.int)
            if input_pos.dim() == 0:
                input_pos = input_pos.view(1)
        start_chunk_index = int(start_chunk_index)
        if start_chunk_index < 0:
            raise ValueError("start_chunk_index must be non-negative")
        prompt_switch_flags: list[bool] = []
        rebase_records: list[tuple[int, bool, torch.Tensor]] = []
        for i in tqdm(
            range(num_new_chunks), desc="transformer sampling....", disable=not os.isatty(2)
        ):
            chunk_index = int(start_chunk_index + i + 1)
            chunk_text_condition = text_condition
            conditioning_audio = cur_token
            (z_out, v_out, audio_slots) = self._sample_audio_video_summary_group(
                conditioning_audio,
                cond_combined,
                chunk_index=chunk_index,
                input_pos=input_pos,
                trans_cfg_scale=trans_cfg_scale,
                prompt_cfg_scale=prompt_cfg_scale,
                inference_noise=inference_noise,
                text_condition=chunk_text_condition,
                video_present=video_present,
                audio_history_null_mask=None,
                preencoded_audio_summaries=None,
                preencoded_audio_slots=None,
            )
            z_token = torch.cat([z_out, v_out], dim=-1)
            if z_token.ndim == 3 and z_token.shape[1] == 1:
                z_token = z_token[:, 0]
            sync_chunk_index = chunk_index
            z_token = z_token.unsqueeze(1).squeeze(1)
            next_token = self.diffusion_sampling(
                z_token,
                local_cond=local_video[:, chunk_index] if local_video is not None else None,
                cfg=diff_cfg_scale,
                dh_y_cfg=dh_y_cfg_scale,
                dh_dino_cfg=dh_dino_cfg_scale,
                dh_prompt_cfg=1.0,
                initial_noise=initial_noise_by_chunk[:, chunk_index]
                if initial_noise_by_chunk is not None
                else None,
                device=z_token.device,
                **edm_kwargs,
            )
            next_token = self._clamp_padded_sampling_slots(
                next_token,
                audio_valid_mask=audio_valid_mask,
                chunk_index=chunk_index,
                previous_chunk=cur_token,
            )
            cur_token = next_token
            new_audio_tokens.append(cur_token.clone())
        return new_audio_tokens

    def _clamp_padded_sampling_slots(
        self,
        chunk: torch.Tensor,
        *,
        audio_valid_mask: Optional[torch.Tensor],
        chunk_index: int,
        previous_chunk: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fill masked dataset-only slots from the latest real latent.

        A 5 s LTX segment has 125 tokens and is padded to 126 solely so the
        three-slot model can process it.  The masked slot has no flow target;
        during finite-clip evaluation it must therefore be deterministic rather
        than an unconstrained diffusion sample.  Natural streaming chunks never
        supply this mask and remain unchanged.
        """
        if audio_valid_mask is None:
            return chunk
        if chunk.ndim != 3 or chunk.shape[1] != 3:
            raise ValueError("sampling validity requires [B,S,D] chunks")
        if audio_valid_mask.ndim != 3 or audio_valid_mask.shape[2] != 3:
            raise ValueError("sampling audio_valid_mask must be [B,N,S]")
        if chunk_index >= audio_valid_mask.shape[1]:
            raise ValueError("sampling validity mask is shorter than generation")
        valid = audio_valid_mask[:, chunk_index].to(device=chunk.device, dtype=torch.bool)
        if valid.shape != chunk.shape[:2]:
            raise ValueError("sampling validity/chunk batch axes differ")
        output = chunk.clone()
        for slot in range(3):
            if slot == 0:
                if previous_chunk is None:
                    replacement = output[:, slot]
                else:
                    replacement = previous_chunk[:, -1]
            else:
                replacement = output[:, slot - 1]
            output[:, slot] = torch.where(valid[:, slot, None], output[:, slot], replacement)
        return output

    @torch.no_grad()
    def offline_sampling(
        self,
        cond: Optional[torch.Tensor],
        max_new_tokens: int,
        context_mode: str = "swa",
        trans_cfg_scale: float = 1.0,
        prompt_cfg_scale: float = 1.0,
        transformer_guidance_method: str = "cfg",
        apg_eta: float = 1.0,
        apg_norm_threshold: float = 0.0,
        apg_video_norm_threshold: Optional[float] = None,
        apg_prompt_norm_threshold: Optional[float] = None,
        dh_y_cfg_scale: float = 1.0,
        dh_dino_cfg_scale: float = 1.0,
        diff_cfg_scale: float = 1.0,
        inference_noise: float = 0.2,
        prompt_embedding: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        prompt_start_audio_chunks: Optional[torch.Tensor] = None,
        prompt_duration_audio_chunks: Optional[torch.Tensor] = None,
        semantic_feature: Optional[torch.Tensor] = None,
        sync_repa_feature: Optional[torch.Tensor] = None,
        sync_repa_valid: Optional[torch.Tensor] = None,
        audio_valid_mask: Optional[torch.Tensor] = None,
        video_present: Optional[torch.Tensor | bool] = None,
        batch_size: Optional[int] = None,
        initial_noise_by_chunk: Optional[torch.Tensor] = None,
        rope_position_offset: int = 0,
        force_ring_kv: bool = False,
        condition_repeat_count: int = 1,
        **edm_kwargs,
    ):
        transformer_guidance_method = str(transformer_guidance_method).strip().lower()
        if transformer_guidance_method not in {"cfg", "apg"}:
            raise ValueError("transformer_guidance_method must be 'cfg' or 'apg'")
        if not 0.0 <= float(apg_eta) <= 1.0:
            raise ValueError("apg_eta must lie in [0,1]")
        for name, value in (
            ("apg_norm_threshold", apg_norm_threshold),
            ("apg_video_norm_threshold", apg_video_norm_threshold),
            ("apg_prompt_norm_threshold", apg_prompt_norm_threshold),
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        self._inference_guidance_method = transformer_guidance_method
        self._inference_apg_eta = float(apg_eta)
        self._inference_apg_norm_threshold = float(apg_norm_threshold)
        self._inference_apg_video_norm_threshold = (
            None if apg_video_norm_threshold is None else float(apg_video_norm_threshold)
        )
        self._inference_apg_prompt_norm_threshold = (
            None if apg_prompt_norm_threshold is None else float(apg_prompt_norm_threshold)
        )
        has_video_input = cond is not None
        if cond is not None:
            expected_max_new_tokens = int(cond.shape[1] * 2 - 1)
            if max_new_tokens != expected_max_new_tokens:
                raise ValueError(
                    f"expected max_new_tokens={expected_max_new_tokens}, got {max_new_tokens}"
                )
            inferred_batch_size = int(cond.shape[0])
            device = cond.device
            null_num_chunks = None
        else:
            if max_new_tokens < 1 or max_new_tokens % 2 != 1:
                raise ValueError("no-video sampling requires a positive odd max_new_tokens")
            input_video_tokens = (int(max_new_tokens) + 1) // 2
            if input_video_tokens % 3:
                raise ValueError(
                    "no-video max_new_tokens does not describe a whole number of audio chunks"
                )
            null_num_chunks = input_video_tokens // 3
            inferred_batch_size = (
                int(prompt_embedding.shape[0])
                if prompt_embedding is not None
                else int(batch_size or 1)
            )
            device = (
                prompt_embedding.device
                if prompt_embedding is not None
                else next(self.parameters()).device
            )
        if batch_size is not None and int(batch_size) != inferred_batch_size:
            raise ValueError(
                f"batch_size disagrees with the supplied condition batch: {batch_size} != {inferred_batch_size}"
            )
        batch_size = inferred_batch_size
        video_present = self._resolve_video_present(
            video_present, batch_size=batch_size, device=device, has_video_input=has_video_input
        )
        if cond is not None:
            cond = self._mask_missing_video_rows(cond, video_present)
        (trans_cfg_scale, prompt_cfg_scale, guidance_mode) = self._resolve_transformer_guidance(
            trans_cfg_scale=trans_cfg_scale, prompt_cfg_scale=prompt_cfg_scale
        )
        semantic_feature = self._mask_missing_video_rows(None, video_present)
        semantic_feature = None
        local_video = None
        if not has_video_input:
            (cond_combined, local_video) = self._empty_video_conditions(
                batch_size=batch_size,
                num_chunks=int(null_num_chunks),
                device=device,
                dtype=self.video_proj[0].weight.dtype,
            )
        else:
            if cond.ndim != 5:
                raise ValueError(f"grid video must be [B,T,H,W,C], got {tuple(cond.shape)}")
            video = cond
            video = self._prepare_video_input(video)
            video = video.permute(0, 4, 1, 2, 3).contiguous(memory_format=torch.channels_last_3d)
            video = self.video_proj_3d(video)
            (B, D, T, H2, W2) = video.shape
            video = rearrange(video, "b c t h w -> (b t) (h w) c")
            video = self.aggregate_transformer(
                video,
                self.aggregated_tokens.unsqueeze(0).expand(B * T, -1, -1),
                spatial_shape=(H2, W2),
            )
            video = video.view(B, T, -1, D)
            if video.shape[2] == 1:
                cond_combined = video.squeeze(2)
            else:
                cond_combined = video.view(B, T * video.shape[2], D)
            (cond_combined, encoded_local_video) = self._encode_video_chunks_with_slots(
                cond_combined
            )
            local_video = encoded_local_video
        condition_repeat_count = int(condition_repeat_count)
        if condition_repeat_count < 1:
            raise ValueError("condition_repeat_count must be positive")
        if condition_repeat_count > 1:
            repeated_auxiliary = {
                "prompt_embedding": prompt_embedding,
                "prompt_mask": prompt_mask,
                "prompt_start_audio_chunks": prompt_start_audio_chunks,
                "prompt_duration_audio_chunks": prompt_duration_audio_chunks,
                "semantic_feature": semantic_feature,
                "sync_repa_feature": sync_repa_feature,
                "sync_repa_valid": sync_repa_valid,
                "audio_valid_mask": audio_valid_mask,
                "h0_inference_noise_by_chunk": None,
                "joint_history_dropout_mask_by_chunk": None,
            }
            supplied = [name for (name, value) in repeated_auxiliary.items() if value is not None]
            if supplied:
                raise ValueError(
                    "condition_repeat_count only supports the no-Prompt, video-only drift diagnostic; supplied "
                    + ", ".join(supplied)
                )
            repeat = (1, condition_repeat_count) + (1,) * (cond_combined.ndim - 2)
            cond_combined = cond_combined.repeat(*repeat)
            if local_video is not None:
                repeat = (1, condition_repeat_count) + (1,) * (local_video.ndim - 2)
                local_video = local_video.repeat(*repeat)
        if cond_combined.shape[1] % 1:
            raise ValueError("video summary token count must divide into generation chunks")
        num_chunks = cond_combined.shape[1] // 1
        if initial_noise_by_chunk is not None:
            expected_noise_shape = (batch_size, num_chunks, 3, 64)
            if tuple(initial_noise_by_chunk.shape) != expected_noise_shape:
                raise ValueError(
                    f"chunk noise schedule differs from the sampling ABI: {tuple(initial_noise_by_chunk.shape)} != {expected_noise_shape}"
                )
            initial_noise_by_chunk = initial_noise_by_chunk.to(
                device=device, dtype=self.video_proj[0].weight.dtype
            )
        sampling_validity = None
        if audio_valid_mask is not None:
            expected_shape = (batch_size, num_chunks * 3)
            if tuple(audio_valid_mask.shape) != expected_shape:
                raise ValueError(
                    f"sampling audio_valid_mask must match generated audio slots: {tuple(audio_valid_mask.shape)} != {expected_shape}"
                )
            sampling_validity = audio_valid_mask.to(device=device, dtype=torch.bool).reshape(
                batch_size, num_chunks, 3
            )
        if sync_repa_feature is not None or sync_repa_valid is not None:
            raise ValueError("sampling Sync tensors reached an injection-disabled model")
        semantic_condition = None
        local_video = self._mask_missing_video_rows(local_video, video_present)
        T_new = num_chunks * 2
        rope_position_offset = int(rope_position_offset)
        force_ring_kv = bool(force_ring_kv)
        if rope_position_offset < 0:
            raise ValueError("rope_position_offset must be non-negative")
        if force_ring_kv and context_mode not in {"swa", "sliding"}:
            raise ValueError("force_ring_kv requires context_mode=swa")
        context_token_count = T_new + rope_position_offset
        if force_ring_kv:
            self.transformer.set_context_extension(
                "swa", window_size=int(self.transformer.causal_window_size)
            )
        else:
            self._configure_inference_context(context_mode, token_count=context_token_count)
        max_seq_length = max(context_token_count, 0)
        max_batch_size = batch_size
        max_batch_size_cfg = max_batch_size * transformer_cfg_branch_count(guidance_mode)
        infer_dtype = self.video_proj[0].weight.dtype
        text_condition = self.transformer.prepare_text_condition(
            prompt_embedding,
            prompt_mask,
            batch_size=max_batch_size,
            device=device,
            dtype=infer_dtype,
            cfg_mode=guidance_mode,
            apply_dropout=False,
            prompt_start_chunks=prompt_start_audio_chunks,
            prompt_duration_chunks=prompt_duration_audio_chunks,
            query_token_count=T_new,
            query_tokens_per_chunk=2,
        )
        (dh_prompt_all, dh_prompt_mask_all) = (None, None)
        dh_prompt_cond = dh_prompt_all
        dh_prompt_mask = dh_prompt_mask_all
        with torch.device(device):
            self.transformer.setup_caches(
                max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=infer_dtype
            )
        seq_shape = (max_batch_size, num_chunks, 3, 64)
        seq = torch.empty(seq_shape, dtype=torch.float, device=device)
        pos = torch.tensor([rope_position_offset], device=device, dtype=torch.int)
        bos_result = self._sample_bos_video_summary_group(
            cond_combined,
            chunk_index=0,
            input_pos=pos,
            trans_cfg_scale=trans_cfg_scale,
            prompt_cfg_scale=prompt_cfg_scale,
            inference_noise=inference_noise,
            text_condition=text_condition,
            video_present=video_present,
            return_audio_summaries=False,
        )
        (z, v_init_token) = bos_result
        z = torch.cat([z, v_init_token], dim=-1)
        if z.ndim == 3 and z.shape[1] == 1:
            z = z[:, 0]
        z = z.unsqueeze(1).squeeze(1)
        next_token = self.diffusion_sampling(
            z,
            local_cond=local_video[:, 0] if local_video is not None else None,
            cfg=diff_cfg_scale,
            dh_y_cfg=dh_y_cfg_scale,
            dh_dino_cfg=dh_dino_cfg_scale,
            dh_prompt_cfg=1.0,
            device=z.device,
            initial_noise=initial_noise_by_chunk[:, 0]
            if initial_noise_by_chunk is not None
            else None,
            **edm_kwargs,
        )
        next_token = self._clamp_padded_sampling_slots(
            next_token, audio_valid_mask=sampling_validity, chunk_index=0, previous_chunk=None
        )
        seq[:, 0] = next_token
        generated_tokens = self.decode_n_tokens(
            cond_combined=cond_combined,
            local_video=local_video,
            cur_token=next_token,
            input_pos=pos,
            num_new_chunks=num_chunks - 1,
            diff_cfg_scale=diff_cfg_scale,
            trans_cfg_scale=trans_cfg_scale,
            prompt_cfg_scale=prompt_cfg_scale,
            inference_noise=inference_noise,
            text_condition=text_condition,
            dh_y_cfg_scale=dh_y_cfg_scale,
            dh_dino_cfg_scale=dh_dino_cfg_scale,
            audio_valid_mask=sampling_validity,
            video_present=video_present,
            initial_noise_by_chunk=initial_noise_by_chunk,
            **edm_kwargs,
        )
        if generated_tokens:
            seq[:, 1:] = torch.stack(generated_tokens, dim=1)
        return seq.flatten(1, 2)
