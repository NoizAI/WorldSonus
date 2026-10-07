from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.jit
from worldsonus.sampling import flow_matching_sampler
from typing import Callable, Optional


def _single_axis_flow_head_apg_update(
    current: torch.Tensor,
    time: torch.Tensor,
    full_velocity: torch.Tensor,
    null_velocity: torch.Tensor,
    previous_momentum: torch.Tensor,
    use_previous_momentum: bool,
    scale: float,
    eta: float,
    norm_threshold: float,
    momentum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one mathematically independent APG axis as a pure tensor graph.

    The FP64 projection and uncapped momentum state deliberately match the
    established eager implementation. Keeping validation and dictionaries
    outside this function lets Inductor fuse the small per-step CUDA kernels.
    """
    dimensions = tuple(range(1, current.ndim))
    clean_factor = (1.0 - time.reshape(-1)).to(device=current.device, dtype=full_velocity.dtype)
    clean_factor = clean_factor.reshape((current.shape[0],) + (1,) * (current.ndim - 1))
    full_clean = current + clean_factor * full_velocity
    difference = clean_factor * (full_velocity - null_velocity)
    if use_previous_momentum:
        difference = difference + float(momentum) * previous_momentum
    next_momentum = difference
    if float(norm_threshold) > 0.0:
        difference_norm = torch.linalg.vector_norm(
            difference.float(), dim=dimensions, keepdim=True
        ).clamp_min(1e-12)
        difference = difference * torch.clamp(float(norm_threshold) / difference_norm, max=1.0).to(
            dtype=difference.dtype
        )
    difference64 = difference.double()
    reference64 = F.normalize(full_clean.double(), dim=dimensions)
    parallel64 = (difference64 * reference64).sum(dim=dimensions, keepdim=True) * reference64
    projected64 = difference64 - parallel64 + float(eta) * parallel64
    guided_velocity = (
        full_velocity.double() + (float(scale) - 1.0) * projected64 / clean_factor.double()
    )
    return (guided_velocity.to(dtype=full_velocity.dtype), next_momentum)


_compiled_single_axis_flow_head_apg_update = torch.compile(
    _single_axis_flow_head_apg_update,
    backend="inductor",
    mode="default",
    fullgraph=True,
    dynamic=False,
)


def _combine_flow_head_guidance(
    current: torch.Tensor,
    time: torch.Tensor,
    full_velocity: torch.Tensor,
    null_velocities: dict[str, torch.Tensor],
    scales: dict[str, float],
    *,
    method: str = "cfg",
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
    apg_momentum: float = 0.0,
    momentum_buffers: Optional[dict[str, torch.Tensor]] = None,
    guidance_update_observer: Optional[Callable[[str, torch.Tensor], None]] = None,
    _trusted_euler_time: bool = False,
) -> torch.Tensor:
    """Combine independent Flow-Head guidance axes at one Euler step.

    The Flow Head is trained with ``x_t=(1-t)*noise+t*clean`` and predicts
    ``velocity=clean-noise``. APG must project denoised predictions rather
    than raw velocity predictions, so the conditional clean prediction is
    reconstructed as ``x_t+(1-t)*velocity``. Each leave-one-axis-out branch
    receives its own APG update before the scaled axis updates are added.

    ``apg_norm_threshold`` applies to the unscaled denoised-prediction
    difference, exactly where the APG paper applies its radius. Setting
    ``eta=1``, ``threshold=0``, and ``momentum=0`` takes an explicit CFG fast
    path so the established arithmetic remains unchanged.
    """
    method = str(method).strip().lower()
    if method not in {"cfg", "apg"}:
        raise ValueError(f"unsupported Flow Head guidance method: {method}")
    if not 0.0 <= float(apg_eta) <= 1.0:
        raise ValueError("Flow Head APG eta must lie in [0,1]")
    if not math.isfinite(float(apg_norm_threshold)) or float(apg_norm_threshold) < 0.0:
        raise ValueError("Flow Head APG norm threshold must be finite and non-negative")
    if not math.isfinite(float(apg_momentum)) or not -1.0 < float(apg_momentum) < 1.0:
        raise ValueError("Flow Head APG momentum must lie in (-1,1)")
    if set(null_velocities) != set(scales):
        raise ValueError("Flow Head guidance branches/scales do not match")
    if any((not math.isfinite(float(scale)) or float(scale) < 0.0 for scale in scales.values())):
        raise ValueError("Flow Head CFG scales must be finite and non-negative")
    cfg_fast_path = method == "cfg" or (
        float(apg_eta) == 1.0 and float(apg_norm_threshold) == 0.0 and (float(apg_momentum) == 0.0)
    )
    cfg_velocity = None
    if cfg_fast_path:
        cfg_velocity = full_velocity
        for axis, null_velocity in null_velocities.items():
            cfg_velocity = cfg_velocity + (float(scales[axis]) - 1.0) * (
                full_velocity - null_velocity
            )
    time = time.reshape(-1)
    if time.numel() == 1:
        time = time.expand(current.shape[0])
    if time.numel() != current.shape[0]:
        raise ValueError("Flow Head APG time must be scalar or one per sample")
    if not _trusted_euler_time:
        validation_factor = (1.0 - time).to(device=current.device, dtype=full_velocity.dtype)
        if bool((validation_factor <= 0).any()):
            raise ValueError("Flow Head APG requires Euler times strictly below one")
    if (
        not cfg_fast_path
        and len(null_velocities) == 1
        and (guidance_update_observer is None)
        and (current.device.type == "cuda")
        and (time.device == current.device)
    ):
        if momentum_buffers is None:
            momentum_buffers = {}
        (axis, null_velocity) = next(iter(null_velocities.items()))
        previous = momentum_buffers.get(axis)
        use_previous = previous is not None and float(apg_momentum) != 0.0
        if previous is None:
            previous = full_velocity
        (guided_velocity, next_momentum) = _compiled_single_axis_flow_head_apg_update(
            current,
            time,
            full_velocity,
            null_velocity,
            previous,
            use_previous,
            float(scales[axis]),
            float(apg_eta),
            float(apg_norm_threshold),
            float(apg_momentum),
        )
        if float(apg_momentum) != 0.0:
            momentum_buffers[axis] = next_momentum
        return guided_velocity
    clean_factor = (1.0 - time).to(device=current.device, dtype=full_velocity.dtype)
    clean_factor = clean_factor.reshape((current.shape[0],) + (1,) * (current.ndim - 1))
    full_clean = current + clean_factor * full_velocity
    raw_clean_differences = {
        axis: clean_factor * (full_velocity - null_velocity)
        for (axis, null_velocity) in null_velocities.items()
    }
    if guidance_update_observer is not None:
        for axis, difference in raw_clean_differences.items():
            guidance_update_observer(axis, difference)
    if cfg_fast_path:
        return cfg_velocity
    if momentum_buffers is None:
        momentum_buffers = {}
    dimensions = tuple(range(1, current.ndim))
    guided_velocity64 = full_velocity.double()
    for axis, raw_difference in raw_clean_differences.items():
        difference = raw_difference
        if float(apg_momentum) != 0.0:
            previous = momentum_buffers.get(axis)
            if previous is not None:
                difference = difference + float(apg_momentum) * previous
            momentum_buffers[axis] = difference
        if float(apg_norm_threshold) > 0.0:
            difference_norm = torch.linalg.vector_norm(
                difference.float(), dim=dimensions, keepdim=True
            ).clamp_min(1e-12)
            difference = difference * torch.clamp(
                float(apg_norm_threshold) / difference_norm, max=1.0
            ).to(dtype=difference.dtype)
        difference64 = difference.double()
        reference64 = F.normalize(full_clean.double(), dim=dimensions)
        parallel64 = (difference64 * reference64).sum(dim=dimensions, keepdim=True) * reference64
        orthogonal64 = difference64 - parallel64
        projected = orthogonal64 + float(apg_eta) * parallel64
        guided_velocity64 = (
            guided_velocity64 + (float(scales[axis]) - 1.0) * projected / clean_factor.double()
        )
    return guided_velocity64.to(dtype=full_velocity.dtype)


def call_diffusion_head_with_runtime_autocast_boundary(diffusion_head: nn.Module, *args, **kwargs):
    """Enter a disabled ambient autocast context for selected compiled heads.

    The runtime marks only targets whose forward recreates BF16 autocast
    internally.  This emulates PyTorch's newer
    ``backward_pass_autocast='off'`` behavior on the production PyTorch 2.6
    stack without changing the eager/default path.
    """
    if not getattr(diffusion_head, "_worldsonus_runtime_backward_autocast_off", False):
        return diffusion_head(*args, **kwargs)
    first_tensor = next((value for value in args if torch.is_tensor(value)))
    with torch.autocast(device_type=first_tensor.device.type, enabled=False):
        return diffusion_head(*args, **kwargs)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, 1280, bias=True), nn.SiLU(), nn.Linear(1280, 1280, bias=True)
        )
        freqs = torch.exp(-9.210340371976184 * torch.arange(128, dtype=torch.float32) / 128)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq.to(dtype=self.mlp[0].weight.dtype))
        return t_emb


@torch.compiler.disable
def mp_sum(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    """
    Magnitude-preserving sum of two tensors (Equation 88).
    Performs a linear interpolation (lerp) between a and b with factor t,
    and then divides by sqrt((1-t)^2 + t^2).
    """
    denom = math.sqrt((1 - t) ** 2 + t**2)
    blended = a.lerp(b, t)
    if a.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        return blended.float() / denom
    return blended / denom


def bounded_modulate(x, shift, scale, gain_limit: float = 1.0):
    """Bound AdaLN gain while preserving ``gain(0)=gain'(0)=1``."""
    return x * (1 + gain_limit * torch.tanh(scale / gain_limit)) + shift


class QKNormMultiheadAttention(nn.MultiheadAttention):
    """Multi-head attention with per-head, parameter-free Q/K RMSNorm."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-06, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-06, elementwise_affine=False)
        self.key_projection: Optional[nn.Module] = None

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        (batch_size, sequence_length, _) = tensor.shape
        return tensor.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = False,
        is_causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        key_condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("QK-normalized temporal attention expects batched rank-3 inputs")
        if need_weights:
            raise ValueError("QK-normalized temporal attention does not return attention weights")
        (q_weight, k_weight, v_weight) = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            (q_bias, k_bias, v_bias) = self.in_proj_bias.chunk(3, dim=0)
        q = self._split_heads(F.linear(query, q_weight, q_bias))
        projected_key = F.linear(key, k_weight, k_bias)
        if key_condition is not None:
            raise ValueError("key_condition was provided to attention without K-projection")
        k = self._split_heads(projected_key)
        v = self._split_heads(F.linear(value, v_weight, v_bias))
        q_dtype = q.dtype
        k_dtype = k.dtype
        (q, k) = (q.float(), k.float())
        q = self.q_norm(q)
        k = self.k_norm(k)
        (q, k) = (q.to(dtype=q_dtype), k.to(dtype=k_dtype))
        h = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        h = h.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[1], self.embed_dim)
        return (self.out_proj(h), None)


class AdaptiveResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_rms = nn.RMSNorm(1280, eps=1e-06)
        self.mlp = nn.Sequential(
            nn.Linear(1280, 5120, bias=True),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(5120, 1280, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(1280, 3840, bias=True))

    def forward(self, x, y):
        (shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(y).chunk(3, dim=-1)
        modulation = bounded_modulate
        h = modulation(self.in_rms(x), shift_mlp, scale_mlp, 2.0)
        h = self.mlp(h)
        gate = torch.tanh(gate_mlp)
        return x + gate * h


class TemporalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_rms = nn.RMSNorm(1280, eps=1e-06)
        self.attention = QKNormMultiheadAttention(1280, 8, dropout=0.1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(1280, 3840, bias=True))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, *, token_valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        (shift, scale, gate) = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = bounded_modulate(self.in_rms(x), shift, scale, 2.0)
        attention_mask = None
        use_causal_fastpath = False
        if token_valid is not None:
            if tuple(token_valid.shape) != x.shape[:2]:
                raise ValueError("temporal attention validity must have shape [B,S]")
            token_valid = token_valid.to(device=x.device, dtype=torch.bool)
            attention_mask = token_valid[:, None, None, :]
            use_causal_fastpath = False
        (h, _) = self.attention(
            h, h, h, need_weights=False, is_causal=use_causal_fastpath, attn_mask=attention_mask
        )
        return x + torch.tanh(gate) * h


class TemporalCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_rms = nn.RMSNorm(1280, eps=1e-06)
        self.context_rms = nn.RMSNorm(1280, eps=1e-06)
        self.attention = QKNormMultiheadAttention(1280, 8, dropout=0.1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(1280, 3840, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
        context_valid: Optional[torch.Tensor] = None,
        key_condition: Optional[torch.Tensor] = None,
        gate_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (shift, scale, gate) = self.adaLN_modulation(y).chunk(3, dim=-1)
        if gate_condition is not None:
            if gate_condition.shape != y.shape:
                raise ValueError("cross-attention gate_condition must match y shape")
            gate_input = self.adaLN_modulation[0](gate_condition)
            gate_linear = self.adaLN_modulation[-1]
            gate = F.linear(
                gate_input,
                gate_linear.weight[2 * x.shape[-1] :],
                None if gate_linear.bias is None else gate_linear.bias[2 * x.shape[-1] :],
            )
        query = bounded_modulate(self.query_rms(x), shift, scale, 2.0)
        context = self.context_rms(context)
        key_condition = None
        (h, _) = self.attention(
            query, context, context, need_weights=False, key_condition=key_condition
        )
        if context_valid is not None:
            h = h * context_valid.to(dtype=h.dtype).reshape(-1, 1, 1)
        return x + torch.tanh(gate) * h


class OutputProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_final = nn.RMSNorm(1280, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(1280, 64, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(1280, 2560, bias=True))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, c):
        (shift, scale) = self.adaLN_modulation(c).chunk(2, dim=-1)
        modulation = bounded_modulate
        x = modulation(self.norm_final(x), shift, scale, 2.0)
        x = self.dropout(x)
        return self.linear(x)


class FlowSampler(nn.Module):
    """Euler flow sampler for the final conditional head."""

    def __init__(self):
        super().__init__()
        self.diffusion_head = FlowHead()

    def sample(
        self,
        z: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        cfg: float = 1.0,
        dh_y_cfg: float = 1.0,
        dh_dino_cfg: float = 1.0,
        dh_prompt_cfg: float = 1.0,
        flow_head_guidance_method: str = "cfg",
        flow_head_apg_eta: float = 1.0,
        flow_head_apg_norm_threshold: float = 0.0,
        flow_head_apg_momentum: float = 0.0,
        device: torch.device = "cuda",
        num_steps: int = 20,
        solver: str = "euler",
        initial_noise: Optional[torch.Tensor] = None,
        sampling_start_time: float = 0.0,
        **unused_sampler_kwargs,
    ) -> torch.Tensor:
        noise_shape = (z.shape[0], 64)
        if self.diffusion_head.chunk_pipeline:
            noise_shape = (z.shape[0], 3, 64)
        if initial_noise is None:
            noise = torch.randn(noise_shape, device=device)
        else:
            if tuple(initial_noise.shape) != noise_shape:
                raise ValueError(
                    f"initial diffusion noise shape differs from the head ABI: {tuple(initial_noise.shape)} != {noise_shape}"
                )
            noise = initial_noise.to(device=device, dtype=z.dtype)
        net_kwargs = {"local_cond": local_cond, "history_cond": None, "history_valid": None}
        multi_cfg = any((float(value) != 1.0 for value in (dh_y_cfg, dh_dino_cfg, dh_prompt_cfg)))
        flow_head_guidance_method = str(flow_head_guidance_method).strip().lower()
        if flow_head_guidance_method not in {"cfg", "apg"}:
            raise ValueError("flow_head_guidance_method must be 'cfg' or 'apg'")
        if flow_head_guidance_method == "apg" and (not multi_cfg):
            raise ValueError("Flow Head APG requires an active y/DINO/Prompt axis")
        if not 0.0 <= float(flow_head_apg_eta) <= 1.0:
            raise ValueError("flow_head_apg_eta must lie in [0,1]")
        if (
            not math.isfinite(float(flow_head_apg_norm_threshold))
            or float(flow_head_apg_norm_threshold) < 0.0
        ):
            raise ValueError("flow_head_apg_norm_threshold must be finite and non-negative")
        if (
            not math.isfinite(float(flow_head_apg_momentum))
            or not -1.0 < float(flow_head_apg_momentum) < 1.0
        ):
            raise ValueError("flow_head_apg_momentum must lie in (-1,1)")
        solver = str(solver).strip().lower()
        if solver not in {"euler", "heun"}:
            raise ValueError(f"unsupported flow matching solver: {solver}")
        sampling_start_time = float(sampling_start_time)
        if not math.isfinite(sampling_start_time) or not 0.0 <= sampling_start_time < 1.0:
            raise ValueError("sampling_start_time must lie in [0,1)")
        if solver != "euler" and (False or multi_cfg):
            raise ValueError("Heun sampling is not implemented for independent DH/history CFG")
        if multi_cfg:
            if float(cfg) != 1.0:
                raise ValueError("legacy diff CFG and independent DH CFG cannot be combined")
            sampled_token_latent = noise
            step_size = (1.0 - sampling_start_time) / int(num_steps)
            apg_momentum_buffers: dict[str, torch.Tensor] = {}
            needs_guidance_components = bool(
                flow_head_guidance_method == "apg"
                or getattr(self, "_record_flow_head_guidance_norms", False)
            )
            for step in range(int(num_steps)):
                time = torch.full(
                    (noise.shape[0],),
                    sampling_start_time + step * step_size,
                    device=noise.device,
                    dtype=torch.float32,
                )
                if float(dh_prompt_cfg) != 1.0:
                    raise ValueError("legacy LH heads have no direct DH Prompt branch")
                velocity = self.diffusion_head.inference_multi_cfg(
                    sampled_token_latent,
                    time,
                    z,
                    y_guidance=dh_y_cfg,
                    dino_guidance=dh_dino_cfg,
                    return_branches=needs_guidance_components,
                    **net_kwargs,
                )
                if needs_guidance_components:
                    (full_velocity, null_velocities) = velocity
                    configured_scales = {
                        "y": float(dh_y_cfg),
                        "dino": float(dh_dino_cfg),
                        "prompt": float(dh_prompt_cfg),
                    }
                    active_scales = {axis: configured_scales[axis] for axis in null_velocities}
                    velocity = _combine_flow_head_guidance(
                        sampled_token_latent,
                        time,
                        full_velocity,
                        null_velocities,
                        active_scales,
                        method=flow_head_guidance_method,
                        apg_eta=flow_head_apg_eta,
                        apg_norm_threshold=flow_head_apg_norm_threshold,
                        apg_momentum=flow_head_apg_momentum,
                        momentum_buffers=apg_momentum_buffers,
                        guidance_update_observer=self._record_flow_head_guidance_update
                        if getattr(self, "_record_flow_head_guidance_norms", False)
                        else None,
                        _trusted_euler_time=True,
                    )
                velocity = velocity.to(dtype=noise.dtype)
                sampled_token_latent = sampled_token_latent + step_size * velocity
        else:
            sampled_token_latent = flow_matching_sampler(
                net=self.diffusion_head,
                noise=noise,
                labels=z,
                num_steps=num_steps,
                guidance=cfg,
                dtype=noise.dtype,
                net_kwargs=net_kwargs,
                solver=solver,
                start_time=sampling_start_time,
            )
        return sampled_token_latent.to(dtype=z.dtype)

    @torch.no_grad()
    def _record_flow_head_guidance_update(self, axis: str, difference: torch.Tensor) -> None:
        dimensions = tuple(range(1, difference.ndim))
        values = torch.linalg.vector_norm(difference.float(), dim=dimensions).detach()
        self._flow_head_guidance_norm_trace.setdefault(str(axis), []).append(values)


class FlowHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_pipeline = True
        self.flow_head_cfg_dropout_p = 0.2
        self.label_drop_prob = 0.1
        self.fake_latent = nn.Parameter(torch.zeros(1, 1280))
        self.input_proj = nn.Sequential(nn.Linear(64, 1280), nn.Dropout(0.1))
        res_blocks = []
        for i in range(8):
            if i < 2:
                res_blocks.append(TemporalSelfAttention())
            else:
                res_blocks.append(AdaptiveResidualBlock())
        self.res_blocks = nn.ModuleList(res_blocks)
        with torch.random.fork_rng(devices=[]):
            self.extra_temporal_attention_blocks = nn.ModuleList(
                [TemporalSelfAttention() for _ in range(1)]
            )
            self.extra_cross_attention_blocks = nn.ModuleList(
                [TemporalCrossAttention() for _ in range(1)]
            )
            late_history_candidates = []
            if True and late_history_candidates:
                object.__setattr__(
                    self, "_disabled_late_history_attention_candidate", late_history_candidates[0]
                )
            self.full_stage_cross_attention_blocks = nn.ModuleList(
                [TemporalCrossAttention() for _ in range(1)]
            )
            self.full_stage_temporal_attention_blocks = nn.ModuleList(
                [TemporalSelfAttention() for _ in range(1)]
            )
            self.full_stage_mlp_blocks = nn.ModuleList([AdaptiveResidualBlock() for _ in range(2)])
        if any((position < 0 or position > len((0, 2, 3, 4, 5, 1, 6, 7)) for position in (3,))):
            raise ValueError("extra attention execution position is out of range")
        extra_by_execution_position = [-1] * 9
        for extra_index, position in enumerate((3,)):
            extra_by_execution_position[position] = extra_index
        self._extra_attention_index_by_execution_position = tuple(extra_by_execution_position)
        self.final_layer = OutputProjection()
        self.time_embed = TimestepEmbedder()
        self.cond_embed = nn.Sequential(nn.Linear(2048, 1280), nn.SiLU(), nn.Linear(1280, 1280))
        self.slot_position = nn.Parameter(torch.randn(1, 3, 1280) / math.sqrt(1280))
        self.local_cond_embed = nn.Sequential(
            nn.Linear(512, 1280), nn.SiLU(), nn.Linear(1280, 1280)
        )
        self.cross_attention_blocks = nn.ModuleList([TemporalCrossAttention() for _ in range(2)])
        history_cross_modules = [TemporalCrossAttention() for _ in range(0)]
        if tuple(sorted(set(()))) != ():
            raise ValueError("history cross-attention block indices must be unique and increasing")
        history_adapter_by_block = [-1] * 8
        for adapter_index, block_index in enumerate(()):
            history_adapter_by_block[block_index] = adapter_index
        self._history_adapter_index_by_block = tuple(history_adapter_by_block)
        self.camera_conditioning_mode: Optional[str] = None
        self.camera_channels: Optional[int] = None
        self.camera_conditioning_active_blocks: tuple[int, ...] = ()
        self.spatial_conditioning_active_blocks: tuple[int, ...] = ()
        self.initialize_weights()

    def initialize_weights(self):

        def _init_linear(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.input_proj.apply(_init_linear)
        self.cond_embed.apply(_init_linear)
        if hasattr(self, "local_cond_embed"):
            self.local_cond_embed.apply(_init_linear)
        if hasattr(self, "history_cond_embed"):
            self.history_cond_embed.apply(_init_linear)
        self.time_embed.mlp.apply(_init_linear)
        if hasattr(self, "logvar_pe"):
            self.logvar_pe.mlp.apply(_init_linear)
        for block in self.res_blocks:
            block.apply(_init_linear)
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        with torch.random.fork_rng(devices=[]):
            for block in self.extra_temporal_attention_blocks:
                block.apply(_init_linear)
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            for block in self.extra_cross_attention_blocks:
                block.apply(_init_linear)
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            late_history_initialization_blocks = list(())
            disabled_late_history_candidate = getattr(
                self, "_disabled_late_history_attention_candidate", None
            )
            if disabled_late_history_candidate is not None:
                late_history_initialization_blocks.append(disabled_late_history_candidate)
            for block in late_history_initialization_blocks:
                block.apply(_init_linear)
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            for block in self.full_stage_cross_attention_blocks:
                block.apply(_init_linear)
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            for block in self.full_stage_temporal_attention_blocks:
                block.apply(_init_linear)
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            for block in self.full_stage_mlp_blocks:
                block.apply(_init_linear)
                nn.init.zeros_(block.adaLN_modulation[-1].weight)
                nn.init.zeros_(block.adaLN_modulation[-1].bias)
            if disabled_late_history_candidate is not None:
                object.__delattr__(self, "_disabled_late_history_attention_candidate")
        for block in self.cross_attention_blocks:
            block.apply(_init_linear)
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        self.final_layer.linear.apply(_init_linear)

    def _network(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        *,
        local_cond: Optional[torch.Tensor] = None,
        drop_condition: Optional[torch.Tensor] = None,
        drop_y: Optional[torch.Tensor] = None,
        drop_dino: Optional[torch.Tensor] = None,
        return_logvar: bool = False,
        return_intermediate_layer: Optional[int] = None,
    ):
        if x.ndim not in (2, 3):
            raise ValueError(f"expected rank-2 or rank-3 diffusion input, got {tuple(x.shape)}")
        if cond.ndim == 3 and cond.shape[1] == 1:
            cond = cond[:, 0]
        if cond.ndim == 2:
            if cond.shape != (x.size(0), 2048):
                raise ValueError(
                    f"expected global AR condition [{x.size(0)},{2048}], got {tuple(cond.shape)}"
                )
        elif cond.ndim == 3:
            expected_cond_shape = (x.size(0), 3, 2048)
            if tuple(cond.shape) != expected_cond_shape:
                raise ValueError(
                    f"expected slot-wise AR condition {expected_cond_shape}, got {tuple(cond.shape)}"
                )
        else:
            raise ValueError(f"AR condition must be [B,C] or [B,S,C], got {tuple(cond.shape)}")

        def normalize_drop_mask(value: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
            if value is None:
                return None
            if value.shape != (x.shape[0],):
                raise ValueError(f"{name} must have one value per batch item")
            return value.to(device=x.device, dtype=torch.bool)

        drop_condition = normalize_drop_mask(drop_condition, "drop_condition")
        drop_y = normalize_drop_mask(drop_condition if drop_y is None else drop_y, "drop_y")
        drop_dino = normalize_drop_mask(
            drop_condition if drop_dino is None else drop_dino, "drop_dino"
        )
        per_token_time = (
            True
            and x.ndim == 3
            and (sigma.ndim == 2)
            and (tuple(sigma.shape) == tuple(x.shape[:2]))
        )
        if per_token_time:
            time = sigma
        else:
            time = sigma.reshape(-1)
            if time.numel() == 1:
                time = time.expand(x.shape[0])
            elif time.numel() != x.shape[0]:
                raise ValueError("time must be scalar or have one value per batch item")
        if return_intermediate_layer is not None:
            if not 0 <= int(return_intermediate_layer) < len((0, 2, 3, 4, 5, 1, 6, 7)):
                raise ValueError("requested intermediate layer is outside the flow head")
            return_intermediate_layer = int(return_intermediate_layer)
        if return_logvar:
            raise ValueError("flow parameterization does not predict log variance")
        time_embedding_input = time * 1000.0
        x_in = x
        if per_token_time:
            t = self.time_embed(time_embedding_input.reshape(-1)).reshape(
                x.shape[0], x.shape[1], -1
            )
        else:
            t = self.time_embed(time_embedding_input)
        model_dtype = self.input_proj[0].weight.dtype
        cond = self.cond_embed(cond.to(dtype=model_dtype))
        if drop_y is not None:
            drop_shape = (cond.shape[0],) + (1,) * (cond.ndim - 1)
            cond = torch.where(
                drop_y.reshape(drop_shape), self.fake_latent.to(cond.dtype).expand_as(cond), cond
            )
        if per_token_time:
            aligned_cond = cond[:, None, :].expand(-1, x.shape[1], -1) if cond.ndim == 2 else cond
            global_y = mp_sum(t, aligned_cond, t=0.5)
        elif cond.ndim == 3:
            global_y = mp_sum(t.unsqueeze(1).expand(-1, cond.shape[1], -1), cond, t=0.5)
        else:
            global_y = mp_sum(t, cond, t=0.5)
        y = global_y
        x_in = self.input_proj(x_in.to(dtype=model_dtype))
        time_audio_condition = None
        audio_time = t.to(dtype=x_in.dtype)
        if x_in.ndim == 3 and audio_time.ndim == 2:
            audio_time = audio_time[:, None, :].expand_as(x_in)
        elif x_in.ndim != audio_time.ndim:
            raise ValueError("time/audio adapter conditioning requires aligned audio slots")
        time_audio_condition = mp_sum(audio_time, x_in, t=0.5)
        local_context = None
        if x.ndim != 3:
            raise ValueError("local chunk conditioning requires rank-3 diffusion input")
        if local_cond is None:
            raise ValueError("local_cond is required for local chunk conditioning")
        expected_shape = (x.shape[0], 3)
        if local_cond.shape[:2] != expected_shape:
            raise ValueError(f"expected local_cond [B,{3},C], got {tuple(local_cond.shape)}")
        local_context = self.local_cond_embed(local_cond.to(dtype=model_dtype))
        if drop_dino is not None:
            local_context = torch.where(
                drop_dino[:, None, None], torch.zeros_like(local_context), local_context
            )
        local_context = local_context.to(dtype=x_in.dtype)
        slot_position = self.slot_position.to(dtype=x_in.dtype)
        x_in = mp_sum(x_in + slot_position, local_context, t=0.5)
        local_context = local_context + slot_position
        if per_token_time:
            y = mp_sum(y.to(dtype=local_context.dtype), local_context, t=0.5)
        else:
            global_slot_y = (
                y.to(dtype=local_context.dtype)
                if y.ndim == 3
                else y.to(dtype=local_context.dtype).unsqueeze(1).expand(-1, 3, -1)
            )
            y = mp_sum(global_slot_y, local_context, t=0.5)
        cross_attention_gate_condition = time_audio_condition
        if True and cross_attention_gate_condition is None:
            raise AssertionError("cross-attention gate condition was not built")
        pass
        attention_prefix_parts = []
        attention_prefix_y_parts = []
        attention_prefix_valid_parts = []
        if attention_prefix_parts:
            attention_prefix = torch.cat(attention_prefix_parts, dim=1)
            attention_prefix_y = torch.cat(attention_prefix_y_parts, dim=1)
            attention_prefix_valid = torch.cat(attention_prefix_valid_parts, dim=1)
            prefix_token_count = attention_prefix.shape[1]
            x_in = torch.cat((attention_prefix, x_in), dim=1)
            y = torch.cat((attention_prefix_y, y), dim=1)
            if cross_attention_gate_condition is not None:
                cross_attention_gate_condition = torch.cat(
                    (
                        cross_attention_gate_condition.new_zeros(
                            x.shape[0], prefix_token_count, cross_attention_gate_condition.shape[-1]
                        ),
                        cross_attention_gate_condition,
                    ),
                    dim=1,
                )
            temporal_token_valid = torch.cat(
                (
                    attention_prefix_valid,
                    torch.ones(x.shape[0], 3, device=x.device, dtype=torch.bool),
                ),
                dim=1,
            )
        else:
            prefix_token_count = 0
            temporal_token_valid = None
        intermediate_hidden = None
        for execution_position, block_index in enumerate((0, 2, 3, 4, 5, 1, 6, 7)):
            block = self.res_blocks[block_index]
            if block_index < len(self.cross_attention_blocks):
                current_x = x_in[:, prefix_token_count:]
                current_y = y[:, prefix_token_count:]
                current_gate_condition = (
                    None
                    if cross_attention_gate_condition is None
                    else cross_attention_gate_condition[:, prefix_token_count:]
                )
                current_x = self.cross_attention_blocks[block_index](
                    current_x, local_context, current_y, gate_condition=current_gate_condition
                )
                if prefix_token_count:
                    x_in = torch.cat((x_in[:, :prefix_token_count], current_x), dim=1)
                else:
                    x_in = current_x
            history_adapter_index = self._history_adapter_index_by_block[block_index]
            extra_attention_index = self._extra_attention_index_by_execution_position[
                execution_position
            ]
            if extra_attention_index >= 0:
                current_x = x_in[:, prefix_token_count:]
                current_y = y[:, prefix_token_count:]
                current_gate_condition = (
                    None
                    if cross_attention_gate_condition is None
                    else cross_attention_gate_condition[:, prefix_token_count:]
                )
                current_x = self.extra_cross_attention_blocks[extra_attention_index](
                    current_x, local_context, current_y, gate_condition=current_gate_condition
                )
                x_in = (
                    torch.cat((x_in[:, :prefix_token_count], current_x), dim=1)
                    if prefix_token_count
                    else current_x
                )
                x_in = self.extra_temporal_attention_blocks[extra_attention_index](
                    x_in, y, token_valid=temporal_token_valid
                )
            if isinstance(block, TemporalSelfAttention):
                x_in = block(x_in, y, token_valid=temporal_token_valid)
            else:
                x_in = block(x_in, y)
            pass
            if prefix_token_count and True and (block_index == 1):
                x_in = x_in[:, prefix_token_count:]
                y = y[:, prefix_token_count:]
                prefix_token_count = 0
                temporal_token_valid = None
            if execution_position == return_intermediate_layer:
                intermediate_hidden = x_in[:, prefix_token_count:]
        final_extra_attention_index = self._extra_attention_index_by_execution_position[
            len((0, 2, 3, 4, 5, 1, 6, 7))
        ]
        if final_extra_attention_index >= 0:
            current_x = x_in[:, prefix_token_count:]
            current_y = y[:, prefix_token_count:]
            current_gate_condition = (
                None
                if cross_attention_gate_condition is None
                else cross_attention_gate_condition[:, prefix_token_count:]
            )
            current_x = self.extra_cross_attention_blocks[final_extra_attention_index](
                current_x, local_context, current_y, gate_condition=current_gate_condition
            )
            x_in = (
                torch.cat((x_in[:, :prefix_token_count], current_x), dim=1)
                if prefix_token_count
                else current_x
            )
            x_in = self.extra_temporal_attention_blocks[final_extra_attention_index](
                x_in, y, token_valid=temporal_token_valid
            )
        for stage_index in range(1):
            current_x = x_in[:, prefix_token_count:]
            current_y = y[:, prefix_token_count:]
            current_gate_condition = (
                None
                if cross_attention_gate_condition is None
                else cross_attention_gate_condition[:, prefix_token_count:]
            )
            current_x = self.full_stage_cross_attention_blocks[stage_index](
                current_x, local_context, current_y, gate_condition=current_gate_condition
            )
            x_in = (
                torch.cat((x_in[:, :prefix_token_count], current_x), dim=1)
                if prefix_token_count
                else current_x
            )
            x_in = self.full_stage_temporal_attention_blocks[stage_index](
                x_in, y, token_valid=temporal_token_valid
            )
            mlp_offset = 2 * stage_index
            x_in = self.full_stage_mlp_blocks[mlp_offset](x_in, y)
            x_in = self.full_stage_mlp_blocks[mlp_offset + 1](x_in, y)
        F_x = self.final_layer(x_in, y)
        if return_intermediate_layer is not None:
            if intermediate_hidden is None:
                raise AssertionError("requested flow-head state was not captured")
            return (F_x, intermediate_hidden)
        return F_x

    def inference(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        history_cond: Optional[torch.Tensor] = None,
        history_valid: Optional[torch.Tensor] = None,
    ):
        """
        Apply the model to an input batch.
        :param x: [(bsz x seq), latent_dim] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param cond: conditioning from AR transformer. [(bsz x seq), latent_dim]
        :param cond_2: conditioning from AR transformer. [(bsz x seq), latent_dim]
        :return: [(bsz x seq), latent_dim] Tensor of outputs.
        """
        return self._network(x, sigma, cond, local_cond=local_cond)

    def inference_multi_cfg(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        *,
        y_guidance: float = 1.0,
        dino_guidance: float = 1.0,
        history_guidance: float = 1.0,
        history_null_cond: Optional[torch.Tensor] = None,
        local_cond: Optional[torch.Tensor] = None,
        history_cond: Optional[torch.Tensor] = None,
        history_valid: Optional[torch.Tensor] = None,
        return_branches: bool = False,
    ):
        """Leave-one-axis-out CFG for legacy LH-style Flow Heads.

        The history axis is unified: its counterfactual swaps in the AR
        previous-audio-null condition *and* disables direct H0/H2 together.
        It is therefore trained by the existing joint AR-token/H0 corruption,
        not by a separate DH-only history dropout.
        """
        scales = {
            "y": float(y_guidance),
            "dino": float(dino_guidance),
            "history": float(history_guidance),
        }
        if any((not math.isfinite(value) or value < 0.0 for value in scales.values())):
            raise ValueError("Flow Head CFG scales must be finite and non-negative")
        active_axes = [name for (name, value) in scales.items() if value != 1.0]
        if not active_axes:
            full = self.inference(
                x,
                sigma,
                cond,
                local_cond=local_cond,
                history_cond=history_cond,
                history_valid=history_valid,
            )
            return (full, {}) if return_branches else full
        if "history" in active_axes:
            if history_cond is None or history_valid is None:
                raise ValueError("unified history CFG requires active H0/H2")
            if history_null_cond is None or history_null_cond.shape != cond.shape:
                raise ValueError("unified history CFG requires history_null_cond matching cond")
        batch = x.shape[0]
        branch_count = 1 + len(active_axes)

        def repeat(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            return torch.cat([value] * branch_count, dim=0)

        sigma = sigma.reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch)
        if sigma.numel() != batch:
            raise ValueError("Flow Head CFG time must be scalar or one per sample")
        zeros = torch.zeros(branch_count * batch, device=x.device, dtype=torch.bool)
        drops = {axis: zeros.clone() for axis in ("y", "dino")}
        repeated_cond = repeat(cond)
        repeated_history_valid = repeat(history_valid)
        for branch_index, axis in enumerate(active_axes, start=1):
            branch_slice = slice(branch_index * batch, (branch_index + 1) * batch)
            if axis in drops:
                drops[axis][branch_slice] = True
            else:
                repeated_cond[branch_slice] = history_null_cond
                repeated_history_valid[branch_slice] = False
        output = self._network(
            repeat(x),
            repeat(sigma),
            repeated_cond,
            local_cond=repeat(local_cond),
            drop_condition=zeros,
            drop_y=drops["y"],
            drop_dino=drops["dino"],
        )
        branches = output.chunk(branch_count, dim=0)
        full = branches[0]
        null_velocities = dict(zip(active_axes, branches[1:]))
        if return_branches:
            return (full, null_velocities)
        combined = full
        for axis, no_axis in null_velocities.items():
            combined = combined + (scales[axis] - 1.0) * (full - no_axis)
        return combined

    def inference_cfg(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        guidance: float,
        local_cond: Optional[torch.Tensor] = None,
        history_cond: Optional[torch.Tensor] = None,
        history_valid: Optional[torch.Tensor] = None,
    ):
        """
        Apply the model to an input batch.
        :param x: [(bsz x seq), latent_dim] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param cond: conditioning from AR transformer. [(bsz x seq), latent_dim]
        :return: [(bsz x seq), latent_dim] Tensor of outputs.
        """
        if x.ndim not in (2, 3):
            raise ValueError(f"expected rank-2 or rank-3 diffusion input, got {tuple(x.shape)}")
        if not hasattr(self, "fake_latent"):
            raise ValueError("diffusion CFG requires label_drop_prob > 0 during training")
        if cond.ndim not in (2, 3) or cond.shape[0] != x.size(0):
            raise ValueError(
                f"CFG AR condition must be [B,C] or [B,S,C], got {tuple(cond.shape)} for batch {x.size(0)}"
            )
        sigma = sigma.reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(cond.shape[0])
        elif sigma.numel() != cond.shape[0]:
            raise ValueError("sigma must be scalar or have one value per batch item")
        x_cfg = torch.cat([x, x], dim=0)
        sigma_cfg = torch.cat([sigma, sigma], dim=0)
        cond_cfg = torch.cat([cond, cond], dim=0)
        local_cond_cfg = None
        if local_cond is not None:
            local_cond_cfg = torch.cat([local_cond, local_cond], dim=0)
        history_cond_cfg = None
        if history_cond is not None:
            history_cond_cfg = torch.cat([history_cond, history_cond], dim=0)
        history_valid_cfg = None
        if history_valid is not None:
            history_valid_cfg = torch.cat([history_valid, history_valid], dim=0)
        drop_condition = torch.cat(
            [
                torch.zeros(cond.shape[0], dtype=torch.bool, device=cond.device),
                torch.ones(cond.shape[0], dtype=torch.bool, device=cond.device),
            ]
        )
        D_x = self._network(
            x_cfg, sigma_cfg, cond_cfg, local_cond=local_cond_cfg, drop_condition=drop_condition
        )
        (D_x_cond, D_x_uncond) = torch.chunk(D_x, 2, dim=0)
        D_x = guidance * D_x_cond + (1 - guidance) * D_x_uncond
        return D_x
