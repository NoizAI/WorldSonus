from __future__ import annotations
import typing as tp
import torch
import torch.nn as nn
from torch.nn import functional as F
from worldsonus.tensor_ops import (
    DropPath,
    precompute_freqs_cis,
    apply_rotary_emb,
    interleave_tokens,
)
import math

TRANSFORMER_CFG_MODES = ("none", "video", "prompt", "video_prompt")
PROMPT_OUTER_GATE_INIT_BIAS = -2.0


def transformer_cfg_branch_count(cfg_mode: str) -> int:
    """Return the inference batch multiplier for a Transformer CFG mode.

    ``video_prompt`` uses three branches in this fixed order: full
    video+prompt, null-video+prompt, and video+null-prompt.  The order is part
    of the sampling contract because :mod:`models.worldsonus` combines the
    corresponding Transformer outputs without a joint-null branch.
    """
    if cfg_mode == "none":
        return 1
    if cfg_mode in {"video", "prompt"}:
        return 2
    if cfg_mode == "video_prompt":
        return 3
    raise ValueError(
        f"invalid Transformer CFG mode: {cfg_mode}; expected one of {TRANSFORMER_CFG_MODES}"
    )


class TypeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_embedding = nn.Embedding(2, 1024)
        self.dropout = nn.Dropout(0.0)

    def forward(self, token_embeds: torch.Tensor, type_id: int) -> torch.Tensor:
        if isinstance(type_id, int):
            type_id = torch.full((token_embeds.size(0),), type_id, device=token_embeds.device)
        type_embeds = self.type_embedding(type_id)
        type_embeds = type_embeds.unsqueeze(1).expand(-1, token_embeds.size(1), -1)
        return self.dropout(token_embeds + type_embeds)


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - n % k


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_dim_multiplier: tp.Optional[float] = None,
        ffn_dropout_p: float = 0.1,
        multiple_of: int = 256,
        hidden_dim: tp.Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model
            hidden_dim = int(2 * hidden_dim / 3)
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = find_multiple(hidden_dim, multiple_of)
        else:
            if ffn_dim_multiplier is not None:
                raise ValueError("set either an exact FFN hidden_dim or ffn_dim_multiplier")
            if isinstance(hidden_dim, bool) or int(hidden_dim) <= 0:
                raise ValueError("FFN hidden_dim must be a positive integer")
            hidden_dim = int(hidden_dim)
        self.hidden_dim = hidden_dim
        self.w13 = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        (a, b) = self.w13(x).chunk(2, dim=-1)
        out = F.silu(a) * b
        return self.ffn_dropout(self.w2(out))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, dtype):
        super().__init__()
        cache_shape = (max_batch_size, 16, max_seq_length, 64)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        if k_val.dtype != k_out.dtype:
            k_val = k_val.to(dtype=k_out.dtype)
        if v_val.dtype != v_out.dtype:
            v_val = v_val.to(dtype=v_out.dtype)
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return (k_out, v_out)


class RingKVCache(nn.Module):
    """Fixed-capacity KV storage with an independent attention window."""

    def __init__(
        self, max_batch_size, window_size, dtype, *, attention_window: tp.Optional[int] = None
    ):
        super().__init__()
        self.capacity = int(window_size)
        self.window_size = self.capacity
        self.storage_capacity = self.capacity + 0
        self.attention_window = int(self.capacity if attention_window is None else attention_window)
        if self.capacity <= 0:
            raise ValueError("RingKVCache capacity must be positive")
        if not 0 < self.attention_window <= self.capacity:
            raise ValueError("RingKVCache attention_window must be in [1, capacity]")
        cache_shape = (max_batch_size, 16, self.storage_capacity, 64)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer(
            "pos", torch.full((max_batch_size,), -1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "valid", torch.zeros((max_batch_size,), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "position_cache",
            torch.full((max_batch_size, self.storage_capacity), -1, dtype=torch.long),
            persistent=False,
        )

    def _storage_slots(self, positions: torch.Tensor) -> torch.Tensor:
        return positions.remainder(self.capacity)

    def update_for_attention(
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        *,
        commit: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return an exact sliding attention view and optionally commit tokens.

        For a paired ``[audio, video]`` update the audio query must not see the
        video token, while the video query may see audio.  Keeping the old ring
        snapshot alongside both new tokens for this attention call also retains
        the one oldest key needed by the audio row before the two-token commit
        overwrites it.  The returned boolean mask applies both causality and the
        per-query sliding-window boundary.
        """
        (batch, heads, token_count, head_dim) = k_val.shape
        if input_pos.ndim != 1 or input_pos.shape[0] != token_count:
            raise ValueError("RingKVCache input positions must match the token dimension")
        if token_count < 1:
            raise ValueError("RingKVCache update requires at least one token")
        if token_count > self.capacity:
            raise ValueError("RingKVCache update cannot exceed its capacity")
        if batch > self.k_cache.shape[0]:
            raise ValueError("RingKVCache batch exceeds its configured capacity")
        if heads != self.k_cache.shape[1] or head_dim != self.k_cache.shape[3]:
            raise ValueError("RingKVCache key/value shape mismatch")
        k_val = k_val.to(dtype=self.k_cache.dtype)
        v_val = v_val.to(dtype=self.v_cache.dtype)
        positions = input_pos.to(device=self.position_cache.device, dtype=torch.long)
        old_positions = self.position_cache[:batch]
        keys = torch.cat((self.k_cache[:batch], k_val), dim=2)
        values = torch.cat((self.v_cache[:batch], v_val), dim=2)
        candidate_positions = torch.cat(
            (old_positions, positions.view(1, token_count).expand(batch, -1)), dim=1
        )
        query_positions = positions.view(1, token_count, 1)
        attention_mask = (
            (candidate_positions[:, None, :] >= 0)
            & (candidate_positions[:, None, :] <= query_positions)
            & (candidate_positions[:, None, :] >= query_positions - self.attention_window + 1)
        ).unsqueeze(1)
        if commit:
            slots = self._storage_slots(positions)
            cache_index = slots.view(1, 1, token_count, 1).expand(
                batch, heads, token_count, head_dim
            )
            self.k_cache[:batch].scatter_(2, cache_index, k_val)
            self.v_cache[:batch].scatter_(2, cache_index, v_val)
            position_index = slots.view(1, token_count).expand(batch, -1)
            self.position_cache[:batch].scatter_(
                1, position_index, positions.view(1, token_count).expand(batch, -1)
            )
            self.pos[:batch].copy_(slots[-1].expand(batch))
            self.valid[:batch].copy_(self.position_cache[:batch].ge(0).sum(dim=1))
        return (keys, values, attention_mask)

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor):
        batch = k_val.shape[0]
        self.update_for_attention(input_pos, k_val, v_val)
        return (self.k_cache[:batch], self.v_cache[:batch])


class MainMultiheadAttention(nn.Module):
    _fsdp_final = True

    def __init__(self):
        super().__init__()
        assert True
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = 64
        assert True, "num_heads must be divisible by num_kv_heads"
        self.wqkv = nn.Linear(1024, 3072, bias=False)
        self.wo = nn.Linear(1024, 1024, bias=False)
        self.kv_cache = None
        self.sliding_window = None
        self.attn_dropout_p = 0.1
        self.resid_dropout = nn.Dropout(0.1)

    def forward(
        self,
        query: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: tp.Optional[torch.Tensor] = None,
        mask: tp.Optional[torch.Tensor] = None,
        cache_commit: bool = True,
    ):
        (bsz, seqlen, _) = query.shape
        (xq, xk, xv) = self.wqkv(query).split([1024, 1024, 1024], dim=-1)
        xq = xq.view(bsz, seqlen, 16, 64)
        xk = xk.view(bsz, seqlen, 16, 64)
        xv = xv.view(bsz, seqlen, 16, 64)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        (xq, xk, xv) = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        if isinstance(self.kv_cache, RingKVCache):
            (keys, values, attn_mask) = self.kv_cache.update_for_attention(
                input_pos, xk, xv, commit=cache_commit
            )
            is_causal = False
        else:
            (keys, values) = self.kv_cache.update(input_pos, xk, xv) if self.kv_cache else (xk, xv)
            if self.sliding_window is not None and self.kv_cache is not None:
                end = int(input_pos[-1].item())
                start = max(0, end - self.sliding_window + 1)
                keys = keys[:, :, start : end + 1, :]
                values = values[:, :, start : end + 1, :]
                attn_mask = None
                is_causal = False
            else:
                attn_mask = mask
                is_causal = mask is None
        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=0.1 if self.training else 0,
            enable_gqa=True,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, 1024)
        output = self.resid_dropout(self.wo(output))
        return output


class TextCrossAttention(nn.Module):
    """Non-causal cross-attention from AR states to a static text sequence."""

    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("text attention heads must divide d_model")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.context_norm = nn.RMSNorm(self.embed_dim, eps=1e-05)
        self.wq = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.wkv = nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=False)
        self.wo = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.attn_dropout_p = float(attn_dropout_p)
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def prepare_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3 or context.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected text context [B,L,{self.embed_dim}], got {tuple(context.shape)}"
            )
        (batch, length, _) = context.shape
        (key, value) = self.wkv(self.context_norm(context)).chunk(2, dim=-1)
        key = key.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        return (key, value)

    def forward(
        self,
        query: torch.Tensor,
        *,
        text_mask: torch.Tensor,
        text_kv: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if query.ndim != 3 or query.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected text query [B,T,{self.embed_dim}], got {tuple(query.shape)}"
            )
        (key, value) = text_kv
        if text_mask.ndim == 2:
            (batch, length) = text_mask.shape
            attention_mask = text_mask[:, None, None, :]
        elif text_mask.ndim == 3:
            (batch, query_length, length) = text_mask.shape
            if query_length != query.shape[1]:
                raise ValueError(
                    f"query-dependent text mask length does not match query: {query_length} != {query.shape[1]}"
                )
            attention_mask = text_mask[:, None, :, :]
        else:
            raise ValueError("text mask must have shape [B,K] or [B,Q,K]")
        if query.shape[0] != batch or key.shape[:2] != (batch, self.num_heads):
            raise ValueError("text condition batch does not match Transformer state")
        if key.shape[2] != length or value.shape != key.shape:
            raise ValueError("text K/V and mask shapes do not match")
        q = (
            self.wq(query)
            .view(batch, query.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        output = F.scaled_dot_product_attention(
            q,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(batch, query.shape[1], self.embed_dim)
        return self.resid_dropout(self.wo(output))


class PromptQGResampler(nn.Module):
    """Compress a masked prompt with one global token and learned queries.

    The global token preserves scene-level context while the remaining tokens
    learn task-specific retrieval from the full frozen text-encoder sequence.
    This module runs once when the static prompt K/V cache is prepared, not on
    every autoregressive token.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 640
        self.num_heads = 8
        self.head_dim = 80
        self.attn_dropout_p = 0.0
        self.source_norm = nn.LayerNorm(640)
        self.query_norm = nn.LayerNorm(640)
        self.learned_queries = nn.Parameter(torch.randn(31, 640) / math.sqrt(640))
        self.wq = nn.Linear(640, 640, bias=False)
        self.wkv = nn.Linear(640, 1280, bias=False)
        self.wo = nn.Linear(640, 640, bias=False)
        self.ffn_norm = nn.LayerNorm(640)
        self.ffn = nn.Sequential(nn.Linear(640, 2560), nn.GELU(), nn.Linear(2560, 640))
        self.output_norm = nn.LayerNorm(640)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != 640:
            raise ValueError(f"expected QG prompt [B,L,{640}], got {tuple(tokens.shape)}")
        if tuple(attention_mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError("QG prompt mask must match the input token sequence")
        valid = attention_mask.to(device=tokens.device, dtype=torch.bool)
        source = self.source_norm(tokens)
        weights = valid.to(dtype=source.dtype).unsqueeze(-1)
        global_token = (source * weights).sum(dim=1, keepdim=True)
        global_token = global_token / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        (batch, source_length, _) = source.shape
        queries = self.learned_queries.to(dtype=source.dtype).unsqueeze(0).expand(batch, -1, -1)
        q = self.wq(self.query_norm(queries)).view(batch, 31, 8, 80).transpose(1, 2)
        (key, value) = self.wkv(source).chunk(2, dim=-1)
        key = key.view(batch, source_length, 8, 80).transpose(1, 2)
        value = value.view(batch, source_length, 8, 80).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            key,
            value,
            attn_mask=valid[:, None, None, :],
            is_causal=False,
            dropout_p=0.0 if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, 31, 640)
        queries = queries + self.wo(attended)
        output = torch.cat((global_token, queries), dim=1)
        output = output + self.ffn(self.ffn_norm(output))
        return self.output_norm(output)


class AlignedCrossAttention(TextCrossAttention):
    """Cross-attention with a query-dependent causal/alignment mask."""

    def forward(
        self,
        query: torch.Tensor,
        *,
        context_mask: torch.Tensor,
        context_kv: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if query.ndim != 3 or query.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected aligned query [B,T,{self.embed_dim}], got {tuple(query.shape)}"
            )
        (key, value) = context_kv
        (batch, query_length) = query.shape[:2]
        if key.shape[:2] != (batch, self.num_heads) or value.shape != key.shape:
            raise ValueError("aligned context batch or K/V shape mismatch")
        context_length = key.shape[2]
        if context_mask.ndim == 2:
            expected = (batch, context_length)
            if tuple(context_mask.shape) != expected:
                raise ValueError(
                    f"expected aligned context mask {expected}, got {tuple(context_mask.shape)}"
                )
            attention_mask = context_mask[:, None, None, :]
        elif context_mask.ndim == 3:
            expected = (batch, query_length, context_length)
            if tuple(context_mask.shape) != expected:
                raise ValueError(
                    f"expected aligned context mask {expected}, got {tuple(context_mask.shape)}"
                )
            attention_mask = context_mask[:, None, :, :]
        else:
            raise ValueError("aligned context mask must be [B,K] or [B,Q,K]")
        q = self.wq(query).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(batch, query_length, self.embed_dim)
        return self.resid_dropout(self.wo(output))


class TextCondition(tp.NamedTuple):
    """Projected text mask plus the static K/V tensors for every main layer."""

    mask: torch.Tensor
    layer_kv: tuple[tp.Optional[tuple[torch.Tensor, torch.Tensor]], ...]
    cfg_mode: str
    projected_context: tp.Optional[torch.Tensor] = None


class MainTransformerLayer(nn.Module):
    def __init__(self, drop_path=0.0):
        super().__init__()
        self.attention = MainMultiheadAttention()
        self.feed_forward = FeedForward(1024, None, 0.1, 256, hidden_dim=4096)
        self.attention_norm = nn.RMSNorm(1024, eps=1e-05)
        self.text_attention_norm = nn.RMSNorm(1024, eps=1e-05)
        self.text_cross_attention = TextCrossAttention(
            embed_dim=1024, num_heads=8, attn_dropout_p=0.0, resid_dropout_p=0.1
        )
        self.text_outer_gate: tp.Optional[nn.Linear] = None
        self.ffn_norm = nn.RMSNorm(1024, eps=1e-05)
        self.semantic_adaln = None
        self.semantic_adaln_active = False
        self.flow_time_adaln = None
        self.ar_dino_attention_norm: tp.Optional[nn.Module] = None
        self.ar_dino_cross_attention: tp.Optional[AlignedCrossAttention] = None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale_1: nn.Module
        self.layer_scale_text: nn.Module
        self.layer_scale_ar_dino: nn.Module
        self.layer_scale_2: nn.Module
        self.layer_scale_1 = nn.Identity()
        self.layer_scale_text = nn.Identity()
        self.layer_scale_ar_dino = nn.Identity()
        self.layer_scale_2 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        mask: tp.Optional[torch.Tensor] = None,
        text_mask: tp.Optional[torch.Tensor] = None,
        text_kv: tp.Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_commit: bool = True,
    ) -> torch.Tensor:
        attention_input = self.attention_norm(x)
        h = x + self.drop_path(
            self.layer_scale_1(
                self.attention(
                    attention_input, freqs_cis, start_pos, mask, cache_commit=cache_commit
                )
            )
        )
        if self.ar_dino_cross_attention is not None:
            raise ValueError("D2 AR layer requires a prepared slot-aligned DINO condition")
        if self.text_cross_attention is not None:
            if text_mask is None or text_kv is None or self.text_attention_norm is None:
                raise ValueError("text-conditioned layer requires a prepared text condition")
            text_query = self.text_attention_norm(h)
            text_output = self.text_cross_attention(
                text_query, text_mask=text_mask, text_kv=text_kv
            )
            if self.text_outer_gate is not None:
                text_output = text_output * torch.sigmoid(self.text_outer_gate(text_query)).to(
                    text_output.dtype
                )
            h = h + self.drop_path(self.layer_scale_text(text_output))
        elif text_mask is not None or text_kv is not None:
            raise ValueError("text condition was passed to a text-disabled layer")
        ffn_input = self.ffn_norm(h)
        out = h + self.drop_path(self.layer_scale_2(self.feed_forward(ffn_input)))
        return out


class MainTransformer(nn.Module):
    def __init__(self, layer_class: tp.Type[MainTransformerLayer] = MainTransformerLayer, **kwargs):
        super().__init__()
        assert True
        self.num_heads = 16
        self.register_parameter("semantic_residual_gate", None)
        self.text_input_proj = nn.Linear(640, 1024, bias=False)
        self.prompt_resampler = PromptQGResampler()
        self.prompt_condition_tokens = 32 if self.prompt_resampler is not None else 192
        self.null_text_embedding = nn.Parameter(
            torch.randn(1, self.prompt_condition_tokens, 640) / math.sqrt(640)
        )
        self.sliding_window = None
        self.kv_cache_capacity = None
        self.rope_scaling = None
        self.null_embedding = nn.Parameter(torch.randn(1, 1024) / math.sqrt(1024))
        self.init_audio_token = nn.Parameter(torch.randn(1, 1024) / math.sqrt(1024))
        self.register_parameter("additional_init_audio_tokens", None)
        self.seq_len = 200
        self.causal_window_size = 100
        self.freqs_cis = precompute_freqs_cis(200, 64, 10000, rope_scaling=None)
        self.x_token_dropout = nn.Dropout(0.0)
        self.y_token_dropout = nn.Dropout(0.0)
        self.type_emb = TypeEmbedding()
        self.output_norm = nn.RMSNorm(1024, eps=1e-05)
        dpr = [0.1 * layer_index / 35 for layer_index in range(36)]
        self.layers = nn.ModuleList()
        for layer_id in range(36):
            selected_layer_class = layer_class
            mmdit_layer_kwargs = {}
            layer = MainTransformerLayer(drop_path=dpr[layer_id])
            layer.semantic_adaln_active = layer_id >= 36
            if layer.semantic_adaln is not None and (not layer.semantic_adaln_active):
                layer.semantic_adaln.requires_grad_(False)
            self.layers.append(layer)
        self.action_conditioning_mode: tp.Optional[str] = None
        self.action_channels: tp.Optional[int] = None
        self.action_conditioning_active_layers: tuple[int, ...] = ()
        self.action_adapters = nn.ModuleList()
        self.ar_dino_input_dim: tp.Optional[int] = None
        self.ar_dino_input_proj: tp.Optional[nn.Module] = None
        self.ar_dino_active_layers: tuple[int, ...] = ()
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.initialize_weights()

    def initial_audio_tokens(self, batch_size: int) -> torch.Tensor:
        """Return one learned causal-start token for each audio summary slot."""
        first = self.init_audio_token.view(1, 1, 1024)
        pass
        return first.expand(batch_size, -1, -1)

    @staticmethod
    def _normalize_video_present(
        video_present: tp.Optional[torch.Tensor | bool],
        *,
        batch_size: int,
        device: torch.device,
        default: bool = True,
    ) -> torch.Tensor:
        """Return one explicit video-valid bit per batch row."""
        if video_present is None:
            return torch.full((batch_size,), default, device=device, dtype=torch.bool)
        if isinstance(video_present, bool):
            return torch.full((batch_size,), video_present, device=device, dtype=torch.bool)
        video_present = torch.as_tensor(video_present, device=device, dtype=torch.bool)
        if video_present.ndim == 0:
            video_present = video_present.expand(batch_size)
        if tuple(video_present.shape) != (batch_size,):
            raise ValueError(
                f"video_present must be a scalar or one value per batch row: expected {(batch_size,)}, got {tuple(video_present.shape)}"
            )
        return video_present

    def set_context_extension(
        self,
        mode: str,
        *,
        factor: float = None,
        window_size: int = None,
        cache_capacity: int = None,
    ):
        if mode == "sliding":
            mode = "swa"
        if mode not in {"pi", "ntk", "swa", "none"}:
            raise ValueError(f"unsupported context extension mode: {mode}")
        self.context_extension_mode = mode
        if mode in {"pi", "ntk"}:
            assert factor is not None and factor >= 1.0
            self.rope_scaling = {"type": "linear" if mode == "pi" else "ntk", "factor": factor}
            self.sliding_window = None
            self.kv_cache_capacity = None
        elif mode == "swa":
            assert window_size is not None and window_size > 0
            capacity = int(window_size if cache_capacity is None else cache_capacity)
            if capacity < int(window_size):
                raise ValueError(
                    "Ring-KV cache capacity cannot be smaller than its attention window"
                )
            self.rope_scaling = None
            self.sliding_window = int(window_size)
            self.kv_cache_capacity = capacity
        else:
            self.rope_scaling = None
            self.sliding_window = None
            self.kv_cache_capacity = None

    def initialize_weights(self):
        self.apply(self._init_weights)
        for layer_module in self.layers:
            mmdit_experts = [
                expert
                for expert in (
                    getattr(layer_module, "audio_expert", None),
                    getattr(layer_module, "video_expert", None),
                )
                if expert is not None
            ]
            for expert in mmdit_experts:
                nn.init.normal_(expert.attention_output.weight, mean=0.0, std=0.0023570226039551587)
                nn.init.normal_(expert.feed_forward.w2.weight, mean=0.0, std=0.0023570226039551587)
                nn.init.zeros_(expert.modulation[-1].weight)
                nn.init.zeros_(expert.modulation[-1].bias)
            if hasattr(layer_module, "feed_forward") and hasattr(layer_module.feed_forward, "w2"):
                nn.init.normal_(
                    layer_module.feed_forward.w2.weight, mean=0.0, std=0.0023570226039551587
                )
                if layer_module.feed_forward.w2.bias is not None:
                    nn.init.zeros_(layer_module.feed_forward.w2.bias)
            if hasattr(layer_module, "attention") and hasattr(layer_module.attention, "wo"):
                nn.init.normal_(
                    layer_module.attention.wo.weight, mean=0.0, std=0.0023570226039551587
                )
                if layer_module.attention.wo.bias is not None:
                    nn.init.zeros_(layer_module.attention.wo.bias)
            if layer_module.text_cross_attention is not None:
                nn.init.normal_(
                    layer_module.text_cross_attention.wo.weight, mean=0.0, std=0.0023570226039551587
                )
            if layer_module.text_outer_gate is not None:
                nn.init.zeros_(layer_module.text_outer_gate.weight)
                nn.init.constant_(layer_module.text_outer_gate.bias, PROMPT_OUTER_GATE_INIT_BIAS)
            if layer_module.semantic_adaln is not None:
                nn.init.zeros_(layer_module.semantic_adaln.modulation[-1].weight)
                nn.init.zeros_(layer_module.semantic_adaln.modulation[-1].bias)
            if layer_module.flow_time_adaln is not None:
                nn.init.zeros_(layer_module.flow_time_adaln.modulation[-1].weight)
                nn.init.zeros_(layer_module.flow_time_adaln.modulation[-1].bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(module, "weight"):
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        if self.sliding_window is not None:
            if self.kv_cache_capacity is None:
                raise AssertionError("missing Ring-KV storage capacity")
            for b in self.layers:
                layer_attention_window = self.sliding_window
                layer_capacity = self.kv_cache_capacity
                b.attention.kv_cache = RingKVCache(
                    max_batch_size, layer_capacity, dtype, attention_window=layer_attention_window
                )
                b.attention.sliding_window = layer_attention_window
                if hasattr(b, "setup_ffn_cache"):
                    b.setup_ffn_cache(
                        max_batch_size, dtype=dtype, device=self.null_embedding.device
                    )
            self.causal_mask = None
        else:
            for b in self.layers:
                layer_seq_length = max_seq_length
                b.attention.kv_cache = KVCache(max_batch_size, layer_seq_length, dtype)
                b.attention.sliding_window = None
                if hasattr(b, "setup_ffn_cache"):
                    b.setup_ffn_cache(
                        max_batch_size, dtype=dtype, device=self.null_embedding.device
                    )
            causal_mask = torch.tril(
                torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
            )
            self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        self.freqs_cis = precompute_freqs_cis(
            self.max_seq_length, 64, 10000, rope_scaling=self.rope_scaling, train_seq_len=200
        )

    def prepare_text_condition(
        self,
        prompt_embedding: tp.Optional[torch.Tensor],
        prompt_mask: tp.Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        cfg_expand: bool = False,
        cfg_mode: str = "none",
        apply_dropout: tp.Optional[bool] = None,
        prompt_dropout_mask: tp.Optional[torch.Tensor] = None,
        prompt_start_chunks: tp.Optional[torch.Tensor] = None,
        prompt_duration_chunks: tp.Optional[torch.Tensor] = None,
        query_token_count: tp.Optional[int] = None,
        query_tokens_per_chunk: tp.Optional[int] = None,
    ) -> tp.Optional[TextCondition]:
        """Build real/null text branches and cache each layer's static K/V."""
        if self.text_input_proj is None or self.null_text_embedding is None:
            if prompt_embedding is not None or prompt_mask is not None:
                raise ValueError("prompt tensors were provided but prompt_embed_dim is disabled")
            return None
        if cfg_expand:
            if cfg_mode != "none":
                raise ValueError("cfg_expand and cfg_mode cannot both select CFG")
            cfg_mode = "prompt"
        if cfg_mode not in TRANSFORMER_CFG_MODES:
            raise ValueError(f"invalid text CFG mode: {cfg_mode}")
        if prompt_embedding is None and prompt_mask is not None:
            raise ValueError("prompt_mask cannot be provided without prompt_embedding")
        if prompt_dropout_mask is not None:
            prompt_dropout_mask = prompt_dropout_mask.to(device=device, dtype=torch.bool)
            if tuple(prompt_dropout_mask.shape) != (batch_size,):
                raise ValueError("prompt_dropout_mask must have one value per sample")
        null_tokens = self.null_text_embedding.to(
            device=device, dtype=self.text_input_proj.weight.dtype
        ).expand(batch_size, -1, -1)
        null_mask = torch.ones(
            batch_size, self.prompt_condition_tokens, device=device, dtype=torch.bool
        )
        segmented = prompt_embedding is not None and prompt_embedding.ndim == 4
        if not segmented and (
            prompt_start_chunks is not None or prompt_duration_chunks is not None
        ):
            raise ValueError("prompt schedule requires segmented [B,S,L,D] prompts")
        if segmented:
            if prompt_start_chunks is None:
                raise ValueError("segmented prompts require prompt_start_chunks")
            if query_token_count is None or query_tokens_per_chunk is None:
                raise ValueError(
                    "segmented prompts require query_token_count and query_tokens_per_chunk"
                )
            query_token_count = int(query_token_count)
            query_tokens_per_chunk = int(query_tokens_per_chunk)
            if query_token_count <= 0 or query_tokens_per_chunk <= 0:
                raise ValueError("segmented prompt query dimensions must be positive")
            expected_prefix = (batch_size, int(prompt_embedding.shape[1]))
            expected = (*expected_prefix, 192, 640)
            if tuple(prompt_embedding.shape) != expected:
                raise ValueError(
                    f"expected segmented prompt {expected}, got {tuple(prompt_embedding.shape)}"
                )
            if prompt_mask is None or tuple(prompt_mask.shape) != expected[:-1]:
                raise ValueError(
                    f"segmented prompt mask must have shape {expected[:-1]}, got {(None if prompt_mask is None else tuple(prompt_mask.shape))}"
                )
            if tuple(prompt_start_chunks.shape) != expected_prefix:
                raise ValueError(
                    f"expected prompt starts {expected_prefix}, got {tuple(prompt_start_chunks.shape)}"
                )
            if prompt_duration_chunks is None:
                prompt_duration_chunks = torch.full_like(prompt_start_chunks, -1)
            if tuple(prompt_duration_chunks.shape) != expected_prefix:
                raise ValueError(
                    f"expected prompt durations {expected_prefix}, got {tuple(prompt_duration_chunks.shape)}"
                )
            segment_count = expected_prefix[1]
            flat_tokens = prompt_embedding.reshape(batch_size * segment_count, 192, 640).to(
                device=device, dtype=self.text_input_proj.weight.dtype
            )
            flat_valid = prompt_mask.reshape(batch_size * segment_count, 192).to(
                device=device, dtype=torch.bool
            )
            use_null = ~flat_valid.any(dim=1)
            dropout_enabled = self.training if apply_dropout is None else apply_dropout
            if prompt_dropout_mask is not None:
                use_null = use_null | prompt_dropout_mask.repeat_interleave(segment_count)
            elif dropout_enabled and True:
                sample_dropout = torch.rand(batch_size, device=device) < 0.1
                use_null = use_null | sample_dropout.repeat_interleave(segment_count)
            flat_null_tokens = (
                null_tokens[:, None]
                .expand(-1, segment_count, -1, -1)
                .reshape(batch_size * segment_count, self.prompt_condition_tokens, 640)
            )
            flat_null_mask = (
                null_mask[:, None]
                .expand(-1, segment_count, -1)
                .reshape(batch_size * segment_count, self.prompt_condition_tokens)
            )
            if self.prompt_resampler is None:
                tokens = torch.where(use_null[:, None, None], flat_null_tokens, flat_tokens)
                valid = torch.where(use_null[:, None], flat_null_mask, flat_valid)
            else:
                safe_valid = torch.where(use_null[:, None], torch.ones_like(flat_valid), flat_valid)
                safe_tokens = torch.where(
                    use_null[:, None, None], torch.zeros_like(flat_tokens), flat_tokens
                )
                resampled = self.prompt_resampler(safe_tokens, safe_valid)
                tokens = torch.where(use_null[:, None, None], flat_null_tokens, resampled)
                valid = flat_null_mask
            condition_tokens = self.prompt_condition_tokens
            tokens = tokens.reshape(batch_size, segment_count, condition_tokens, 640)
            valid = valid.reshape(batch_size, segment_count, condition_tokens)
            tokens = torch.cat([tokens, null_tokens[:, None]], dim=1)
            valid = torch.cat([valid, null_mask[:, None]], dim=1)
            projected = self.text_input_proj(tokens).to(dtype=dtype)
            starts = prompt_start_chunks.to(device=device, dtype=torch.long)
            durations = prompt_duration_chunks.to(device=device, dtype=torch.long)
            query_chunks = (
                torch.arange(query_token_count, device=device, dtype=torch.long)
                // query_tokens_per_chunk
            )
            active = starts[:, None, :] >= 0
            active = active & (query_chunks[None, :, None] >= starts[:, None, :])
            active = active & (
                (durations[:, None, :] < 0)
                | (query_chunks[None, :, None] < starts[:, None, :] + durations[:, None, :])
            )
            ordinal = torch.arange(segment_count, device=device, dtype=torch.long)
            candidate_start = starts[:, None, :, None]
            replacement_start = starts[:, None, None, :]
            candidate_ordinal = ordinal[None, None, :, None]
            replacement_ordinal = ordinal[None, None, None, :]
            higher_priority = (
                (replacement_start > candidate_start)
                | (replacement_start == candidate_start) & (replacement_ordinal > candidate_ordinal)
            ) & (replacement_start >= 0)
            replacement_has_started = query_chunks[None, :, None, None] >= replacement_start
            superseded = (higher_priority & replacement_has_started).any(dim=-1)
            active = active & ~superseded
            score = starts[:, None, :] * (segment_count + 1) + ordinal
            score = torch.where(active, score, torch.full_like(score, -1))
            selected = score.argmax(dim=-1)
            has_active = active.any(dim=-1)
            selected = torch.where(has_active, selected, torch.full_like(selected, segment_count))
            selected_segments = torch.nn.functional.one_hot(
                selected, num_classes=segment_count + 1
            ).to(dtype=torch.bool)
            query_mask = (selected_segments[:, :, :, None] & valid[:, None, :, :]).flatten(2, 3)
            projected = projected.flatten(1, 2)
            null_query_mask = torch.zeros_like(query_mask)
            null_begin = segment_count * condition_tokens
            null_query_mask[:, :, null_begin:] = null_mask[:, None, :]
            if cfg_mode == "prompt":
                projected = torch.cat([projected, projected], dim=0)
                query_mask = torch.cat([query_mask, null_query_mask], dim=0)
            elif cfg_mode == "video":
                projected = torch.cat([projected, projected], dim=0)
                query_mask = torch.cat([query_mask, query_mask], dim=0)
            elif cfg_mode == "video_prompt":
                projected = torch.cat([projected, projected, projected], dim=0)
                query_mask = torch.cat([query_mask, query_mask, null_query_mask], dim=0)
            layer_kv = tuple(
                (
                    None
                    if layer.text_cross_attention is None
                    else layer.text_cross_attention.prepare_kv(projected)
                    for layer in self.layers
                )
            )
            return TextCondition(
                mask=query_mask, layer_kv=layer_kv, cfg_mode=cfg_mode, projected_context=projected
            )
        if prompt_embedding is None:
            tokens = null_tokens
            valid = null_mask
        else:
            expected = (batch_size, 192, 640)
            if tuple(prompt_embedding.shape) != expected:
                raise ValueError(f"expected prompt {expected}, got {tuple(prompt_embedding.shape)}")
            input_tokens = prompt_embedding.to(
                device=device, dtype=self.text_input_proj.weight.dtype
            )
            if prompt_mask is None:
                input_valid = torch.ones(batch_size, 192, device=device, dtype=torch.bool)
            else:
                if tuple(prompt_mask.shape) != expected[:2]:
                    raise ValueError(
                        f"expected prompt mask {expected[:2]}, got {tuple(prompt_mask.shape)}"
                    )
                input_valid = prompt_mask.to(device=device, dtype=torch.bool)
            use_null = ~input_valid.any(dim=1)
            dropout_enabled = self.training if apply_dropout is None else apply_dropout
            if prompt_dropout_mask is not None:
                use_null = use_null | prompt_dropout_mask
            elif dropout_enabled and True:
                use_null = use_null | (torch.rand(batch_size, device=device) < 0.1)
            if self.prompt_resampler is None:
                tokens = torch.where(use_null[:, None, None], null_tokens, input_tokens)
                valid = torch.where(use_null[:, None], null_mask, input_valid)
            else:
                safe_valid = torch.where(
                    use_null[:, None], torch.ones_like(input_valid), input_valid
                )
                safe_tokens = torch.where(
                    use_null[:, None, None], torch.zeros_like(input_tokens), input_tokens
                )
                resampled = self.prompt_resampler(safe_tokens, safe_valid)
                tokens = torch.where(use_null[:, None, None], null_tokens, resampled)
                valid = null_mask
        projected = self.text_input_proj(tokens).to(dtype=dtype)
        if cfg_mode == "prompt":
            null_projected = self.text_input_proj(null_tokens).to(dtype=dtype)
            projected = torch.cat([projected, null_projected], dim=0)
            valid = torch.cat([valid, null_mask], dim=0)
        elif cfg_mode == "video":
            projected = torch.cat([projected, projected], dim=0)
            valid = torch.cat([valid, valid], dim=0)
        elif cfg_mode == "video_prompt":
            null_projected = self.text_input_proj(null_tokens).to(dtype=dtype)
            projected = torch.cat([projected, projected, null_projected], dim=0)
            valid = torch.cat([valid, valid, null_mask], dim=0)
        layer_kv = tuple(
            (
                None
                if layer.text_cross_attention is None
                else layer.text_cross_attention.prepare_kv(projected)
                for layer in self.layers
            )
        )
        return TextCondition(
            mask=valid, layer_kv=layer_kv, cfg_mode=cfg_mode, projected_context=projected
        )

    @staticmethod
    def _text_mask_for_query(
        text_condition: tp.Optional[TextCondition],
        *,
        input_pos: tp.Optional[torch.Tensor],
        query_length: int,
    ) -> tp.Optional[torch.Tensor]:
        if text_condition is None:
            return None
        text_mask = text_condition.mask
        if text_mask.ndim == 2:
            return text_mask
        if text_mask.ndim != 3:
            raise ValueError("prepared text mask must be [B,K] or [B,Q,K]")
        if input_pos is None:
            if text_mask.shape[1] != query_length:
                raise ValueError(
                    f"prepared segmented prompt query length mismatch: {text_mask.shape[1]} != {query_length}"
                )
            return text_mask
        positions = input_pos.to(device=text_mask.device, dtype=torch.long).reshape(-1)
        if positions.numel() != query_length:
            raise ValueError(
                f"input_pos count does not match segmented prompt query length: {positions.numel()} != {query_length}"
            )
        if (
            positions.numel()
            and (not torch.compiler.is_compiling())
            and (int(positions.max()) >= text_mask.shape[1])
        ):
            raise ValueError("input_pos exceeds the prepared segmented prompt schedule")
        return text_mask.index_select(1, positions)

    def prepare_inference_semantic_condition(self) -> tp.Optional[torch.Tensor]:
        """Project one causal semantic condition for incremental AR decoding.

        The video-null branch must also null the auxiliary SigLIP2 AdaLN
        stream.  The prompt-null branch retains it, matching the training-time
        coupling between video-token dropout and semantic dropout.
        """
        return None

    def forward(
        self,
        video=None,
        audio=None,
        input_pos=None,
        inference_pair=False,
        trans_cfg_scale=1.0,
        audio_cfg_expanded=False,
        text_condition=None,
        semantic_feature=None,
        video_present=None,
    ):
        """Dispatch a cached audio/video token pair; teacher forcing is not released."""
        if not inference_pair:
            raise RuntimeError("This release provides inference only")
        return self.inference_audio_video_pair(
            audio_token=audio,
            video_token=video,
            input_pos=input_pos,
            trans_cfg_scale=trans_cfg_scale,
            audio_cfg_expanded=audio_cfg_expanded,
            text_condition=text_condition,
            video_present=video_present,
        )

    def inference_audio_video_pair(
        self,
        audio_token: torch.Tensor,
        video_token: torch.Tensor,
        input_pos: torch.Tensor,
        trans_cfg_scale: float = 1.0,
        audio_cfg_expanded: bool = False,
        text_condition: tp.Optional[TextCondition] = None,
        video_present: tp.Optional[torch.Tensor | bool] = None,
    ) -> torch.Tensor:
        """Decode one historical causal ``[audio, video]`` pair."""
        return self._inference_audio_video_group(
            audio_token=audio_token,
            video_token=video_token,
            input_pos=input_pos,
            summary_count=1,
            trans_cfg_scale=trans_cfg_scale,
            audio_cfg_expanded=audio_cfg_expanded,
            text_condition=text_condition,
            video_present=video_present,
        )

    def _inference_audio_video_group(
        self,
        audio_token: torch.Tensor,
        video_token: torch.Tensor,
        input_pos: torch.Tensor,
        summary_count: int,
        trans_cfg_scale: float = 1.0,
        audio_cfg_expanded: bool = False,
        text_condition: tp.Optional[TextCondition] = None,
        video_present: tp.Optional[torch.Tensor | bool] = None,
    ) -> torch.Tensor:
        """Decode one A/V summary group in a single layer pass.

        Both token inputs are available before AR decoding starts.  Processing
        them together with a two-row causal mask is mathematically equivalent
        to two one-token cache updates, while each Transformer layer and its
        weight matrices are launched only once.

        ``audio_cfg_expanded`` is used for the BOS path, whose historical
        implementation generated independent inference noise for the
        conditional and unconditional cache rows before entering Transformer.
        Keeping that expanded input preserves the existing RNG semantics.
        """
        if audio_token is None or video_token is None or input_pos is None:
            raise ValueError("paired AR inference requires audio, video, and input positions")
        if audio_token.ndim == 2:
            audio_token = audio_token.unsqueeze(1)
        if video_token.ndim == 2:
            video_token = video_token.unsqueeze(1)
        summary_count = int(summary_count)
        if summary_count <= 0:
            raise ValueError("summary_count must be positive")
        if (
            audio_token.ndim != 3
            or video_token.ndim != 3
            or audio_token.shape[1] != summary_count
            or (video_token.shape[1] != summary_count)
        ):
            raise ValueError(
                f"grouped AR tokens have the wrong summary count: expected {summary_count}, audio={tuple(audio_token.shape)} video={tuple(video_token.shape)}"
            )
        grouped_token_count = summary_count * 2
        if input_pos.ndim != 1 or input_pos.shape[0] != grouped_token_count:
            raise ValueError(
                f"grouped AR input_pos must match the fused/interleaved token count: expected [{grouped_token_count}], got {tuple(input_pos.shape)}"
            )
        video_batch = video_token.shape[0]
        real_video_mask = self._normalize_video_present(
            video_present, batch_size=video_batch, device=video_token.device
        )
        audio_hidden = self.type_emb(audio_token, 1)
        audio_hidden = self.y_token_dropout(audio_hidden)
        video_hidden = self.type_emb(video_token, 0)
        video_hidden = self.x_token_dropout(video_hidden)
        conditional_null_video = (
            self.null_embedding.view(1, 1, -1)
            .expand(video_batch, summary_count, -1)
            .to(dtype=video_hidden.dtype)
        )
        video_hidden = torch.where(
            real_video_mask[:, None, None], video_hidden, conditional_null_video
        )
        cfg_mode = (
            ("video" if trans_cfg_scale > 1.0 else "none")
            if text_condition is None
            else text_condition.cfg_mode
        )
        cfg_branches = transformer_cfg_branch_count(cfg_mode)
        if trans_cfg_scale > 1.0:
            if cfg_branches == 1:
                raise ValueError("CFG inference requires an active CFG mode")
            if audio_cfg_expanded:
                if audio_hidden.shape[0] != video_batch * cfg_branches:
                    raise ValueError(
                        f"CFG-expanded BOS audio has the wrong branch batch: audio={audio_hidden.shape[0]} video={video_batch}"
                    )
            else:
                if audio_hidden.shape[0] != video_batch:
                    raise ValueError(
                        f"audio/video batches must match before CFG expansion: audio={audio_hidden.shape[0]} video={video_batch}"
                    )
                audio_hidden = torch.cat([audio_hidden] * cfg_branches, dim=0)
            if cfg_mode == "video":
                null_video = self.null_embedding.view(1, 1, -1).expand(
                    video_batch, summary_count, -1
                )
                if null_video.dtype != video_hidden.dtype:
                    null_video = null_video.to(dtype=video_hidden.dtype)
                video_hidden = torch.cat([video_hidden, null_video], dim=0)
            elif cfg_mode == "prompt":
                video_hidden = torch.cat([video_hidden, video_hidden], dim=0)
            elif cfg_mode == "video_prompt":
                null_video = self.null_embedding.view(1, 1, -1).expand(
                    video_batch, summary_count, -1
                )
                if null_video.dtype != video_hidden.dtype:
                    null_video = null_video.to(dtype=video_hidden.dtype)
                video_hidden = torch.cat([video_hidden, null_video, video_hidden], dim=0)
            else:
                raise ValueError(f"unsupported CFG mode: {cfg_mode}")
        else:
            if cfg_mode != "none":
                raise ValueError("a CFG-expanded text condition requires CFG scale > 1")
            if audio_cfg_expanded:
                raise ValueError("audio_cfg_expanded is only valid when Transformer CFG is enabled")
            if audio_hidden.shape[0] != video_batch:
                raise ValueError(
                    f"audio/video batches must match: audio={audio_hidden.shape[0]} video={video_batch}"
                )
        if text_condition is None:
            text_condition = self.prepare_text_condition(
                None,
                None,
                batch_size=video_batch,
                device=audio_hidden.device,
                dtype=audio_hidden.dtype,
                cfg_mode=cfg_mode,
                apply_dropout=False,
            )
        if text_condition is not None and text_condition.mask.shape[0] != audio_hidden.shape[0]:
            raise ValueError(
                f"prepared text batch does not match paired inference batch: {text_condition.mask.shape[0]} != {audio_hidden.shape[0]}"
            )
        (audio_hidden, video_hidden) = (audio_hidden, video_hidden)
        if summary_count == 1:
            hidden_state = torch.cat([audio_hidden, video_hidden], dim=1)
        else:
            (hidden_state, _) = interleave_tokens(audio_hidden, video_hidden)
        semantic_condition = self.prepare_inference_semantic_condition()
        ar_dino_condition = None
        if self.causal_mask is not None and self.causal_mask.device != hidden_state.device:
            self.causal_mask = self.causal_mask.to(hidden_state.device)
        self.freqs_cis = self.freqs_cis.to(hidden_state.device)
        mask = (
            None
            if self.sliding_window is not None
            else self.causal_mask[: hidden_state.shape[0], None, input_pos]
        )
        freqs_cis = self.freqs_cis[input_pos]
        text_mask = self._text_mask_for_query(
            text_condition, input_pos=input_pos, query_length=hidden_state.shape[1]
        )
        for layer_index, layer in enumerate(self.layers):
            hidden_state = layer(
                hidden_state,
                freqs_cis,
                input_pos,
                mask,
                text_mask=text_mask,
                text_kv=None if text_condition is None else text_condition.layer_kv[layer_index],
            )
            hidden_state = hidden_state
        return self.output_norm(hidden_state)

    def inference(
        self,
        input_token: torch.Tensor,
        type_id: int,
        input_pos: tp.Optional[torch.Tensor] = None,
        mask: tp.Optional[torch.Tensor] = None,
        trans_cfg_scale: float = 1.0,
        input_cfg_expanded: bool = False,
        text_condition: tp.Optional[TextCondition] = None,
        video_present: tp.Optional[torch.Tensor | bool] = None,
        *args,
        **kwargs,
    ):
        input_batch = input_token.size(0)
        input_token = input_token.view(input_batch, 1, -1)
        assert self.freqs_cis is not None, "Caches must be initialized first"
        input_emb = self.type_emb(input_token, type_id)
        cfg_mode = (
            ("video" if trans_cfg_scale > 1.0 else "none")
            if text_condition is None
            else text_condition.cfg_mode
        )
        cfg_branches = transformer_cfg_branch_count(cfg_mode)
        cfg_enabled = trans_cfg_scale > 1.0
        if cfg_enabled != (cfg_branches > 1):
            raise ValueError(
                f"Transformer CFG scale and prepared condition mode disagree: scale={trans_cfg_scale} mode={cfg_mode}"
            )
        if input_cfg_expanded:
            if not cfg_enabled:
                raise ValueError("input_cfg_expanded requires Transformer CFG to be enabled")
            if input_batch % cfg_branches:
                raise ValueError(
                    f"CFG-expanded input batch is not divisible by its branch count: batch={input_batch} branches={cfg_branches}"
                )
            bs = input_batch // cfg_branches
        else:
            bs = input_batch
        if type_id == 0:
            (B, T, C) = input_emb.shape
            real_video_mask = self._normalize_video_present(
                video_present, batch_size=B, device=input_emb.device
            )
            conditional_video = self.x_token_dropout(input_emb)
            conditional_video = torch.where(
                real_video_mask[:, None, None],
                conditional_video,
                self.null_embedding.expand(B, T, C).to(dtype=conditional_video.dtype),
            )
            if cfg_enabled:
                if input_cfg_expanded:
                    raise ValueError("pre-expanded CFG input is only supported for audio tokens")
                hidden_state = conditional_video
                if cfg_mode == "video":
                    cond_null = self.null_embedding.expand(B, T, C)
                    hidden_state = torch.cat([hidden_state, cond_null], dim=0)
                elif cfg_mode == "prompt":
                    hidden_state = torch.cat([hidden_state, hidden_state], dim=0)
                elif cfg_mode == "video_prompt":
                    cond_null = self.null_embedding.expand(B, T, C)
                    hidden_state = torch.cat([hidden_state, cond_null, hidden_state], dim=0)
                else:
                    raise ValueError("CFG inference requires an active CFG mode")
            else:
                hidden_state = conditional_video
        elif type_id == 1:
            real_video_mask = None
            hidden_state = self.y_token_dropout(input_emb)
            if cfg_enabled and (not input_cfg_expanded):
                hidden_state = torch.cat([hidden_state] * cfg_branches, dim=0)
        else:
            raise ValueError(f"invalid token type id: {type_id}")
        if text_condition is None:
            text_condition = self.prepare_text_condition(
                None,
                None,
                batch_size=bs,
                device=hidden_state.device,
                dtype=hidden_state.dtype,
                cfg_mode=cfg_mode,
                apply_dropout=False,
            )
        if text_condition is not None and text_condition.mask.shape[0] != hidden_state.shape[0]:
            raise ValueError(
                f"prepared text batch does not match inference batch: {text_condition.mask.shape[0]} != {hidden_state.shape[0]}"
            )
        semantic_condition = self.prepare_inference_semantic_condition()
        ar_dino_condition = None
        if self.causal_mask is not None and self.causal_mask.device != hidden_state.device:
            self.causal_mask = self.causal_mask.to(hidden_state.device)
        mask = (
            None
            if self.sliding_window is not None
            else self.causal_mask[: hidden_state.size(0), None, input_pos]
        )
        self.freqs_cis = self.freqs_cis.to(hidden_state.device)
        freqs_cis = self.freqs_cis[input_pos]
        text_mask = self._text_mask_for_query(
            text_condition, input_pos=input_pos, query_length=hidden_state.shape[1]
        )
        for layer_index, layer in enumerate(self.layers):
            hidden_state = layer(
                hidden_state,
                freqs_cis,
                input_pos,
                mask,
                text_mask=text_mask,
                text_kv=None if text_condition is None else text_condition.layer_kv[layer_index],
            )
            hidden_state = hidden_state
        hidden_state = self.output_norm(hidden_state)
        return hidden_state


class MultiheadAttention(nn.Module):
    _fsdp_final = True

    def __init__(self):
        super().__init__()
        assert True
        self.embed_dim = 512
        self.num_heads = 8
        self.head_dim = 64
        self.attn_dropout_p = 0.1
        self.resid_dropout = nn.Dropout(0.1)
        self.wqkv = nn.Linear(512, 1536, bias=False)
        self.wo = nn.Linear(512, 512, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        rotary_cos: tp.Optional[torch.Tensor] = None,
        rotary_sin: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (bsz, seqlen_q, _) = query.shape
        (xq, xk, xv) = self.wqkv(query).split(512, dim=-1)
        xq = xq.view(bsz, seqlen_q, 8, 64)
        xk = xk.view(bsz, seqlen_q, 8, 64)
        xv = xv.view(bsz, seqlen_q, 8, 64).transpose(1, 2)
        if rotary_cos is not None or rotary_sin is not None:
            if rotary_cos is None or rotary_sin is None:
                raise ValueError("rotary_cos and rotary_sin must be provided together")
            expected_shape = (seqlen_q, 64)
            if rotary_cos.shape != expected_shape or rotary_sin.shape != expected_shape:
                raise ValueError(
                    f"2D RoPE cache shape mismatch: expected {expected_shape}, got {tuple(rotary_cos.shape)} and {tuple(rotary_sin.shape)}"
                )
            cos = rotary_cos[None, :, None, :]
            sin = rotary_sin[None, :, None, :]
            xq = xq * cos + _rotate_pairs(xq) * sin
            xk = xk * cos + _rotate_pairs(xk) * sin
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        (keys, values) = (xk, xv)
        output = F.scaled_dot_product_attention(
            xq, keys, values, dropout_p=0.1 if self.training else 0, is_causal=False
        )
        output = output.transpose(1, 2).reshape(bsz, seqlen_q, 512)
        output = self.resid_dropout(self.wo(output))
        return output


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs by 90 degrees for real-valued RoPE."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE dimensions must be even")
    paired = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
    return rotated.flatten(-2)


def build_2d_rope_cache(
    spatial_shape: tuple[int, int],
    *,
    num_special_tokens: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build axial 2D RoPE values for row-major grid tokens and special tokens."""
    (height, width) = (int(spatial_shape[0]), int(spatial_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"spatial_shape must be positive, got {spatial_shape}")
    if num_special_tokens < 0:
        raise ValueError("num_special_tokens must be non-negative")
    if head_dim % 4:
        raise ValueError(f"2D RoPE requires attention head_dim divisible by 4, got {head_dim}")
    axis_dim = head_dim // 2
    inv_freq = 1.0 / theta ** (
        torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32) / axis_dim
    )
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    (row_grid, col_grid) = torch.meshgrid(rows, cols, indexing="ij")
    row_angles = row_grid.reshape(-1, 1) * inv_freq.reshape(1, -1)
    col_angles = col_grid.reshape(-1, 1) * inv_freq.reshape(1, -1)
    angles = torch.cat(
        (row_angles.repeat_interleave(2, dim=-1), col_angles.repeat_interleave(2, dim=-1)), dim=-1
    )
    if num_special_tokens:
        angles = torch.cat(
            (angles, torch.zeros(num_special_tokens, head_dim, device=device, dtype=angles.dtype)),
            dim=0,
        )
    return (angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype))


class AggregateTransformerLayer(nn.Module):
    """TransformerLayer."""

    def __init__(self):
        super().__init__()
        self.attention = MultiheadAttention()
        self.drop_path = nn.Identity()
        self.feed_forward = FeedForward(512, None, 0.1, 256)
        self.attention_norm = nn.RMSNorm(512, eps=1e-05)
        self.ffn_norm = nn.RMSNorm(512, eps=1e-05)
        self.layer_scale_1: nn.Module
        self.layer_scale_2: nn.Module
        self.layer_scale_1 = nn.Identity()
        self.layer_scale_2 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos: tp.Optional[torch.Tensor] = None,
        rotary_sin: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.drop_path(
            self.layer_scale_1(
                self.attention(self.attention_norm(x), rotary_cos=rotary_cos, rotary_sin=rotary_sin)
            )
        )
        out = h + self.drop_path(self.layer_scale_2(self.feed_forward(self.ffn_norm(h))))
        return out


class AggregateTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        assert True
        self.num_heads = 8
        self.positional_embedding = nn.Parameter(
            0.04419417382415922 * torch.randn(96, 512), requires_grad=False
        )
        self.aggregated_token_positional_embedding = nn.Parameter(
            0.04419417382415922 * torch.randn(1, 512)
        )
        self.output_norm = nn.RMSNorm(512, eps=1e-05)
        self.layers = nn.ModuleList()
        for layer_id in range(1):
            self.layers.append(AggregateTransformerLayer())
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        x: tp.Optional[torch.Tensor] = None,
        y: tp.Optional[torch.Tensor] = None,
        spatial_shape: tp.Optional[tuple[int, int]] = None,
        *args,
        **kwargs,
    ):
        if x is None or y is None:
            raise ValueError("AggregateTransformer requires both grid and query tokens")
        rotary_cos = None
        rotary_sin = None
        if spatial_shape is None:
            raise ValueError("rope_2d requires spatial_shape=(height, width)")
        expected_tokens = int(spatial_shape[0]) * int(spatial_shape[1])
        if x.shape[1] != expected_tokens:
            raise ValueError(
                f"rope_2d spatial token count mismatch: shape {spatial_shape} implies {expected_tokens} tokens, got {x.shape[1]}"
            )
        (rotary_cos, rotary_sin) = build_2d_rope_cache(
            spatial_shape,
            num_special_tokens=y.shape[1],
            head_dim=64,
            device=x.device,
            dtype=x.dtype,
            theta=10000.0,
        )
        y = y + self.aggregated_token_positional_embedding
        hidden_state = torch.cat((x, y), dim=1)
        for layer in self.layers:
            hidden_state = layer(hidden_state, rotary_cos=rotary_cos, rotary_sin=rotary_sin)
        hidden_state = self.output_norm(hidden_state)
        hidden_state = hidden_state[:, -1:]
        return hidden_state
