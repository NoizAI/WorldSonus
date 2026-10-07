from __future__ import annotations
import torch
import typing as tp
import math


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, True)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    rope_scaling: tp.Optional[dict] = None,
    train_seq_len: tp.Optional[int] = None,
):
    """
    Returns:
      freqs_cis: (seq_len, n_elem//2, 2) where [:, :, 0]=cos, [:, :, 1]=sin
    rope_scaling (if not None):
      {"type": "linear" | "ntk" | "yarn",
       "factor": s (>=1),
       # for yarn:
       "alpha": 1.0, "beta": 32.0,
       "attn_temperature_t": float}
    """
    half = n_elem // 2
    dim_index = torch.arange(0, n_elem, 2)[:half].float()
    device = dim_index.device
    t_dtype = torch.float32

    def build_inv_freq(_base: float) -> torch.Tensor:
        inv = 1.0 / _base ** (dim_index / n_elem)
        return inv.to(dtype=t_dtype, device=device)

    method = None
    s = 1.0
    alpha = 1.0
    beta = 32.0
    t_temp = None
    if rope_scaling is not None:
        method = rope_scaling.get("type", None)
        s = float(rope_scaling.get("factor", 1.0))
        alpha = float(rope_scaling.get("alpha", alpha))
        beta = float(rope_scaling.get("beta", beta))
        t_temp = rope_scaling.get("attn_temperature_t", None)
    if method == "ntk":
        d = float(n_elem)
        exp = d / max(d - 2.0, 2.0)
        base_prime = base * s**exp
        inv_freq = build_inv_freq(base_prime)
        t_scaled = torch.arange(seq_len, dtype=t_dtype, device=device)
    elif method == "yarn":
        inv_freq_orig = build_inv_freq(float(base))
        L_train = float(train_seq_len) if train_seq_len is not None else float(seq_len)
        r = L_train * inv_freq_orig / 6.283185307179586
        gamma = torch.where(
            r < alpha,
            torch.zeros_like(r),
            torch.where(r > beta, torch.ones_like(r), (r - alpha) / max(beta - alpha, 1e-06)),
        )
        scale_d = gamma + (1.0 - gamma) / max(s, 1.0)
        inv_freq = inv_freq_orig * scale_d
        t_scaled = torch.arange(seq_len, dtype=t_dtype, device=device)
    else:
        inv_freq = build_inv_freq(float(base))
        t_scaled = torch.arange(seq_len, dtype=t_dtype, device=device) / max(s, 1.0)
    freqs = torch.outer(t_scaled, inv_freq)
    amp = 1.0
    if method == "yarn":
        if t_temp is None:
            amp = 0.1 * math.log(max(s, 1.0)) + 1.0
        else:
            amp = 1.0 / math.sqrt(float(t_temp))
    freqs_cis = torch.polar(torch.full_like(freqs, fill_value=amp), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    if freqs_cis.size(0) != xshaped.size(1):
        if freqs_cis.size(0) < xshaped.size(1):
            raise ValueError(
                f"RoPE cache length {freqs_cis.size(0)} is shorter than sequence length {xshaped.size(1)}"
            )
        freqs_cis = freqs_cis[: xshaped.size(1)]
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def interleave_tokens(x: torch.Tensor, y: torch.Tensor) -> tp.Tuple[torch.Tensor, int]:
    (bsz, t1, c) = x.shape
    (_, t2, c) = y.shape
    assert x.size(2) == y.size(2), "Channel dimensions must match"
    assert t2 % t1 == 0, "Audio token count must be an integer multiple of video token count"
    r = t2 // t1
    y_reshaped = y.view(bsz, t1, r, c)
    x_unsq = x.unsqueeze(2)
    combined = torch.cat((x_unsq, y_reshaped), dim=2)
    merged = combined.view(bsz, t1 * (r + 1), c)
    return (merged, r)


def noise_augment(h: torch.Tensor, k_max: float) -> torch.Tensor:
    """
    Noise augmentation on latent representation.
    Modifiled from https://arxiv.org/pdf/2411.18447

    Args:
        h (torch.Tensor): latent representation with shape [B, T, C]
        k_max (float): maximum scaling factor; scaling parameter k_t is sampled from Uniform[0, k_max]

    Returns:
        torch.Tensor: augmented latent representation with the same shape as h.
    """
    (B, T, C) = h.shape
    noise = torch.randn_like(h)
    k = torch.rand(B, T, 1, device=h.device, dtype=h.dtype) * k_max
    h_aug = k * noise + (1 - k) * h
    return h_aug
