"""Continuous stereo 48 kHz decoder for the 64-channel causal audio codec.

The complete latent sequence is decoded in one causal pass: no independently
decoded five-second pieces and no waveform crossfades. Convolution names match
the codec checkpoint. See THIRD_PARTY_NOTICES.md for upstream attribution.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        if getattr(self, "_streaming", False):
            if self.stride != (1,):
                raise ValueError("Streaming decoder convolutions must have stride 1")
            previous = self._stream_tail
            if previous is None:
                previous = x.new_zeros(*x.shape[:-1], padding)
            joined = torch.cat((previous, x), dim=-1)
            self._stream_tail = joined[..., -padding:].clone() if padding else None
            return super().forward(joined)
        return super().forward(F.pad(x, (padding, 0)))


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        trim = self.kernel_size[0] - self.stride[0]
        if getattr(self, "_streaming", False) and trim:
            if self._stream_tail is not None:
                y[..., :trim] += self._stream_tail
            # Bias is already present in the next block; overlap only the signal.
            self._stream_tail = y[..., -trim:].clone()
            if self.bias is not None:
                self._stream_tail -= self.bias[None, :, None]
        return y[..., :-trim] if trim else y


class SnakeBeta(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp()[None, :, None].to(x.dtype)
        beta = self.beta.exp()[None, :, None].to(x.dtype)
        return x + (beta + 1e-9).reciprocal() * torch.sin(x * alpha).square()


def conv(in_channels: int, out_channels: int, kernel: int, **kwargs) -> nn.Module:
    return weight_norm(CausalConv1d(in_channels, out_channels, kernel, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            SnakeBeta(channels),
            conv(channels, channels, 7, dilation=dilation),
            SnakeBeta(channels),
            conv(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            SnakeBeta(in_channels),
            weight_norm(
                CausalConvTranspose1d(
                    in_channels,
                    out_channels,
                    2 * stride,
                    stride=stride,
                )
            ),
            *(ResidualUnit(out_channels, dilation) for dilation in (1, 3, 9)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AudioDecoder(nn.Module):
    sample_rate = 48000
    samples_per_latent = 1600

    def __init__(self):
        super().__init__()
        widths = (128, 128, 256, 512, 1024, 2048)
        strides = (2, 4, 5, 5, 8)
        self.layers = nn.Sequential(
            conv(64, widths[-1], 7),
            *(DecoderBlock(widths[i], widths[i - 1], strides[i - 1]) for i in range(5, 0, -1)),
            SnakeBeta(128),
            conv(128, 2, 7, bias=False),
            nn.Identity(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Unnormalized ``[B,64,T]`` latents -> ``[B,2,T*1600]`` audio."""
        if latent.ndim != 3 or latent.shape[1] != 64 or latent.shape[-1] < 1:
            raise ValueError("Expected nonempty codec latents [B,64,T]")
        # TF32's reduced mantissa makes whole-clip and chunked convolution paths
        # diverge. Full FP32 keeps continuous/streaming decoder error near 1e-6.
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=False,
        ):
            return self.layers(latent)

    def stream(self):
        """Create a bounded-state decoder session; close it before reusing this model."""
        return StreamingAudioDecoder(self)

    @classmethod
    def from_checkpoint(cls, path: str, *, device="cpu") -> AudioDecoder:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload)
        # Accept a decoder-only export or an original complete codec checkpoint.
        if any(key.startswith("decoder.") for key in state):
            state = {
                key.removeprefix("decoder."): value
                for key, value in state.items()
                if key.startswith("decoder.")
            }
        model = cls()
        model.load_state_dict(state, strict=True)
        return model.eval().to(device)


def load_latent_stats(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    stats = torch.load(path, map_location="cpu", weights_only=True)
    mean, std = stats["z_mean"].float().reshape(-1), stats["z_std"].float().reshape(-1)
    if mean.numel() != 64 or std.numel() != 64:
        raise ValueError("Latent statistics must contain 64 means and standard deviations")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid latent normalization statistics")
    return mean, std


def denormalize_latents(latent, mean, std):
    """Invert the model's (z - mean) / (2 * std) training normalization."""
    return latent.float() * (2 * std.to(latent.device)) + mean.to(latent.device)


class StreamingAudioDecoder:
    """Decode latent chunks without restarting convolutions at chunk boundaries.

    Each convolution retains its left context; each transposed convolution
    overlaps its unfinished tail. No full-prefix recomputation or crossfade.
    One session owns an AudioDecoder until close().
    """

    def __init__(self, decoder):
        if decoder.training:
            raise ValueError("Call decoder.eval() before streaming")
        self.decoder = decoder
        self.layers = [
            m for m in decoder.modules() if isinstance(m, (CausalConv1d, CausalConvTranspose1d))
        ]
        if any(getattr(m, "_streaming", False) for m in self.layers):
            raise RuntimeError("Decoder already has an active stream")
        self.closed = False
        self.batch_size = None
        for module in self.layers:
            module._streaming = True
            module._stream_tail = None

    @torch.inference_mode()
    def push(self, latents):
        if self.closed:
            raise RuntimeError("Decoder stream is closed")
        if self.batch_size is not None and latents.shape[0] != self.batch_size:
            raise ValueError("Batch size cannot change within a stream")
        self.batch_size = latents.shape[0]
        return self.decoder(latents)

    def close(self):
        for module in self.layers:
            module._streaming = False
            module._stream_tail = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
