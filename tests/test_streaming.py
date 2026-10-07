import torch
from torch import nn

from worldsonus.codec import AudioDecoder, CausalConv1d, CausalConvTranspose1d, denormalize_latents


def tiny_decoder():
    decoder = AudioDecoder.__new__(AudioDecoder)
    nn.Module.__init__(decoder)
    decoder.layers = nn.Sequential(
        CausalConv1d(64, 8, 7),
        nn.SiLU(),
        CausalConvTranspose1d(8, 4, 10, stride=5, bias=True),
        nn.SiLU(),
        CausalConv1d(4, 2, 7, dilation=3),
    )
    return decoder.eval()


def test_incremental_decode_matches_continuous_and_resets():
    torch.manual_seed(2)
    decoder = tiny_decoder()
    latents = torch.randn(1, 64, 11)
    with torch.inference_mode():
        expected = decoder(latents)
        for pieces in ((3, 3, 3, 2), (1,) * 11):
            with decoder.stream() as stream:
                actual = torch.cat([stream.push(x) for x in latents.split(pieces, dim=-1)], dim=-1)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(decoder(latents), expected)


def test_latent_denormalization_matches_training_scale():
    latent = torch.ones(1, 3, 64)
    mean, std = torch.full((64,), 3.0), torch.full((64,), 4.0)
    torch.testing.assert_close(
        denormalize_latents(latent, mean, std), torch.full_like(latent, 11.0)
    )


def test_stream_ownership_and_close():
    import pytest

    decoder = tiny_decoder()
    with decoder.stream() as stream:
        with pytest.raises(RuntimeError, match="active"):
            decoder.stream()
    with pytest.raises(RuntimeError, match="closed"):
        stream.push(torch.zeros(1, 64, 3))
