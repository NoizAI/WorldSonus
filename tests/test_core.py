import json
from pathlib import Path

import torch
import pytest

from worldsonus.model import WorldSonus
from worldsonus.video_delta import concat_video_with_delta
from worldsonus.flow import _single_axis_flow_head_apg_update


def test_final_checkpoint_shapes():
    with torch.device("meta"):
        model = WorldSonus()
    expected = json.loads((Path(__file__).parent / "model_shapes.json").read_text())
    actual = {key: list(value.shape) for key, value in model.state_dict().items()}
    assert actual == expected
    assert len(actual) == 638
    assert not any("history" in key for key in actual)


def test_causal_delta():
    video = torch.tensor([[[1.0, 2.0], [4.0, 8.0], [7.0, 9.0]]])
    output = concat_video_with_delta(video)
    torch.testing.assert_close(output[..., :2], video)
    torch.testing.assert_close(
        output[..., 2:], torch.tensor([[[1.0, 2.0], [3.0, 6.0], [3.0, 1.0]]])
    )
    changed = video.clone()
    changed[:, -1] += 100
    torch.testing.assert_close(concat_video_with_delta(changed)[:, :-1], output[:, :-1])


def test_apg_scale_one_is_identity():
    torch.manual_seed(7)
    current, full, null = [torch.randn(2, 3, 64) for _ in range(3)]
    update, momentum = _single_axis_flow_head_apg_update(
        current,
        torch.tensor([0.1, 0.4]),
        full,
        null,
        torch.zeros_like(full),
        False,
        1.0,
        0.0,
        0.12421,
        0.25,
    )
    torch.testing.assert_close(update, full)
    assert torch.isfinite(momentum).all()


def test_codec_length_meta():
    from worldsonus.codec import AudioDecoder

    with torch.device("meta"):
        decoder = AudioDecoder()
        output = decoder(torch.zeros(1, 64, 3))
    assert output.shape == (1, 2, 4800)


def test_codec_preserves_cudnn_acceleration_and_restores_tf32():
    from worldsonus.codec import AudioDecoder

    seen = []

    class ObserveBackend(torch.nn.Module):
        def forward(self, value):
            seen.append((torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32))
            return value

    with torch.device("meta"):
        decoder = AudioDecoder()
    decoder.layers = ObserveBackend()
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=True):
        decoder(torch.zeros(1, 64, 3))
        assert torch.backends.cudnn.enabled
        assert torch.backends.cudnn.allow_tf32
    assert seen == [(True, False)]


@pytest.mark.parametrize("capacity", [4, 80, 100])
def test_ring_cache_causality_across_wraparound(capacity):
    from worldsonus.transformer import RingKVCache

    cache = RingKVCache(1, capacity, torch.float32)
    for start in range(0, capacity * 3, 2):
        positions = torch.tensor([start, start + 1])
        tokens = positions.float()[None, None, :, None].expand(1, 16, 2, 64)
        keys, _, mask = cache.update_for_attention(positions, tokens, tokens)
        for query in range(2):
            visible = keys[0, 0, mask[0, 0, query], 0].sort().values
            end = start + query
            expected = torch.arange(max(0, end - capacity + 1), end + 1).float()
            torch.testing.assert_close(visible, expected)
    assert cache.k_cache.shape == (1, 16, capacity, 64)
