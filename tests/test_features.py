import shutil
import subprocess

import pytest
import torch

from worldsonus.features import VideoFrames, prepare_pixels


@pytest.mark.parametrize(
    "height,width,grid", [(36, 64, (15, 26)), (64, 96, (16, 24)), (48, 64, (18, 23))]
)
def test_reference_spatial_grids(height, width, grid):
    pixels = prepare_pixels(torch.zeros(1, height, width, 3, dtype=torch.uint8), "cpu")
    assert pixels.shape[2] % 16 == pixels.shape[3] % 16 == 0
    assert tuple(round(n / 16 / 2) for n in pixels.shape[2:]) == grid
    assert torch.count_nonzero(pixels) == 0


def test_preprocessing_independent_of_frame_batch():
    frames = torch.randint(0, 256, (6, 36, 64, 3), dtype=torch.uint8)
    whole = prepare_pixels(frames, "cpu")
    chunks = torch.cat([prepare_pixels(x, "cpu") for x in frames.split(3)])
    torch.testing.assert_close(whole, chunks, rtol=0, atol=0)


def test_preprocessing_rejects_non_rgb_uint8():
    with pytest.raises(ValueError, match="uint8"):
        prepare_pixels(torch.zeros(3, 36, 64, 3), "cpu")


@pytest.mark.parametrize("fps", [24, 25, 30, 60])
def test_video_reader_uses_causal_frames(tmp_path, fps):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is needed for video decoding")
    source = tmp_path / "source.nut"
    data = b"".join(bytes([i]) * (4 * 4 * 3) for i in range(fps))
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "4x4",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            str(source),
        ],
        input=data,
        capture_output=True,
        check=True,
    )
    frames = torch.cat(list(VideoFrames(source, seconds=1).batches(3)))
    expected = torch.tensor([i * fps // 30 for i in range(30)], dtype=torch.uint8)
    torch.testing.assert_close(frames[:, 0, 0, 0], expected, rtol=0, atol=0)
