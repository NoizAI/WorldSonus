"""Raw video and prompt encoders shared by offline and streaming inference."""

import json
import math
import subprocess

import numpy as np
import torch
from torch.nn import functional as F


def read_exact(stream, size):
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError("Video ended before the requested continuous window")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


class VideoFrames:
    """Lazy aspect-preserving 30 Hz decoding; never loads the full video."""

    def __init__(self, path, *, seconds=None, start=0.0, long_edge=None):
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration:format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = json.loads(probe.stdout)
        stream = metadata["streams"][0]
        if not math.isfinite(start) or start < 0 or (long_edge is not None and long_edge < 32):
            raise ValueError("Invalid start time or image size")
        if seconds is None:
            duration = stream.get("duration", metadata.get("format", {}).get("duration"))
            if duration in (None, "N/A"):
                raise ValueError("Video duration unavailable; supply --seconds")
            seconds = math.floor((float(duration) - start + 1e-6) * 10) / 10
        if not math.isfinite(seconds):
            raise ValueError("Duration must be finite")
        self.frames = round(seconds * 30)
        if self.frames < 3 or self.frames % 3 or abs(self.frames / 30 - seconds) > 1e-6:
            raise ValueError("Duration must be a positive multiple of 100 ms")
        self.height, self.width = int(stream["height"]), int(stream["width"])
        self.long_edge = long_edge
        if long_edge is not None:
            scale = long_edge / max(self.width, self.height)
            self.height = max(32, round(self.height * scale / 32) * 32)
            self.width = max(32, round(self.width * scale / 32) * 32)
        self.path, self.start = str(path), start

    def batches(self, batch_size=3):
        if batch_size < 1:
            raise ValueError("Frame batch size must be positive")
        # Round source timestamps upward onto the 30 Hz clock: each tick uses
        # the most recent available frame, never a future nearest neighbour.
        filters = "fps=30:round=up:start_time=0,setsar=1"
        if self.long_edge is not None:
            filters = (
                f"fps=30:round=up:start_time=0,scale={self.width}:{self.height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{self.height},setsar=1"
            )
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(self.start),
            "-i",
            self.path,
            "-an",
            "-vf",
            filters,
            "-frames:v",
            str(self.frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "2",
            "pipe:1",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        try:
            for first in range(0, self.frames, batch_size):
                count = min(batch_size, self.frames - first)
                data = read_exact(process.stdout, count * self.height * self.width * 3)
                array = np.frombuffer(data, dtype=np.uint8).copy()
                yield torch.from_numpy(array.reshape(count, self.height, self.width, 3))
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed while reading the video")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()


def prepare_pixels(frames, device="cuda", target_pixels=399_360):
    """Match the reference area resize and aspect-preserving patch alignment."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.uint8:
        raise ValueError("Expected RGB uint8 frames [T,H,W,3]")
    pixels = frames.to(device).permute(0, 3, 1, 2).contiguous()
    height, width = pixels.shape[-2:]
    scale = math.sqrt(target_pixels / (height * width))
    resized = max(1, round(height * scale)), max(1, round(width * scale))
    if (height, width) != resized:
        pixels = (
            F.interpolate(
                pixels.float(), size=resized, mode="bilinear", align_corners=False, antialias=True
            )
            .round()
            .clamp_(0, 255)
            .to(torch.uint8)
        )
    height, width = resized
    patch_h, patch_w = math.ceil(height / 16) * 16, math.ceil(width / 16) * 16
    scale = max(patch_h / height, patch_w / width)
    cover_h, cover_w = math.ceil(height * scale), math.ceil(width * scale)
    pixels = (pixels.float() / 255).contiguous(memory_format=torch.channels_last)
    pixels = F.interpolate(pixels, size=(cover_h, cover_w), mode="bicubic", antialias=True)
    top, left = (cover_h - patch_h) // 2, (cover_w - patch_w) // 2
    return pixels[:, :, top : top + patch_h, left : left + patch_w]


class VideoEncoder:
    def __init__(self, path, device="cuda"):
        from transformers import AutoModel

        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = (
            AutoModel.from_pretrained(
                path, local_files_only=True, dtype=self.dtype, attn_implementation="sdpa"
            )
            .to(self.device)
            .eval()
        )
        if self.model.config.hidden_size != 384 or self.model.config.patch_size != 16:
            raise ValueError("Expected DINOv3 ViT-S+/16")
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

    @torch.inference_mode()
    def __call__(self, frames):
        pixels = prepare_pixels(frames, self.device)
        count, _, height, width = pixels.shape
        with torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"
        ):
            tokens = self.model(pixel_values=(pixels - self.mean) / self.std).last_hidden_state
        prefix = 1 + self.model.config.num_register_tokens
        grid = tokens[:, prefix:].reshape(count, height // 16, width // 16, 384)
        return (
            F.interpolate(
                grid.permute(0, 3, 1, 2),
                size=(round((height // 16) / 2), round((width // 16) / 2)),
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .half()
        )


@torch.inference_mode()
def encode_prompt(prompt, path, device="cuda"):
    if prompt is None:
        return {}
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "right"
    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        path, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
    )
    encoder = text_model.get_encoder().to(device).eval()
    del text_model
    text = tokenizer(
        [prompt], max_length=192, truncation=True, padding="max_length", return_tensors="pt"
    ).to(device)
    embedding = encoder(
        input_ids=text.input_ids, attention_mask=text.attention_mask.bool()
    ).last_hidden_state
    if tuple(embedding.shape) != (1, 192, 640):
        raise ValueError("Expected T5Gemma 2 features [1,192,640]")
    return {
        "prompt_embedding": embedding.cpu().half(),
        "prompt_mask": text.attention_mask.cpu().bool(),
    }


def extract_features(
    video,
    prompt,
    *,
    dino,
    text_encoder,
    seconds=None,
    start=0.0,
    long_edge=None,
    frame_batch=3,
    device="cuda",
):
    source = VideoFrames(video, seconds=seconds, start=start, long_edge=long_edge)
    text = encode_prompt(prompt, text_encoder, device)
    encoder = VideoEncoder(dino, device)
    batches = source.batches(frame_batch)
    try:
        video = torch.cat([encoder(frames).cpu() for frames in batches])
    finally:
        batches.close()
    return {"video": video, **text}
