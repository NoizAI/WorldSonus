"""Generate continuous audio from DINOv3 and prompt feature tensors."""

from __future__ import annotations

import torch


def compile_model(model):
    """Compile the AR pair and guided Flow Head; sampling and weights are unchanged.

    Compilation is lazy. The first chunks also warm CUDA graphs, so callers
    should not include those chunks when reporting steady-state latency.
    """
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("Compiled inference requires a CUDA model")
    if getattr(model, "_inference_compiled", False):
        return model
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    model.transformer.inference_audio_video_pair = torch.compile(
        model.transformer.inference_audio_video_pair, mode="reduce-overhead", dynamic=False
    )
    model.diffloss.diffusion_head.inference_multi_cfg = torch.compile(
        model.diffloss.diffusion_head.inference_multi_cfg, mode="reduce-overhead", dynamic=False
    )
    model._inference_compiled = True
    return model


@torch.inference_mode()
def generate(model, record, *, seed=5031, steps=15):
    if getattr(model, "_active_stream", None) is not None:
        raise RuntimeError("Close the streaming session before offline generation")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    video = record["video"].to(device=device, dtype=dtype)
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5 or video.shape[-1] != 384 or video.shape[1] % 3:
        raise ValueError("DINOv3 features must be [B,T,H,W,384], T divisible by 3")
    fields = {}
    for key in (
        "prompt_embedding",
        "prompt_mask",
        "prompt_start_audio_chunks",
        "prompt_duration_audio_chunks",
    ):
        if key in record:
            fields[key] = record[key].to(device)
    from worldsonus.streaming import StreamingGenerator

    # Use identical chunk shapes and operations in offline and streaming mode.
    # BF16 batching differences can otherwise grow through autoregressive feedback.
    with StreamingGenerator(
        model,
        num_chunks=video.shape[1] // 3,
        batch_size=video.shape[0],
        seed=seed,
        steps=steps,
        **fields,
    ) as stream:
        return torch.cat([stream.push(chunk) for chunk in video.split(3, dim=1)], dim=1)


def main():
    from worldsonus.cli import main as run

    run()


if __name__ == "__main__":
    main()
