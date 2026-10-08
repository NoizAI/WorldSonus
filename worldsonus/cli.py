"""Video + prompt to audio, with optional incremental generation and decoding."""

import argparse
from contextlib import ExitStack, closing
from pathlib import Path
import sys

import soundfile as sf
import torch

from worldsonus.checkpoint import load_model
from worldsonus.codec import AudioDecoder, load_latent_stats, denormalize_latents
from worldsonus.infer import compile_model, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Video path; features are extracted automatically")
    source.add_argument("--features", help="Optional precomputed tensor-only feature file")
    parser.add_argument("--prompt", help="Audio description; omit for null-text conditioning")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    for name in ("checkpoint", "codec", "z-stats", "dino", "text-encoder"):
        parser.add_argument("--" + name)
    parser.add_argument(
        "--seconds", type=float, help="Default: video duration rounded down to 100 ms"
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--output", help="Output WAV file")
    parser.add_argument(
        "--stream", action="store_true", help="Generate and decode each 100 ms chunk immediately"
    )
    parser.add_argument(
        "--pcm-stdout", action="store_true", help="Stream raw stereo 48 kHz float32 PCM to stdout"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=None,
        help="Compile inference (default: enabled on CUDA; slower first-time startup)",
    )
    parser.add_argument("--seed", type=int, default=5031)
    args = parser.parse_args()
    if not args.output and not args.pcm_stdout:
        parser.error("Supply --output or --pcm-stdout")
    if args.pcm_stdout and not args.stream:
        parser.error("--pcm-stdout requires --stream")
    if args.features and args.prompt is not None:
        parser.error("Precomputed features already contain prompt tensors")
    if args.output and Path(args.output).exists():
        parser.error("Output exists; choose a new file")
    checkpoint = args.checkpoint or str(args.assets / "worldsonus_150k_kv4.pt")
    codec = args.codec or str(args.assets / "audio_codec.pt")
    stats = args.z_stats or str(args.assets / "z_stats.pt")
    dino = args.dino or str(args.assets / "dino")
    text_encoder = args.text_encoder or str(args.assets / "text")
    device = torch.device(args.device)
    if args.compile is None:
        args.compile = device.type == "cuda"
    if args.compile and device.type != "cuda":
        parser.error("--compile requires a CUDA device")
    if device.type == "cuda":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    mean, std = load_latent_stats(stats)
    with ExitStack() as stack:
        if args.features:
            record = torch.load(args.features, map_location="cpu", weights_only=True)
            video = record["video"]
            if video.ndim == 5 and video.shape[0] == 1:
                video = video[0]
            if video.ndim != 4 or not video.shape[0] or video.shape[0] % 3:
                raise ValueError("Expected one clip with a multiple of three DINO frames")
            record["video"] = video
            num_chunks = video.shape[0] // 3
            chunks = (video[i : i + 3] for i in range(0, len(video), 3))
            prompt = {k: v for k, v in record.items() if k.startswith("prompt_")}
        elif args.stream:
            from worldsonus.features import VideoFrames, VideoEncoder, encode_prompt

            source = VideoFrames(args.video, seconds=args.seconds, start=args.start)
            prompt = encode_prompt(args.prompt, text_encoder, device)
            encoder = VideoEncoder(dino, device)
            frames = stack.enter_context(closing(source.batches(3)))
            chunks = (encoder(batch) for batch in frames)
            num_chunks = source.frames // 3
        else:
            from worldsonus.features import extract_features

            record = extract_features(
                args.video,
                args.prompt,
                dino=dino,
                text_encoder=text_encoder,
                seconds=args.seconds,
                start=args.start,
                device=device,
            )
        model = load_model(checkpoint, device=device, dtype=dtype)
        if args.compile:
            compile_model(model)
        if args.stream:
            from worldsonus.streaming import stream_audio

            decoder = AudioDecoder.from_checkpoint(codec, device=device)
            output = stack.enter_context(
                closing(
                    stream_audio(
                        model,
                        decoder,
                        chunks,
                        num_chunks=num_chunks,
                        mean=mean,
                        std=std,
                        seed=args.seed,
                        **prompt,
                    )
                )
            )
        else:
            latent = generate(model, record, seed=args.seed)
            del model
            decoder = AudioDecoder.from_checkpoint(codec, device=device)
            with torch.inference_mode():
                output = [decoder(denormalize_latents(latent, mean, std).transpose(1, 2))]
        writer = None
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            writer = stack.enter_context(
                sf.SoundFile(args.output, mode="w", samplerate=48000, channels=2, subtype="FLOAT")
            )
        samples = 0
        for waveform in output:
            array = waveform.detach().float().cpu()[0].T.numpy()
            if writer is not None:
                writer.write(array)
                writer.flush()
            if args.pcm_stdout:
                sys.stdout.buffer.write(array.astype("<f4").tobytes())
                sys.stdout.buffer.flush()
            samples += len(array)
    print(f"Generated {samples / 48000:.3f} s of stereo 48 kHz audio", file=sys.stderr)
