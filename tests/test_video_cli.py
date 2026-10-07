import sys

import pytest
import torch
import soundfile as sf

from worldsonus import cli, features


@pytest.mark.parametrize(
    "device,flags,compiled",
    [("cpu", [], False), ("cpu", ["--no-compile"], False),
     ("cuda", [], True), ("cuda", ["--compile"], True),
     ("cuda", ["--no-compile"], False)],
)
def test_video_prompt_cli_runs_encoders_automatically(tmp_path, monkeypatch, device, flags, compiled):
    calls = []
    compile_calls = []

    def extract(video, prompt, **kwargs):
        calls.append((video, prompt))
        return {"video": torch.zeros(3, 2, 2, 384)}

    class Decoder:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            return cls()

        def __call__(self, latent):
            # Ensure the final CLI applies the model's 2*std normalization.
            torch.testing.assert_close(latent, torch.full_like(latent, 11.0))
            return torch.zeros(1, 2, 4800)

    monkeypatch.setattr(features, "extract_features", extract)
    monkeypatch.setattr(cli, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "compile_model", lambda model: compile_calls.append(model))
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(cli, "generate", lambda *a, **k: torch.ones(1, 3, 64))
    monkeypatch.setattr(
        cli, "load_latent_stats", lambda _: (torch.full((64,), 3.0), torch.full((64,), 4.0))
    )
    monkeypatch.setattr(cli, "AudioDecoder", Decoder)
    output = tmp_path / "out.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldsonus-infer",
            "--video",
            "clip.mp4",
            "--prompt",
            "Water",
            "--device",
            device,
            "--output",
            str(output),
            *flags,
        ],
    )
    cli.main()
    assert calls == [("clip.mp4", "Water")]
    assert bool(compile_calls) is compiled
    info = sf.info(output)
    assert (info.frames, info.channels, info.samplerate) == (4800, 2, 48000)


@pytest.mark.parametrize(
    "device,flags,compiled",
    [("cpu", [], False), ("cuda", [], True), ("cuda", ["--no-compile"], False)],
)
def test_stream_video_cli_consumes_one_chunk_at_a_time(tmp_path, monkeypatch, device, flags, compiled):
    from worldsonus import streaming

    events = []
    compile_calls = []

    class Frames:
        frames = 6

        def __init__(self, *args, **kwargs):
            pass

        def batches(self, count):
            assert count == 3
            for i in range(2):
                events.append(f"read{i}")
                yield torch.zeros(3, 32, 32, 3)

    class Encoder:
        def __init__(self, *args):
            pass

        def __call__(self, batch):
            events.append("encode")
            return torch.zeros(3, 1, 1, 384)

    def stream(model, decoder, chunks, **kwargs):
        for i, video in enumerate(chunks):
            assert len(events) == (i + 1) * 3 - 1
            events.append("generate_decode")
            yield torch.zeros(1, 2, 4800)

    monkeypatch.setattr(features, "VideoFrames", Frames)
    monkeypatch.setattr(features, "VideoEncoder", Encoder)
    monkeypatch.setattr(features, "encode_prompt", lambda *a: {})
    monkeypatch.setattr(cli, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "compile_model", lambda model: compile_calls.append(model))
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(cli.AudioDecoder, "from_checkpoint", lambda *a, **k: object())
    monkeypatch.setattr(cli, "load_latent_stats", lambda _: (torch.zeros(64), torch.ones(64)))
    monkeypatch.setattr(streaming, "stream_audio", stream)
    output = tmp_path / "stream.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worldsonus-infer",
            "--video",
            "clip.mp4",
            "--stream",
            "--device",
            device,
            "--output",
            str(output),
            *flags,
        ],
    )
    cli.main()
    assert events == ["read0", "encode", "generate_decode", "read1", "encode", "generate_decode"]
    assert bool(compile_calls) is compiled
    assert sf.info(output).frames == 9600
