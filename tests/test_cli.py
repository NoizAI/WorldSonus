import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["worldsonus.infer"])
def test_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--no-compile" in result.stdout


def test_explicit_compile_rejects_cpu(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "worldsonus.infer", "--video", "unused.mp4",
         "--output", str(tmp_path / "out.wav"), "--device", "cpu", "--compile"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "--compile requires a CUDA device" in result.stderr


def test_wav_preserves_decoder_peaks(tmp_path, monkeypatch):
    import soundfile as sf
    import torch
    from worldsonus import cli

    features, output = tmp_path / "features.pt", tmp_path / "out.wav"
    torch.save({"video": torch.zeros(3, 2, 2, 384)}, features)
    monkeypatch.setattr(sys, "argv", ["infer", "--features", str(features),
                                      "--device", "cpu", "--output", str(output)])
    monkeypatch.setattr(cli, "load_latent_stats", lambda _: (torch.zeros(64), torch.ones(64)))
    monkeypatch.setattr(cli, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(cli, "generate", lambda *a, **k: torch.zeros(1, 3, 64))
    monkeypatch.setattr(cli.AudioDecoder, "from_checkpoint",
                        lambda *a, **k: lambda z: torch.full((1, 2, 4800), 1.25))
    cli.main()
    wave, rate = sf.read(output, dtype="float32")
    assert rate == 48000 and sf.info(output).subtype == "FLOAT"
    assert (wave == 1.25).all()
