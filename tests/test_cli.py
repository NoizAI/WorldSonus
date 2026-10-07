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
