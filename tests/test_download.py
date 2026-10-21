import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub


def test_download_current_bundle_pins_one_revision(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        model_info=lambda repo, revision: SimpleNamespace(sha="pinned-revision")))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, name, **kwargs: calls.append((repo, name, kwargs)))
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--output", str(tmp_path)])
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/download_model.py"),
                  run_name="__main__")
    assert [name for repo, name, kwargs in calls] == [
        "worldsonus.pt", "audio_codec.pt", "z_stats.pt"]
    assert all(repo == "FF2416/WorldSonus" and kwargs == {
        "revision": "pinned-revision", "local_dir": tmp_path,
    } for repo, name, kwargs in calls)
