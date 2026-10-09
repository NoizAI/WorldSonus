import pytest
import torch

from worldsonus.checkpoint import ARCHITECTURE, export_checkpoint, load_model, model_config
from worldsonus.model import WorldSonus


def test_legacy_and_kv4_configuration():
    assert model_config({}) == {"context_window_chunks": 50}
    for chunks in (40, 50):
        with torch.device("meta"):
            model = WorldSonus(context_window_chunks=chunks)
        assert model.transformer.causal_window_size == chunks * 2


@pytest.mark.parametrize("value", [0, -1, 40.5, True, "40", None])
def test_invalid_configuration(value):
    with pytest.raises(ValueError):
        model_config({"config": {"context_window_chunks": value}})


def test_unknown_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        model_config({"config": {"window": 40}})


@pytest.mark.parametrize("chunks", [40, 50])
def test_new_export_keeps_window_and_removes_linear_projector(tmp_path, chunks):
    with torch.device("meta"):
        state = WorldSonus().state_dict()
        for name in ("weight", "bias"):
            state[f"sync_repa_projector.0.{name}"] = torch.empty(2048)
            state[f"sync_repa_projector.1.{name}"] = torch.empty(6144)
        for name in ("adaptive_weight_value", "gradient_ratio_ema", "gradient_ratio_initialized"):
            state[f"_sync_repa_{name}"] = torch.empty(())
    source, dest = tmp_path / "training.pt", tmp_path / "release.pt"
    torch.save({"ema_model": state, "steps": 150000,
                "args": {"ar_context_window_chunks": chunks}}, source)
    assert export_checkpoint(str(source), str(dest)) == (638, 7)
    result = torch.load(dest, weights_only=True)
    assert result["architecture"] == ARCHITECTURE
    assert result["step"] == 150000
    assert model_config(result) == {"context_window_chunks": chunks}
    with torch.device("meta"):
        model = load_model(dest, device="meta")
    assert model.transformer.causal_window_size == chunks * 2
