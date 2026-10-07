"""Strict, tensor-only release checkpoints and explicit no-H0 conversion."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
import torch

ARCHITECTURE = "worldsonus-c3-no-h0"
HISTORY_PREFIXES = (
    "diffloss.diffusion_head.history_cond_embed.",
    "diffloss.diffusion_head.history_cross_attention_blocks.",
    "diffloss.diffusion_head.extra_history_cross_attention_blocks.",
)
TRAINING_PREFIXES = ("sync_repa_projector.", "_sync_repa_")


def atomic_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def load_model(path: str | Path, *, device="cpu", dtype=torch.float32, ema=True):
    from worldsonus.model import WorldSonus

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError("Not a WorldSonus C3 no-H0 release checkpoint; convert it first")
    state = payload["ema" if ema and "ema" in payload else "model"]
    model = WorldSonus()
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def export_checkpoint(source: str, destination: str, *, trusted_pickle: bool = False):
    """Drop inactive history and training-only tensors; reject other mismatches.

    Original training checkpoints contain an argparse object. Loading those
    requires explicit opt-in to pickle execution; never opt in for unknown files.
    The exported checkpoint contains tensors and primitive metadata only.
    """
    from worldsonus.model import WorldSonus

    payload = torch.load(source, map_location="cpu", weights_only=not trusted_pickle)
    state = payload.get("ema_model", payload.get("model", payload))
    dropped = {key: value for key, value in state.items() if key.startswith(HISTORY_PREFIXES)}
    if len(dropped) not in (0, 20):
        raise ValueError(f"Unexpected inactive history tensor count: {len(dropped)}")
    training = {key: value for key, value in state.items() if key.startswith(TRAINING_PREFIXES)}
    if len(training) not in (0, 9):
        raise ValueError(f"Unexpected training-only tensor count: {len(training)}")
    clean = {
        key: value for key, value in state.items() if key not in dropped and key not in training
    }
    with torch.device("meta"):
        reference = WorldSonus()
    expected = reference.state_dict()
    if set(clean) != set(expected):
        raise ValueError(
            f"Checkpoint keys differ: missing={set(expected) - set(clean)}, "
            f"extra={set(clean) - set(expected)}"
        )
    for key, value in clean.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"Checkpoint shape mismatch at {key}")
    atomic_save(
        {
            "architecture": ARCHITECTURE,
            "model": clean,
            "step": int(payload.get("steps", payload.get("step", 150000))),
        },
        destination,
    )
    return len(clean), len(dropped) + len(training)
