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


def model_config(payload):
    """Legacy releases used 50 chunks; newer weights carry their own window."""
    config = payload.get("config", {})
    if not isinstance(config, dict) or set(config) - {"context_window_chunks"}:
        raise ValueError("Unsupported WorldSonus checkpoint configuration")
    chunks = config.get("context_window_chunks", 50)
    if type(chunks) is not int or chunks < 1:
        raise ValueError("context_window_chunks must be a positive integer")
    return {"context_window_chunks": chunks}


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
    model = WorldSonus(**model_config(payload))
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
    if len(training) not in (0, 7, 9):
        raise ValueError(f"Unexpected training-only tensor count: {len(training)}")
    clean = {
        key: value for key, value in state.items() if key not in dropped and key not in training
    }
    config = model_config(payload)
    args = payload.get("args")
    if args is not None:
        args = args if isinstance(args, dict) else vars(args)
        chunks = args.get("ar_context_window_chunks")
        if chunks is None:
            seconds = args.get("ar_context_window_seconds")
            seconds = 5.0 if seconds is None else seconds
            chunks = round(float(seconds) * 10)
            if abs(chunks / 10 - float(seconds)) > 1e-6:
                raise ValueError("Context duration must be a multiple of 100 ms")
        config = model_config({"config": {"context_window_chunks": chunks}})
    with torch.device("meta"):
        reference = WorldSonus(**config)
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
            "config": config,
            "model": clean,
            "step": int(payload.get("steps", payload.get("step", 150000))),
        },
        destination,
    )
    return len(clean), len(dropped) + len(training)
