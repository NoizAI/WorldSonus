from __future__ import annotations
import torch


def concat_video_with_delta(video: torch.Tensor, *, lag: int = 1) -> torch.Tensor:
    """Concatenate features with a causal lagged temporal difference.

    ``lag=1`` is exactly the historical WorldSonus behavior.  Missing
    history at the beginning of a clip is represented by an all-zero feature,
    so the first ``lag`` difference entries equal their appearance features.
    """
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError("temporal-difference lag must be a positive integer")
    if video.ndim < 2:
        raise ValueError("video must contain batch and temporal dimensions")
    if lag >= video.shape[1]:
        raise ValueError(
            f"temporal-difference lag {lag} must be smaller than the sequence length {video.shape[1]}"
        )
    video_shift = torch.zeros_like(video)
    video_shift[:, lag:] = video[:, :-lag]
    return torch.cat([video, video - video_shift], dim=-1)
