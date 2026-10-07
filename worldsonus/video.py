"""Learned DINOv3 bottleneck; the causal temporal difference is added afterward."""

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class VideoInputSpec:
    name: str = "bottleneck128_delta"
    input_dim: int = 384
    output_dim: int = 128
    use_delta: bool = True


def resolve_video_input_spec(**fixed_contract):
    """The public model has one visual representation, not a variant selector."""
    return VideoInputSpec()


class VideoInputAdapter(nn.Module):
    def __init__(self, spec=None):
        super().__init__()
        self.projection = nn.Linear(384, 128, bias=True)

    def reset_projection_parameters(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.shape[-1] != 384:
            raise ValueError("Expected 384-dimensional DINOv3 features")
        return self.projection(video)
