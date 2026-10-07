from __future__ import annotations
import numpy as np
import torch
from tqdm import tqdm


def flow_matching_sampler(
    net,
    noise: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int = 20,
    guidance: float = 1.0,
    dtype: torch.dtype = torch.float32,
    net_kwargs=None,
    solver: str = "euler",
    start_time: float = 0.0,
):
    """Integrate a noise-to-data rectified-flow ODE.

    ``heun`` intentionally uses an Euler final interval.  This is the usual
    diffusion-sampler convention and gives ``2 * num_steps - 1`` model
    evaluations, so Heun-8/10 can be compared fairly with Euler-15/20.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    start_time = float(start_time)
    if not np.isfinite(start_time) or not 0.0 <= start_time < 1.0:
        raise ValueError("flow matching start_time must lie in [0,1)")
    solver = str(solver).strip().lower()
    if solver not in {"euler", "heun"}:
        raise ValueError(f"unsupported flow matching solver: {solver}")
    net_kwargs = {} if net_kwargs is None else net_kwargs

    def velocity(x, t):
        if guidance == 1:
            return net.inference(x, t, labels, **net_kwargs).to(dtype)
        return net.inference_cfg(x, t, labels, guidance, **net_kwargs).to(dtype)

    x_next = noise.to(dtype)
    step_size = (1.0 - start_time) / num_steps
    for step in tqdm(range(num_steps), desc="flow matching sampling...."):
        t = torch.full(
            (noise.shape[0],),
            start_time + step * step_size,
            device=noise.device,
            dtype=torch.float32,
        )
        current_velocity = velocity(x_next, t)
        euler_next = x_next + step_size * current_velocity
        if solver == "heun" and step < num_steps - 1:
            t_next = torch.full(
                (noise.shape[0],),
                start_time + (step + 1) * step_size,
                device=noise.device,
                dtype=torch.float32,
            )
            next_velocity = velocity(euler_next, t_next)
            x_next = x_next + step_size * 0.5 * (current_velocity + next_velocity)
        else:
            x_next = euler_next
    return x_next
