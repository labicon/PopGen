import torch

def agent0_policy(obs):
    # Fallback defaults in case the observation is unexpectedly small
    device = obs.device if isinstance(obs, torch.Tensor) else None
    dtype = obs.dtype if isinstance(obs, torch.Tensor) else torch.float32

    if not isinstance(obs, torch.Tensor):
        obs = torch.zeros(100, dtype=dtype, device=device)

    # Direction towards the ball (approximate from observation structure)
    if obs.numel() >= 4:
        dir_to_ball = -obs[2:4]
        dir_to_ball = dir_to_ball.to(dtype=dtype)
        if dir_to_ball.numel() != 2:
            dir_to_ball = torch.zeros(2, dtype=dtype, device=device)
    else:
        dir_to_ball = torch.zeros(2, dtype=dtype, device=device)

    # Direction towards teammate (from encoded teammate relative position)
    if obs.numel() >= 8:
        dir_to_teammate = -obs[-8:-6]
        dir_to_teammate = dir_to_teammate.to(dtype=dtype)
        if dir_to_teammate.numel() != 2:
            dir_to_teammate = torch.zeros(2, dtype=dtype, device=device)
    else:
        dir_to_teammate = torch.zeros(2, dtype=dtype, device=device)

    # Combine heuristics: prioritize quick one-two with teammate, while not ignoring ball
    action = 0.6 * dir_to_ball + 0.9 * dir_to_teammate

    # Normalize to reasonable magnitude and keep tensor on correct device
    mag = torch.norm(action)
    if mag > 0:
        action = action / mag
    action = action.clamp(-1.0, 1.0)

    return action.view(1, 2)

import torch

def agent1_policy(obs):
    device = obs.device if isinstance(obs, torch.Tensor) else None
    dtype = obs.dtype if isinstance(obs, torch.Tensor) else torch.float32

    if not isinstance(obs, torch.Tensor):
        obs = torch.zeros(100, dtype=dtype, device=device)

    # Direction towards the ball (approximate from observation structure)
    if obs.numel() >= 4:
        dir_to_ball = -obs[2:4]
        dir_to_ball = dir_to_ball.to(dtype=dtype)
        if dir_to_ball.numel() != 2:
            dir_to_ball = torch.zeros(2, dtype=dtype, device=device)
    else:
        dir_to_ball = torch.zeros(2, dtype=dtype, device=device)

    # Direction towards teammate (from encoded teammate relative position)
    if obs.numel() >= 8:
        dir_to_teammate = -obs[-8:-6]
        dir_to_teammate = dir_to_teammate.to(dtype=dtype)
        if dir_to_teammate.numel() != 2:
            dir_to_teammate = torch.zeros(2, dtype=dtype, device=device)
    else:
        dir_to_teammate = torch.zeros(2, dtype=dtype, device=device)

    # If teammate direction is weak, bias toward ball; otherwise combine
    if torch.norm(dir_to_teammate) < 0.2 and torch.norm(dir_to_ball) > 0:
        action = 0.9 * dir_to_ball
    else:
        action = 0.5 * dir_to_ball + 0.9 * dir_to_teammate

    # Normalize to reasonable magnitude and keep tensor on correct device
    mag = torch.norm(action)
    if mag > 0:
        action = action / mag
    action = action.clamp(-1.0, 1.0)

    return action.view(1, 2)