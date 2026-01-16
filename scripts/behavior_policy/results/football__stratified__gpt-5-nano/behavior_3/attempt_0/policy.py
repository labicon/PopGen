import torch

def agent0_policy(obs):
    # Wing Pressor policy: aggressive yet controlled wing pressure
    # Strategy implemented: move toward the ball to contest possession while biasing motion toward the sideline
    # to occupy wide space and shadow ball handlers.
    if obs is None:
        return torch.zeros((1, 2), dtype=torch.float32)

    o = obs
    if not isinstance(o, torch.Tensor):
        o = torch.tensor(obs, dtype=torch.float32)

    if o.dim() != 1:
        o = o.view(-1)

    device = o.device
    dtype = o.dtype

    # Try to extract vectors from the observation strawman layout described in observation_base:
    # o[2:4] = agent_pos - ball_pos -> so -o[2:4] gives ball direction relative to agent
    if o.numel() >= 8:
        toward_ball = -o[2:4]  # 2D vector towards the ball
        # Lateral (sideline) bias: push sideways relative to ball position
        lateral_bias = torch.tensor([0.0, -torch.sign(o[3]).item() if o[3].dim() == 0 else -torch.sign(o[3])], dtype=dtype, device=device)
        # Combine: stronger forward/ball pressure with lateral dispersion
        action_vec = 0.8 * toward_ball + 0.4 * lateral_bias
    else:
        # Fallback: simple wing pressure when observation is shorter than expected
        action_vec = torch.tensor([0.0, 0.5], dtype=dtype, device=device)

    # Clip to environment action limits
    action_vec = torch.clamp(action_vec, -1.0, 1.0)

    return action_vec.view(1, 2)

import torch

def agent1_policy(obs):
    # Supportive mid-field runner policy to coordinate with Wing Pressor
    # Strategy: position to offer central passing lane and width-balanced support
    # by moving toward the ball vicinity while keeping a slight lateral bias to avoid crowding
    if obs is None:
        return torch.zeros((1, 2), dtype=torch.float32)

    o = obs
    if not isinstance(o, torch.Tensor):
        o = torch.tensor(obs, dtype=torch.float32)

    if o.dim() != 1:
        o = o.view(-1)

    device = o.device
    dtype = o.dtype

    # Robust extraction of ball-relative vectors from the observation layout used by the environment
    action_vec = None
    if o.numel() >= 6:
        # Prefer using agent_pos - ball_pos when available (observed as a 2D vector)
        toward_ball = -o[1:3]  # 2D vector toward the ball
        action_vec = 0.7 * toward_ball
        lateral_bias = torch.tensor([0.0, 0.25], dtype=dtype, device=device)
        action_vec = action_vec + 0.25 * lateral_bias
    elif o.numel() >= 4:
        # Fallback to using the next available pair as ball direction
        toward_ball = -o[2:4]
        action_vec = 0.6 * toward_ball
        lateral_bias = torch.tensor([0.0, 0.2], dtype=dtype, device=device)
        action_vec = action_vec + lateral_bias
    else:
        action_vec = torch.tensor([0.0, 0.2], dtype=dtype, device=device)

    # Clip to environment action limits
    action_vec = torch.clamp(action_vec, -1.0, 1.0)

    return action_vec.view(1, 2)