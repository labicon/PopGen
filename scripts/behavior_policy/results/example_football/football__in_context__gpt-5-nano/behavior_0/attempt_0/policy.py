import torch

def agent0_policy(obs):
    # Ensure tensor
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs)
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    batch = obs.size(0)

    # Fallback in case of unexpected observation size
    if obs.size(-1) < 41:
        action = torch.zeros((batch, 2), dtype=obs.dtype, device=obs.device)
        action[:, 0] = 0.4  # gentle forward push
        action[:, 1] = 0.0
        return action

    # Segments based on observation construction
    # s_obs corresponds to first 13 features
    s_obs = obs[:, :13]
    ball_to_goal = s_obs[:, 6:8]  # ball_pos - goal_pos
    a_to_ball = s_obs[:, 2:4]     # agent_pos - ball_pos
    # Teammate information (first 8 features of teammates block)
    team_block = obs[:, 33:41]

    # Compute arc-side direction around the ball
    v = ball_to_goal  # vector from goal to ball
    perp = torch.cat([-v[:, 1:2], v[:, 0:1]], dim=1)  # rotate 90 degrees
    norm_perp = torch.norm(perp, dim=1, keepdim=True) + 1e-6
    side = perp / norm_perp  # unit vector tangent to arc around the ball

    arc_radius = 1.0
    near_goal = (torch.norm(v, dim=1, keepdim=True) < 1.6).float()
    arc_mult = 1.0 + 0.8 * near_goal
    offset = side * arc_radius * arc_mult  # offset from ball to create arc

    target_delta = -a_to_ball + offset  # move to an offset around the ball

    # Teammate influence: bias to align with teammate's lane (keep open passing lane)
    te_vec = team_block[:, 0:2]  # agent_pos - teammate_pos
    to_teammate = -te_vec
    norm_teammate = torch.norm(to_teammate, dim=1, keepdim=True) + 1e-6
    to_teammate_dir = to_teammate / norm_teammate
    target_delta = target_delta + to_teammate_dir * 0.25

    action = target_delta

    # Normalize and clip to reasonable force magnitudes
    mag = torch.norm(action, dim=1, keepdim=True) + 1e-6
    max_mag = torch.tensor(3.0, dtype=obs.dtype, device=obs.device)
    scale = torch.clamp(mag, min=1e-6)
    action = action / scale * torch.clamp(mag, max=max_mag)

    # Ensure shape is (batch, 2)
    action = action[:, :2]
    return action

import torch

def agent1_policy(obs):
    # Ensure tensor
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs)
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    batch = obs.size(0)

    # Fallback in case of unexpected observation size
    if obs.size(-1) < 41:
        action = torch.zeros((batch, 2), dtype=obs.dtype, device=obs.device)
        action[:, 0] = 0.4  # gentle forward push
        action[:, 1] = 0.0
        return action

    # Segments based on observation construction
    # s_obs corresponds to first 13 features
    s_obs = obs[:, :13]
    ball_to_goal = s_obs[:, 6:8]  # ball_pos - goal_pos
    a_to_ball = s_obs[:, 2:4]     # agent_pos - ball_pos
    # Teammate information (first 8 features of teammates block)
    team_block = obs[:, 33:41]

    # Compute arc-side direction around the ball
    v = ball_to_goal  # vector from goal to ball
    perp = torch.cat([-v[:, 1:2], v[:, 0:1]], dim=1)  # rotate 90 degrees
    norm_perp = torch.norm(perp, dim=1, keepdim=True) + 1e-6
    side = perp / norm_perp  # unit vector tangent to arc around the ball

    arc_radius = 1.0
    near_goal = (torch.norm(v, dim=1, keepdim=True) < 1.6).float()
    arc_mult = 1.0 + 0.8 * near_goal
    offset = side * arc_radius * arc_mult  # offset from ball to create arc

    target_delta = -a_to_ball + offset  # move to an offset around the ball

    # Teammate influence: bias to align with teammate's lane
    te_vec = team_block[:, 0:2]  # agent_pos - teammate_pos
    to_teammate = -te_vec
    norm_teammate = torch.norm(to_teammate, dim=1, keepdim=True) + 1e-6
    to_teammate_dir = to_teammate / norm_teammate
    target_delta = target_delta + to_teammate_dir * 0.25

    action = target_delta

    # Normalize and clip to reasonable force magnitudes
    mag = torch.norm(action, dim=1, keepdim=True) + 1e-6
    max_mag = torch.tensor(3.0, dtype=obs.dtype, device=obs.device)
    scale = torch.clamp(mag, min=1e-6)
    action = action / scale * torch.clamp(mag, max=max_mag)

    # Ensure shape is (batch, 2)
    action = action[:, :2]
    return action