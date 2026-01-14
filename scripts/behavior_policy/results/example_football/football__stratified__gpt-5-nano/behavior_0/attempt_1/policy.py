import torch

def agent0_policy(obs):
    device = obs.device if isinstance(obs, torch.Tensor) else torch.device('cpu')
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

    if obs.dim() > 1:
        obs = obs.view(-1)
    if obs.dim() == 0:
        obs = obs.unsqueeze(0)

    x = obs

    def safe_norm(v):
        if v.numel() == 0:
            return v
        n = torch.norm(v)
        if isinstance(n, torch.Tensor) and n.item() == 0:
            return v
        return v / (n + 1e-8)

    action_dir = torch.tensor([1.0, 0.0], device=device)

    # Prefer direction toward goal using ball_to_goal if available
    if x.numel() >= 8:
        ball_to_goal = -x[6:8]
        dir_to_goal = safe_norm(ball_to_goal)
        if dir_to_goal.numel() == 2:
            action_dir = dir_to_goal

    # Adjust for spacing with teammate if data available
    if x.numel() >= 27:
        teammate_vec = x[25:27]  # agent_pos - teammate_pos
        dist = torch.norm(teammate_vec).item()
        if dist > 0.0 and dist < 2.0:
            teammate_dir = safe_norm(teammate_vec)
            away_dir = -teammate_dir
            action_dir = away_dir * 0.6 + action_dir * 0.4

    # Dribble hint when ball is nearly stationary
    mag = 0.22
    if x.numel() >= 10:
        ball_vel = x[8:10]
        if torch.norm(ball_vel).item() < 0.05:
            perp = torch.tensor([-action_dir[1], action_dir[0]], device=device)
            action_dir = (action_dir * 0.6 + perp * 0.4)

    action = action_dir * mag
    action = action.clamp(-1.0, 1.0)

    action = action.view(1, -1)
    if action.shape[1] > 2:
        action = action[:, :2]
    return action

import torch

def agent1_policy(obs):
    device = obs.device if isinstance(obs, torch.Tensor) else torch.device('cpu')
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

    if obs.dim() > 1:
        obs = obs.view(-1)
    if obs.dim() == 0:
        obs = obs.unsqueeze(0)

    x = obs

    def safe_norm(v):
        if v.numel() == 0:
            return v
        n = torch.norm(v)
        if isinstance(n, torch.Tensor) and n.item() == 0:
            return v
        return v / (n + 1e-8)

    # Default action direction (stay ready to support/playmaker)
    action_dir = torch.tensor([0.0, 0.0], device=device)

    # Prefer alignment with goal direction if ball_to_goal is available
    if x.numel() >= 8:
        ball_to_goal = -x[6:8]
        dir_to_goal = safe_norm(ball_to_goal)
        if dir_to_goal.numel() == 2:
            action_dir = dir_to_goal

    # Coordinate with teammate (agent0) to maintain spacing and support
    if x.numel() >= 27:
        teammate_vec = x[25:27]  # agent_pos - teammate_pos
        dist = torch.norm(teammate_vec).item()
        if dist > 0.0 and dist < 2.0:
            toward_teammate = -teammate_vec
            toward_teammate = safe_norm(toward_teammate)
            action_dir = (action_dir * 0.6) + (toward_teammate * 0.4)

    # If action_dir is zero (no strong cue), default to a forward-right bias
    if action_dir.numel() == 0 or torch.norm(action_dir).item() == 0.0:
        action_dir = torch.tensor([1.0, 0.0], device=device)

    # Gentle dribble bias to keep possession when ball velocity is low
    mag = 0.20
    action = action_dir * mag
    action = action.clamp(-1.0, 1.0)

    action = action.view(1, -1)
    if action.shape[1] > 2:
        action = action[:, :2]
    return action