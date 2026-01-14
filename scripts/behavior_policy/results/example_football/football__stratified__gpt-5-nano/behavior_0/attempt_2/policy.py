import torch

def agent0_policy(obs):
    device = obs.device if isinstance(obs, torch.Tensor) else torch.device('cpu')
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

    # Normalize observation shape to 1D
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

    # Start with a forward-aim towards goal when possible
    action_dir = torch.tensor([1.0, 0.0], dtype=obs.dtype, device=device)

    # Align with goal direction if ball_to_goal is available
    if x.numel() >= 8:
        ball_to_goal = -x[6:8]  # ball_pos - goal_pos
        dir_to_goal = safe_norm(ball_to_goal)
        if dir_to_goal.numel() == 2:
            action_dir = dir_to_goal

    # Maintain spacing with teammate to avoid crowding (signal via space)
    if x.numel() >= 27:
        teammate_vec = x[25:27]  # agent_pos - teammate_pos
        dist = torch.norm(teammate_vec).item()
        if dist > 0.0 and dist < 2.0:
            toward_teammate = -teammate_vec
            toward_teammate = safe_norm(toward_teammate)
            # Blend toward-teammate direction to suggest a through-ball/space cue
            action_dir = safe_norm(action_dir * 0.5 + toward_teammate * 0.5)

    # Dribble hint when ball is nearly stationary
    if x.numel() >= 10:
        ball_vel = x[8:10]
        if torch.norm(ball_vel).item() < 0.05:
            perp = torch.tensor([-action_dir[1], action_dir[0]], dtype=obs.dtype, device=device)
            action_dir = safe_norm(action_dir * 0.6 + perp * 0.4)

    # Movement magnitude for patient buildup
    mag = 0.22
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

    # Normalize observation shape to 1D
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

    # Start with a forward-aim towards goal when possible
    action_dir = torch.tensor([1.0, 0.0], dtype=obs.dtype, device=device)

    # Align with goal direction if ball_to_goal is available
    if x.numel() >= 8:
        ball_to_goal = -x[6:8]  # ball_pos - goal_pos or similar
        dir_to_goal = safe_norm(ball_to_goal)
        if dir_to_goal.numel() == 2:
            action_dir = dir_to_goal

    # Maintain spacing with teammate to avoid crowding (signal away from teammate)
    if x.numel() >= 27:
        teammate_vec = x[25:27]  # agent_pos - teammate_pos
        dist = torch.norm(teammate_vec).item()
        if dist > 0.0 and dist < 2.0:
            away_from_teammate = teammate_vec
            away_from_teammate = safe_norm(away_from_teammate)
            action_dir = safe_norm(action_dir * 0.4 + away_from_teammate * 0.6)

    # Dribble hint when ball is nearly stationary
    if x.numel() >= 10:
        ball_vel = x[8:10]
        if torch.norm(ball_vel).item() < 0.05:
            perp = torch.tensor([-action_dir[1], action_dir[0]], dtype=obs.dtype, device=device)
            action_dir = safe_norm(action_dir * 0.6 + perp * 0.4)

    # Movement magnitude for patient buildup
    mag = 0.22
    action = action_dir * mag
    action = action.clamp(-1.0, 1.0)

    action = action.view(1, -1)
    if action.shape[1] > 2:
        action = action[:, :2]
    return action