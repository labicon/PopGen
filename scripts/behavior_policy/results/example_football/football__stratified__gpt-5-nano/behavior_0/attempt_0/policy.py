import torch

def agent0_policy(obs):
    device = obs.device if isinstance(obs, torch.Tensor) else torch.device('cpu')
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

    if obs.dim() > 1:
        obs = obs.squeeze(0)
    if obs.dim() == 0:
        obs = obs.unsqueeze(0)

    def safe_norm(v):
        n = torch.norm(v)
        if isinstance(n, torch.Tensor) and n.item() == 0:
            return v
        return v / (n + 1e-8)

    x = obs

    # Default goal-directed direction
    if x.numel() >= 8:
        ball_to_goal = -x[6:8]  # ball_pos - goal_pos; negate to get goal direction
        goal_dir = safe_norm(ball_to_goal)
        if goal_dir.numel() != 2:
            goal_dir = torch.tensor([1.0, 0.0], device=device)
    else:
        goal_dir = torch.tensor([1.0, 0.0], device=device)

    action_dir = goal_dir

    # Consider teammate if data available
    if x.numel() >= 30:
        teammate_vec = -x[26:28]  # agent_pos - teammate_pos; negate to get agent->teammate
        dist_to_teammate = torch.norm(teammate_vec).item()
        if 0.0 < dist_to_teammate < 2.0:
            teammate_dir = safe_norm(teammate_vec)
            # Combine direction: bias toward teammate to signal through-ball when possible
            action_dir = teammate_dir * 0.7 + goal_dir * 0.3
        else:
            action_dir = goal_dir

    # Movement magnitude (keep it patient and moderate)
    mag = 0.25
    action = (action_dir * mag).clamp(-1.0, 1.0)

    # Ensure shape (1,2)
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
        obs = obs.squeeze(0)
    if obs.dim() == 0:
        obs = obs.unsqueeze(0)

    def safe_norm(v):
        n = torch.norm(v)
        if isinstance(n, torch.Tensor) and n.item() == 0:
            return v
        return v / (n + 1e-8)

    x = obs

    # Default goal-directed direction
    if x.numel() >= 8:
        ball_to_goal = -x[6:8]  # ball_pos - goal_pos; negate to get goal direction
        goal_dir = safe_norm(ball_to_goal)
        if goal_dir.numel() != 2:
            goal_dir = torch.tensor([1.0, 0.0], device=device)
    else:
        goal_dir = torch.tensor([1.0, 0.0], device=device)

    action_dir = goal_dir

    # Consider teammate if data available
    if x.numel() >= 30:
        teammate_vec = -x[26:28]  # agent_pos - teammate_pos; negate to get agent->teammate
        dist_to_teammate = torch.norm(teammate_vec).item()
        if 0.0 < dist_to_teammate < 2.0:
            teammate_dir = safe_norm(teammate_vec)
            away_dir = -teammate_dir  # move away from teammate to maintain spacing
            action_dir = away_dir * 0.7 + goal_dir * 0.3
        else:
            action_dir = goal_dir

    # Movement magnitude (patient and moderate)
    mag = 0.22
    action = (action_dir * mag).clamp(-1.0, 1.0)

    # Ensure shape (1,2)
    action = action.view(1, -1)
    if action.shape[1] > 2:
        action = action[:, :2]
    return action