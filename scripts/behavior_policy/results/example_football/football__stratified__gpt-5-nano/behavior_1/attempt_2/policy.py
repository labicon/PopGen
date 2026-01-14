def agent0_policy(obs):
    import torch

    # Ensure obs is a tensor
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    # Flatten to 1D vector if needed
    if obs.dim() > 1:
        x = obs.view(-1)
    else:
        x = obs

    device = x.device
    dtype = x.dtype

    if x.numel() >= 12:
        # Direction toward the ball: ball_pos - agent_pos
        to_ball = -x[2:4]
        to_ball_norm = torch.norm(to_ball)
        if to_ball_norm > 0:
            dir_ball = to_ball / to_ball_norm
        else:
            dir_ball = torch.zeros(2, device=device, dtype=dtype)

        # Direction toward the goal from the ball
        ball_to_goal = x[6:8]
        dir_goal = -ball_to_goal
        dir_goal_norm = torch.norm(dir_goal)
        if dir_goal_norm > 0:
            dir_goal = dir_goal / dir_goal_norm
        else:
            dir_goal = torch.zeros(2, device=device, dtype=dtype)

        near_ball = (to_ball_norm.item() < 0.6)
        perp = torch.tensor([-dir_ball[1], dir_ball[0]], dtype=dtype, device=device)

        # Burst/dribble strategy with quick decision tendencies
        if near_ball:
            action = 0.7 * dir_goal + 0.5 * dir_ball + 0.25 * perp
        else:
            action = 0.75 * dir_ball + 0.15 * dir_goal + 0.15 * perp

        # Quick decisive shot if a direct path to goal is available
        ball_to_goal_norm = torch.norm(ball_to_goal)
        if ball_to_goal_norm.item() < 0.5 and dir_goal_norm > 0:
            action = dir_goal

        # Quick pivot/pass opportunity to teammate if available
        if x.numel() >= 34:
            teammate_rel = x[32:34]  # relative position component used by environment
            teammate_dist = torch.norm(teammate_rel)
            if teammate_dist < 0.6:
                pass_dir = -teammate_rel
                pass_dir_norm = torch.norm(pass_dir)
                if pass_dir_norm > 0:
                    pass_dir = pass_dir / pass_dir_norm
                action = 0.5 * pass_dir + 0.3 * dir_goal + 0.2 * dir_ball

        # Normalize to keep within force limits
        mag = torch.norm(action)
        if mag > 1.0:
            action = action / mag

        return action.clamp(-1.0, 1.0).view(1, 2)

    # Fallback if observation is smaller than expected
    return torch.tensor([[0.6, 0.0]], dtype=dtype, device=device)

def agent1_policy(obs):
    import torch

    # Ensure obs is a tensor
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    # Flatten to 1D vector if needed
    if obs.dim() > 1:
        x = obs.view(-1)
    else:
        x = obs

    device = x.device
    dtype = x.dtype

    if x.numel() >= 12:
        # Direction toward the ball: ball_pos - agent_pos
        to_ball = -x[2:4]
        to_ball_norm = torch.norm(to_ball)
        if to_ball_norm > 0:
            dir_ball = to_ball / to_ball_norm
        else:
            dir_ball = torch.zeros(2, device=device, dtype=dtype)

        # Direction toward the goal from the ball
        ball_to_goal = x[6:8]
        dir_goal = -ball_to_goal
        dir_goal_norm = torch.norm(dir_goal)
        if dir_goal_norm > 0:
            dir_goal = dir_goal / dir_goal_norm
        else:
            dir_goal = torch.zeros(2, device=device, dtype=dtype)

        near_ball = (to_ball_norm.item() < 0.6)
        perp = torch.tensor([-dir_ball[1], dir_ball[0]], dtype=dtype, device=device)

        # Dribble/transition strategy with coordinated pacing
        if near_ball:
            action = 0.65 * dir_goal + 0.4 * dir_ball + 0.15 * perp
        else:
            action = 0.6 * dir_ball + 0.25 * dir_goal + 0.15 * perp

        # Quick decisive shot if a direct path to goal is available
        ball_to_goal_norm = torch.norm(ball_to_goal)
        if ball_to_goal_norm.item() < 0.5 and dir_goal_norm > 0:
            action = dir_goal

        # Quick pivot/pass opportunity to teammate if available
        if x.numel() >= 34:
            teammate_rel = x[32:34]  # relative position component used by environment
            teammate_dist = torch.norm(teammate_rel)
            if teammate_dist < 0.65:
                pass_dir = -teammate_rel
                pass_dir_norm = torch.norm(pass_dir)
                if pass_dir_norm > 0:
                    pass_dir = pass_dir / pass_dir_norm
                action = 0.55 * pass_dir + 0.25 * dir_goal + 0.15 * dir_ball

        # Normalize to keep within force limits
        mag = torch.norm(action)
        if mag > 1.0:
            action = action / mag

        return action.clamp(-1.0, 1.0).view(1, 2)

    # Fallback if observation is smaller than expected
    return torch.tensor([[0.6, 0.0]], dtype=dtype, device=device)