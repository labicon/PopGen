def agent0_policy(obs):
    import torch

    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    if obs.dim() > 1:
        x = obs.view(-1)
    else:
        x = obs

    device = x.device
    dtype = x.dtype

    if x.numel() >= 12:
        # Direction toward the ball: ball_pos - agent_pos (derived from agent_pos - ball_pos)
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

        if near_ball:
            # Burst dribble with a decisive bias toward goal and ball control
            action = 0.95 * dir_goal + 0.5 * dir_ball + 0.2 * perp
        else:
            # Maintain tempo with ball toward ball and some lateral spacing
            action = 0.8 * dir_ball + 0.2 * dir_goal + 0.15 * perp

        # If ball is very close to the goal, take a direct shot (decisive action)
        ball_to_goal_norm = torch.norm(ball_to_goal)
        if ball_to_goal_norm.item() < 0.6 and dir_goal_norm > 0:
            action = dir_goal

        mag = torch.norm(action)
        if mag > 1.0:
            action = action / mag

        return action.clamp(-1.0, 1.0).view(1, 2)

    # Fallback policy when observation is smaller than expected
    return torch.tensor([[0.6, 0.0]], dtype=dtype, device=device)

def agent1_policy(obs):
    import torch

    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    if obs.dim() > 1:
        x = obs.view(-1)
    else:
        x = obs

    device = x.device
    dtype = x.dtype

    if x.numel() >= 12:
        # Direction toward the ball
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

        # Perpendicular to dir_ball
        perp = torch.tensor([-dir_ball[1], dir_ball[0]], dtype=dtype, device=device)

        # Teammate relative position (to coordinate with partner)
        teammate_rel = x[32:34]
        teammate_dist = torch.norm(teammate_rel)

        near_ball = (to_ball_norm.item() < 0.6)

        if teammate_dist < 0.5:
            # Cooperative offset to avoid both chasing the ball; provide support
            offset = torch.tensor([-dir_goal[1], dir_goal[0]], dtype=dtype, device=device)
            action = 0.7 * dir_goal + 0.25 * offset
        elif near_ball:
            action = 0.5 * dir_goal + 0.3 * dir_ball + 0.2 * perp
        else:
            action = 0.6 * dir_ball + 0.25 * dir_goal + 0.15 * perp

        # If ball is very close to the goal, take a direct shot
        ball_to_goal_norm = torch.norm(ball_to_goal)
        if ball_to_goal_norm.item() < 0.6 and dir_goal_norm > 0:
            action = dir_goal

        mag = torch.norm(action)
        if mag > 1.0:
            action = action / mag

        return action.clamp(-1.0, 1.0).view(1, 2)

    # Fallback
    return torch.tensor([[0.6, 0.0]], dtype=dtype, device=device)