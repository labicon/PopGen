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

    # Heuristic policy:
    # Use the first 12 features which encode relation to ball and goal.
    # Structure (as per observation_base building):
    # [agent_force(2), agent_pos-ball_pos(2), agent_vel-ball_vel(2),
    #  ball_pos-goal_pos(2), ball_vel(2), ball_force(2)]
    if x.numel() >= 12:
        # Direction toward the ball
        to_ball = -x[2:4]
        to_ball_norm = torch.norm(to_ball)
        if to_ball_norm > 0:
            dir_ball = to_ball / to_ball_norm
        else:
            dir_ball = torch.zeros(2, device=device, dtype=dtype)

        # Direction toward the goal from ball
        ball_to_goal = x[6:8]
        dir_to_goal = -ball_to_goal
        dir_goal_norm = torch.norm(dir_to_goal)
        if dir_goal_norm > 0:
            dir_goal = dir_to_goal / dir_goal_norm
        else:
            dir_goal = torch.zeros(2, device=device, dtype=dtype)

        near_ball = (to_ball_norm.item() < 0.6)

        if near_ball:
            # Dribble_pivot: favor goal direction a bit, but keep ball control
            action = 0.75 * dir_goal + 0.25 * dir_ball
        else:
            # Dribble maneuver: push toward ball, with some tempo toward goal
            action = 0.9 * dir_ball + 0.2 * dir_goal

        # Normalize to keep within reasonable force limits
        mag = torch.norm(action)
        if mag > 1.0:
            action = action / mag

        return action.clamp(-1.0, 1.0).view(1, 2)

    # Fallback if observation is smaller than expected
    return torch.tensor([[0.8, 0.0]], dtype=dtype, device=device)

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

    # Complementary heuristic policy:
    # Mirrors the teammate/dribble strategy with a slight lateral bias to provide passing/spacing options.
    # Uses the same initial framing as the partner policy to leverage awareness of ball and goal.
    if x.numel() >= 12:
        # Direction toward the ball
        to_ball = -x[2:4]
        to_ball_norm = torch.norm(to_ball)
        if to_ball_norm > 0:
            dir_ball = to_ball / to_ball_norm
        else:
            dir_ball = torch.zeros(2, device=device, dtype=dtype)

        # Direction toward the goal from ball
        ball_to_goal = x[6:8]
        dir_goal = -ball_to_goal
        dir_goal_norm = torch.norm(dir_goal)
        if dir_goal_norm > 0:
            dir_goal = dir_goal / dir_goal_norm
        else:
            dir_goal = torch.zeros(2, device=device, dtype=dtype)

        near_ball = (to_ball_norm.item() < 0.6)

        # Perpendicular (to add spacing and potential passing lanes)
        perp = torch.tensor([-dir_ball[1], dir_ball[0]], dtype=dtype, device=device)

        if near_ball:
            # Slight bias toward goal with lateral spacing to create passing options
            action = 0.6 * dir_goal + 0.3 * dir_ball + 0.1 * perp
        else:
            # Move toward ball with a bit of lateral bias to coordinate with teammate's drive
            action = 0.8 * dir_ball + 0.1 * dir_goal + 0.1 * perp

        # Normalize to keep within force limits
        mag = torch.norm(action)
        if mag > 1.0:
            action = action / mag

        return action.clamp(-1.0, 1.0).view(1, 2)

    # Fallback if observation is smaller than expected
    return torch.tensor([[-0.4, 0.4]], dtype=dtype, device=device)