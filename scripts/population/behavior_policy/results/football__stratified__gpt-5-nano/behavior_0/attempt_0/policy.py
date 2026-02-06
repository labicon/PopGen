import torch


def agent0_policy(obs):
    """
    Aggressive Ambush Attacker policy.
    - Always pursues the ball when not too close.
    - When the ball is close, pushes toward the opponent's goal to capitalize on opportunities.
    - Uses internal observation channels available in the environment:
      agent_pos - ball_pos is at obs[2:4]; ball_pos - goal_pos is at obs[6:8].
    """
    # Ensure we work with a tensor
    o = obs
    if not isinstance(o, torch.Tensor):
        o = torch.tensor(obs, dtype=torch.float32)

    # Basic safety: if shape is unexpected, fallback to zero action
    if o.numel() < 8:
        return torch.zeros((1, 2), dtype=o.dtype if isinstance(o, torch.Tensor) else torch.float32)

    # Vectors from observation components (based on observation_base layout)
    to_ball = o[2:4]        # agent_pos - ball_pos
    ball_to_goal = o[6:8]     # ball_pos - goal_pos

    dir_to_ball = -to_ball
    dist_to_ball = torch.norm(dir_to_ball, p=2)

    if dist_to_ball < 0.6:
        # Close to ball: push toward the opponent's goal (ambush maneuver)
        dir_to_goal = -(to_ball + ball_to_goal)  # direction from agent toward goal
        norm = torch.norm(dir_to_goal, p=2)
        if norm > 0:
            action_dir = dir_to_goal / norm
        else:
            action_dir = torch.zeros(2, dtype=o.dtype)
        action = action_dir * 1.0
    else:
        # Farther from ball: chase the ball
        norm = torch.norm(dir_to_ball, p=2)
        if norm > 0:
            action = dir_to_ball / norm
        else:
            action = torch.zeros(2, dtype=o.dtype)

    return action.reshape(1, 2)

import torch


def agent1_policy(obs):
    """Coordinated Supporter policy.
    - When ball is near, provide side-positioned support while advancing toward the goal.
    - When ball is far, shadow the ball with a slight lateral offset to maintain spacing with the aggressive partner.
    - Uses the same observation components as the aggressive agent: o[2:4] is agent_pos - ball_pos, o[6:8] is ball_pos - goal_pos.
    """
    o = obs
    if not isinstance(o, torch.Tensor):
        o = torch.tensor(obs, dtype=torch.float32)

    if o.numel() < 8:
        return torch.zeros((1, 2), dtype=o.dtype if isinstance(o, torch.Tensor) else torch.float32)

    to_ball = o[2:4]
    ball_to_goal = o[6:8]

    dir_to_ball = -to_ball
    dist_to_ball = torch.norm(dir_to_ball, p=2)

    if dist_to_ball < 0.6:
        # Near the ball: push toward the goal but with a bias to propagate support
        dir_to_goal = -(to_ball + ball_to_goal)
        norm = torch.norm(dir_to_goal, p=2)
        if norm > 0:
            action_dir = dir_to_goal / norm
        else:
            action_dir = torch.zeros(2, dtype=o.dtype)
        action = action_dir * 1.0
    else:
        # Far from ball: shadow the ball with a slight lateral offset to keep spacing with partner
        norm = torch.norm(dir_to_ball, p=2)
        if norm > 0:
            base = dir_to_ball / norm
        else:
            base = torch.zeros(2, dtype=o.dtype)
        # Perpendicular direction to introduce spacing
        perp = torch.tensor([-base[1], base[0]], dtype=o.dtype)
        p_norm = torch.norm(perp, p=2)
        if p_norm > 0:
            perp = perp / p_norm
        action_dir = base + 0.25 * perp
        norm2 = torch.norm(action_dir, p=2)
        if norm2 > 0:
            action = action_dir / norm2
        else:
            action = torch.zeros(2, dtype=o.dtype)

    return action.reshape(1, 2)