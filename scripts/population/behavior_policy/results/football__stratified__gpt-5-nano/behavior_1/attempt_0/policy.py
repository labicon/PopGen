import torch

def agent0_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    Defensive Anchor policy (short description):
    - Keeps conservative position between the ball and our goal by moving toward a defined anchor on the line from the goal to the ball.
    - Uses only the information available in the observation to compute a small corrective force.
    - Escalates pressure modestly as the ball approaches the goal region to intercept passes and block advances, but remains conservative overall.
    """
    # Guard against malformed observations
    if obs is None or obs.numel() < 8:
        return torch.zeros(1, 2)

    # Observation layout (inferred from environment's observation_base):
    # obs[2:4]  -> agent_pos - ball_pos  (p)
    # obs[6:8]  -> ball_pos - goal_pos   (d)

    p = obs[2:4]  # 2D vector: agent_pos - ball_pos
    d = obs[6:8]  # 2D vector: ball_pos - goal_pos

    # Anchor from ball toward goal: anchor_pos - agent_pos ~ -p - 0.5*d
    v = -p - 0.5 * d  # desired direction to move (2D)

    d_norm = torch.norm(d)
    # Conservative escalation: increase force when ball is near goal
    if d_norm < 1.5:
        alpha = 0.8
    elif d_norm < 3.0:
        alpha = 0.4
    else:
        alpha = 0.2

    a = alpha * v
    a = torch.clamp(a, -1.0, 1.0)

    return a.view(1, 2)

import torch

def agent1_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    Coordinated policy for Agent 1 to reach the same landmark together with the Defensive Anchor partner.
    Strategy:
    - Use available observation to infer the ball and goal geometry (via p = agent_pos - ball_pos, d = ball_pos - goal_pos).
    - Move toward the goal while providing a slight lateral offset (perpendicular component) to avoid collision and to help align with the defender.
    - Modulate aggressiveness based on how close the ball is to the goal, promoting coordination in critical regions.
    """
    # Guard against malformed observations
    if obs is None or obs.numel() < 8:
        return torch.zeros(1, 2)

    # Observation layout (inferred from environment's observation_base):
    # obs[2:4]  -> agent_pos - ball_pos  (p)
    # obs[6:8]  -> ball_pos - goal_pos   (d)
    p = obs[2:4]  # 2D vector: agent_pos - ball_pos
    d = obs[6:8]  # 2D vector: ball_pos - goal_pos

    # Vector from agent toward the goal (anchor direction)
    v = -(p + d)  # agent -> goal direction

    d_norm = torch.norm(d)
    v_norm = torch.norm(v)

    # Perpendicular (lateral) component for a slight offset to avoid tight packing
    if v_norm > 0:
        perp = torch.tensor([-v[1], v[0]], dtype=obs.dtype, device=obs.device)
        perp = perp / torch.norm(perp)
    else:
        perp = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    # Compute target with a bias toward the goal and a small lateral offset
    if v_norm < 1e-6:
        target = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    else:
        if d_norm < 1.5:
            alpha = 0.6
        elif d_norm < 3.0:
            alpha = 0.45
        else:
            alpha = 0.25
        target = alpha * v + 0.25 * perp

    # Clamp to allowable action space
    a = torch.clamp(target, -1.0, 1.0)

    return a.view(1, 2)