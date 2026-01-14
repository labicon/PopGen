import torch

def agent0_policy(obs):
    """
    Reactive Ball Chaser policy.
    obs: 1D torch tensor as produced by observation_base (see prompt).
    Returns: torch tensor shape (1,2) giving force to apply.
    """
    o = obs.flatten()
    device = o.device
    dtype = o.dtype

    # Index map derived from observation_base construction (all 2D subvectors):
    # 0-1: agent_force
    # 2-3: agent_pos - ball_pos
    # 4-5: agent_vel - ball_vel
    # 6-7: ball_pos - goal_pos
    # 8-9: ball_vel
    # 10-11: ball_force
    # 12-13: agent_rot
    # 14-15: agent_pos - goal_pos
    # 16-17: agent_vel
    # 18-25: adversary0 chunk (8): [agent_pos - adv_pos, agent_vel - adv_vel, adv_vel, adv_force]
    # 26-33: adversary1 chunk (8)
    # 34-41: teammate chunk (8): [agent_pos - team_pos, agent_vel - team_vel, team_vel, team_force]

    # Extract relative vectors we need
    agent_pos_minus_ball = o[2:4]       # agent_pos - ball_pos
    ball_vec = -agent_pos_minus_ball    # ball_pos - agent_pos (direction toward ball)

    agent_pos_minus_goal = o[14:16]     # agent_pos - goal_pos
    goal_vec = -agent_pos_minus_goal    # goal_pos - agent_pos (direction toward goal)

    # Nearby agents (vectors equal agent_pos - other_pos => points from other to ego)
    teammate_vec = o[34:36]
    adv0_vec = o[18:20]
    adv1_vec = o[26:28]

    # Parameters (tunable)
    max_force = 1.0          # maximum thrust magnitude
    capture_radius = 0.5     # if closer than this to ball, push toward goal
    safety_radius = 0.6      # radius to trigger repulsion from other agents
    k_repulse = 0.25         # repulsion strength (small)
    eps = 1e-6

    # Choose reactive target: ball unless within capture radius, then goal
    dist_ball = torch.norm(ball_vec)
    if dist_ball.item() < capture_radius:
        target = goal_vec
    else:
        target = ball_vec

    # Desired force: normalized toward target scaled by max_force
    t_norm = torch.norm(target)
    if t_norm.item() > eps:
        desired = target / (t_norm + eps) * max_force
    else:
        desired = torch.zeros(2, dtype=dtype, device=device)

    # Short-range repulsion from teammate and adversaries (push away from close agents)
    avoid = torch.zeros(2, dtype=dtype, device=device)
    for vec in (teammate_vec, adv0_vec, adv1_vec):
        dist = torch.norm(vec)
        # vec == agent_pos - other_pos; pointing away from other agent toward ego
        if dist.item() < safety_radius and dist.item() > eps:
            # add repulsive component that decays with distance squared
            avoid = avoid + k_repulse * vec / (dist * dist + eps)

    # Combine desired thrust and avoidance
    action = desired + avoid

    # Clip magnitude to max_force
    a_norm = torch.norm(action)
    if a_norm.item() > max_force and a_norm.item() > eps:
        action = action / (a_norm + eps) * max_force

    return action.unsqueeze(0)

import torch

def agent1_policy(obs):
    """
    Cooperative Reactive Ball Chaser for Agent 1.
    Strategy to coordinate with a reactive partner (also a ball chaser):
    - If our teammate is significantly closer to the ball, move to a support position behind
      the ball (along ball->goal direction) so both agents can push the ball toward goal
      without colliding.
    - Otherwise chase the ball like a standard reactive chaser. If within capture radius,
      push toward the goal.
    - Always include short-range repulsion from teammate and adversaries to avoid blocking.

    Input: obs (1D torch tensor from observation_base). Output: torch tensor shape (1,2).
    """
    o = obs.flatten()
    device = o.device
    dtype = o.dtype

    # Index map (2D subvectors as in partner policy)
    # 0-1: agent_force
    # 2-3: agent_pos - ball_pos
    # 4-5: agent_vel - ball_vel
    # 6-7: ball_pos - goal_pos
    # 8-9: ball_vel
    # 10-11: ball_force
    # 12-13: agent_rot
    # 14-15: agent_pos - goal_pos
    # 16-17: agent_vel
    # 18-25: adversary0 chunk (8): [agent_pos - adv_pos, agent_vel - adv_vel, adv_vel, adv_force]
    # 26-33: adversary1 chunk (8)
    # 34-41: teammate chunk (8): [agent_pos - team_pos, agent_vel - team_vel, team_vel, team_force]

    # Extract vectors
    agent_pos_minus_ball = o[2:4]
    ball_vec = -agent_pos_minus_ball            # vector from agent to ball

    ball_minus_goal = o[6:8]                    # ball_pos - goal_pos
    # direction from ball toward goal
    bg_norm = torch.norm(ball_minus_goal)
    if bg_norm.item() > 1e-6:
        dir_bg = ball_minus_goal / (bg_norm + 1e-6)
    else:
        dir_bg = torch.zeros(2, dtype=dtype, device=device)

    agent_pos_minus_goal = o[14:16]
    goal_vec = -agent_pos_minus_goal            # vector from agent to goal

    teammate_vec = o[34:36]                     # agent_pos - teammate_pos
    adv0_vec = o[18:20]
    adv1_vec = o[26:28]

    # Compute teammate position relative to ball: teammate_pos - ball_pos = (agent_pos - ball_pos) - (agent_pos - teammate_pos)
    teammate_pos_minus_ball = agent_pos_minus_ball - teammate_vec

    # Distances
    dist_self_ball = torch.norm(ball_vec)
    dist_teammate_ball = torch.norm(teammate_pos_minus_ball)

    # Parameters
    max_force = 1.0
    capture_radius = 0.5
    safety_radius = 0.6
    k_repulse = 0.35   # slightly larger to ensure spacing
    support_margin = 0.25  # if teammate is this much closer, adopt support role
    behind_offset = 0.6    # distance behind ball along ball->goal to position for support
    eps = 1e-6

    # Decide role: support behind the ball if teammate is noticeably closer
    support_role = (dist_teammate_ball + support_margin) < dist_self_ball

    # If within capture radius, always try to push toward goal (same as partner behavior)
    if dist_self_ball.item() < capture_radius:
        target = goal_vec
    else:
        if support_role:
            # target point = ball_pos + dir_bg * behind_offset; make vector from agent to that point
            target = -agent_pos_minus_ball + dir_bg * behind_offset
        else:
            # chase ball
            target = ball_vec

    # Desired force: normalized toward target scaled by max_force
    t_norm = torch.norm(target)
    if t_norm.item() > eps:
        desired = target / (t_norm + eps) * max_force
    else:
        desired = torch.zeros(2, dtype=dtype, device=device)

    # Short-range repulsion from teammate and adversaries
    avoid = torch.zeros(2, dtype=dtype, device=device)
    for vec in (teammate_vec, adv0_vec, adv1_vec):
        dist = torch.norm(vec)
        if dist.item() < safety_radius and dist.item() > eps:
            # vec is agent_pos - other_pos, points away from other -> push away
            avoid = avoid + k_repulse * vec / (dist * dist + eps)

    action = desired + avoid

    # Clip magnitude to max_force
    a_norm = torch.norm(action)
    if a_norm.item() > max_force and a_norm.item() > eps:
        action = action / (a_norm + eps) * max_force

    return action.unsqueeze(0)