def agent0_policy(obs):
    import torch

    eps = 1e-6

    # Persistent tempo state
    global tempo_state
    if 'tempo_state' not in globals():
        tempo_state = {"phase": 0, "steps": 0, "phase_len": 40}

    tempo_state["steps"] += 1
    if tempo_state["steps"] >= tempo_state["phase_len"]:
        tempo_state["steps"] = 0
        tempo_state["phase"] = 1 - tempo_state["phase"]

    phase = tempo_state["phase"]

    # Safeguard for small observations
    if obs.numel() < 8:
        return torch.zeros((1, 2), dtype=obs.dtype, device=obs.device)

    # pos_rel = ball_pos - agent_pos
    pos_rel = -obs[2:4]
    dist = torch.norm(pos_rel) + eps

    # Direction from agent to ball
    dir_to_ball = -pos_rel / dist

    # Tangential direction around the ball (orbit)
    tangent = torch.stack([-pos_rel[1], pos_rel[0]], dim=0) / dist

    # Bias toward goal direction (ball to goal)
    dir_to_goal = -obs[6:8]
    norm_goal = torch.norm(dir_to_goal) + eps
    dir_to_goal = dir_to_goal / norm_goal
    bias = dir_to_goal * 0.15

    # Orbit radius maintenance
    target_radius = 0.45
    radial = dir_to_ball * (dist - target_radius) * 0.6  # radial adjustment to keep radius

    # Compose force based on tempo phase
    if phase == 0:
        # Proactive forward-press toward ball with slight tangential drift
        force = dir_to_ball * 0.9 + tangent * 0.25
    else:
        # Spaced sideways loop around the ball with radius maintenance
        force = tangent * 0.9 + radial * 0.6

    final_force = force + bias
    final_force = torch.clamp(final_force, -1.0, 1.0)

    return final_force.unsqueeze(0)

def agent1_policy(obs):
    import torch

    eps = 1e-6

    # Safeguard for small observations
    if obs.numel() < 8:
        return torch.zeros((1, 2), dtype=obs.dtype, device=obs.device)

    # pos_rel = ball_pos - agent_pos (as in partner policy)
    pos_rel = -obs[2:4]
    dist = torch.norm(pos_rel) + eps

    # Direction from agent to ball
    dir_to_ball = -pos_rel / dist

    # Tangential direction around the ball (orbit)
    tangent = torch.stack([-pos_rel[1], pos_rel[0]], dim=0) / dist

    # Bias toward goal direction (ball to goal)
    dir_to_goal = -obs[6:8]
    norm_goal = torch.norm(dir_to_goal) + eps
    dir_to_goal = dir_to_goal / norm_goal
    bias = dir_to_goal * 0.15

    # Orbit radius maintenance
    target_radius = 0.45
    radial = dir_to_ball * (dist - target_radius) * 0.6

    # Compose force
    if dist > 0.3:
        force = dir_to_goal * 0.9 + tangent * 0.2
    else:
        force = tangent * 0.9 + radial * 0.6

    final_force = force + bias
    final_force = torch.clamp(final_force, -1.0, 1.0)

    return final_force.unsqueeze(0)