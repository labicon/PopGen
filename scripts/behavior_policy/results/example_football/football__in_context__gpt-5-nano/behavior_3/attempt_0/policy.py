def agent0_policy(obs):
    import torch

    # Initialize tempo state (persists across calls)
    global tempo_state
    if 'tempo_state' not in globals():
        tempo_state = {"phase": 0, "steps": 0, "phase_len": 30}

    # Update tempo phase
    tempo_state["steps"] += 1
    if tempo_state["steps"] >= tempo_state["phase_len"]:
        tempo_state["steps"] = 0
        tempo_state["phase"] = 1 - tempo_state["phase"]

    phase = tempo_state["phase"]

    # Basic safeguard
    if obs.numel() < 14:
        return torch.zeros((1, 2), dtype=torch.float32)

    eps = 1e-6

    # v: vector from agent to ball (ball_pos - agent_pos)
    v = -obs[2:4]
    norm_v = torch.norm(v) + eps
    dir_to_ball = v / norm_v  # forward to ball

    # Tangential (sideways) direction around the ball
    vx, vy = v[0], v[1]
    tangent = torch.tensor([-vy, vx], dtype=obs.dtype, device=obs.device) / norm_v

    # Bias toward goal: direction from ball to goal (goal_pos - ball_pos)
    dir_to_goal = -obs[6:8]
    norm_goal = torch.norm(dir_to_goal) + eps
    dir_to_goal = dir_to_goal / norm_goal

    bias = dir_to_goal * 0.2

    if phase == 0:
        force = dir_to_ball * 0.8  # proactive forward-press toward the ball
    else:
        force = tangent * 0.8      # sideways loop around the ball

    final_force = force + bias
    final_force = torch.clamp(final_force, -1.0, 1.0)

    return final_force.unsqueeze(0)

def agent1_policy(obs):
    import torch

    # Basic safeguard
    if obs.numel() < 14:
        return torch.zeros((1, 2), dtype=torch.float32)

    eps = 1e-6

    # v: vector from agent to ball (ball_pos - agent_pos) but as in partner policy we use -obs[2:4]
    v = -obs[2:4]
    norm_v = torch.norm(v) + eps

    # Tangential (sideways) direction around the ball
    vx, vy = v[0], v[1]
    tangent = torch.tensor([-vy, vx], dtype=obs.dtype, device=obs.device) / norm_v

    # Bias toward goal: direction from ball to goal (goal_pos - ball_pos)
    dir_to_goal = -obs[6:8]
    norm_goal = torch.norm(dir_to_goal) + eps
    dir_to_goal = dir_to_goal / norm_goal

    bias = dir_to_goal * 0.2

    # Combine directions to encourage both approaching the goal and circling around the ball
    force = tangent * 0.8 + dir_to_goal * 0.6
    final_force = force + bias
    final_force = torch.clamp(final_force, -1.0, 1.0)

    return final_force.unsqueeze(0)