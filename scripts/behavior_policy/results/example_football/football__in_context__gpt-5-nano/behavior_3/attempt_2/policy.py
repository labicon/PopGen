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

    # Vector from agent to ball (as per environment's observation layout)
    v = -obs[2:4]
    dist = torch.norm(v) + eps
    dir_to_ball = v / dist

    # Tangential direction around the ball (perpendicular to dir_to_ball)
    tangent = torch.stack([-dir_to_ball[1], dir_to_ball[0]], dim=0)

    # Bias toward goal (ball to goal)
    dir_to_goal = -obs[6:8]
    norm_goal = torch.norm(dir_to_goal) + eps
    dir_to_goal = dir_to_goal / norm_goal
    bias = dir_to_goal * 0.15

    # Orbit radius maintenance
    target_radius = 0.45
    radial = dir_to_ball * (dist - target_radius) * 0.6

    if phase == 0:
        # Proactive forward-press toward the ball with slight tangential drift
        force = dir_to_ball * 0.95 + tangent * 0.25
    else:
        # Spaced sideways loop around the ball with radius maintenance
        force = tangent * 0.95 + radial * 0.6

    final_force = force + bias
    final_force = torch.clamp(final_force, -1.0, 1.0)

    return final_force.unsqueeze(0)

def agent1_policy(obs):
    import torch

    eps = 1e-6

    # Separate tempo for agent1 to coordinate-ish behavior
    global tempo_state_agent1
    if 'tempo_state_agent1' not in globals():
        tempo_state_agent1 = {"phase": 0, "steps": 0, "phase_len": 50}

    tempo_state_agent1["steps"] += 1
    if tempo_state_agent1["steps"] >= tempo_state_agent1["phase_len"]:
        tempo_state_agent1["steps"] = 0
        tempo_state_agent1["phase"] = 1 - tempo_state_agent1["phase"]

    phase = tempo_state_agent1["phase"]

    # Safeguard for small observations
    if obs.numel() < 8:
        return torch.zeros((1, 2), dtype=obs.dtype, device=obs.device)

    # Vector from agent to ball (ball relative to agent)
    v = -obs[2:4]
    dist = torch.norm(v) + eps
    dir_to_ball = v / dist

    # Tangential direction around the ball (perpendicular to dir_to_ball)
    tangent = torch.stack([-dir_to_ball[1], dir_to_ball[0]], dim=0)

    # Bias toward goal (ball to goal)
    dir_to_goal = -obs[6:8]
    norm_goal = torch.norm(dir_to_goal) + eps
    dir_to_goal = dir_to_goal / norm_goal
    bias = dir_to_goal * 0.15

    # Orbit radius maintenance
    target_radius = 0.45
    radial = dir_to_ball * (dist - target_radius) * 0.6

    if phase == 0:
        # Proactive movement towards the landmark with a bias toward the goal
        force = dir_to_ball * 0.65 + tangent * 0.3 + bias * 0.25
    else:
        # Spaced sideways loop around the ball with radius maintenance
        force = tangent * 0.9 + radial * 0.6 + bias * 0.1

    final_force = torch.clamp(force, -1.0, 1.0)

    return final_force.unsqueeze(0)