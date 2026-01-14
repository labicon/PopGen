import torch

def agent0_policy(obs):
    # obs: [agent_pos(2), agent_vel(2), 4*landmark_rel_pos(8), other_rel_pos(2)]
    device = obs.device if hasattr(obs, "device") else None
    dtype = obs.dtype if hasattr(obs, "dtype") else torch.float32

    agent_pos = obs[0:2]
    agent_vel = obs[2:4]
    landmarks_rel = obs[4:12].view(-1, 2)  # shape (4,2)
    other_rel = obs[12:14]

    # Absolute positions of landmarks and other agent
    landmarks = landmarks_rel + agent_pos.unsqueeze(0)  # (4,2)
    other_pos = other_rel + agent_pos  # (2,)

    # Predict other future position using ego velocity (decisive-opportunistic simplification)
    predicted_other = other_pos + 0.5 * agent_vel

    # Scoring parameters
    alpha = 0.6  # weight for predicted_other proximity
    eps = 1e-6

    # Distances
    d_e = torch.norm(landmarks - agent_pos.unsqueeze(0), dim=1)  # ego distances (4,)
    d_o = torch.norm(landmarks - predicted_other.unsqueeze(0), dim=1)  # predicted other distances (4,)

    scores = d_e + alpha * d_o
    preferred_idx = int(torch.argmin(scores).item())

    # Check if other clearly is moving to a different landmark; if so, re-evaluate once with smaller alpha
    other_best_idx = int(torch.argmin(d_o).item())
    # Heuristic: if other is clearly closer to some different landmark (by margin), re-evaluate using smaller alpha
    margin = 0.5
    if other_best_idx != preferred_idx and (d_o[other_best_idx] + 0.1 < d_o[preferred_idx] - margin):
        alpha2 = 0.2
        scores2 = d_e + alpha2 * d_o
        preferred_idx = int(torch.argmin(scores2).item())

    target = landmarks[preferred_idx]

    # PD-style decisive move toward target: proportional to displacement, damped by current velocity
    Kp = 2.0
    Kv = 1.0
    force = Kp * (target - agent_pos) - Kv * agent_vel

    # Limit force magnitude to reasonable bounds
    max_force = 1.0
    mag = torch.norm(force)
    if mag > max_force:
        force = force / (mag + eps) * max_force

    return force.unsqueeze(0)

import torch

def agent1_policy(obs):
    # obs: [agent_pos(2), agent_vel(2), 4*landmark_rel_pos(8), other_rel_pos(2)]
    device = obs.device if hasattr(obs, "device") else None
    dtype = obs.dtype if hasattr(obs, "dtype") else torch.float32

    agent_pos = obs[0:2]
    agent_vel = obs[2:4]
    landmarks_rel = obs[4:12].view(-1, 2)  # shape (4,2)
    other_rel = obs[12:14]

    # Absolute positions of landmarks and other agent
    landmarks = landmarks_rel + agent_pos.unsqueeze(0)  # (4,2)
    other_pos = other_rel + agent_pos  # (2,)

    # Predict other future position using ego velocity (same simplification as partner)
    predicted_other = other_pos + 0.5 * agent_vel

    # Scoring parameters (match partner's decisiveness/opportunism)
    alpha = 0.6
    eps = 1e-6

    # Distances
    d_e = torch.norm(landmarks - agent_pos.unsqueeze(0), dim=1)  # ego distances (4,)
    d_o = torch.norm(landmarks - predicted_other.unsqueeze(0), dim=1)  # predicted other distances (4,)

    scores = d_e + alpha * d_o
    preferred_idx = int(torch.argmin(scores).item())

    # Check if other clearly is moving to a different landmark; if so, re-evaluate once with smaller alpha
    other_best_idx = int(torch.argmin(d_o).item())
    margin = 0.5
    if other_best_idx != preferred_idx and (d_o[other_best_idx] + 0.1 < d_o[preferred_idx] - margin):
        alpha2 = 0.2
        scores2 = d_e + alpha2 * d_o
        preferred_idx = int(torch.argmin(scores2).item())

    target = landmarks[preferred_idx]

    # PD-style decisive move toward target: proportional to displacement, damped by current velocity
    Kp = 2.2
    Kv = 1.0
    force = Kp * (target - agent_pos) - Kv * agent_vel

    # Limit force magnitude
    max_force = 1.0
    mag = torch.norm(force)
    if mag > max_force:
        force = force / (mag + eps) * max_force

    return force.unsqueeze(0)