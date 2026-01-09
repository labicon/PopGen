import torch

def agent0_policy(obs):
    agent_pos = obs[0:2]
    agent_vel = obs[2:4]
    land_rel_flat = obs[4:-2]
    other_rel = obs[-2:]

    n_landmarks = land_rel_flat.numel() // 2
    if n_landmarks == 0:
        return torch.zeros(1, 2, device=obs.device, dtype=obs.dtype)

    land_rel = land_rel_flat.view(n_landmarks, 2)
    land_abs = land_rel + agent_pos.unsqueeze(0)

    x_coords = land_abs[:, 0]
    max_x = x_coords.max()
    tol = 1e-6
    candidate_mask = (x_coords - max_x).abs() <= tol

    if candidate_mask.any():
        midpoint = agent_pos + 0.5 * other_rel
        dists_sq = ((land_abs - midpoint.unsqueeze(0)) ** 2).sum(dim=1)
        dists_sq_masked = dists_sq.masked_fill(~candidate_mask, float('inf'))
        target_idx = int(torch.argmin(dists_sq_masked))
    else:
        target_idx = int(torch.argmax(x_coords))

    target_pos = land_abs[target_idx]
    kp = 1.5
    kd = 0.5
    force = kp * (target_pos - agent_pos) - kd * agent_vel

    norm = torch.norm(force)
    if torch.isfinite(norm) and norm > 1.0:
        force = force / norm
    force = torch.clamp(force, -1.0, 1.0)

    return force.view(1, 2)

def agent1_policy(obs):
    agent_pos = obs[0:2]
    agent_vel = obs[2:4]
    land_rel_flat = obs[4:-2]
    other_rel = obs[-2:]

    n_landmarks = land_rel_flat.numel() // 2
    if n_landmarks == 0:
        return torch.zeros(1, 2, device=obs.device, dtype=obs.dtype)

    land_rel = land_rel_flat.view(n_landmarks, 2)
    land_abs = land_rel + agent_pos.unsqueeze(0)

    x_coords = land_abs[:, 0]
    max_x = x_coords.max()
    tol = 1e-6
    candidate_mask = (x_coords - max_x).abs() <= tol

    if candidate_mask.any():
        midpoint = agent_pos + 0.5 * other_rel
        dists_sq = ((land_abs - midpoint.unsqueeze(0)) ** 2).sum(dim=1)
        dists_sq_masked = dists_sq.masked_fill(~candidate_mask, float('inf'))
        target_idx = int(torch.argmin(dists_sq_masked))
    else:
        target_idx = int(torch.argmax(x_coords))

    target_pos = land_abs[target_idx]
    kp = 1.5
    kd = 0.5
    force = kp * (target_pos - agent_pos) - kd * agent_vel

    norm = torch.norm(force)
    if torch.isfinite(norm) and norm > 1.0:
        force = force / norm
    force = torch.clamp(force, -1.0, 1.0)

    return force.view(1, 2)