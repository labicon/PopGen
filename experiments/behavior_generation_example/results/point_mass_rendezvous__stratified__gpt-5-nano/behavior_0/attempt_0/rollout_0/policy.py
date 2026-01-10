import torch

def agent0_policy(obs):
    # Ego leader policy
    eps = 1e-6
    obs = obs.clone()
    dim = obs.shape[0]
    if dim < 6:
        # Fallback: no information
        return torch.zeros((1, 2))

    # infer number of landmarks from observation size
    L = int((dim - 6) / 2)
    if L <= 0:
        L = 4  # default to 4 landmarks if parsing fails

    ego_pos = obs[0:2]
    ego_vel = obs[2:4]

    landmarks_rel = obs[4:4 + 2 * L].view(L, 2)
    other_rel = obs[4 + 2 * L: 4 + 2 * L + 2]

    landmark_positions = ego_pos.unsqueeze(0) + landmarks_rel  # (L, 2)
    other_pos = ego_pos + other_rel  # (2,)

    # Choose landmark i* that minimizes sum of distances
    d_ego = torch.norm(landmark_positions - ego_pos, dim=1)
    d_other = torch.norm(landmark_positions - other_pos, dim=1)
    total = d_ego + d_other
    i_star = int(torch.argmin(total).item())
    target = landmark_positions[i_star]

    # Direction from ego to target
    dir_E = target - ego_pos
    distE = torch.norm(dir_E)
    if distE > eps:
        dir_E_norm = dir_E / distE
    else:
        dir_E_norm = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    s_E = torch.norm(ego_vel)

    t_E = distE / max(s_E, eps)

    distO = torch.norm(target - other_pos)
    v_O_est = torch.tensor(1.0, dtype=obs.dtype, device=obs.device)
    t_O = distO / max(v_O_est, eps)

    # Desired velocity for ego
    if distE < 1e-3:
        desired_vel_E = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    else:
        if t_E < t_O - 0.01:
            # Ego would arrive earlier: slow down
            desired_vel_E = dir_E_norm * max(s_E * 0.5, 0.0)
        elif t_E > t_O + 0.01:
            # Ego would arrive later: speed up
            desired_vel_E = dir_E_norm * max(s_E * 1.5, 0.1)
        else:
            # Maintain current speed toward target
            desired_vel_E = dir_E_norm * s_E

    a_E = desired_vel_E - ego_vel
    max_acc = 1.0
    a_norm = torch.norm(a_E)
    if a_norm > max_acc and a_norm > 0:
        a_E = a_E / a_norm * max_acc

    force = a_E
    if distE < 1e-3:
        force = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    return force.view(1, 2)

import torch

def agent1_policy(obs):
    eps = 1e-6
    obs = obs.clone()
    dim = obs.shape[0]
    if dim < 6:
        return torch.zeros((1, 2))
    
    L = int((dim - 6) / 2)
    if L <= 0:
        L = 4

    pos1 = obs[0:2]
    vel1 = obs[2:4]

    landmarks_rel = obs[4:4 + 2 * L].view(L, 2)
    other_rel = obs[4 + 2 * L: 4 + 2 * L + 2]

    leader_pos = pos1 + other_rel
    landmark_positions = pos1.unsqueeze(0) + landmarks_rel  # (L, 2)

    d1 = torch.norm(landmark_positions - pos1, dim=1)
    d2 = torch.norm(landmark_positions - leader_pos, dim=1)
    total = d1 + d2
    i_star = int(torch.argmin(total).item())
    target = landmark_positions[i_star]

    dir_vec = target - pos1
    dist = torch.norm(dir_vec)
    if dist > eps:
        dir_norm = dir_vec / dist
    else:
        dir_norm = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    vel1_mag = torch.norm(vel1)
    if dist < 1e-3:
        desired_vel = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    else:
        desired_vel = dir_norm * max(vel1_mag, 0.1)

    a = desired_vel - vel1
    max_acc = 1.0
    a_norm = torch.norm(a)
    if a_norm > max_acc and a_norm > 0:
        a = a / a_norm * max_acc

    force = a
    if dist < 1e-3:
        force = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    return force.view(1, 2)