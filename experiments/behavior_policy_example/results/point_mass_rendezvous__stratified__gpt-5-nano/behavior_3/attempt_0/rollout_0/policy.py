import torch

def agent0_policy(obs):
    # Egalitarian decision with tolerant synchrony
    # obs layout:
    # 0-1: ego position
    # 2-3: ego velocity
    # 4-11: landmarks relative positions (4 landmarks * 2 dims)
    # 10-11? wait, for 4 landmarks it's 8 elements: 4*2
    # last 2: other agent relative position
    # Ensure we have at least expected size
    ego_pos = obs[0:2]
    ego_vel = obs[2:4]

    # landmarks relative positions
    landmark_rels = []
    idx = 4
    for _ in range(4):
        landmark_rels.append(obs[idx:idx+2])
        idx += 2

    other_rel = obs[-2:]

    # distances to ego and to other for each landmark
    dist_E_list = [torch.norm(rel) for rel in landmark_rels]
    dist_O_list = [torch.norm(rel - other_rel) for rel in landmark_rels]

    sums = [dist_E_list[i] + dist_O_list[i] for i in range(4)]
    i_star = int(torch.argmin(torch.tensor(sums)).item())

    Li_rel = landmark_rels[i_star]
    dist_E = torch.norm(Li_rel)
    dist_O = torch.norm(Li_rel - other_rel)

    v_E = torch.norm(ego_vel)
    t_E = dist_E / max(v_E, 1e-3)
    t_O = dist_O / max(v_E, 1e-3)  # assume other speed ~ ego speed for estimation

    delta = 0.5
    max_speed = 1.0

    # Convert to python floats for comparison
    t_E_val = float(t_E) if isinstance(t_E, torch.Tensor) else t_E
    t_O_val = float(t_O) if isinstance(t_O, torch.Tensor) else t_O

    if abs(t_E_val - t_O_val) > delta:
        if t_E_val < t_O_val:
            target_speed = max_speed * 0.5  # ego arrives earlier, slow down
        else:
            target_speed = max_speed  # ego slower, go full speed
    else:
        target_speed = max_speed * 0.9  # proceed with plan with slight speed

    # If ego is near the landmark, wait a bit
    if dist_E < 0.15:
        target_speed = 0.0

    Li_dist = dist_E
    if Li_dist > 1e-6:
        dir_vec = Li_rel / Li_dist
    else:
        dir_vec = torch.zeros(2)

    action = dir_vec * target_speed
    action = torch.clamp(action, -1.0, 1.0)

    return action.view(1, 2)

import torch

def agent1_policy(obs):
    # Egalitarian decision with tolerant synchrony for Agent 1
    # obs layout:
    # 0-1: ego position
    # 2-3: ego velocity
    # 4-11: landmarks relative positions (4 landmarks * 2 dims)
    # 12-13: other agent relative position

    ego_pos = obs[0:2]
    ego_vel = obs[2:4]

    landmark_rels = []
    idx = 4
    for _ in range(4):
        landmark_rels.append(obs[idx:idx+2])
        idx += 2

    other_rel = obs[-2:]

    # distances to ego and to other for each landmark
    dist_E_list = [torch.norm(rel) for rel in landmark_rels]
    dist_O_list = [torch.norm(rel - other_rel) for rel in landmark_rels]

    sums = [dist_E_list[i] + dist_O_list[i] for i in range(4)]
    i_star = int(torch.argmin(torch.tensor(sums)).item())

    Li_rel = landmark_rels[i_star]
    dist_E = torch.norm(Li_rel)
    dist_O = torch.norm(Li_rel - other_rel)

    v_E = torch.norm(ego_vel)
    t_E = dist_E / max(v_E, 1e-3)
    t_O = dist_O / max(v_E, 1e-3)  # assume other speed similar to ego's

    delta = 0.5
    max_speed = 1.0

    t_E_val = float(t_E) if isinstance(t_E, torch.Tensor) else t_E
    t_O_val = float(t_O) if isinstance(t_O, torch.Tensor) else t_O

    if abs(t_E_val - t_O_val) > delta:
        if t_E_val < t_O_val:
            target_speed = max_speed * 0.5  # ego arrives earlier, slow down
        else:
            target_speed = max_speed  # ego arrives later, go full speed
    else:
        target_speed = max_speed * 0.9  # proceed with plan with slight speed

    # If ego is near the landmark, wait a bit
    if dist_E < 0.15:
        target_speed = 0.0

    Li_dist = dist_E
    if Li_dist > 1e-6:
        dir_vec = Li_rel / Li_dist
    else:
        dir_vec = torch.zeros(2)

    action = dir_vec * target_speed
    action = torch.clamp(action, -1.0, 1.0)

    return action.view(1, 2)