def agent0_policy(obs):
    import torch
    device = obs.device

    # extract observations
    ego_pos = obs[0:2]
    L_rel = [obs[4:6], obs[6:8], obs[8:10], obs[10:12]]
    other_rel = obs[12:14]

    # compute distances to each landmark
    D_self_vals = [float(torch.norm(l).item()) for l in L_rel]
    D_other_vals = [float(torch.norm(l - other_rel).item()) for l in L_rel]

    # scoring: product of distances
    scores = [D_self_vals[i] * D_other_vals[i] for i in range(4)]
    min_score = min(scores)
    cand_indices = [i for i, s in enumerate(scores) if abs(s - min_score) < 1e-9]

    if len(cand_indices) == 1:
        idx = cand_indices[0]
    else:
        # tie-break by smallest absolute landmark position (x, then y)
        L_abs_x = []
        L_abs_y = []
        for i in cand_indices:
            L_rel_i = L_rel[i]
            L_abs = ego_pos + L_rel_i
            L_abs_x.append(float(L_abs[0].item()))
            L_abs_y.append(float(L_abs[1].item()))
        best_t = 0
        best_x = L_abs_x[0]
        best_y = L_abs_y[0]
        for k in range(1, len(cand_indices)):
            x = L_abs_x[k]
            y = L_abs_y[k]
            if x < best_x or (abs(x - best_x) < 1e-9 and y < best_y):
                best_t = k
                best_x = x
                best_y = y
        idx = cand_indices[best_t]

    D_self = D_self_vals[idx]
    D_other = D_other_vals[idx]
    L_rel_target = L_rel[idx]

    # movement direction towards landmark with curved pursuit
    D_self_eps = D_self + 1e-6
    dir_to_L = L_rel_target / D_self_eps  # 2D

    # perpendicular direction
    perp = torch.tensor([-dir_to_L[1], dir_to_L[0]], dtype=obs.dtype, device=device)

    max_D = max(D_self, D_other)
    c_raw = (D_other - D_self) / (max_D + 1e-6)
    c_raw = max(-1.0, min(1.0, c_raw.item() if isinstance(c_raw, torch.Tensor) else c_raw))
    curvature_scale = 0.5
    c = c_raw * curvature_scale

    Move_dir = dir_to_L + c * perp
    Move_dir_norm = torch.norm(Move_dir)
    if Move_dir_norm.item() < 1e-6:
        Move_dir = dir_to_L
        Move_dir_norm = torch.norm(Move_dir)
    Move_dir = Move_dir / (Move_dir_norm + 1e-6)

    # speed shaping
    max_speed = 1.0
    base_speed = max_speed * 0.65
    D_sum = D_self + D_other + 1e-6
    speed_factor = 0.4 + 0.6 * (D_other / D_sum)
    velocity_mag = base_speed * speed_factor
    velocity_mag = min(float(velocity_mag), max_speed)

    velocity = Move_dir * velocity_mag

    # arrival handling
    arrival_threshold = 0.05
    if D_self <= arrival_threshold:
        velocity = torch.zeros(2, dtype=obs.dtype, device=device)

    action = velocity.unsqueeze(0)
    return action

def agent1_policy(obs):
    import torch
    device = obs.device

    # extract observations
    ego_pos = obs[0:2]
    # ego_vel not used directly in this policy but available
    _ = obs[2:4]

    # landmarks relative positions to ego
    L_rel = [obs[4:6], obs[6:8], obs[8:10], obs[10:12]]
    # other agent relative position to ego
    other_rel = obs[12:14]

    # compute distances to each landmark for self and other
    D_self_vals = [float(torch.norm(l).item()) for l in L_rel]
    D_other_vals = [float(torch.norm(l - other_rel).item()) for l in L_rel]

    # scoring: product of distances
    scores = [D_self_vals[i] * D_other_vals[i] for i in range(4)]
    min_score = min(scores)
    cand_indices = [i for i, s in enumerate(scores) if abs(s - min_score) < 1e-9]

    if len(cand_indices) == 1:
        idx = cand_indices[0]
    else:
        # tie-break by smallest absolute landmark position (x, then y)
        L_abs_x = []
        L_abs_y = []
        for i in cand_indices:
            L_rel_i = L_rel[i]
            L_abs = ego_pos + L_rel_i
            L_abs_x.append(float(L_abs[0].item()))
            L_abs_y.append(float(L_abs[1].item()))
        best_t = 0
        best_x = L_abs_x[0]
        best_y = L_abs_y[0]
        for k in range(1, len(cand_indices)):
            x = L_abs_x[k]
            y = L_abs_y[k]
            if x < best_x or (abs(x - best_x) < 1e-9 and y < best_y):
                best_t = k
                best_x = x
                best_y = y
        idx = cand_indices[best_t]

    D_self = D_self_vals[idx]
    D_other = D_other_vals[idx]
    L_rel_target = L_rel[idx]

    # movement direction towards landmark with curved pursuit
    D_self_eps = D_self + 1e-6
    dir_to_L = L_rel_target / D_self_eps  # 2D

    # perpendicular direction
    perp = torch.tensor([-dir_to_L[1], dir_to_L[0]], dtype=obs.dtype, device=device)

    max_D = max(D_self, D_other)
    c_raw = (D_other - D_self) / (max_D + 1e-6)
    if isinstance(c_raw, torch.Tensor):
        c_raw = c_raw.item()
    c_raw = max(-1.0, min(1.0, c_raw))
    curvature_scale = 0.5
    c = c_raw * curvature_scale

    Move_dir = dir_to_L + c * perp
    Move_dir_norm = torch.norm(Move_dir)
    if Move_dir_norm.item() < 1e-6:
        Move_dir = dir_to_L
        Move_dir_norm = torch.norm(Move_dir)
    Move_dir = Move_dir / (Move_dir_norm + 1e-6)

    # speed shaping
    max_speed = 1.0
    base_speed = max_speed * 0.65
    D_sum = D_self + D_other + 1e-6
    speed_factor = 0.4 + 0.6 * (D_other / D_sum)
    velocity_mag = base_speed * speed_factor
    velocity_mag = min(float(velocity_mag), max_speed)

    velocity = Move_dir * velocity_mag

    # arrival handling
    arrival_threshold = 0.05
    if D_self <= arrival_threshold:
        velocity = torch.zeros(2, dtype=obs.dtype, device=device)

    action = velocity.unsqueeze(0)
    return action