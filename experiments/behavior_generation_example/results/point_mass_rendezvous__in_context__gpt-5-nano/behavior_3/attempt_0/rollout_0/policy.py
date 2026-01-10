def agent0_policy(obs):
    import torch
    eps = 1e-6
    arrival_threshold = 0.1
    curvature_scale = 0.65
    max_speed = 1.0

    # Parse observation
    ego_pos = obs[0:2]                  # (2,)
    # ego_vel = obs[2:4]               # not used directly
    landmarks_rel = obs[4:12].reshape(4, 2)  # (4,2)
    other_rel = obs[-2:]                 # (2,)

    # Deterministic landmark selection via minimax criterion
    best_score = None
    best_L_abs = None
    best_idx = 0

    for i in range(4):
        Li = landmarks_rel[i]  # (2,)
        D_self = torch.norm(Li)
        D_other = torch.norm(Li - other_rel)
        score = torch.max(D_self, D_other)
        L_abs = ego_pos + Li  # absolute landmark position

        if best_score is None:
            best_score = score.item()
            best_L_abs = L_abs
            best_idx = i
        else:
            score_val = score.item()
            tie = torch.allclose(score, torch.tensor(best_score, dtype=score.dtype, device=score.device), atol=1e-6)
            if (score_val < best_score - 1e-6) or (tie and (L_abs[0].item() < best_L_abs[0].item() or (L_abs[0].item() == best_L_abs[0].item() and L_abs[1].item() < best_L_abs[1].item()))):
                best_score = score_val
                best_L_abs = L_abs
                best_idx = i

    L_star = landmarks_rel[best_idx]  # chosen landmark (2,)
    D_self = torch.norm(L_star)
    D_other = torch.norm(L_star - other_rel)

    # Arrival handling
    if D_self <= arrival_threshold:
        vel = torch.zeros((2,), dtype=obs.dtype, device=obs.device)
        return vel.view(1, 2)

    L_star_abs = ego_pos + L_star
    C = ego_pos + other_rel / 2.0
    W = (C + L_star_abs) / 2.0  # waypoint

    dir_to_W = W - ego_pos
    dir_norm = torch.norm(dir_to_W) + eps
    dir_unit = dir_to_W / dir_norm

    # Tangent (rotate by 90 degrees)
    tang = torch.tensor([-dir_unit[1], dir_unit[0]], dtype=obs.dtype, device=obs.device)

    max_D = torch.max(D_self, D_other) + eps
    bias_scalar = (D_other - D_self) / max_D
    bias_scalar = torch.clamp(bias_scalar, -1.0, 1.0) * curvature_scale

    move_vec = dir_unit + bias_scalar * tang
    move_norm = torch.norm(move_vec) + eps
    move_dir = move_vec / move_norm

    base_speed = max_speed * 0.65
    denom = torch.max(D_self, D_other) + eps
    speed_factor = 0.85 + 0.65 * torch.tanh((D_other - D_self) / denom)
    velocity_mag = base_speed * speed_factor
    velocity_mag = torch.clamp(velocity_mag, 0.0, max_speed)

    velocity = velocity_mag * move_dir
    if torch.isnan(velocity).any():
        velocity = torch.zeros((2,), dtype=obs.dtype, device=obs.device)

    return velocity.view(1, 2)

def agent1_policy(obs):
    import torch
    eps = 1e-6
    arrival_threshold = 0.1
    curvature_scale = 0.65
    max_speed = 1.0

    # Parse observation
    ego_pos = obs[0:2]  # (2,)
    # ego_vel = obs[2:4]  # not used directly
    landmarks_rel = obs[4:12].reshape(4, 2)  # (4,2)
    other_rel = obs[-2:]  # (2,)

    # Deterministic landmark selection via minimax criterion
    best_score = None
    best_L_abs = None
    best_idx = 0

    for i in range(4):
        Li = landmarks_rel[i]  # (2,)
        D_self = torch.norm(Li)
        D_other = torch.norm(Li - other_rel)
        score = torch.max(D_self, D_other)
        L_abs = ego_pos + Li  # absolute landmark position

        if best_score is None:
            best_score = score.item()
            best_L_abs = L_abs
            best_idx = i
        else:
            score_val = score.item()
            tie = torch.allclose(score, torch.tensor(best_score, dtype=score.dtype, device=score.device), atol=1e-6)
            if (score_val < best_score - 1e-6) or (tie and (L_abs[0].item() < best_L_abs[0].item() or (L_abs[0].item() == best_L_abs[0].item() and L_abs[1].item() < best_L_abs[1].item()))):
                best_score = score_val
                best_L_abs = L_abs
                best_idx = i

    L_star = landmarks_rel[best_idx]  # chosen landmark (2,)
    D_self = torch.norm(L_star)
    D_other = torch.norm(L_star - other_rel)

    # Arrival handling
    if D_self <= arrival_threshold:
        vel = torch.zeros((2,), dtype=obs.dtype, device=obs.device)
        return vel.view(1, 2)

    L_star_abs = ego_pos + L_star
    C = ego_pos + other_rel / 2.0
    W = (C + L_star_abs) / 2.0  # waypoint

    dir_to_W = W - ego_pos
    dir_norm = torch.norm(dir_to_W) + eps
    dir_unit = dir_to_W / dir_norm

    # Tangent (rotate by 90 degrees)
    tang = torch.tensor([-dir_unit[1], dir_unit[0]], dtype=obs.dtype, device=obs.device)

    max_D = torch.max(D_self, D_other) + eps
    bias_scalar = (D_other - D_self) / max_D
    bias_scalar = torch.clamp(bias_scalar, -1.0, 1.0) * curvature_scale

    move_vec = dir_unit + bias_scalar * tang
    move_norm = torch.norm(move_vec) + eps
    move_dir = move_vec / move_norm

    base_speed = max_speed * 0.65
    denom = torch.max(D_self, D_other) + eps
    speed_factor = 0.85 + 0.65 * torch.tanh((D_other - D_self) / denom)
    velocity_mag = base_speed * speed_factor
    velocity_mag = torch.clamp(velocity_mag, 0.0, max_speed)

    velocity = velocity_mag * move_dir
    if torch.isnan(velocity).any():
        velocity = torch.zeros((2,), dtype=obs.dtype, device=obs.device)

    return velocity.view(1, 2)