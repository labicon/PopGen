import torch

def agent0_policy(obs):
    # OAR policy for agent0
    # obs layout:
    # [ego_pos(2), ego_vel(2), landmarks_rel(4*2), other_rel(2)]
    # landmarks_rel is 4 vectors: (dx, dy) to each landmark
    # other_rel is the relative position to the other agent: (dx, dy)

    # Extract ego position
    ego_x = float(obs[0].item())
    ego_y = float(obs[1].item())

    # Extract landmarks relative positions
    landmarks_rel = []
    for i in range(4):
        idx = 4 + 2 * i
        lr_x = float(obs[idx].item())
        lr_y = float(obs[idx + 1].item())
        landmarks_rel.append((lr_x, lr_y))

    # Extract other agent relative position
    other_rel = (float(obs[12].item()), float(obs[13].item()))

    arrival_threshold = 0.05

    # Select landmark L* by combined proximity with tie-breaking by absolute coordinates
    best_score = None
    best_L_rel = None
    best_L_pos = None

    for lr in landmarks_rel:
        D_self = (lr[0] ** 2 + lr[1] ** 2) ** 0.5
        D_other = ((other_rel[0] - lr[0]) ** 2 + (other_rel[1] - lr[1]) ** 2) ** 0.5
        score = D_self + D_other
        L_pos = (ego_x + lr[0], ego_y + lr[1])
        if best_score is None or score < best_score - 1e-6:
            best_score = score
            best_L_rel = lr
            best_L_pos = L_pos
        elif abs(score - best_score) <= 1e-6:
            if L_pos[0] < best_L_pos[0] - 1e-6 or (abs(L_pos[0] - best_L_pos[0]) <= 1e-6 and L_pos[1] < best_L_pos[1] - 1e-6):
                best_L_rel = lr
                best_L_pos = L_pos

    L_rel = best_L_rel
    D_self = (L_rel[0] ** 2 + L_rel[1] ** 2) ** 0.5
    D_other = ((other_rel[0] - L_rel[0]) ** 2 + (other_rel[1] - L_rel[1]) ** 2) ** 0.5

    vx = 0.0
    vy = 0.0

    if D_self <= arrival_threshold:
        vx = 0.0
        vy = 0.0
    else:
        # Direction to landmark
        dir_to_L = (L_rel[0] / (D_self + 1e-6), L_rel[1] / (D_self + 1e-6))

        # Arc offset magnitude
        max_perp_offset = 1.0
        min_perp_offset = 0.0
        r = (D_self / (D_self + D_other + 1e-6)) * max_perp_offset
        r = max(min(r, max_perp_offset), min_perp_offset)

        # Determine side using cross product
        cross = dir_to_L[0] * other_rel[1] - dir_to_L[1] * other_rel[0]
        rot90 = (-dir_to_L[1], dir_to_L[0])  # +90 deg rotation

        if cross > 0:
            offset_dir = (-rot90[0], -rot90[1])  # other is left, offset to the right
        elif cross < 0:
            offset_dir = rot90  # other is right, offset to the left
        else:
            offset_dir = rot90  # fallback

        offset = (r * offset_dir[0], r * offset_dir[1])

        # Target point T along the arc
        L_pos = (ego_x + L_rel[0], ego_y + L_rel[1])
        T = (L_pos[0] + offset[0], L_pos[1] + offset[1])

        vec_T = (T[0] - ego_x, T[1] - ego_y)
        dist_T = (vec_T[0] ** 2 + vec_T[1] ** 2) ** 0.5

        if dist_T > 1e-6:
            v_dir = (vec_T[0] / dist_T, vec_T[1] / dist_T)
            max_speed = 1.0
            base_speed = 0.65 * max_speed

            denom = D_self + D_other + 1e-6
            speed_factor = 0.6 + 0.4 * (D_other / denom)
            speed_factor = max(0.0, min(1.0, speed_factor))

            velocity_mag = base_speed * speed_factor
            vx = velocity_mag * v_dir[0]
            vy = velocity_mag * v_dir[1]
        else:
            vx = 0.0
            vy = 0.0

    action = torch.tensor([[vx, vy]], dtype=obs.dtype, device=obs.device)
    return action

import torch

def agent1_policy(obs):
    # OAR policy for agent1 (symmetric to agent0)
    # obs layout:
    # [ego_pos(2), ego_vel(2), landmarks_rel(4*2), other_rel(2)]
    # landmarks_rel is 4 vectors: (dx, dy) to each landmark
    # other_rel is the relative position to the other agent: (dx, dy)

    # Extract ego position
    ego_x = float(obs[0].item())
    ego_y = float(obs[1].item())

    # Extract landmarks relative positions
    landmarks_rel = []
    for i in range(4):
        idx = 4 + 2 * i
        lr_x = float(obs[idx].item())
        lr_y = float(obs[idx + 1].item())
        landmarks_rel.append((lr_x, lr_y))

    # Extract other agent relative position
    other_rel = (float(obs[12].item()), float(obs[13].item()))

    arrival_threshold = 0.05

    # Select landmark L* by combined proximity with tie-breaking by absolute coordinates
    best_score = None
    best_L_rel = None
    best_L_pos = None

    for lr in landmarks_rel:
        D_self = (lr[0] ** 2 + lr[1] ** 2) ** 0.5
        D_other = ((other_rel[0] - lr[0]) ** 2 + (other_rel[1] - lr[1]) ** 2) ** 0.5
        score = D_self + D_other
        L_pos = (ego_x + lr[0], ego_y + lr[1])
        if best_score is None or score < best_score - 1e-6:
            best_score = score
            best_L_rel = lr
            best_L_pos = L_pos
        elif abs(score - best_score) <= 1e-6:
            if L_pos[0] < best_L_pos[0] - 1e-6 or (abs(L_pos[0] - best_L_pos[0]) <= 1e-6 and L_pos[1] < best_L_pos[1] - 1e-6):
                best_L_rel = lr
                best_L_pos = L_pos

    L_rel = best_L_rel
    D_self = (L_rel[0] ** 2 + L_rel[1] ** 2) ** 0.5
    D_other = ((other_rel[0] - L_rel[0]) ** 2 + (other_rel[1] - L_rel[1]) ** 2) ** 0.5

    vx = 0.0
    vy = 0.0

    if D_self <= arrival_threshold:
        vx = 0.0
        vy = 0.0
    else:
        # Direction to landmark
        dir_to_L = (L_rel[0] / (D_self + 1e-6), L_rel[1] / (D_self + 1e-6))

        # Arc offset magnitude
        max_perp_offset = 1.0
        min_perp_offset = 0.0
        r = (D_self / (D_self + D_other + 1e-6)) * max_perp_offset
        r = max(min(r, max_perp_offset), min_perp_offset)

        # Determine side using cross product
        cross = dir_to_L[0] * other_rel[1] - dir_to_L[1] * other_rel[0]
        rot90 = (-dir_to_L[1], dir_to_L[0])  # +90 deg rotation

        if cross > 0:
            offset_dir = (-rot90[0], -rot90[1])  # other is left, offset to the right
        elif cross < 0:
            offset_dir = rot90  # other is right, offset to the left
        else:
            offset_dir = rot90  # fallback

        offset = (r * offset_dir[0], r * offset_dir[1])

        # Target point T along the arc
        L_pos = (ego_x + L_rel[0], ego_y + L_rel[1])
        T = (L_pos[0] + offset[0], L_pos[1] + offset[1])

        vec_T = (T[0] - ego_x, T[1] - ego_y)
        dist_T = (vec_T[0] ** 2 + vec_T[1] ** 2) ** 0.5

        if dist_T > 1e-6:
            v_dir = (vec_T[0] / dist_T, vec_T[1] / dist_T)
            max_speed = 1.0
            base_speed = 0.65 * max_speed

            denom = D_self + D_other + 1e-6
            speed_factor = 0.6 + 0.4 * (D_other / denom)
            speed_factor = max(0.0, min(1.0, speed_factor))

            velocity_mag = base_speed * speed_factor
            vx = velocity_mag * v_dir[0]
            vy = velocity_mag * v_dir[1]
        else:
            vx = 0.0
            vy = 0.0

    action = torch.tensor([[vx, vy]], dtype=obs.dtype, device=obs.device)
    return action