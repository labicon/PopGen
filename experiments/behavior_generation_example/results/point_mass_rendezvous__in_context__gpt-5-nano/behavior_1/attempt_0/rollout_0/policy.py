def agent0_policy(obs):
    import torch

    # Parse observation
    ego_pos = obs[0:2]
    ego_vel = obs[2:4]  # not used, but kept for completeness
    landmark_rel = obs[4:12].reshape(4, 2)  # relative positions of landmarks to ego
    other_rel = obs[12:14]

    # Absolute landmark positions
    landmark_abs = landmark_rel + ego_pos  # shape (4, 2)

    # Other agent absolute position
    other_pos = ego_pos + other_rel  # shape (2,)

    # Distances from ego to landmarks
    dist_to_ego = torch.sqrt((landmark_rel * landmark_rel).sum(dim=1))  # (4,)

    # Choose landmark L* with minimum distance to centroid proxy
    # Break ties by smallest x, then smallest y (in absolute coordinates)
    min_dist = float('inf')
    min_index = 0
    min_coord = None
    for i in range(4):
        d = float(dist_to_ego[i].item())
        coord = landmark_abs[i]
        if d < min_dist - 1e-6:
            min_dist = d
            min_index = i
            min_coord = coord.clone()
        elif abs(d - min_dist) <= 1e-6:
            if min_coord is None:
                min_coord = coord.clone()
                min_index = i
            else:
                if (coord[0].item() < min_coord[0].item()) or (abs(coord[0].item() - min_coord[0].item()) <= 1e-6 and coord[1].item() < min_coord[1].item()):
                    min_index = i
                    min_coord = coord.clone()

    Lstar = landmark_abs[min_index]  # absolute position of selected landmark
    D_self = float(dist_to_ego[min_index].item())  # distance ego -> L*

    D_other = float(torch.norm(Lstar - other_pos).item())  # distance other -> L*

    # Direction toward landmark
    dir_vec = Lstar - ego_pos
    if D_self > 1e-6:
        unit_dir = dir_vec / D_self
    else:
        unit_dir = torch.tensor([0.0, 0.0], dtype=obs.dtype)

    # Tangential component (rotate 90 degrees)
    tang = torch.tensor([-unit_dir[1], unit_dir[0]], dtype=obs.dtype)

    # Alpha blending factor
    max_dist = float(dist_to_ego.max().item()) if dist_to_ego.numel() > 0 else 1.0
    max_dist = max(max_dist, 1e-6)
    alpha = D_self / max_dist
    alpha = max(0.4, min(0.95, alpha))

    # Blended direction and normalization
    blended_dir = alpha * unit_dir + (1.0 - alpha) * tang
    blended_norm = torch.norm(blended_dir)
    if blended_norm.item() > 1e-6:
        dir_unit = blended_dir / blended_norm
    else:
        dir_unit = torch.tensor([1.0, 0.0], dtype=obs.dtype)

    # Speed shaping
    max_speed = 1.0
    base_speed = max_speed * 0.6

    diff = D_other - D_self
    den = max(D_self, D_other) + 1e-6
    tanh_arg = diff / den
    tanh_val = torch.tanh(torch.tensor(tanh_arg, dtype=obs.dtype)).item()
    speed_factor = 1.0 + 0.4 * tanh_val
    velocity_mag = base_speed * speed_factor
    velocity_mag = max(0.0, min(max_speed, velocity_mag))

    # Velocity vector
    vel = velocity_mag * dir_unit

    # Arrival handling
    arrival_threshold = 0.05
    if D_self <= arrival_threshold:
        vel = torch.tensor([0.0, 0.0], dtype=obs.dtype)

    action = vel.view(1, 2)
    return action

def agent1_policy(obs):
    import torch

    # Parse observation
    ego_pos = obs[0:2]
    ego_vel = obs[2:4]  # not used, but kept for completeness
    landmark_rel = obs[4:12].reshape(4, 2)  # relative positions of landmarks to ego
    other_rel = obs[12:14]

    # Absolute landmark positions
    landmark_abs = landmark_rel + ego_pos  # shape (4, 2)

    # Other agent absolute position
    other_pos = ego_pos + other_rel  # shape (2,)

    # Centroid of the two agents
    C = (ego_pos + other_pos) * 0.5

    # Distances from landmarks to the centroid
    diff_to_centroid = landmark_abs - C  # (4,2)
    dist_to_centroid = torch.sqrt((diff_to_centroid * diff_to_centroid).sum(dim=1))  # (4,)

    # Choose landmark L* with minimum distance to centroid
    min_dist = float('inf')
    min_index = 0
    min_coord = None
    for i in range(4):
        d = float(dist_to_centroid[i].item())
        coord = landmark_abs[i]
        if d < min_dist - 1e-6:
            min_dist = d
            min_index = i
            min_coord = coord.clone()
        elif abs(d - min_dist) <= 1e-6:
            if min_coord is None:
                min_coord = coord.clone()
                min_index = i
            else:
                if (coord[0].item() < min_coord[0].item()) or (abs(coord[0].item() - min_coord[0].item()) <= 1e-6 and coord[1].item() < min_coord[1].item()):
                    min_index = i
                    min_coord = coord.clone()

    Lstar = landmark_abs[min_index]  # absolute position of selected landmark
    D_self = float(torch.norm(Lstar - ego_pos).item())  # distance ego -> L*

    D_other = float(torch.norm(Lstar - other_pos).item())  # distance other -> L*

    # Direction toward landmark
    dir_vec = Lstar - ego_pos
    if D_self > 1e-6:
        unit_dir = dir_vec / D_self
    else:
        unit_dir = torch.tensor([0.0, 0.0], dtype=obs.dtype)

    # Tangential component (rotate 90 degrees)
    tang = torch.tensor([-unit_dir[1], unit_dir[0]], dtype=obs.dtype)

    # Alpha blending factor
    max_dist = float(dist_to_centroid.max().item()) if dist_to_centroid.numel() > 0 else 1.0
    max_dist = max(max_dist, 1e-6)
    alpha = D_self / max_dist
    alpha = max(0.4, min(0.95, alpha))

    # Blended direction and normalization
    blended_dir = alpha * unit_dir + (1.0 - alpha) * tang
    blended_norm = torch.norm(blended_dir)
    if blended_norm.item() > 1e-6:
        dir_unit = blended_dir / blended_norm
    else:
        dir_unit = torch.tensor([1.0, 0.0], dtype=obs.dtype)

    # Speed shaping
    max_speed = 1.0
    base_speed = max_speed * 0.6

    diff = D_other - D_self
    den = max(D_self, D_other) + 1e-6
    tanh_arg = diff / den
    tanh_val = torch.tanh(torch.tensor(tanh_arg, dtype=obs.dtype)).item()
    speed_factor = 1.0 + 0.4 * tanh_val
    velocity_mag = base_speed * speed_factor
    velocity_mag = max(0.0, min(max_speed, velocity_mag))

    # Velocity vector
    vel = velocity_mag * dir_unit

    # Arrival handling
    arrival_threshold = 0.05
    if D_self <= arrival_threshold:
        vel = torch.tensor([0.0, 0.0], dtype=obs.dtype)

    action = vel.view(1, 2)
    return action