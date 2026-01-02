import torch

def agent1_policy(obs):
    # Parse observation
    pos = obs[0:2]
    vel = obs[2:4]
    if obs.numel() <= 6:
        # Not enough information; stay still
        return vel.new_zeros(1, 2)
    # Assume single other agent (last two entries)
    other_rel = obs[-2:]
    lm_flat = obs[4:-2]
    L = lm_flat.numel() // 2
    if L <= 0:
        # No landmarks; mild damping
        force = -0.3 * vel
        return force.view(1, 2)

    landmarks_rel = lm_flat.view(L, 2)

    # Distances
    d_ego = torch.norm(landmarks_rel, dim=1)
    d_other = torch.norm(landmarks_rel - other_rel.unsqueeze(0), dim=1)

    # Selection with tie-breakers:
    # 1) smallest distance to ego
    # 2) smallest sum of distances to ego and other
    # 3) lowest index
    tol = 1e-6
    min_d_ego = d_ego.min()
    mask1 = (d_ego - min_d_ego).abs() <= tol

    if mask1.sum() > 1:
        sum_scores = d_ego + d_other
        min_sum = sum_scores[mask1].min()
        mask2 = mask1 & ((sum_scores - min_sum).abs() <= tol)
        idxs = torch.nonzero(mask2, as_tuple=False).squeeze(-1)
        selected_idx = idxs.min()
    else:
        selected_idx = torch.argmin(d_ego)

    target_rel = landmarks_rel[selected_idx]

    # Proportional-derivative control toward selected landmark
    k_p = obs.new_tensor(1.5)
    k_d = obs.new_tensor(0.3)
    force = k_p * target_rel - k_d * vel

    # Saturate force magnitude
    max_force = obs.new_tensor(1.0)
    eps = obs.new_tensor(1e-8)
    norm = torch.norm(force)
    scale = torch.clamp(max_force / (norm + eps), max=1.0)
    force = force * scale

    return force.view(1, 2)