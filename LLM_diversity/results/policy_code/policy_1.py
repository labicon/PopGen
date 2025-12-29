import torch

def agent1_policy(obs):
    # Parse base components
    pos = obs[0:2]
    vel = obs[2:4]
    D = obs.shape[0]
    rem = max(0, D - 4)
    n_pairs = rem // 2

    # Assume exactly one other agent (typical setting); infer landmarks count
    n_landmarks = max(0, n_pairs - 1)

    # Extract landmarks relative positions
    lm_start = 4
    lm_end = lm_start + 2 * n_landmarks
    if n_landmarks > 0:
        landmarks_rel = obs[lm_start:lm_end].reshape(n_landmarks, 2)
    else:
        landmarks_rel = torch.zeros((0, 2), dtype=obs.dtype, device=obs.device)

    # Extract other agent relative position (fallback to zero if missing)
    if lm_end + 2 <= D:
        other_rel = obs[lm_end:lm_end + 2]
    else:
        other_rel = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    # If no landmarks available, just damp velocity
    if landmarks_rel.shape[0] == 0:
        u = -0.3 * vel
        return u.view(1, 2)

    # Compute objective: minimize ||rLi||^2 + ||rLi - rOther||^2
    rL = landmarks_rel
    rO = other_rel
    s = (rL.pow(2).sum(dim=1)) + ((rL - rO).pow(2).sum(dim=1))

    # Tie-breaker: smaller squared distance to max(pos, other_pos) under lexicographic order
    p1 = pos
    p2 = pos + other_rel
    eps = 1e-8
    p1x, p1y = p1[0].item(), p1[1].item()
    p2x, p2y = p2[0].item(), p2[1].item()
    if p1x > p2x + eps:
        use_p1 = True
    elif p2x > p1x + eps:
        use_p1 = False
    else:
        # x equal within tolerance; compare y
        use_p1 = p1y >= p2y - eps

    if use_p1:
        d_tie = rL.pow(2).sum(dim=1)
    else:
        d_tie = (rL - rO).pow(2).sum(dim=1)

    # Select index with tie-breaking: min s, then min d_tie, then smallest index
    min_s = s.min()
    tie_mask = (s - min_s).abs() <= 1e-8
    candidate_idx = torch.nonzero(tie_mask, as_tuple=False).flatten()
    if candidate_idx.numel() == 1:
        idx = candidate_idx.item()
    else:
        d_candidates = d_tie[candidate_idx]
        # argmin returns first occurrence -> smallest index due to ascending candidate_idx
        rel_choice = torch.argmin(d_candidates)
        idx = candidate_idx[rel_choice].item()

    # Control towards chosen landmark (PD control)
    target_rel = rL[idx]
    kp = 1.2
    kd = 0.3
    u = kp * target_rel - kd * vel

    # Clip to reasonable max force
    max_force = 1.0
    norm = torch.norm(u)
    if norm > max_force:
        u = u * (max_force / (norm + 1e-8))

    return u.view(1, 2)