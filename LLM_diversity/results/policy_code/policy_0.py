import torch

def agent1_policy(obs):
    # obs: [ego_pos(2), ego_vel(2), landmarks_rel(2*L), other_rel(2)]
    ego_pos = obs[0:2]
    ego_vel = obs[2:4]
    obs_dim = obs.shape[0]
    # Assume exactly one other agent
    L = int((obs_dim - 6) // 2)
    if L <= 0:
        force = -0.5 * ego_vel
        return force.view(1, 2)
    lm_start = 4
    lm_end = lm_start + 2 * L
    landmarks_rel = obs[lm_start:lm_end].view(L, 2)
    other_rel = obs[lm_end:lm_end + 2]

    # Costs: sum of squared distances to ego and other
    d2_ego = (landmarks_rel.pow(2)).sum(dim=1)
    d2_other = (landmarks_rel - other_rel).pow(2).sum(dim=1)
    costs = d2_ego + d2_other

    # Tie-break: by smaller squared distance to lexicographically smaller agent position
    other_pos = ego_pos + other_rel
    eps = 1e-8
    # Determine lexicographic min between ego_pos and other_pos
    def lex_is_ego_min(a, b):
        if a[0] < b[0] - eps:
            return True
        if abs(float(a[0] - b[0])) <= eps:
            if a[1] < b[1] - eps:
                return True
            if abs(float(a[1] - b[1])) <= eps:
                # Equal positions; arbitrarily choose ego as min
                return True
        return False

    min_is_ego = lex_is_ego_min(ego_pos, other_pos)
    tie_metric = d2_ego if min_is_ego else d2_other

    # Select landmark index with lexicographic tie-breaking
    min_cost = costs.min()
    scale1 = 1.0 + float(costs.abs().max()) if costs.numel() > 0 else 1.0
    tol1 = 1e-6 * scale1
    mask1 = (costs - min_cost).abs() <= tol1
    candidate_idxs = torch.nonzero(mask1, as_tuple=False).squeeze(-1)
    if candidate_idxs.numel() == 1:
        i_star = int(candidate_idxs.item())
    else:
        cand_ties = tie_metric[candidate_idxs]
        min_tie = cand_ties.min()
        scale2 = 1.0 + float(cand_ties.abs().max()) if cand_ties.numel() > 0 else 1.0
        tol2 = 1e-6 * scale2
        mask2 = (cand_ties - min_tie).abs() <= tol2
        final_candidates = candidate_idxs[mask2]
        i_star = int(final_candidates.min().item())

    target_rel = landmarks_rel[i_star]
    # PD control towards target landmark
    kp = 1.5
    kd = 0.3
    force = kp * target_rel - kd * ego_vel
    force = torch.clamp(force, -1.0, 1.0)
    return force.view(1, 2)