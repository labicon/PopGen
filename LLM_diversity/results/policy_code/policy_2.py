import torch

def agent1_policy(obs):
    # obs: [ego_pos(2), ego_vel(2), landmarks_rel(2*L), other_rel(2)] assuming one other agent
    d = obs.shape[-1]
    pos = obs[0:2]
    vel = obs[2:4]

    # Assume exactly one other agent if possible
    if d >= 6:
        L = max(0, (d - 6) // 2)
        landmarks_flat_end = 4 + 2 * L
        landmarks_rel = obs[4:landmarks_flat_end].reshape(L, 2) if L > 0 else obs.new_zeros((0, 2))
        other_rel = obs[-2:]
    else:
        # Fallback: no other agent present
        L = max(0, (d - 4) // 2)
        landmarks_flat_end = 4 + 2 * L
        landmarks_rel = obs[4:landmarks_flat_end].reshape(L, 2) if L > 0 else obs.new_zeros((0, 2))
        other_rel = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    if L == 0:
        # No landmarks: just damp velocity
        u = -0.3 * vel
        return torch.clamp(u, -1.0, 1.0).view(1, 2)

    # Primary criterion: minimize squared distance to ego (i.e., norm of landmark_rel)
    dists_ego = (landmarks_rel ** 2).sum(dim=1)
    min_val, _ = torch.min(dists_ego, dim=0)
    tol = 1e-9
    tie_mask = torch.isclose(dists_ego, min_val, rtol=0.0, atol=tol)
    tie_indices = torch.nonzero(tie_mask, as_tuple=False).flatten()

    if tie_indices.numel() > 1:
        # Determine lexicographic min between ego and other
        # Compare x first, then y using other_rel = other - ego
        if other_rel[0] > 0:
            ego_is_lex_min = True
        elif other_rel[0] < 0:
            ego_is_lex_min = False
        else:
            if other_rel[1] > 0:
                ego_is_lex_min = True
            elif other_rel[1] < 0:
                ego_is_lex_min = False
            else:
                ego_is_lex_min = True  # identical positions, treat ego as min

        if ego_is_lex_min:
            tie_dists = dists_ego[tie_indices]
        else:
            # Distance from lexicographic min (the other agent) to landmark:
            # landmark - other = (landmark - ego) - (other - ego) = landmarks_rel - other_rel
            diffs = landmarks_rel[tie_indices] - other_rel.unsqueeze(0)
            tie_dists = (diffs ** 2).sum(dim=1)

        tie_min_val, _ = torch.min(tie_dists, dim=0)
        tie_winners_mask = torch.isclose(tie_dists, tie_min_val, rtol=0.0, atol=tol)
        # Choose smallest index among winners
        winners = tie_indices[tie_winners_mask]
        chosen_idx = int(winners.min().item())
    else:
        chosen_idx = int(tie_indices.item())

    target_rel = landmarks_rel[chosen_idx]

    # PD controller towards selected landmark relative position
    kp = 1.2
    kd = 0.3
    u = kp * target_rel - kd * vel

    # Clip action
    u = torch.clamp(u, -1.0, 1.0)
    return u.view(1, 2)