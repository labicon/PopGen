import torch

def agent0_policy(obs):
    # persistent parameters
    if not hasattr(agent0_policy, 'preferred_index'):
        agent0_policy.preferred_index = 0               # anchored preferred landmark
        agent0_policy.timeout_remaining = 0
        agent0_policy.timeout_max = 10                  # steps to maintain commitment when contested
        agent0_policy.conflict_threshold = 0.8          # distance threshold to consider conflict
        agent0_policy.arrival_threshold = 0.1           # consider arrived if closer than this
        agent0_policy.force_mag = 1.0                   # decisive force magnitude (limited to 1.0)
        agent0_policy.max_force = 1.0                   # action limit enforced by environment

    # observation layout:
    # 0:2   -> agent pos
    # 2:4   -> agent vel
    # 4:12  -> 4 landmarks relative pos (each 2D)
    # 12:14 -> other agent relative pos
    base = 4
    i = agent0_policy.preferred_index
    lm_start = base + i * 2
    lm_rel = obs[lm_start:lm_start + 2]                # landmark relative position (2,)
    other_rel = obs[base + 4 * 2: base + 4 * 2 + 2]    # other agent relative pos (2,)

    # distances
    ego_to_lm_dist = torch.norm(lm_rel)
    other_to_lm_dist = torch.norm(lm_rel - other_rel)

    # detect conflict: other agent is closer to the same landmark
    if (other_to_lm_dist < ego_to_lm_dist) and (other_to_lm_dist < agent0_policy.conflict_threshold):
        agent0_policy.timeout_remaining = agent0_policy.timeout_max

    if agent0_policy.timeout_remaining > 0:
        agent0_policy.timeout_remaining -= 1

    # if already at the landmark, apply no force
    if ego_to_lm_dist < agent0_policy.arrival_threshold:
        force = obs.new_zeros(2)
    else:
        dist = ego_to_lm_dist
        if dist.item() < 1e-6:
            force = obs.new_zeros(2)
        else:
            direction = lm_rel / (dist + 1e-8)
            force = direction * agent0_policy.force_mag
            # enforce environment action limits
            norm = torch.norm(force)
            if norm.item() > agent0_policy.max_force:
                force = force * (agent0_policy.max_force / (norm + 1e-8))

    return force.unsqueeze(0)

def agent1_policy(obs):
    # persistent parameters
    if not hasattr(agent1_policy, 'preferred_index'):
        agent1_policy.preferred_index = 0               # anchored preferred landmark
        agent1_policy.timeout_remaining = 0
        agent1_policy.timeout_max = 10                  # steps to maintain commitment when contested
        agent1_policy.conflict_threshold = 0.8          # distance threshold to consider conflict
        agent1_policy.arrival_threshold = 0.1           # consider arrived if closer than this
        agent1_policy.force_mag = 1.0                   # decisive force magnitude (limited to 1.0)
        agent1_policy.max_force = 1.0                   # action limit enforced by environment

    # observation layout:
    # 0:2   -> agent pos
    # 2:4   -> agent vel
    # 4:12  -> 4 landmarks relative pos (each 2D)
    # 12:14 -> other agent relative pos
    base = 4
    i = agent1_policy.preferred_index
    lm_start = base + i * 2
    lm_rel = obs[lm_start:lm_start + 2]                # landmark relative position (2,)
    other_rel = obs[base + 4 * 2: base + 4 * 2 + 2]    # other agent relative pos (2,)

    # distances
    ego_to_lm_dist = torch.norm(lm_rel)
    other_to_lm_dist = torch.norm(lm_rel - other_rel)

    # detect conflict: other agent is closer to the same landmark
    if (other_to_lm_dist < ego_to_lm_dist) and (other_to_lm_dist < agent1_policy.conflict_threshold):
        agent1_policy.timeout_remaining = agent1_policy.timeout_max

    if agent1_policy.timeout_remaining > 0:
        agent1_policy.timeout_remaining -= 1

    # if already at the landmark, apply no force
    if ego_to_lm_dist < agent1_policy.arrival_threshold:
        force = obs.new_zeros(2)
    else:
        dist = ego_to_lm_dist
        if dist.item() < 1e-6:
            force = obs.new_zeros(2)
        else:
            direction = lm_rel / (dist + 1e-8)
            force = direction * agent1_policy.force_mag
            # enforce environment action limits
            norm = torch.norm(force)
            if norm.item() > agent1_policy.max_force:
                force = force * (agent1_policy.max_force / (norm + 1e-8))

    return force.unsqueeze(0)