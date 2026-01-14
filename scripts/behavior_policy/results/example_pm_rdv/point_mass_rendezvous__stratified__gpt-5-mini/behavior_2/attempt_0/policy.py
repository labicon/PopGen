import torch

def agent0_policy(obs):
    # obs: tensor shape (14,)
    device = obs.device if hasattr(obs, 'device') else torch.device('cpu')
    eps = 1e-6

    # parse observation
    agent_pos = obs[0:2]
    agent_vel = obs[2:4]
    landmarks_rel = obs[4:12].view(4, 2)  # 4 landmarks, relative positions
    other_rel = obs[12:14]
    other_pos = agent_pos + other_rel
    landmarks_abs = landmarks_rel + agent_pos.unsqueeze(0)  # reconstruct absolute landmark positions

    # persistent state initialization
    if not hasattr(agent0_policy, "preferred_index"):
        agent0_policy.preferred_index = 1  # anchored preference
    if not hasattr(agent0_policy, "prev_other_pos"):
        agent0_policy.prev_other_pos = other_pos.clone()
    if not hasattr(agent0_policy, "prev_other_dists"):
        # initialize with current distances
        agent0_policy.prev_other_dists = torch.norm(landmarks_abs - other_pos.unsqueeze(0), dim=1)

    # parameters
    slow_speed = 0.5
    kp = 2.0
    max_force = 1.0
    v_thresh = 0.05
    commit_dot_thresh = 0.75  # cosine similarity threshold to consider heading aligned
    dist_decrease_thresh = 0.02  # absolute distance decrease to indicate commitment

    # estimate other agent velocity (approximate, dt assumed 1)
    other_vel = other_pos - agent0_policy.prev_other_pos
    other_speed = torch.norm(other_vel) + eps

    # compute distances from other to each landmark
    d_other = torch.norm(landmarks_abs - other_pos.unsqueeze(0), dim=1)

    # compute heading alignment: dot( vector_to_landmark, other_vel )
    vec_to_land = landmarks_abs - other_pos.unsqueeze(0)
    vec_norms = torch.norm(vec_to_land, dim=1) + eps
    dot_vals = (vec_to_land * other_vel.unsqueeze(0)).sum(dim=1)
    cos_sim = dot_vals / (vec_norms * other_speed + eps)

    # distance decrease evidence
    dist_decrease = agent0_policy.prev_other_dists - d_other

    # commitment criteria per landmark
    heading_commit = (cos_sim > commit_dot_thresh) & (other_speed > v_thresh)
    dist_commit = dist_decrease > dist_decrease_thresh
    commit_flags = heading_commit | dist_commit

    # If other commits strongly to some other landmark, switch to that one
    preferred = int(agent0_policy.preferred_index)
    # ignore commitment to our preferred (we only yield if they commit to a different one)
    commit_flags_preferred_ignored = commit_flags.clone()
    commit_flags_preferred_ignored[preferred] = False

    if commit_flags_preferred_ignored.any():
        # choose the landmark with strongest evidence: prioritize heading (cos_sim) then distance decrease
        # compute a score: large cos_sim and large decrease get higher score
        score = cos_sim.clone().clamp(min=0.0) * 1.0 + dist_decrease.clamp(min=0.0) * 0.5
        score[preferred] = -1e6  # ensure we don't pick current preferred
        new_idx = int(torch.argmax(score).item())
        agent0_policy.preferred_index = new_idx
        preferred = new_idx

    # target is the preferred landmark absolute position
    target = landmarks_abs[preferred]

    # compute desired velocity: cautious slow approach
    to_target = target - agent_pos
    dist_to_target = torch.norm(to_target)
    if dist_to_target > 1e-3:
        desired_dir = to_target / (dist_to_target + eps)
        desired_vel = desired_dir * slow_speed
    else:
        desired_vel = torch.zeros_like(agent_vel)

    # PD-like force: proportional to velocity error (no explicit mass/inertia known)
    force = kp * (desired_vel - agent_vel)
    # clip force magnitude
    force_norm = torch.norm(force)
    if force_norm > max_force:
        force = force * (max_force / (force_norm + eps))

    # update persistent info for next step
    agent0_policy.prev_other_pos = other_pos.clone()
    agent0_policy.prev_other_dists = d_other.clone()

    return force.unsqueeze(0)

import torch

def agent1_policy(obs):
    device = obs.device if hasattr(obs, "device") else torch.device("cpu")
    eps = 1e-6

    # parse observation
    agent_pos = obs[0:2]
    agent_vel = obs[2:4]
    landmarks_rel = obs[4:12].view(4, 2)  # 4 landmarks, relative positions
    other_rel = obs[12:14]
    other_pos = agent_pos + other_rel
    landmarks_abs = landmarks_rel + agent_pos.unsqueeze(0)  # absolute landmark positions

    # persistent state initialization
    if not hasattr(agent1_policy, "preferred_index"):
        agent1_policy.preferred_index = 1  # anchored preference (match partner's anchor)
    if not hasattr(agent1_policy, "prev_other_pos"):
        agent1_policy.prev_other_pos = other_pos.clone()
    if not hasattr(agent1_policy, "prev_other_dists"):
        agent1_policy.prev_other_dists = torch.norm(landmarks_abs - other_pos.unsqueeze(0), dim=1)

    # parameters
    slow_speed = 0.5
    kp = 2.0
    max_force = 1.0
    v_thresh = 0.05
    commit_dot_thresh = 0.75
    dist_decrease_thresh = 0.02

    # estimate other agent velocity (dt ~ 1)
    other_vel = other_pos - agent1_policy.prev_other_pos
    other_speed = torch.norm(other_vel) + eps

    # distances from other to each landmark
    d_other = torch.norm(landmarks_abs - other_pos.unsqueeze(0), dim=1)

    # heading alignment: dot(vector_to_landmark, other_vel)
    vec_to_land = landmarks_abs - other_pos.unsqueeze(0)
    vec_norms = torch.norm(vec_to_land, dim=1) + eps
    dot_vals = (vec_to_land * other_vel.unsqueeze(0)).sum(dim=1)
    cos_sim = dot_vals / (vec_norms * other_speed + eps)

    # distance decrease evidence
    dist_decrease = agent1_policy.prev_other_dists - d_other

    # commitment criteria
    heading_commit = (cos_sim > commit_dot_thresh) & (other_speed > v_thresh)
    dist_commit = dist_decrease > dist_decrease_thresh
    commit_flags = heading_commit | dist_commit

    preferred = int(agent1_policy.preferred_index)
    commit_flags_preferred_ignored = commit_flags.clone()
    commit_flags_preferred_ignored[preferred] = False

    if commit_flags_preferred_ignored.any():
        score = cos_sim.clone().clamp(min=0.0) * 1.0 + dist_decrease.clamp(min=0.0) * 0.5
        score[preferred] = -1e6
        new_idx = int(torch.argmax(score).item())
        agent1_policy.preferred_index = new_idx
        preferred = new_idx

    target = landmarks_abs[preferred]

    # cautious slow approach toward target
    to_target = target - agent_pos
    dist_to_target = torch.norm(to_target)
    if dist_to_target > 1e-3:
        desired_dir = to_target / (dist_to_target + eps)
        desired_vel = desired_dir * slow_speed
    else:
        desired_vel = torch.zeros_like(agent_vel)

    force = kp * (desired_vel - agent_vel)
    force_norm = torch.norm(force)
    if force_norm > max_force:
        force = force * (max_force / (force_norm + eps))

    # update persistent info
    agent1_policy.prev_other_pos = other_pos.clone()
    agent1_policy.prev_other_dists = d_other.clone()

    return force.unsqueeze(0)