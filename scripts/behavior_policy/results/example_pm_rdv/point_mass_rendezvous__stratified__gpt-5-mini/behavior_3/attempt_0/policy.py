import torch

def agent0_policy(obs):
    # Ensure obs is a tensor
    if not torch.is_tensor(obs):
        obs = torch.tensor(obs, dtype=torch.float32)
    device = obs.device
    dtype = obs.dtype

    # Initialize persistent memory on the function object
    if not hasattr(agent0_policy, "memory"):
        agent0_policy.memory = {
            "other_history": [],        # list of other agent absolute positions (tensors)
            "steps": 0,
            "committed_index": None,
            "window": 6,                # observation window length (frames)
            "commit_consistency": 3,    # required frames showing commitment
            "heading_threshold": 0.08,  # threshold for heading dot product
            "dist_eps": 1e-3,           # minimal decrease for distance to count
            "max_force": 1.0,           # action magnitude cap
        }

    mem = agent0_policy.memory
    mem["steps"] += 1

    # Parse observation
    # obs layout: [ego_pos(2), ego_vel(2), landmarks_rel(2*num_landmarks), other_rel(2)]
    ego_pos = obs[0:2]
    ego_vel = obs[2:4]

    # Determine number of landmarks (assume single other agent)
    remaining = obs.numel() - 4 - 2
    num_landmarks = max(0, int(remaining // 2))

    # landmarks relative start at index 4
    lm_start = 4
    landmarks_rel = []
    for i in range(num_landmarks):
        idx = lm_start + 2 * i
        landmarks_rel.append(obs[idx: idx + 2])
    other_rel = obs[lm_start + 2 * num_landmarks : lm_start + 2 * num_landmarks + 2]

    # Compute absolute positions for landmarks and other agent
    landmarks_abs = [ego_pos + lr for lr in landmarks_rel]
    other_abs = ego_pos + other_rel

    # Update history of other agent absolute positions
    mem["other_history"].append(other_abs.clone())
    # Keep at most window+1 entries (we need pairs to compute velocities)
    max_hist = mem["window"] + 1
    if len(mem["other_history"]) > max_hist:
        mem["other_history"].pop(0)

    committed_idx = None

    # Only attempt to detect commitment if we have at least 2 history points
    hist = mem["other_history"]
    if len(hist) >= 2:
        # We'll check across up to last (window) frames (i.e., up to window comparisons)
        checks = min(len(hist) - 1, mem["window"])
        # For each landmark, count number of recent frames where heading>threshold and distance decreased
        counts = [0] * num_landmarks
        for t in range(len(hist) - checks, len(hist)):
            prev = hist[t - 1]
            curr = hist[t]
            vel = curr - prev  # approximate velocity
            # For each landmark, compute heading and distance change (use absolute landmark position)
            for i, lm in enumerate(landmarks_abs):
                vec_to_lm = lm - curr
                heading = float(torch.dot(vec_to_lm, vel))
                dist_curr = float(torch.norm(curr - lm))
                dist_prev = float(torch.norm(prev - lm))
                if heading > mem["heading_threshold"] and (dist_curr < dist_prev - mem["dist_eps"]):
                    counts[i] += 1
        # If any landmark has count >= commit_consistency, declare commitment
        for i, c in enumerate(counts):
            if c >= mem["commit_consistency"]:
                committed_idx = i
                break

    # Update stored committed index (allow switching if partner commits elsewhere)
    mem["committed_index"] = committed_idx

    target = None
    # If partner committed, match partner
    if mem["committed_index"] is not None:
        target = landmarks_abs[mem["committed_index"]]
    else:
        # If still within observation window, be patient (no strong movement)
        if mem["steps"] <= mem["window"]:
            # stay still / small passive action
            action = torch.zeros((1, 2), dtype=dtype, device=device)
            return action
        # Otherwise choose compromise: minimize sum of distances (ego + other)
        sumd = []
        for lm in landmarks_abs:
            d1 = float(torch.norm(ego_pos - lm))
            d2 = float(torch.norm(other_abs - lm))
            sumd.append(d1 + d2)
        if len(sumd) == 0:
            action = torch.zeros((1, 2), dtype=dtype, device=device)
            return action
        preferred = int(min(range(len(sumd)), key=lambda i: sumd[i]))
        target = landmarks_abs[preferred]

    # Compute action: move directly toward target, capped magnitude
    to_target = target - ego_pos
    dist = torch.norm(to_target)
    if dist.item() < 1e-4:
        action_vec = torch.zeros(2, dtype=dtype, device=device)
    else:
        action_vec = to_target / (dist + 1e-8) * mem["max_force"]

    return action_vec.unsqueeze(0)

import torch

def agent1_policy(obs):
    if not torch.is_tensor(obs):
        obs = torch.tensor(obs, dtype=torch.float32)
    device = obs.device
    dtype = obs.dtype

    # Initialize persistent memory
    if not hasattr(agent1_policy, "memory"):
        agent1_policy.memory = {
            "other_history": [],        # list of other agent absolute positions (tensors)
            "steps": 0,
            "committed_index": None,
            "window": 6,                # observation window length (frames)
            "commit_consistency": 3,    # required frames showing commitment
            "heading_threshold": 0.08,  # threshold for heading dot product
            "dist_eps": 1e-3,           # minimal decrease for distance to count
            "max_force": 1.0,           # action magnitude cap
        }

    mem = agent1_policy.memory
    mem["steps"] += 1

    # Parse observation
    # obs layout: [ego_pos(2), ego_vel(2), landmarks_rel(2*num_landmarks), other_rel(2)]
    ego_pos = obs[0:2]
    ego_vel = obs[2:4]

    remaining = obs.numel() - 4 - 2
    num_landmarks = max(0, int(remaining // 2))

    lm_start = 4
    landmarks_rel = []
    for i in range(num_landmarks):
        idx = lm_start + 2 * i
        landmarks_rel.append(obs[idx: idx + 2])
    other_rel = obs[lm_start + 2 * num_landmarks : lm_start + 2 * num_landmarks + 2]

    # Compute absolute positions
    landmarks_abs = [ego_pos + lr for lr in landmarks_rel]
    other_abs = ego_pos + other_rel

    # Update history of other agent positions
    mem["other_history"].append(other_abs.clone())
    max_hist = mem["window"] + 1
    if len(mem["other_history"]) > max_hist:
        mem["other_history"].pop(0)

    committed_idx = None

    # Detect commitment from other's recent motion
    hist = mem["other_history"]
    if len(hist) >= 2 and num_landmarks > 0:
        checks = min(len(hist) - 1, mem["window"])
        counts = [0] * num_landmarks
        for t in range(len(hist) - checks, len(hist)):
            prev = hist[t - 1]
            curr = hist[t]
            vel = curr - prev
            for i, lm in enumerate(landmarks_abs):
                vec_to_lm = lm - curr
                heading = float(torch.dot(vec_to_lm, vel))
                dist_curr = float(torch.norm(curr - lm))
                dist_prev = float(torch.norm(prev - lm))
                if heading > mem["heading_threshold"] and (dist_curr < dist_prev - mem["dist_eps"]):
                    counts[i] += 1
        for i, c in enumerate(counts):
            if c >= mem["commit_consistency"]:
                committed_idx = i
                break

    mem["committed_index"] = committed_idx

    target = None
    if mem["committed_index"] is not None:
        target = landmarks_abs[mem["committed_index"]]
    else:
        if mem["steps"] <= mem["window"]:
            return torch.zeros((1, 2), dtype=dtype, device=device)
        # choose compromise (minimize sum distances)
        sumd = []
        for lm in landmarks_abs:
            d1 = float(torch.norm(ego_pos - lm))
            d2 = float(torch.norm(other_abs - lm))
            sumd.append(d1 + d2)
        if len(sumd) == 0:
            return torch.zeros((1, 2), dtype=dtype, device=device)
        preferred = int(min(range(len(sumd)), key=lambda i: sumd[i]))
        target = landmarks_abs[preferred]

    # Move toward target, capped
    to_target = target - ego_pos
    dist = torch.norm(to_target)
    if dist.item() < 1e-4:
        action_vec = torch.zeros(2, dtype=dtype, device=device)
    else:
        action_vec = to_target / (dist + 1e-8) * mem["max_force"]

    return action_vec.unsqueeze(0)