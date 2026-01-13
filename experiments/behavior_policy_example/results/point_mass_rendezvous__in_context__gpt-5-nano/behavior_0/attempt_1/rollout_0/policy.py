def agent0_policy(obs):
    import torch

    if not hasattr(agent0_policy, "GLOBAL_STEP"):
        agent0_policy.GLOBAL_STEP = 0
        agent0_policy.GRACE_END = {}

    # Parse observations
    ego_pos = obs[0:2]  # (2,)
    landmarks_rel = [obs[4 + i*2 : 4 + i*2 + 2] for i in range(4)]  # list of 4 tensors (2,)
    other_rel = obs[12:14]  # (2,)

    # Deterministic landmark selection: minimize (x+y), break ties by smaller x, then smaller y
    best_i = 0
    best_score = None
    best_x = None
    best_y = None

    for i in range(4):
        rel = landmarks_rel[i]
        lx = (ego_pos[0] + rel[0]).item()
        ly = (ego_pos[1] + rel[1]).item()
        score = lx + ly
        if best_score is None or score < best_score - 1e-6 or (
            abs(score - best_score) <= 1e-6 and (lx < best_x - 1e-6 or (
                abs(lx - best_x) <= 1e-6 and ly < best_y - 1e-6
            ))
        ):
            best_i = i
            best_score = score
            best_x = lx
            best_y = ly

    landmark_target = ego_pos + landmarks_rel[best_i]
    dir_vec = landmark_target - ego_pos
    dist_tensor = torch.norm(dir_vec)

    D_self = float(dist_tensor.item()) if dist_tensor.numel() > 0 else 0.0
    D_other = float(torch.norm(landmarks_rel[best_i] - other_rel).item()) if (landmarks_rel[best_i] - other_rel).numel() > 0 else 0.0

    current_step = agent0_policy.GLOBAL_STEP
    grace_end = agent0_policy.GRACE_END.get(best_i, -1)

    threshold = 0.05
    if D_self < threshold:
        if grace_end == -1 or current_step > grace_end:
            agent0_policy.GRACE_END[best_i] = current_step + 20

    grace_end_now = agent0_policy.GRACE_END.get(best_i, -1)
    hold = (grace_end_now != -1 and current_step <= grace_end_now)

    # Movement toward target with explicit timing
    if dist_tensor.numel() > 0 and dist_tensor.item() > 0 and not hold:
        unit_dir = dir_vec / dist_tensor
        base_speed = 0.8
        alpha = 0.6
        max_speed = 2.0
        # Use D_self - D_other to speed up if farther (self farther -> positive)
        speed = base_speed + alpha * (D_self - D_other)
        speed = max(0.0, min(speed, max_speed))
        action_vec = unit_dir * speed
    else:
        action_vec = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    action = action_vec.view(1, 2)

    agent0_policy.GLOBAL_STEP += 1
    return action

def agent1_policy(obs):
    import torch

    if not hasattr(agent1_policy, "GLOBAL_STEP"):
        agent1_policy.GLOBAL_STEP = 0
        agent1_policy.GRACE_END = {}

    # Parse observations
    ego_pos = obs[0:2]  # (2,)
    landmarks_rel = [obs[4 + i*2 : 4 + i*2 + 2] for i in range(4)]  # list of 4 tensors (2,)
    other_rel = obs[12:14]  # (2,)

    # Deterministic landmark selection: minimize (x+y), break ties by smaller x, then smaller y
    best_i = 0
    best_score = None
    best_x = None
    best_y = None

    for i in range(4):
        rel = landmarks_rel[i]
        lx = (ego_pos[0] + rel[0]).item()
        ly = (ego_pos[1] + rel[1]).item()
        score = lx + ly
        if best_score is None or score < best_score - 1e-6 or (
            abs(score - best_score) <= 1e-6 and (lx < best_x - 1e-6 or (
                abs(lx - best_x) <= 1e-6 and ly < best_y - 1e-6
            ))
        ):
            best_i = i
            best_score = score
            best_x = lx
            best_y = ly

    landmark_target = ego_pos + landmarks_rel[best_i]
    dir_vec = landmark_target - ego_pos
    dist_tensor = torch.norm(dir_vec)

    D_self = float(dist_tensor.item()) if dist_tensor.numel() > 0 else 0.0
    D_other = float(torch.norm(landmarks_rel[best_i] - other_rel).item()) if (landmarks_rel[best_i] - other_rel).numel() > 0 else 0.0

    current_step = agent1_policy.GLOBAL_STEP
    grace_end = agent1_policy.GRACE_END.get(best_i, -1)

    threshold = 0.05
    if D_self < threshold:
        if grace_end == -1 or current_step > grace_end:
            agent1_policy.GRACE_END[best_i] = current_step + 20

    grace_end_now = agent1_policy.GRACE_END.get(best_i, -1)
    hold = (grace_end_now != -1 and current_step <= grace_end_now)

    # Movement toward target with explicit timing
    if dist_tensor.numel() > 0 and dist_tensor.item() > 0 and not hold:
        unit_dir = dir_vec / dist_tensor
        base_speed = 0.8
        alpha = 0.6
        max_speed = 2.0
        speed = base_speed + alpha * (D_self - D_other)
        speed = max(0.0, min(speed, max_speed))
        action_vec = unit_dir * speed
    else:
        action_vec = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    action = action_vec.view(1, 2)

    agent1_policy.GLOBAL_STEP += 1
    return action