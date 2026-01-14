def agent0_policy(obs):
    # Ensure tensor
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs)
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    batch = obs.size(0)

    # Fallback in case of unexpected observation size
    if obs.size(-1) < 41:
        action = torch.zeros((batch, 2), dtype=obs.dtype, device=obs.device)
        action[:, 0] = 0.4  # gentle forward push
        action[:, 1] = 0.0
        return action

    # Segments based on observation construction
    # s_obs corresponds to first 13 features
    s_obs = obs[:, :13]
    ball_to_goal = s_obs[:, 6:8]  # ball_pos - goal_pos
    a_to_ball = s_obs[:, 2:4]     # agent_pos - ball_pos
    # Teammate information (first 8 features of teammates block)
    team_block = obs[:, 33:41]

    # Compute arc-side direction around the ball
    v = ball_to_goal  # vector from goal to ball
    perp = torch.cat([-v[:, 1:2], v[:, 0:1]], dim=1)  # rotate 90 degrees
    norm_perp = torch.norm(perp, dim=1, keepdim=True) + 1e-6
    side = perp / norm_perp  # unit vector tangent to arc around the ball

    arc_radius = 1.0
    v_norm = torch.norm(v, dim=1, keepdim=True)
    near_goal = (v_norm < 1.6).float()
    arc_mult = 1.0 + 0.8 * near_goal
    offset = side * arc_radius * arc_mult  # offset from ball to create arc

    target_delta = -a_to_ball + offset  # move to an offset around the ball

    # Teammate influence: adjust to emphasize teammate's pocket (space creation)
    te_vec = team_block[:, 0:2]  # agent_pos - teammate_pos
    dist_to_teammate = torch.norm(te_vec, dim=1, keepdim=True) + 1e-6
    to_teammate = -te_vec
    to_teammate_dir = to_teammate / dist_to_teammate
    corr = torch.clamp(2.0 - dist_to_teammate, min=0.0, max=0.8)
    target_delta = target_delta + to_teammate_dir * corr

    # Lane/pocket influence: widen/narrow arc toward teammate pocket
    dot_side = (te_vec * side).sum(dim=1, keepdim=True)
    pocket_influence = torch.clamp(torch.abs(dot_side) / 2.0, 0.0, 1.0)  # 0..0.5
    lane_dir = torch.sign(dot_side)
    offset_lane = side * lane_dir * pocket_influence * 0.4
    target_delta = target_delta + offset_lane

    # Defensive projection: adapt arc to adversaries near ball-goal line
    adv1_rel = obs[:, 17:19]
    adv2_rel = obs[:, 25:27]
    adv1_rel = -adv1_rel
    adv2_rel = -adv2_rel
    dot1 = (adv1_rel * side).sum(dim=1, keepdim=True)
    dot2 = (adv2_rel * side).sum(dim=1, keepdim=True)
    avg_dot = (dot1 + dot2) * 0.5
    offset_dir = torch.where(avg_dot > 0, -side, side)
    weight = torch.abs(avg_dot) * 0.5
    weight = torch.clamp(weight, 0.0, 1.0)
    arc_adjust = offset_dir * weight * 0.5
    offset = offset + arc_adjust
    target_delta = -a_to_ball + offset  # recalc with arc adjust

    action = target_delta
    mag = torch.norm(action, dim=1, keepdim=True) + 1e-6
    max_mag = torch.tensor(3.0, dtype=obs.dtype, device=obs.device)
    scale = torch.clamp(mag, min=1e-6)
    action = action / scale * torch.clamp(mag, max=max_mag)

    # Ensure shape is (batch, 2)
    action = action[:, :2]
    return action

import torch

def agent1_policy(obs):
    # Ensure tensor
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs)
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    batch = obs.size(0)

    # Fallback in case of unexpected observation size
    if obs.size(-1) < 41:
        action = torch.zeros((batch, 2), dtype=obs.dtype, device=obs.device)
        action[:, 0] = 0.4  # gentle forward push
        action[:, 1] = 0.0
        return action

    # Segments based on observation construction
    # s_obs corresponds to first 13 features
    s_obs = obs[:, :13]
    ball_to_goal = s_obs[:, 6:8]  # ball_pos - goal_pos
    a_to_ball = s_obs[:, 2:4]     # agent_pos - ball_pos
    # Teammate information (first 8 features of teammates block)
    team_block = obs[:, 33:41]

    # Compute arc-side direction around the ball
    v = ball_to_goal  # vector from goal to ball
    perp = torch.cat([-v[:, 1:2], v[:, 0:1]], dim=1)  # rotate 90 degrees
    norm_perp = torch.norm(perp, dim=1, keepdim=True) + 1e-6
    side = perp / norm_perp  # unit vector tangent to arc around the ball

    # Choose arc side opposite to teammate's relative position to reduce crowding
    te_vec = team_block[:, 0:2]  # agent_pos - teammate_pos
    dot_side = (te_vec * side).sum(dim=1, keepdim=True)
    flip = (dot_side > 0).float()
    side_to_use = side * (1.0 - 2.0 * flip)  # flip side if teammate is on the side

    arc_radius = 1.0
    v_norm = torch.norm(v, dim=1, keepdim=True)
    near_goal = (v_norm < 1.6).float()
    arc_mult = 1.0 + 0.8 * near_goal
    offset = side_to_use * arc_radius * arc_mult  # initial arc offset

    target_delta = -a_to_ball + offset  # move to an offset around the ball

    # Teammate influence: emphasize teammate's pocket
    dist_to_teammate = torch.norm(te_vec, dim=1, keepdim=True) + 1e-6
    to_teammate = -te_vec
    to_teammate_dir = to_teammate / dist_to_teammate
    corr = torch.clamp(2.0 - dist_to_teammate, min=0.0, max=0.8)
    target_delta = target_delta + to_teammate_dir * corr

    # Lane/pocket influence: widen/narrow arc toward teammate pocket
    dot_side = (te_vec * side_to_use).sum(dim=1, keepdim=True)
    pocket_influence = torch.clamp(torch.abs(dot_side) / 2.0, 0.0, 1.0)
    lane_dir = torch.sign(dot_side)
    offset_lane = side_to_use * lane_dir * pocket_influence * 0.4
    target_delta = target_delta + offset_lane

    # Defensive projection: adapt arc to adversaries near ball-goal line
    adv1_rel = obs[:, 17:19]
    adv2_rel = obs[:, 25:27]
    adv1_rel = -adv1_rel
    adv2_rel = -adv2_rel
    dot1 = (adv1_rel * side_to_use).sum(dim=1, keepdim=True)
    dot2 = (adv2_rel * side_to_use).sum(dim=1, keepdim=True)
    avg_dot = (dot1 + dot2) * 0.5
    offset_dir = torch.where(avg_dot > 0, -side_to_use, side_to_use)
    weight = torch.abs(avg_dot) * 0.5
    weight = torch.clamp(weight, 0.0, 1.0)
    arc_adjust = offset_dir * weight * 0.5
    offset = offset + arc_adjust
    target_delta = -a_to_ball + offset  # recalc with arc adjust

    action = target_delta
    mag = torch.norm(action, dim=1, keepdim=True) + 1e-6
    max_mag = torch.tensor(3.0, dtype=obs.dtype, device=obs.device)
    scale = torch.clamp(mag, min=1e-6)
    action = action / scale * torch.clamp(mag, max=max_mag)

    # Ensure shape is (batch, 2)
    action = action[:, :2]
    return action