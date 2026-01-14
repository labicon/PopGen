import torch

def agent0_policy(obs):
    """
    Anticipatory Ball Interceptor (agent0).

    Key points:
    - Predict short-horizon ball position and pick an intercept time t (blend of projection and
      distance/speed estimate).
    - If a teammate is clearly closer to that intercept, yield and move to a supporting intercept
      (midway toward goal with a lateral offset to open a passing lane).
    - If an adversary is better positioned, bias the intercept toward the goal to cut off clearance.
    - Defensive fallback: when the ball is on our defensive half, bias toward blocking the most
      threatening adversary approach.
    - Add local repulsive forces from nearby agents and damping.
    - Ensure final action magnitude is clipped to the environment's action range (<= 1.0).

    Input: obs - 1D torch tensor as produced by observation_base
    Output: (1,2) torch tensor force (magnitude <= 1.0)
    """

    # Ensure consistent dtype/device
    obs = obs.float()
    device = obs.device

    total_len = obs.numel()

    # Layout assumptions (from observation_base):
    # obs consists of a variable-length obs_block followed by: pos(2), vel(2), adversaries(2*8), teammates(8)
    adversary_count = 2
    adversary_block = 8 * adversary_count
    teammate_block = 8
    tail_fixed = 2 + 2 + adversary_block + teammate_block

    len_obs_block = int(total_len - tail_fixed)
    if len_obs_block < 12:
        len_obs_block = 12

    i = 0
    obs_block = obs[i : i + len_obs_block]
    i += len_obs_block

    agent_pos_rel_goal = obs[i : i + 2].clone()
    i += 2
    agent_vel = obs[i : i + 2].clone()
    i += 2

    adversaries_block = obs[i : i + adversary_block].clone()
    i += adversary_block
    teammates_block = obs[i : i + teammate_block].clone()

    # parse obs_block (ordering from observation_base)
    # [agent_force (2), agent_pos - ball_pos (2), agent_vel - ball_vel (2),
    #  ball_pos - goal_pos (2), ball_vel (2), ball_force (2), agent_rot (...)]
    # We only need some of these
    agent_pos_minus_ball = obs_block[2:4].clone()
    ball_pos_rel_goal = obs_block[6:8].clone()
    ball_vel = obs_block[8:10].clone()

    # Represent positions in goal-centered frame (goal at origin) as in observation: agent_pos = agent_pos_rel_goal
    agent_pos = agent_pos_rel_goal
    ball_pos = ball_pos_rel_goal

    # Parse adversaries (2 adversaries each 8 dims)
    adversaries = []
    for k in range(adversary_count):
        chunk = adversaries_block[k * 8 : (k + 1) * 8]
        rel_agent_minus_advpos = chunk[0:2]
        adv_vel = chunk[4:6]
        # adv_pos_rel_goal = agent_pos - rel_agent_minus_advpos
        adv_pos_rel_goal = agent_pos - rel_agent_minus_advpos
        adversaries.append({"pos": adv_pos_rel_goal, "vel": adv_vel})

    # Parse single teammate
    tchunk = teammates_block[0:8]
    rel_agent_minus_teamp = tchunk[0:2]
    team_vel = tchunk[4:6]
    team_pos_rel_goal = agent_pos - rel_agent_minus_teamp
    teammate = {"pos": team_pos_rel_goal, "vel": team_vel}

    # PARAMETERS (ensure final max_force <= environment action magnitude limit)
    max_horizon = 3.0
    nominal_speed = 3.0
    repel_radius = 1.2
    repel_strength = 1.0  # reduced so repulsion won't push beyond limit
    max_force = 1.0  # action magnitude limit enforced by environment

    # Compute candidate intercept time t
    rel_agent_ball = agent_pos - ball_pos
    v_b = ball_vel
    v_b_norm2 = (v_b * v_b).sum()

    if v_b_norm2.item() > 1e-6:
        t_proj = torch.clamp((rel_agent_ball * v_b).sum() / (v_b_norm2 + 1e-6), 0.0, max_horizon)
    else:
        t_proj = torch.tensor(0.0, dtype=obs.dtype, device=device)

    t_est = torch.clamp(torch.norm(rel_agent_ball) / (nominal_speed + 1e-6), 0.0, max_horizon)
    t = 0.6 * t_proj + 0.4 * t_est

    # Predicted ball position
    predicted_ball = ball_pos + v_b * t

    # Distances to predicted ball
    ego_dist = torch.norm(agent_pos - predicted_ball)

    adv_dists = []
    for adv in adversaries:
        adv_pred = adv["pos"] + adv["vel"] * t
        to_ball = predicted_ball - adv["pos"]
        vel_alignment = 0.0
        adv_vel_norm = torch.norm(adv["vel"])
        if adv_vel_norm > 1e-6 and torch.norm(to_ball) > 1e-6:
            vel_alignment = torch.dot(adv["vel"] / (adv_vel_norm + 1e-9), to_ball / (torch.norm(to_ball) + 1e-9)).item()
        eff_dist = torch.norm(adv_pred - predicted_ball) - 0.5 * max(0.0, vel_alignment)
        adv_dists.append(eff_dist)

    team_pred = teammate["pos"] + teammate["vel"] * t
    team_dist = torch.norm(team_pred - predicted_ball)

    intercept_point = predicted_ball.clone()

    # Yield to teammate if clearly closer -> supporting intercept
    if team_dist + 0.6 < ego_dist:
        to_goal = -predicted_ball
        support_base = predicted_ball * 0.5
        if torch.norm(to_goal) > 1e-6:
            perp = torch.tensor([-to_goal[1], to_goal[0]], dtype=obs.dtype, device=device)
            perp = perp / (torch.norm(perp) + 1e-6)
            side_val = torch.dot((teammate["pos"] - predicted_ball), perp).item()
            lateral = perp * 0.6 * (1.0 if side_val >= 0.0 else -1.0)
            support_point = support_base + lateral
        else:
            support_point = support_base
        intercept_point = support_point
    else:
        # If adversary is significantly closer to intercept, bias point toward goal to cut off clearance
        if len(adv_dists) > 0:
            min_adv_dist = min(adv_dists)
            if min_adv_dist + 0.4 < ego_dist:
                # Move intercept a fraction toward goal (goal at origin)
                intercept_point = predicted_ball + (-predicted_ball) * 0.35

    # Defensive fallback: if ball is on our defensive half (ball x < 0), block threatening adversary
    defensive_flag = False
    try:
        defensive_flag = (ball_pos[0].item() < 0.0)
    except Exception:
        defensive_flag = (ball_pos[0] < 0.0)

    if defensive_flag and len(adversaries) > 0:
        adv_vals = [torch.norm(adv["pos"] - predicted_ball) for adv in adversaries]
        adv_idx = int(torch.argmin(torch.stack(adv_vals)).item())
        threat = adversaries[adv_idx]
        threat_pred = threat["pos"] + threat["vel"] * t
        block_point = 0.5 * (threat_pred + torch.zeros_like(threat_pred))
        intercept_point = 0.6 * intercept_point + 0.4 * block_point

    # Compute desired force toward intercept_point
    to_target = intercept_point - agent_pos
    dist_to_target = torch.norm(to_target)
    if dist_to_target < 1e-4:
        move_dir = torch.zeros_like(to_target)
    else:
        move_dir = to_target / (dist_to_target + 1e-6)

    # Scale base force modestly relative to distance but ensure it never exceeds max_force
    # Use a gentle sigmoid-like scale: s = 1 - exp(-d / 1.0) so small distances give small forces
    dist_scalar = (1.0 - torch.exp(-dist_to_target / (1.0 + 1e-6))).item()
    base_mag = float(min(max_force, 0.9 * dist_scalar))
    base_force = move_dir * base_mag

    # Repulsive forces from nearby agents (small magnitude)
    repulse = torch.zeros(2, dtype=obs.dtype, device=device)
    d_team = torch.norm(agent_pos - teammate["pos"]).item()
    if d_team < repel_radius and d_team > 1e-6:
        dir_away = (agent_pos - teammate["pos"]) / (d_team + 1e-6)
        repulse += dir_away * repel_strength * (1.0 - d_team / repel_radius) ** 2

    for adv in adversaries:
        d_adv = torch.norm(agent_pos - adv["pos"]).item()
        if d_adv < repel_radius and d_adv > 1e-6:
            dir_away = (agent_pos - adv["pos"]) / (d_adv + 1e-6)
            repulse += dir_away * repel_strength * (1.0 - d_adv / repel_radius) ** 2

    # damping (opposite current velocity)
    damping = -0.2 * agent_vel

    force = base_force + repulse + damping

    # Clip to maximum force magnitude allowed by environment
    force_mag = torch.norm(force)
    if force_mag > max_force:
        force = force / (force_mag + 1e-9) * max_force

    # ensure shape (1,2)
    return force.unsqueeze(0)

import torch

def agent1_policy(obs):
    """
    Coordinating Anticipatory Interceptor (agent1).

    Strategy / coordination points (summary implemented below):
    - Predict a short-horizon ball position using a blend of projection and distance/speed estimate.
    - If our teammate (agent0) is clearly closer to the predicted intercept (team_dist + 0.6 < ego_dist), yield and move to a supporting intercept that opens a passing lane (midway toward goal with a lateral offset).
    - Otherwise act as the primary interceptor and move toward the predicted intercept; if an adversary is better positioned, bias intercept toward the goal to cut off opponent clearance.
    - Defensive fallback: if the ball is on our defensive half, bias toward blocking the most threatening adversary approach.
    - Add small repulsive forces from nearby agents and damping.
    - Clip final force magnitude to 1.0 and return shape (1,2).
    """

    obs = obs.float()
    device = obs.device

    total_len = obs.numel()
    adversary_count = 2
    adversary_block = 8 * adversary_count
    teammate_block = 8
    tail_fixed = 2 + 2 + adversary_block + teammate_block

    len_obs_block = int(total_len - tail_fixed)
    if len_obs_block < 12:
        len_obs_block = 12

    i = 0
    obs_block = obs[i : i + len_obs_block]
    i += len_obs_block

    agent_pos_rel_goal = obs[i : i + 2].clone()
    i += 2
    agent_vel = obs[i : i + 2].clone()
    i += 2

    adversaries_block = obs[i : i + adversary_block].clone()
    i += adversary_block
    teammates_block = obs[i : i + teammate_block].clone()

    # parse some fields from obs_block
    # ordering: [agent_force(2), agent_pos - ball_pos(2), agent_vel - ball_vel(2), ball_pos - goal_pos(2), ball_vel(2), ball_force(2), agent_rot(...)]
    agent_pos_minus_ball = obs_block[2:4].clone()
    ball_pos_rel_goal = obs_block[6:8].clone()
    ball_vel = obs_block[8:10].clone()

    # Represent positions in goal-centered frame (goal at origin)
    agent_pos = agent_pos_rel_goal
    ball_pos = ball_pos_rel_goal

    # Parse adversaries
    adversaries = []
    for k in range(adversary_count):
        chunk = adversaries_block[k * 8 : (k + 1) * 8]
        rel_agent_minus_advpos = chunk[0:2]
        adv_vel = chunk[4:6]
        adv_pos_rel_goal = agent_pos - rel_agent_minus_advpos
        adversaries.append({"pos": adv_pos_rel_goal, "vel": adv_vel})

    # Parse single teammate (agent0)
    tchunk = teammates_block[0:8]
    rel_agent_minus_teamp = tchunk[0:2]
    team_vel = tchunk[4:6]
    team_pos_rel_goal = agent_pos - rel_agent_minus_teamp
    teammate = {"pos": team_pos_rel_goal, "vel": team_vel}

    # Parameters
    max_horizon = 3.0
    nominal_speed = 3.0
    repel_radius = 1.2
    repel_strength = 1.0
    max_force = 1.0

    # Compute candidate intercept time t (blend projection and estimate)
    rel_agent_ball = agent_pos - ball_pos
    v_b = ball_vel
    v_b_norm2 = (v_b * v_b).sum()

    if v_b_norm2.item() > 1e-6:
        t_proj = torch.clamp((rel_agent_ball * v_b).sum() / (v_b_norm2 + 1e-6), 0.0, max_horizon)
    else:
        t_proj = torch.tensor(0.0, dtype=obs.dtype, device=device)

    t_est = torch.clamp(torch.norm(rel_agent_ball) / (nominal_speed + 1e-6), 0.0, max_horizon)
    t = 0.6 * t_proj + 0.4 * t_est

    predicted_ball = ball_pos + v_b * t

    ego_dist = torch.norm(agent_pos - predicted_ball)

    # Compute adversary effective distances to predicted ball (considering velocity alignment)
    adv_dists = []
    for adv in adversaries:
        adv_pred = adv["pos"] + adv["vel"] * t
        to_ball = predicted_ball - adv["pos"]
        vel_alignment = 0.0
        adv_vel_norm = torch.norm(adv["vel"])
        if adv_vel_norm > 1e-6 and torch.norm(to_ball) > 1e-6:
            vel_alignment = torch.dot(adv["vel"] / (adv_vel_norm + 1e-9), to_ball / (torch.norm(to_ball) + 1e-9)).item()
        eff_dist = torch.norm(adv_pred - predicted_ball) - 0.5 * max(0.0, vel_alignment)
        adv_dists.append(eff_dist)

    team_pred = teammate["pos"] + teammate["vel"] * t
    team_dist = torch.norm(team_pred - predicted_ball)

    intercept_point = predicted_ball.clone()

    # If our teammate is clearly closer, yield and take a supporting intercept that opens a passing lane
    if team_dist + 0.6 < ego_dist:
        to_goal = -predicted_ball
        support_base = predicted_ball * 0.5
        if torch.norm(to_goal) > 1e-6:
            perp = torch.tensor([-to_goal[1], to_goal[0]], dtype=obs.dtype, device=device)
            perp = perp / (torch.norm(perp) + 1e-6)
            side_val = torch.dot((teammate["pos"] - predicted_ball), perp).item()
            lateral = perp * 0.6 * (1.0 if side_val >= 0.0 else -1.0)
            support_point = support_base + lateral
        else:
            support_point = support_base
        intercept_point = support_point
    else:
        # Primary intercept: if an adversary is in a better position, bias toward goal to cut off clearance
        if len(adv_dists) > 0:
            min_adv_dist = min(adv_dists)
            if min_adv_dist + 0.4 < ego_dist:
                intercept_point = predicted_ball + (-predicted_ball) * 0.35

    # Defensive fallback: if ball is on our defensive half (x < 0) bias toward blocking the most threatening adversary
    defensive_flag = False
    try:
        defensive_flag = (ball_pos[0].item() < 0.0)
    except Exception:
        defensive_flag = (ball_pos[0] < 0.0)

    if defensive_flag and len(adversaries) > 0:
        adv_vals = [torch.norm(adv["pos"] - predicted_ball) for adv in adversaries]
        adv_idx = int(torch.argmin(torch.stack(adv_vals)).item())
        threat = adversaries[adv_idx]
        threat_pred = threat["pos"] + threat["vel"] * t
        block_point = 0.5 * (threat_pred + torch.zeros_like(threat_pred))
        intercept_point = 0.6 * intercept_point + 0.4 * block_point

    # Compute desired force toward intercept_point
    to_target = intercept_point - agent_pos
    dist_to_target = torch.norm(to_target)
    if dist_to_target < 1e-4:
        move_dir = torch.zeros_like(to_target)
    else:
        move_dir = to_target / (dist_to_target + 1e-6)

    # gentle scaling: small when near, stronger when far
    dist_scalar = (1.0 - torch.exp(-dist_to_target / (1.0 + 1e-6))).item()
    base_mag = float(min(max_force, 0.9 * dist_scalar))
    base_force = move_dir * base_mag

    # Repulsive forces from nearby agents (teammate + adversaries)
    repulse = torch.zeros(2, dtype=obs.dtype, device=device)
    d_team = torch.norm(agent_pos - teammate["pos"]).item()
    if d_team < repel_radius and d_team > 1e-6:
        dir_away = (agent_pos - teammate["pos"]) / (d_team + 1e-6)
        repulse += dir_away * repel_strength * (1.0 - d_team / repel_radius) ** 2

    for adv in adversaries:
        d_adv = torch.norm(agent_pos - adv["pos"]).item()
        if d_adv < repel_radius and d_adv > 1e-6:
            dir_away = (agent_pos - adv["pos"]) / (d_adv + 1e-6)
            repulse += dir_away * (repel_strength * 0.9) * (1.0 - d_adv / repel_radius) ** 2

    # damping to avoid overshoot
    damping = -0.2 * agent_vel

    force = base_force + repulse + damping

    # Clip to maximum magnitude
    force_mag = torch.norm(force)
    if force_mag > max_force:
        force = force / (force_mag + 1e-9) * max_force

    return force.unsqueeze(0)