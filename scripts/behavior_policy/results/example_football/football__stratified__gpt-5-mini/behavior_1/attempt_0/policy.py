import torch

def agent0_policy(obs):
    """
    Anticipatory Ball Interceptor policy for agent0.

    Behavior summary (implemented):
    - Predict short-horizon ball position and choose an interception time t that approximately
      minimizes distance to that predicted point (uses projection onto ball velocity and
      fallback to distance/speed estimate).
    - If an adversary is better positioned to reach the chosen intercept, bias the intercept
      point toward the goal to cut off clearance.
    - If teammate is already converging to the same intercept (teammate closer), yield and
      choose a supporting intercept point opening a passing lane (between ball path and goal).
    - If the ball is on our defensive half (ball is between agent and goal, judged by ball_pos_rel_goal.x < 0), prioritize blocking opponent approach lines.
    - Add short-range repulsive forces from nearby agents (teammate + adversaries) to avoid collisions.
    - Output is a (1,2) torch tensor representing the force vector.

    Assumptions about observation layout (derived from observation_base):
    - obs is a 1D torch tensor built from concatenation of the following blocks:
      [obs_block, pos(2), vel(2), adversaries(2*8), teammates(8)].
    - obs_block contains (in order):
      agent_force(2), agent_pos - ball_pos(2), agent_vel - ball_vel(2),
      ball_pos - goal_pos(2), ball_vel(2), ball_force(2), agent_rot(N)
      (agent_rot dimension is inferred automatically).
    - There are exactly 2 adversaries (each 8-dim chunk) and 1 teammate (8-dim chunk).

    The code is robust to the agent_rot dimensionality (1 or 2 typically).
    """

    # Ensure float tensor
    obs = obs.float()
    total_len = obs.numel()

    # fixed sizes for tail blocks
    adversary_count = 2
    adversary_block = 8 * adversary_count  # 16
    teammate_block = 8  # one teammate
    tail_fixed = 2 + 2 + adversary_block + teammate_block  # pos(2) + vel(2) + adversaries + teammates

    # compute obs_block length (may include agent_rot of variable size)
    len_obs_block = int(total_len - tail_fixed)
    if len_obs_block < 12:
        # fallback safety
        len_obs_block = 12

    # slicing indices
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

    # Parse obs_block components according to observation_base ordering
    # expected positions inside obs_block:
    # 0:2 agent_force, 2:4 agent_pos - ball_pos, 4:6 agent_vel - ball_vel,
    # 6:8 ball_pos - goal_pos, 8:10 ball_vel, 10:12 ball_force, 12:... agent_rot
    agent_force = obs_block[0:2].clone()
    agent_pos_minus_ball = obs_block[2:4].clone()
    agent_vel_minus_ball_vel = obs_block[4:6].clone()
    ball_pos_rel_goal = obs_block[6:8].clone()
    ball_vel = obs_block[8:10].clone()
    # ball_force = obs_block[10:12]

    # For calculations we'll treat goal position as origin in this frame, so
    # agent_pos = agent_pos_rel_goal, ball_pos = ball_pos_rel_goal
    agent_pos = agent_pos_rel_goal
    ball_pos = ball_pos_rel_goal

    # Parse adversaries (2 adversaries, each 8-long)
    adversaries = []
    for k in range(adversary_count):
        chunk = adversaries_block[k * 8 : (k + 1) * 8]
        rel_agent_minus_advpos = chunk[0:2]
        # rel_agent_minus_advvel = chunk[2:4]
        adv_vel = chunk[4:6]
        # adv_force = chunk[6:8]
        # adv_pos_rel_goal = agent_pos_rel_goal - (agent_pos - adv_pos) = agent_pos_rel_goal - rel
        adv_pos_rel_goal = agent_pos - rel_agent_minus_advpos
        adversaries.append({
            "pos": adv_pos_rel_goal,
            "vel": adv_vel,
        })

    # Parse teammate (single teammate)
    teammate = None
    tchunk = teammates_block[0:8]
    rel_agent_minus_teamp = tchunk[0:2]
    # rel_agent_minus_teamvel = tchunk[2:4]
    team_vel = tchunk[4:6]
    # team_force = tchunk[6:8]
    team_pos_rel_goal = agent_pos - rel_agent_minus_teamp
    teammate = {"pos": team_pos_rel_goal, "vel": team_vel}

    # PARAMETERS
    max_horizon = 3.0
    nominal_speed = 3.0  # nominal travel speed used for time estimate
    repel_radius = 1.2
    repel_strength = 2.0
    max_force = 4.0

    # Compute candidate intercept time t
    rel_agent_ball = (agent_pos - ball_pos)
    v_b = ball_vel
    v_b_norm2 = (v_b * v_b).sum()

    if v_b_norm2.item() > 1e-6:
        t_proj = torch.clamp((rel_agent_ball * v_b).sum() / (v_b_norm2 + 1e-6), 0.0, max_horizon)
    else:
        t_proj = torch.tensor(0.0)

    t_est = torch.clamp(torch.norm(rel_agent_ball) / (nominal_speed + 1e-6), 0.0, max_horizon)
    # blend projection and distance-based estimate
    t = 0.6 * t_proj + 0.4 * t_est

    # Predicted ball position
    predicted_ball = ball_pos + v_b * t

    # Distances to predicted ball (using agents' predicted positions)
    ego_dist = torch.norm(agent_pos - predicted_ball)

    adv_dists = []
    for adv in adversaries:
        adv_pred = adv["pos"] + adv["vel"] * t
        # include small directional bias: if adv velocity points toward predicted_ball, they are more threatening
        to_ball = predicted_ball - adv["pos"]
        vel_alignment = 0.0
        if torch.norm(adv["vel"]) > 1e-6:
            vel_alignment = torch.dot(torch.nn.functional.normalize(adv["vel"], dim=0),
                                      torch.nn.functional.normalize(to_ball, dim=0)).item()
        # lower effective distance if velocity aligned (more likely to intercept)
        eff_dist = torch.norm(adv_pred - predicted_ball) - 0.5 * max(0.0, vel_alignment)
        adv_dists.append(eff_dist)

    team_pred = teammate["pos"] + teammate["vel"] * t
    team_dist = torch.norm(team_pred - predicted_ball)

    # Choose intercept point logic
    intercept_point = predicted_ball.clone()

    # If teammate is closer by margin, yield and choose supporting intercept (open passing lane)
    if team_dist + 0.6 < ego_dist:
        # supporting intercept: midpoint between predicted_ball and goal (goal at origin)
        to_goal = -predicted_ball  # vector from ball to goal (goal at origin)
        support_base = predicted_ball * 0.5  # halfway to goal
        # add lateral offset perpendicular to to_goal to open lane; choose side based on teammate position
        if torch.norm(to_goal) > 1e-6:
            perp = torch.tensor([-to_goal[1], to_goal[0]])
            perp = perp / (torch.norm(perp) + 1e-6)
            side = torch.sign(torch.dot((teammate["pos"] - predicted_ball), perp))
            lateral = perp * 0.6 * (1.0 if side >= 0 else -1.0)
            support_point = support_base + lateral
        else:
            support_point = support_base
        intercept_point = support_point
    else:
        # If an adversary is closer (consider their effective distance), bias intercept toward goal
        min_adv_dist = min(adv_dists)
        # compute margin: if adversary significantly closer, bias toward goal
        if min_adv_dist + 0.4 < ego_dist:
            # shift intercept toward goal to cut off opponent clearance
            alpha = 0.6
            intercept_point = predicted_ball * (1.0 - alpha) + (torch.zeros_like(predicted_ball)) * alpha
            # alpha toward goal; this effectively moves intercept closer to goal
            # we can also nudge slightly along direction from ball to goal
            intercept_point = predicted_ball + ( -predicted_ball ) * 0.35

    # Defensive fallback: if ball is on our defensive half, try to block opponent approach lines
    # Heuristic: if ball_pos_rel_goal[0] < 0 (between agent and goal) treat as defensive
    try:
        defensive_flag = (ball_pos[0].item() < 0.0)
    except Exception:
        defensive_flag = (ball_pos[0] < 0.0)

    if defensive_flag:
        # Compute the most threatening adversary and place intercept between them and goal
        adv_vals = []
        for adv in adversaries:
            # distance from adv to line between ball and goal (goal at 0)
            a_to_goal = -adv["pos"]  # vector from adv to goal
            # simple measure: closeness to predicted ball path
            adv_score = torch.norm(adv["pos"] - predicted_ball)
            adv_vals.append(adv_score)
        # choose best adversary
        adv_idx = int(torch.argmin(torch.stack(adv_vals)).item())
        threat = adversaries[adv_idx]
        # blocking point: point between threat predicted pos and goal, slightly toward goal
        threat_pred = threat["pos"] + threat["vel"] * t
        block_point = 0.5 * (threat_pred + torch.zeros_like(threat_pred))
        # blend with current intercept
        intercept_point = 0.6 * intercept_point + 0.4 * block_point

    # Compute desired force toward intercept_point
    to_target = intercept_point - agent_pos
    dist_to_target = torch.norm(to_target)
    if dist_to_target < 1e-4:
        move_dir = torch.zeros_like(to_target)
    else:
        move_dir = to_target / (dist_to_target + 1e-6)

    # base propulsion proportional to distance but capped
    base_force = move_dir * min(max_force, 2.0 + dist_to_target.item())

    # Repulsive forces from nearby agents
    repulse = torch.zeros(2, dtype=obs.dtype)
    # teammate
    d_team = torch.norm(agent_pos - teammate["pos"]).item()
    if d_team < repel_radius and d_team > 1e-6:
        dir_away = (agent_pos - teammate["pos"]) / (d_team + 1e-6)
        repulse += dir_away * repel_strength * (1.0 - d_team / repel_radius) ** 2
    # adversaries
    for adv in adversaries:
        d_adv = torch.norm(agent_pos - adv["pos"]).item()
        if d_adv < repel_radius and d_adv > 1e-6:
            dir_away = (agent_pos - adv["pos"]) / (d_adv + 1e-6)
            repulse += dir_away * repel_strength * (1.0 - d_adv / repel_radius) ** 2

    # small damping to avoid oscillations (opposite current velocity)
    damping = -0.3 * agent_vel

    force = base_force + repulse + damping

    # Clip to maximum force magnitude
    force_mag = torch.norm(force)
    if force_mag > max_force:
        force = force / (force_mag + 1e-6) * max_force

    # return shape (1,2)
    return force.unsqueeze(0)

import torch


def agent1_policy(obs):
    """
    Coordinating Anticipatory Interceptor for Agent 1.

    Coordination strategy (implemented):
    - Mirror the partner's anticipatory interception logic: predict a short-horizon ball position
      and compute an intercept point using a blended projection/time estimate.
    - If the teammate is clearly closer, yield by choosing a supporting intercept (midway toward
      goal with a lateral offset to open a passing lane).
    - If both agents appear to be about equally well-positioned (within a small margin), take a
      complementary approach: move to the predicted intercept region but with a lateral offset
      chosen so the two agents approach from different sides (reduces collision and stabilizes
      role selection when both think they should go).
    - If an adversary is significantly closer to the intercept, bias the intercept toward the goal
      to cut off clearances (same as partner's idea).
    - Defensive fallback: if the ball is on our defensive half, move to block the most threatening
      adversary's approach line.
    - Add short-range repulsive forces from nearby agents and damping.

    This policy intentionally includes a small complementary lateral-offset rule when distances are
    similar to avoid both agents repeatedly yielding to each other (a deterministic tie-break based on
    relative positions of ego vs teammate). The rest of the math follows the partner's conventions
    so both agents behave compatibly.

    Input: obs - 1D torch tensor as produced by observation_base
    Output: (1,2) torch tensor force
    """

    obs = obs.float()
    total_len = obs.numel()

    # fixed layout assumptions consistent with observation_base and partner's parser
    adversary_count = 2
    adv_block = 8 * adversary_count
    teammate_block = 8
    tail_fixed = 2 + 2 + adv_block + teammate_block

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

    adversaries_block = obs[i : i + adv_block].clone()
    i += adv_block
    teammates_block = obs[i : i + teammate_block].clone()

    # parse obs_block (same ordering as partner)
    agent_force = obs_block[0:2].clone()
    agent_pos_minus_ball = obs_block[2:4].clone()
    agent_vel_minus_ball_vel = obs_block[4:6].clone()
    ball_pos_rel_goal = obs_block[6:8].clone()
    ball_vel = obs_block[8:10].clone()

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

    # Parse teammate
    tchunk = teammates_block[0:8]
    rel_agent_minus_teamp = tchunk[0:2]
    team_vel = tchunk[4:6]
    team_pos_rel_goal = agent_pos - rel_agent_minus_teamp
    teammate = {"pos": team_pos_rel_goal, "vel": team_vel}

    # PARAMETERS
    max_horizon = 3.0
    nominal_speed = 3.0
    repel_radius = 1.2
    repel_strength = 2.0
    max_force = 4.0

    # Compute candidate intercept time t (blend projection and distance/speed estimate)
    rel_agent_ball = agent_pos - ball_pos
    v_b = ball_vel
    v_b_norm2 = (v_b * v_b).sum()

    if v_b_norm2.item() > 1e-6:
        t_proj = torch.clamp((rel_agent_ball * v_b).sum() / (v_b_norm2 + 1e-6), 0.0, max_horizon)
    else:
        t_proj = torch.tensor(0.0, dtype=obs.dtype)

    t_est = torch.clamp(torch.norm(rel_agent_ball) / (nominal_speed + 1e-6), 0.0, max_horizon)
    t = 0.6 * t_proj + 0.4 * t_est

    # Predicted ball
    predicted_ball = ball_pos + v_b * t

    # Distances
    ego_dist = torch.norm(agent_pos - predicted_ball)

    adv_dists = []
    for adv in adversaries:
        adv_pred = adv["pos"] + adv["vel"] * t
        to_ball = predicted_ball - adv["pos"]
        vel_alignment = 0.0
        if torch.norm(adv["vel"]) > 1e-6:
            vel_alignment = torch.dot(
                torch.nn.functional.normalize(adv["vel"], dim=0),
                torch.nn.functional.normalize(to_ball, dim=0),
            ).item()
        eff_dist = torch.norm(adv_pred - predicted_ball) - 0.5 * max(0.0, vel_alignment)
        adv_dists.append(eff_dist)

    team_pred = teammate["pos"] + teammate["vel"] * t
    team_dist = torch.norm(team_pred - predicted_ball)

    intercept_point = predicted_ball.clone()

    # Yielding thresholds: be slightly complementary to partner by using a small complementary rule
    # If teammate is clearly closer, yield and take supporting intercept
    if team_dist + 0.6 < ego_dist:
        # supporting intercept: halfway to goal with lateral offset to open lane
        to_goal = -predicted_ball
        support_base = predicted_ball * 0.5
        if torch.norm(to_goal) > 1e-6:
            perp = torch.tensor([-to_goal[1], to_goal[0]], dtype=obs.dtype)
            perp = perp / (torch.norm(perp) + 1e-6)
            # choose lateral side so that support complements teammate position
            side = torch.sign(torch.dot((teammate["pos"] - predicted_ball), perp))
            lateral = perp * 0.6 * (1.0 if side >= 0 else -1.0)
            support_point = support_base + lateral
        else:
            support_point = support_base
        intercept_point = support_point
    else:
        # If adversary more threatening, bias intercept toward goal
        min_adv_dist = min(adv_dists) if len(adv_dists) > 0 else float('inf')
        if min_adv_dist + 0.4 < ego_dist:
            intercept_point = predicted_ball + (-predicted_ball) * 0.35

    # If both agents are similarly positioned, pick complementary lateral offsets so agents approach
    # the same intercept region from different sides (reduces oscillation / mutual yielding)
    if torch.abs(team_dist - ego_dist) < 0.35:
        # lateral relative to ball->goal direction
        to_goal = -predicted_ball
        if torch.norm(to_goal) > 1e-6:
            perp = torch.tensor([-to_goal[1], to_goal[0]], dtype=obs.dtype)
            perp = perp / (torch.norm(perp) + 1e-6)
            # pick side based on relative position to teammate (deterministic tie-break)
            side_sign = torch.sign(torch.dot((agent_pos - teammate["pos"]), perp))
            # if side_sign is zero, pick by x coordinate to be deterministic
            if side_sign == 0:
                side_sign = 1.0 if agent_pos[0] >= teammate["pos"][0] else -1.0
            lateral = perp * 0.5 * (1.0 if side_sign >= 0 else -1.0)
            intercept_point = predicted_ball + lateral
        else:
            # fallback: small lateral using agent_pos - teammate_pos
            diff = agent_pos - teammate["pos"]
            if torch.norm(diff) > 1e-6:
                lateral = torch.nn.functional.normalize(diff, dim=0) * 0.2
                intercept_point = predicted_ball + lateral

    # Defensive fallback similar to partner: block the most threatening adversary path
    defensive_flag = False
    try:
        defensive_flag = (ball_pos[0].item() < 0.0)
    except Exception:
        defensive_flag = (ball_pos[0] < 0.0)

    if defensive_flag:
        adv_vals = []
        for adv in adversaries:
            adv_vals.append(torch.norm(adv["pos"] - predicted_ball))
        if len(adv_vals) > 0:
            adv_idx = int(torch.argmin(torch.stack(adv_vals)).item())
            threat = adversaries[adv_idx]
            threat_pred = threat["pos"] + threat["vel"] * t
            block_point = 0.5 * (threat_pred + torch.zeros_like(threat_pred))
            intercept_point = 0.6 * intercept_point + 0.4 * block_point

    # Desired force toward intercept_point
    to_target = intercept_point - agent_pos
    dist_to_target = torch.norm(to_target)
    if dist_to_target < 1e-4:
        move_dir = torch.zeros_like(to_target)
    else:
        move_dir = to_target / (dist_to_target + 1e-6)

    base_force = move_dir * min(max_force, 2.0 + dist_to_target.item())

    # Repulsive forces
    repulse = torch.zeros(2, dtype=obs.dtype)
    d_team = torch.norm(agent_pos - teammate["pos"]).item()
    if d_team < repel_radius and d_team > 1e-6:
        dir_away = (agent_pos - teammate["pos"]) / (d_team + 1e-6)
        repulse += dir_away * repel_strength * (1.0 - d_team / repel_radius) ** 2
    for adv in adversaries:
        d_adv = torch.norm(agent_pos - adv["pos"]).item()
        if d_adv < repel_radius and d_adv > 1e-6:
            dir_away = (agent_pos - adv["pos"]) / (d_adv + 1e-6)
            repulse += dir_away * repel_strength * (1.0 - d_adv / repel_radius) ** 2

    # Small cooperative attraction when supporting: if the teammate is going to the same intercept and is
    # slightly ahead, add a small lateral attraction toward teammate goal approach to help converge together
    coop_attract = torch.zeros(2, dtype=obs.dtype)
    if team_dist + 0.6 < ego_dist:
        # gently align velocity toward teammate's predicted approach direction
        vec_to_team = teammate["pos"] - agent_pos
        if torch.norm(vec_to_team) > 1e-4:
            coop_attract = torch.nn.functional.normalize(vec_to_team, dim=0) * 0.4

    # damping
    damping = -0.3 * agent_vel

    force = base_force + repulse + damping + coop_attract

    force_mag = torch.norm(force)
    if force_mag > max_force:
        force = force / (force_mag + 1e-6) * max_force

    return force.unsqueeze(0)