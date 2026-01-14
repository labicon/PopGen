import torch

def agent0_policy(obs):
    """
    Anticipatory Ball Interceptor (agent0).

    Implements:
    - Predict short-horizon ball position using a time horizon t chosen so that the agent's
      travel time to the predicted point matches the ball arrival time (search over candidate t).
    - If an adversary can reach the intercept sooner (considering small velocity alignment bonus),
      bias the intercept toward the goal to cut off clearance.
    - If a teammate is closer/converging to the same intercept, yield and choose a supporting
      intercept between the ball path and the goal with a lateral offset to open a passing lane.
    - Defensive fallback: when the ball is on our defensive half (ball_pos.x < 0), move to
      block a threatening adversary by blending toward a block point between that adversary and goal.
    - Small repulsive forces from nearby agents to avoid collisions.
    - Final action is clipped to magnitude <= 1.0 (environment limit).

    Input: obs - 1D torch tensor produced by observation_base
    Output: (1,2) torch tensor force
    """

    obs = obs.float()
    device = obs.device

    # Layout assumptions consistent with observation_base
    adversary_count = 2
    adversary_block = 8 * adversary_count
    teammate_block = 8  # single teammate
    tail_fixed = 2 + 2 + adversary_block + teammate_block

    total_len = obs.numel()
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

    # Parse relevant fields from obs_block (ordering follows observation_base)
    # [agent_force(2), agent_pos - ball_pos(2), agent_vel - ball_vel(2), ball_pos - goal_pos(2), ball_vel(2), ball_force(2), agent_rot(...)]
    # We need ball_pos_rel_goal and ball_vel and agent_pos (which is agent_pos_rel_goal) and some relative fields.
    ball_pos_rel_goal = obs_block[6:8].clone()
    ball_vel = obs_block[8:10].clone()

    # Coordinates are in goal-centered frame (goal at origin)
    agent_pos = agent_pos_rel_goal
    ball_pos = ball_pos_rel_goal

    # Parse adversaries
    adversaries = []
    for k in range(adversary_count):
        chunk = adversaries_block[k * 8 : (k + 1) * 8]
        rel_agent_minus_advpos = chunk[0:2]
        # rel_agent_minus_advvel = chunk[2:4]
        adv_vel = chunk[4:6]
        # adv_force = chunk[6:8]
        adv_pos_rel_goal = agent_pos - rel_agent_minus_advpos
        adversaries.append({"pos": adv_pos_rel_goal, "vel": adv_vel})

    # Parse teammate (single teammate)
    tchunk = teammates_block[0:8]
    rel_agent_minus_teamp = tchunk[0:2]
    team_vel = tchunk[4:6]
    team_pos_rel_goal = agent_pos - rel_agent_minus_teamp
    teammate = {"pos": team_pos_rel_goal, "vel": team_vel}

    # Parameters
    max_horizon = 3.0
    nominal_speed = 3.0  # nominal travel speed for time estimates
    repel_radius = 1.0
    repel_strength = 0.5
    max_force = 1.0  # environment action magnitude limit

    # Choose intercept time by searching candidate times and minimizing mismatch between
    # agent travel time and ball arrival time at that point: minimize |t_agent - t_ball|
    candidates = torch.linspace(0.0, max_horizon, steps=9, device=device)

    if torch.norm(ball_vel) < 1e-6:
        # stationary ball -> prefer t=0 (current position)
        t_opt = torch.tensor(0.0, device=device)
    else:
        best_t = candidates[0]
        best_score = float('inf')
        for t_c in candidates:
            pred_ball = ball_pos + ball_vel * t_c
            dist = torch.norm(pred_ball - agent_pos)
            t_agent = dist / (nominal_speed + 1e-6)
            score = torch.abs(t_agent - t_c).item()
            # prefer slightly earlier intercepts when similar (be proactive)
            if score < best_score - 1e-6:
                best_score = score
                best_t = t_c
        t_opt = best_t

    predicted_ball = ball_pos + ball_vel * t_opt

    # Compute ego time-to-intercept (time agent needs to reach predicted point)
    ego_dist = torch.norm(predicted_ball - agent_pos)
    ego_time = (ego_dist / (nominal_speed + 1e-6)).item()

    # Compute teammate time and adversary effective times
    team_dist = torch.norm(predicted_ball - teammate['pos'])
    team_time = (team_dist / (nominal_speed + 1e-6)).item()

    adv_times = []
    for adv in adversaries:
        adv_dist = torch.norm(predicted_ball - adv['pos']).item()
        # velocity alignment bonus: if adv velocity points toward the intercept, reduce effective time
        adv_vel_norm = torch.norm(adv['vel']).item()
        if adv_vel_norm > 1e-6:
            to_point = (predicted_ball - adv['pos'])
            to_point_norm = torch.norm(to_point).item()
            if to_point_norm > 1e-6:
                align = (torch.dot(adv['vel'], to_point) / (adv_vel_norm * to_point_norm)).item()
            else:
                align = 0.0
        else:
            align = 0.0
        # reduce time a bit if velocity aligned (makes adversary more threatening)
        adv_time = max(0.0, adv_dist / (nominal_speed + 1e-6) - 0.25 * max(0.0, align))
        adv_times.append(adv_time)

    # Start with intercept point at predicted ball
    intercept_point = predicted_ball.clone()

    # Yielding/support logic: if teammate can get there significantly sooner, move to support point
    # margin chosen so teammate clearly closer
    yield_margin = 0.6
    if team_time + yield_margin < ego_time:
        # supporting intercept: point between predicted_ball and goal (goal at origin) with lateral offset
        to_goal = -predicted_ball
        support_base = 0.5 * predicted_ball  # halfway to goal
        if torch.norm(to_goal) > 1e-6:
            perp = torch.tensor([-to_goal[1], to_goal[0]], dtype=obs.dtype, device=device)
            perp = perp / (torch.norm(perp) + 1e-6)
            # choose side based on teammate relative position so we open the lane on the other side
            side_val = torch.dot((teammate['pos'] - predicted_ball), perp).item()
            side = 1.0 if side_val >= 0.0 else -1.0
            lateral = perp * (0.6 * side)
            support_point = support_base + lateral
        else:
            support_point = support_base
        intercept_point = support_point
    else:
        # If an adversary can reach the intercept notably sooner, bias toward goal to cut them off
        if len(adv_times) > 0:
            min_adv_time = min(adv_times)
            adv_margin = 0.4
            if min_adv_time + adv_margin < ego_time:
                # move intercept some fraction toward goal (goal at origin) to cut off opponent clearance
                alpha = 0.35
                intercept_point = predicted_ball + (-predicted_ball) * alpha

    # Defensive fallback: if ball is on our defensive half (ball_pos.x < 0) try to block the most threatening adversary
    defensive_flag = False
    try:
        defensive_flag = (ball_pos[0].item() < 0.0)
    except Exception:
        defensive_flag = (ball_pos[0] < 0.0)

    if defensive_flag and len(adversaries) > 0:
        # threat measured by closeness to predicted ball
        adv_vals = [torch.norm(adv['pos'] - predicted_ball) for adv in adversaries]
        adv_idx = int(torch.argmin(torch.stack(adv_vals)).item())
        threat = adversaries[adv_idx]
        threat_pred = threat['pos'] + threat['vel'] * t_opt
        block_point = 0.5 * (threat_pred + torch.zeros_like(threat_pred))
        # blend block point in
        intercept_point = 0.55 * intercept_point + 0.45 * block_point

    # If teammate and ego are very close in ability to intercept, adopt complementary lateral offset so
    # the two agents approach from different sides (reduces oscillation / both going same path)
    if abs(team_time - ego_time) < 0.35:
        to_goal = -predicted_ball
        if torch.norm(to_goal) > 1e-6:
            perp = torch.tensor([-to_goal[1], to_goal[0]], dtype=obs.dtype, device=device)
            perp = perp / (torch.norm(perp) + 1e-6)
            side_sign = torch.sign(torch.dot((agent_pos - teammate['pos']), perp))
            if side_sign == 0:
                # deterministic tie-break
                side_sign = 1.0 if (agent_pos[0] >= teammate['pos'][0]) else -1.0
            lateral = perp * (0.45 * side_sign)
            intercept_point = predicted_ball + lateral

    # Compute force toward intercept_point
    to_target = intercept_point - agent_pos
    dist_to_target = torch.norm(to_target)
    if dist_to_target < 1e-6:
        move_dir = torch.zeros_like(to_target)
    else:
        move_dir = to_target / (dist_to_target + 1e-6)

    # Base magnitude: stronger when further, gentle when near. Keep below max_force.
    # Use monotonic map: mag = max_force * (1 - exp(-d / 1.2)) but leave headroom for repulsion/damping
    mag = (1.0 - torch.exp(-dist_to_target / (1.2 + 1e-6))).item() * 0.9 * max_force
    base_force = move_dir * float(mag)

    # Small predictive velocity match: if intercept requires leading, add component along ball velocity
    lead = 0.12 * (ball_vel / (torch.norm(ball_vel) + 1e-6))
    lead_component = lead * float(min(0.5, dist_to_target.item()))

    # Repulsive forces from nearby agents (small)
    repulse = torch.zeros(2, dtype=obs.dtype, device=device)
    # teammate
    d_team = torch.norm(agent_pos - teammate['pos']).item()
    if d_team < repel_radius and d_team > 1e-6:
        dir_away = (agent_pos - teammate['pos']) / (d_team + 1e-6)
        repulse += dir_away * repel_strength * (1.0 - d_team / repel_radius) ** 2
    # adversaries
    for adv in adversaries:
        d_adv = torch.norm(agent_pos - adv['pos']).item()
        if d_adv < repel_radius and d_adv > 1e-6:
            dir_away = (agent_pos - adv['pos']) / (d_adv + 1e-6)
            repulse += dir_away * (repel_strength * 1.1) * (1.0 - d_adv / repel_radius) ** 2

    # Damping opposite to current velocity to reduce oscillations
    damping = -0.18 * agent_vel

    force = base_force + lead_component + repulse + damping

    # Clip force magnitude to max_force
    force_mag = torch.norm(force)
    if force_mag > max_force:
        force = force / (force_mag + 1e-9) * max_force

    return force.unsqueeze(0)

import torch

def agent1_policy(obs):
    """
    Agent1 policy coordinated to work with the provided partner (agent0).

    Strategy (coordination specifics):
    - Use the same anticipatory interception idea as the partner (search over short horizon t
      for the predicted ball position that best matches agent travel time).
    - If the teammate (partner) can reach the predicted intercept noticeably sooner, yield and
      move to a supporting intercept point between the ball and the goal that opens a passing
      lane. When yielding, choose the lateral offset on the opposite side of the teammate so
      the two agents open complementary lanes.
    - If we are closer to the intercept (or only marginally slower), commit to the intercept.
    - If both are very close in ability to intercept, adopt complementary lateral offsets so
      agents approach from different sides (reducing collisions/oscillations).
    - Respect defensive fallback: if the ball is on our defensive half, bias to block the
      most threatening adversary.
    - Small repulsive terms from nearby agents and light velocity damping.
    - Clip action magnitude to <= 1.0 and return shape (1,2).
    """

    obs = obs.float()
    device = obs.device

    # Parsing layout consistent with observation_base and partner implementation
    adversary_count = 2
    adversary_block = 8 * adversary_count
    teammate_block = 8  # single teammate
    tail_fixed = 2 + 2 + adversary_block + teammate_block

    total_len = obs.numel()
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

    # Parse relevant fields from obs_block (ordering follows observation_base)
    # [agent_force(2), agent_pos - ball_pos(2), agent_vel - ball_vel(2), ball_pos - goal_pos(2), ball_vel(2), ball_force(2), agent_rot(...)]
    ball_pos_rel_goal = obs_block[6:8].clone()
    ball_vel = obs_block[8:10].clone()

    # Coordinates are in goal-centered frame (goal at origin)
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

    # Parse teammate (single teammate) - this is partner agent0
    tchunk = teammates_block[0:8]
    rel_agent_minus_teamp = tchunk[0:2]
    team_vel = tchunk[4:6]
    team_pos_rel_goal = agent_pos - rel_agent_minus_teamp
    teammate = {"pos": team_pos_rel_goal, "vel": team_vel}

    # Parameters
    max_horizon = 3.0
    nominal_speed = 3.0
    repel_radius = 1.0
    repel_strength = 0.5
    max_force = 1.0

    # Choose intercept time (same scheme as partner)
    candidates = torch.linspace(0.0, max_horizon, steps=9, device=device)
    if torch.norm(ball_vel) < 1e-6:
        t_opt = torch.tensor(0.0, device=device)
    else:
        best_t = candidates[0]
        best_score = float('inf')
        for t_c in candidates:
            pred_ball = ball_pos + ball_vel * t_c
            dist = torch.norm(pred_ball - agent_pos)
            t_agent = dist / (nominal_speed + 1e-6)
            score = torch.abs(t_agent - t_c).item()
            if score < best_score - 1e-6:
                best_score = score
                best_t = t_c
        t_opt = best_t

    predicted_ball = ball_pos + ball_vel * t_opt

    # Compute times
    ego_dist = torch.norm(predicted_ball - agent_pos)
    ego_time = (ego_dist / (nominal_speed + 1e-6)).item()

    team_dist = torch.norm(predicted_ball - teammate['pos'])
    team_time = (team_dist / (nominal_speed + 1e-6)).item()

    adv_times = []
    for adv in adversaries:
        adv_dist = torch.norm(predicted_ball - adv['pos']).item()
        adv_vel_norm = torch.norm(adv['vel']).item()
        if adv_vel_norm > 1e-6:
            to_point = (predicted_ball - adv['pos'])
            to_point_norm = torch.norm(to_point).item()
            if to_point_norm > 1e-6:
                align = (torch.dot(adv['vel'], to_point) / (adv_vel_norm * to_point_norm)).item()
            else:
                align = 0.0
        else:
            align = 0.0
        adv_time = max(0.0, adv_dist / (nominal_speed + 1e-6) - 0.25 * max(0.0, align))
        adv_times.append(adv_time)

    # Decide intercept_point, coordinating with teammate
    intercept_point = predicted_ball.clone()

    # Determine if teammate is actively converging onto predicted_ball (velocity alignment)
    team_vel_norm = torch.norm(teammate['vel']).item()
    team_align = 0.0
    to_pred_team = (predicted_ball - teammate['pos'])
    to_pred_team_norm = torch.norm(to_pred_team).item()
    if team_vel_norm > 1e-6 and to_pred_team_norm > 1e-6:
        team_align = (torch.dot(teammate['vel'], to_pred_team) / (team_vel_norm * to_pred_team_norm)).item()

    # Yield/support logic: if teammate is clearly better (or clearly converging) yield
    yield_margin = 0.5
    if (team_time + yield_margin < ego_time) or (team_align > 0.55 and team_time + 0.25 < ego_time):
        # supporting intercept: between predicted_ball and goal with lateral offset
        to_goal = -predicted_ball
        support_base = 0.55 * predicted_ball  # slightly different blend than partner to diversify
        if torch.norm(to_goal) > 1e-6:
            perp = torch.tensor([-to_goal[1], to_goal[0]], dtype=obs.dtype, device=device)
            perp = perp / (torch.norm(perp) + 1e-6)
            # choose side opposite to teammate so lanes open complementarily
            side_val = torch.dot((teammate['pos'] - predicted_ball), perp).item()
            side = -1.0 if side_val >= 0.0 else 1.0
            lateral = perp * (0.6 * side)
            support_point = support_base + lateral
        else:
            support_point = support_base
        intercept_point = support_point
    else:
        # If an adversary can reach the intercept notably sooner and teammate is farther, bias toward goal
        if len(adv_times) > 0:
            min_adv_time = min(adv_times)
            adv_margin = 0.4
            # if adversary is threatening AND teammate not clearly closer, bias to cut them off
            if min_adv_time + adv_margin < ego_time and not (team_time + 0.6 < ego_time):
                alpha = 0.36
                intercept_point = predicted_ball + (-predicted_ball) * alpha

    # Defensive fallback: if ball is on our defensive half (ball_pos.x < 0) bias to block
    defensive_flag = False
    try:
        defensive_flag = (ball_pos[0].item() < 0.0)
    except Exception:
        defensive_flag = (ball_pos[0] < 0.0)

    if defensive_flag and len(adversaries) > 0:
        adv_vals = [torch.norm(adv['pos'] - predicted_ball) for adv in adversaries]
        adv_idx = int(torch.argmin(torch.stack(adv_vals)).item())
        threat = adversaries[adv_idx]
        threat_pred = threat['pos'] + threat['vel'] * t_opt
        block_point = 0.5 * (threat_pred + torch.zeros_like(threat_pred))
        intercept_point = 0.5 * intercept_point + 0.5 * block_point

    # Complementary lateral if both close in ability
    if abs(team_time - ego_time) < 0.35:
        to_goal = -predicted_ball
        if torch.norm(to_goal) > 1e-6:
            perp = torch.tensor([-to_goal[1], to_goal[0]], dtype=obs.dtype, device=device)
            perp = perp / (torch.norm(perp) + 1e-6)
            # pick side opposite to teammate relative sign so we approach from different sides
            side_sign = torch.sign(torch.dot((agent_pos - teammate['pos']), perp))
            if side_sign == 0:
                side_sign = 1.0 if (agent_pos[0] >= teammate['pos'][0]) else -1.0
            # invert sign compared to partner's tie-break so lanes complement
            side_sign = -side_sign
            lateral = perp * (0.48 * side_sign)
            intercept_point = predicted_ball + lateral

    # Compute movement force toward intercept_point
    to_target = intercept_point - agent_pos
    dist_to_target = torch.norm(to_target)
    if dist_to_target < 1e-6:
        move_dir = torch.zeros_like(to_target)
    else:
        move_dir = to_target / (dist_to_target + 1e-6)

    mag = (1.0 - torch.exp(-dist_to_target / (1.2 + 1e-6))).item() * 0.9 * max_force
    base_force = move_dir * float(mag)

    # small lead toward ball motion to anticipate
    lead = 0.12 * (ball_vel / (torch.norm(ball_vel) + 1e-6))
    lead_component = lead * float(min(0.5, dist_to_target.item()))

    # Repulsive forces
    repulse = torch.zeros(2, dtype=obs.dtype, device=device)
    d_team = torch.norm(agent_pos - teammate['pos']).item()
    if d_team < repel_radius and d_team > 1e-6:
        dir_away = (agent_pos - teammate['pos']) / (d_team + 1e-6)
        repulse += dir_away * repel_strength * (1.0 - d_team / repel_radius) ** 2
    for adv in adversaries:
        d_adv = torch.norm(agent_pos - adv['pos']).item()
        if d_adv < repel_radius and d_adv > 1e-6:
            dir_away = (agent_pos - adv['pos']) / (d_adv + 1e-6)
            repulse += dir_away * (repel_strength * 1.1) * (1.0 - d_adv / repel_radius) ** 2

    damping = -0.18 * agent_vel

    force = base_force + lead_component + repulse + damping

    force_mag = torch.norm(force)
    if force_mag > max_force:
        force = force / (force_mag + 1e-9) * max_force

    return force.unsqueeze(0)