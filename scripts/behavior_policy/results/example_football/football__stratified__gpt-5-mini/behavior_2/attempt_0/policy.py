import torch

def agent0_policy(obs):
    """
    Reactive Support / Zone Maintainer policy for agent0.
    Input: obs: 1D torch tensor (observation_dim,)
    Output: torch tensor shape (1,2) - force to apply

    Logic summary (implemented):
    - Parse the observation vector into meaningful relative positions and velocities
      using the documented observation layout.
    - Compute a desired support position located along the line from ball toward
      opponent goal: desired = interpolate(ball, goal, alpha).
    - Add a lateral (perpendicular) offset to create a passing angle; choose the
      sign so the offset increases distance from the nearest defender.
    - If the ball is judged in our defensive half (heuristic) or defenders are
      advancing toward our goal, compute a blocking (drop-back) position between
      the ball and an estimated own-goal location.
    - Respect teammate proximity: if teammate is in close contest for the ball,
      move slightly further away to avoid collision; otherwise move a bit closer
      to present as a passing option.
    - Return a force = kp*(desired - agent_pos) - kd*agent_vel, clamped to a
      maximum magnitude.
    """

    # Ensure 1D
    if obs.dim() > 1:
        obs = obs.view(-1)
    total_len = obs.shape[0]

    # Known block sizes based on environment description
    pos_len = 2
    vel_len = 2
    # adversaries: 2 adversaries * (agent_pos-adversary_pos(2) + agent_vel-adversary_vel(2) + adversary_vel(2) + adversary_force(2)) = 16
    adv_len = 16
    # teammates: 1 teammate * (agent_pos-teammate_pos(2) + agent_vel-teammate_vel(2) + teammate_vel(2) + teammate_force(2)) = 8
    team_len = 8

    obs_chunk_len = total_len - (pos_len + vel_len + adv_len + team_len)
    # Inside obs chunk we expect:
    # [agent_force(2), agent_pos - ball_pos(2), agent_vel - ball_vel(2), ball_pos - goal_pos(2),
    #  ball_vel(2), ball_force(2), agent_rot(rot_dim)]
    # The fixed part before agent_rot is 12 dims; compute rot_dim dynamically
    fixed_before_rot = 12
    rot_dim = max(0, obs_chunk_len - fixed_before_rot)

    # Offsets
    idx = 0
    agent_force = obs[idx:idx+2]; idx += 2
    agent_pos_minus_ball = obs[idx:idx+2]; idx += 2
    agent_vel_minus_ball = obs[idx:idx+2]; idx += 2
    ball_pos_minus_goal = obs[idx:idx+2]; idx += 2
    ball_vel = obs[idx:idx+2]; idx += 2
    ball_force = obs[idx:idx+2]; idx += 2
    agent_rot = obs[idx: idx+rot_dim]; idx += rot_dim

    # pos (agent_pos - goal_pos)
    agent_pos_minus_goal = obs[idx: idx+pos_len]; idx += pos_len
    # vel (agent_vel)
    agent_vel = obs[idx: idx+vel_len]; idx += vel_len
    # adversaries block
    adversary_block = obs[idx: idx+adv_len]; idx += adv_len
    # teammates block
    teammate_block = obs[idx: idx+team_len]; idx += team_len

    # Recover adversary positions relative to goal and adversary velocities
    # adversary_block layout per adversary: [agent_pos - adv_pos (2), agent_vel - adv_vel (2), adv_vel (2), adv_force (2)]
    # Two adversaries concatenated
    advs = []
    num_adv = adv_len // 8
    for i in range(num_adv):
        b = adversary_block[i*8:(i+1)*8]
        agent_pos_minus_adv = b[0:2]
        agent_vel_minus_adv = b[2:4]
        adv_vel = b[4:6]
        adv_force = b[6:8]
        # adv_pos_minus_goal = adv_pos - goal = agent_pos - goal - (agent_pos - adv_pos)
        adv_pos_minus_goal = agent_pos_minus_goal - agent_pos_minus_adv
        advs.append({
            'pos_rel_goal': adv_pos_minus_goal,
            'vel': adv_vel,
            'agent_pos_minus_adv': agent_pos_minus_adv,
        })

    # Recover teammate info (one teammate expected)
    teammate = None
    if team_len >= 8:
        b = teammate_block[0:8]
        agent_pos_minus_teammate = b[0:2]
        agent_vel_minus_teammate = b[2:4]
        teammate_vel = b[4:6]
        teammate_force = b[6:8]
        teammate_pos_minus_goal = agent_pos_minus_goal - agent_pos_minus_teammate
        teammate = {
            'pos_rel_goal': teammate_pos_minus_goal,
            'vel': teammate_vel,
            'agent_pos_minus_teammate': agent_pos_minus_teammate,
        }

    # All positions are in coordinates relative to opponent goal (goal_pos chosen in obs)
    # agent_pos_minus_goal, ball_pos_minus_goal, adversary pos are in same frame.
    # We will compute desired position in this same frame.

    # Parameters
    alpha_base = 0.35  # interpolation factor toward goal (0 = ball, 1 = goal)
    offering_alpha = 0.5
    contest_dist = 1.0
    perp_scale_factor = 0.35
    kp = 5.0
    kd = 1.2
    max_force = 4.0

    # Compute desired basic position: interpolation between ball and goal
    # Goal in this frame is at (0,0). So desired_rel_goal = (1 - alpha) * ball_pos_rel_goal
    alpha = alpha_base

    # If teammate is closer to ball, act as offering (move forward toward goal)
    if teammate is not None:
        # compute teammate distance to ball
        teammate_pos_rel_goal = teammate['pos_rel_goal']
        teammate_pos_rel_ball = teammate_pos_rel_goal - ball_pos_minus_goal
        teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)
        if teammate_dist_to_ball < contest_dist:
            # teammate contesting: increase alpha a bit to present as forward option
            alpha = offering_alpha
        else:
            alpha = alpha_base

    desired_rel_goal = (1.0 - alpha) * ball_pos_minus_goal

    # Compute perpendicular offset away from nearest defender
    # Find nearest defender to the candidate desired position (in rel-goal frame)
    if len(advs) > 0:
        def_dists = []
        for adv in advs:
            d = adv['pos_rel_goal'] - desired_rel_goal
            def_dists.append(torch.norm(d))
        def_dists = torch.stack(def_dists)
        nearest_idx = torch.argmin(def_dists)
        nearest_adv = advs[nearest_idx]

        # Vector from defender to desired
        v = desired_rel_goal - nearest_adv['pos_rel_goal']
        v_norm = torch.norm(v)
        if v_norm.item() < 1e-6:
            # if coincident, pick a small arbitrary vector
            v = torch.tensor([1.0, 0.0], dtype=obs.dtype)
            v_norm = torch.norm(v)
        # perpendicular
        perp = torch.tensor([-v[1], v[0]], dtype=obs.dtype)
        perp_norm = torch.norm(perp)
        if perp_norm.item() > 1e-6:
            perp = perp / perp_norm
        else:
            perp = torch.zeros_like(perp)

        # Choose sign that increases distance to the defender
        cand1 = desired_rel_goal + perp
        cand2 = desired_rel_goal - perp
        d1 = torch.norm(cand1 - nearest_adv['pos_rel_goal'])
        d2 = torch.norm(cand2 - nearest_adv['pos_rel_goal'])
        if d1 > d2:
            chosen_perp = perp
        else:
            chosen_perp = -perp

        # Offset magnitude proportional to ball-goal distance, clamped
        ball_goal_dist = torch.norm(ball_pos_minus_goal)
        offset_mag = torch.clamp(perp_scale_factor * ball_goal_dist, 0.3, 1.5)
        desired_rel_goal = desired_rel_goal + chosen_perp * offset_mag

    # Defensive condition: heuristic - if ball is on our half (ball is left of midpoint toward our goal)
    # Since goal_pos in the obs is opponent goal, ball_pos_minus_goal[0] < 0 indicates ball is left of opponent goal
    defensive = False
    try:
        if ball_pos_minus_goal[0].item() < -0.5:
            defensive = True
    except Exception:
        # fallback if scalar access fails
        if ball_pos_minus_goal[0] < -0.5:
            defensive = True

    # Also treat as defensive if any adversary is advancing toward our goal (approx):
    # check dot(adv_vel, -adv_pos_rel_goal) > threshold meaning moving toward our side
    for adv in advs:
        adv_pos = adv['pos_rel_goal']
        adv_vel = adv['vel']
        toward_our_goal = torch.dot(adv_vel, -adv_pos)
        if toward_our_goal.item() > 0.5:
            defensive = True
            break

    if defensive:
        # Drop back to a central blocking position between ball and an estimated own goal
        # We approximate our goal relative to opponent goal as some negative x position
        our_goal_rel_goal = torch.tensor([-8.0, 0.0], dtype=obs.dtype)
        desired_rel_goal = 0.5 * (ball_pos_minus_goal + our_goal_rel_goal)

    # Teammate contest avoidance: if teammate near ball, increase distance from ball
    if teammate is not None:
        teammate_pos_rel_goal = teammate['pos_rel_goal']
        teammate_pos_rel_ball = teammate_pos_rel_goal - ball_pos_minus_goal
        teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)
        agent_pos_rel_ball = agent_pos_minus_goal - ball_pos_minus_goal
        if teammate_dist_to_ball < contest_dist:
            # move away from ball a bit
            away_dir = agent_pos_rel_ball
            nd = torch.norm(away_dir)
            if nd.item() < 1e-6:
                away_dir = torch.tensor([1.0, 0.0], dtype=obs.dtype)
                nd = torch.norm(away_dir)
            away_dir = away_dir / nd
            desired_rel_goal = desired_rel_goal + away_dir * 0.8
        else:
            # present as pass option: nudge a bit toward ball
            toward_ball = ball_pos_minus_goal - agent_pos_minus_goal
            nt = torch.norm(toward_ball)
            if nt.item() > 1e-6:
                desired_rel_goal = desired_rel_goal - 0.15 * (desired_rel_goal - (agent_pos_minus_goal + 0.2 * toward_ball / nt))

    # Compute action: force toward desired with damping on velocity
    delta = desired_rel_goal - agent_pos_minus_goal
    force = kp * delta - kd * agent_vel

    # Clamp magnitude
    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force * (max_force / fmag)

    # Return as shape (1,2)
    return force.view(1, 2)

import torch

def agent1_policy(obs):
    """
    Coordinating Reactive Support / Zone Maintainer for agent1.

    Strategy to coordinate with partner (agent0):
    - Parse the same observation layout as the partner policy.
    - Compute a candidate desired support position along the line from ball to
      opponent goal (same interpolation idea as partner).
    - Compute a lateral (perpendicular) offset from that candidate using the
      nearest defender, but break symmetry intentionally: choose the perpendicular
      direction that moves our desired position closer to our teammate's current
      position (so both agents converge to the same landmark). If no teammate
      info is available, fall back to moving away from the nearest defender
      like the partner.
    - Respect teammate contest heuristics (if teammate is near the ball, back
      off to avoid collision; otherwise nudge forward to present a pass).
    - If a defensive situation is detected (ball in defensive half or an
      opponent advancing), drop back to a central blocking position.
    - Output is force = kp*(desired - agent_pos) - kd*agent_vel, clamped.

    Input: obs: 1D torch tensor (observation_dim,)
    Output: torch tensor shape (1,2)
    """

    # Ensure 1D
    if obs.dim() > 1:
        obs = obs.view(-1)
    total_len = obs.shape[0]

    # Known block sizes
    pos_len = 2
    vel_len = 2
    adv_len = 16  # 2 adversaries * 8
    team_len = 8  # 1 teammate * 8

    obs_chunk_len = total_len - (pos_len + vel_len + adv_len + team_len)
    fixed_before_rot = 12
    rot_dim = max(0, obs_chunk_len - fixed_before_rot)

    idx = 0
    agent_force = obs[idx:idx+2]; idx += 2
    agent_pos_minus_ball = obs[idx:idx+2]; idx += 2
    agent_vel_minus_ball = obs[idx:idx+2]; idx += 2
    ball_pos_minus_goal = obs[idx:idx+2]; idx += 2
    ball_vel = obs[idx:idx+2]; idx += 2
    ball_force = obs[idx:idx+2]; idx += 2
    agent_rot = obs[idx: idx+rot_dim]; idx += rot_dim

    # pos (agent_pos - goal_pos)
    agent_pos_minus_goal = obs[idx: idx+pos_len]; idx += pos_len
    # vel (agent_vel)
    agent_vel = obs[idx: idx+vel_len]; idx += vel_len
    # adversaries block
    adversary_block = obs[idx: idx+adv_len]; idx += adv_len
    # teammates block
    teammate_block = obs[idx: idx+team_len]; idx += team_len

    # Recover adversaries
    advs = []
    num_adv = adv_len // 8
    for i in range(num_adv):
        b = adversary_block[i*8:(i+1)*8]
        agent_pos_minus_adv = b[0:2]
        agent_vel_minus_adv = b[2:4]
        adv_vel = b[4:6]
        adv_force = b[6:8]
        adv_pos_minus_goal = agent_pos_minus_goal - agent_pos_minus_adv
        advs.append({
            'pos_rel_goal': adv_pos_minus_goal,
            'vel': adv_vel,
            'agent_pos_minus_adv': agent_pos_minus_adv,
        })

    # Recover teammate (if present)
    teammate = None
    if team_len >= 8:
        b = teammate_block[0:8]
        agent_pos_minus_teammate = b[0:2]
        agent_vel_minus_teammate = b[2:4]
        teammate_vel = b[4:6]
        teammate_force = b[6:8]
        teammate_pos_rel_goal = agent_pos_minus_goal - agent_pos_minus_teammate
        teammate = {
            'pos_rel_goal': teammate_pos_rel_goal,
            'vel': teammate_vel,
            'agent_pos_minus_teammate': agent_pos_minus_teammate,
        }

    # Parameters (similar to partner but with a coordination tie-break)
    alpha_base = 0.35
    offering_alpha = 0.5
    contest_dist = 1.0
    perp_scale_factor = 0.35
    kp = 5.0
    kd = 1.2
    max_force = 4.0

    # Base desired position (in frame relative to opponent goal)
    alpha = alpha_base

    # If teammate is closer to ball, present as offering
    if teammate is not None:
        teammate_pos_rel_goal = teammate['pos_rel_goal']
        teammate_pos_rel_ball = teammate_pos_rel_goal - ball_pos_minus_goal
        teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)
        if teammate_dist_to_ball < contest_dist:
            alpha = offering_alpha
        else:
            alpha = alpha_base

    desired_rel_goal = (1.0 - alpha) * ball_pos_minus_goal

    # Compute perpendicular offset using nearest defender, but choose sign to
    # move desired position closer to teammate (to converge to same landmark).
    if len(advs) > 0:
        def_dists = []
        for adv in advs:
            d = adv['pos_rel_goal'] - desired_rel_goal
            def_dists.append(torch.norm(d))
        def_dists = torch.stack(def_dists)
        nearest_idx = torch.argmin(def_dists)
        nearest_adv = advs[nearest_idx]

        v = desired_rel_goal - nearest_adv['pos_rel_goal']
        v_norm = torch.norm(v)
        if v_norm.item() < 1e-6:
            v = torch.tensor([1.0, 0.0], dtype=obs.dtype, device=obs.device)
            v_norm = torch.norm(v)
        perp = torch.tensor([-v[1], v[0]], dtype=obs.dtype, device=obs.device)
        perp_norm = torch.norm(perp)
        if perp_norm.item() > 1e-6:
            perp = perp / perp_norm
        else:
            perp = torch.zeros_like(perp)

        # Candidate offsets
        cand1 = desired_rel_goal + perp
        cand2 = desired_rel_goal - perp

        choose_by_teammate = False
        if teammate is not None:
            # pick the candidate that is closer to teammate's current position
            tpos = teammate['pos_rel_goal']
            d1 = torch.norm(cand1 - tpos)
            d2 = torch.norm(cand2 - tpos)
            if d1 < d2:
                chosen_perp = perp
            else:
                chosen_perp = -perp
            choose_by_teammate = True
        else:
            # fallback: choose the sign that increases distance from defender
            d1 = torch.norm(cand1 - nearest_adv['pos_rel_goal'])
            d2 = torch.norm(cand2 - nearest_adv['pos_rel_goal'])
            if d1 > d2:
                chosen_perp = perp
            else:
                chosen_perp = -perp

        # Offset magnitude scaled with ball-goal distance, clamped
        ball_goal_dist = torch.norm(ball_pos_minus_goal)
        offset_mag = torch.clamp(perp_scale_factor * ball_goal_dist, 0.2, 1.2)
        # If we purposely chose to move toward teammate, use slightly larger offset
        if choose_by_teammate:
            offset_mag = offset_mag * 1.05

        desired_rel_goal = desired_rel_goal + chosen_perp * offset_mag

    # Defensive detection (similar heuristic): ball far toward our side or advs moving in
    defensive = False
    try:
        if ball_pos_minus_goal[0].item() < -0.5:
            defensive = True
    except Exception:
        if ball_pos_minus_goal[0] < -0.5:
            defensive = True

    for adv in advs:
        adv_pos = adv['pos_rel_goal']
        adv_vel = adv['vel']
        toward_our_goal = torch.dot(adv_vel, -adv_pos)
        if toward_our_goal.item() > 0.5:
            defensive = True
            break

    if defensive:
        our_goal_rel_goal = torch.tensor([-8.0, 0.0], dtype=obs.dtype, device=obs.device)
        desired_rel_goal = 0.5 * (ball_pos_minus_goal + our_goal_rel_goal)

    # Teammate contest avoidance or presenting option
    if teammate is not None:
        teammate_pos_rel_goal = teammate['pos_rel_goal']
        teammate_pos_rel_ball = teammate_pos_rel_goal - ball_pos_minus_goal
        teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)
        agent_pos_rel_ball = agent_pos_minus_goal - ball_pos_minus_goal
        if teammate_dist_to_ball < contest_dist:
            # move away from ball a bit to avoid collision
            away_dir = agent_pos_rel_ball
            nd = torch.norm(away_dir)
            if nd.item() < 1e-6:
                away_dir = torch.tensor([1.0, 0.0], dtype=obs.dtype, device=obs.device)
                nd = torch.norm(away_dir)
            away_dir = away_dir / nd
            desired_rel_goal = desired_rel_goal + away_dir * 0.8
        else:
            # nudge to present as pass option (small tweak)
            toward_ball = ball_pos_minus_goal - agent_pos_minus_goal
            nt = torch.norm(toward_ball)
            if nt.item() > 1e-6:
                desired_rel_goal = desired_rel_goal - 0.12 * (desired_rel_goal - (agent_pos_minus_goal + 0.2 * toward_ball / nt))

    # Compute force with damping and clamp
    delta = desired_rel_goal - agent_pos_minus_goal
    force = kp * delta - kd * agent_vel

    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force * (max_force / fmag)

    return force.view(1, 2)