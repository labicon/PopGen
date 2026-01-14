import torch

def agent0_policy(obs):
    """
    Reactive Support / Zone Maintainer policy for agent0.

    Input: obs: 1D torch tensor (observation_dim,)
    Output: torch tensor shape (1,2) - force to apply (clamped to [-1,1] magnitude)

    Behavior implemented:
    - Compute a desired support position between the ball and the opponent goal:
        desired = (1 - alpha) * ball + alpha * goal  (converted into goal-relative frame)
    - If our teammate is closer to the ball than we are -> increase alpha to move forward and offer.
    - Add a perpendicular lateral offset away from nearest defender to create a passing angle.
    - If the ball is closer to our own goal than the opponent goal OR an adversary is clearly moving toward our goal,
      drop back to a central blocking position between ball and our goal (y set to 0 to keep it central).
    - If teammate is in close contest (very near the ball), move a bit away to avoid collision; otherwise nudge
      closer to present a pass option.
    - Output is a PD-style force: kp*(desired - pos) - kd*vel, clamped to the environment action limit (1.0).
    """

    # Flatten observation
    if obs.dim() > 1:
        obs = obs.view(-1)
    total_len = obs.shape[0]

    # Known block sizes from observation_base
    pos_len = 2
    vel_len = 2
    adv_len = 16  # 2 adversaries * 8
    team_len = 8  # 1 teammate * 8

    # Compute rotation dim from total length
    obs_chunk_len = total_len - (pos_len + vel_len + adv_len + team_len)
    fixed_before_rot = 12
    rot_dim = max(0, obs_chunk_len - fixed_before_rot)

    idx = 0
    agent_force = obs[idx: idx + 2]; idx += 2
    agent_pos_minus_ball = obs[idx: idx + 2]; idx += 2
    agent_vel_minus_ball = obs[idx: idx + 2]; idx += 2
    ball_pos_minus_goal = obs[idx: idx + 2]; idx += 2
    ball_vel = obs[idx: idx + 2]; idx += 2
    ball_force = obs[idx: idx + 2]; idx += 2
    agent_rot = obs[idx: idx + rot_dim]; idx += rot_dim

    # agent pos relative to opponent goal and agent vel
    agent_pos_minus_goal = obs[idx: idx + pos_len]; idx += pos_len
    agent_vel = obs[idx: idx + vel_len]; idx += vel_len

    adversary_block = obs[idx: idx + adv_len]; idx += adv_len
    teammate_block = obs[idx: idx + team_len]; idx += team_len

    device = obs.device
    dtype = obs.dtype

    # Reconstruct adversaries (each block: agent_pos - adv_pos (2), agent_vel - adv_vel (2), adv_vel (2), adv_force (2))
    advs = []
    num_adv = adv_len // 8
    for i in range(num_adv):
        b = adversary_block[i * 8: (i + 1) * 8]
        agent_pos_minus_adv = b[0:2]
        agent_vel_minus_adv = b[2:4]
        adv_vel = b[4:6]
        adv_force = b[6:8]
        # adv position relative to opponent goal
        adv_pos_rel_goal = agent_pos_minus_goal - agent_pos_minus_adv
        advs.append({
            'pos_rel_goal': adv_pos_rel_goal,
            'vel': adv_vel,
            'agent_pos_minus_adv': agent_pos_minus_adv,
            'force': adv_force,
        })

    # Reconstruct teammate (single)
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
            'force': teammate_force,
        }

    # Parameters
    alpha_base = 0.35
    offering_alpha = 0.55
    contest_dist = 0.9
    perp_scale_factor = 0.35
    kp = 2.8
    kd = 0.9
    max_force = 1.0

    # distances to ball
    my_pos_rel_ball = agent_pos_minus_goal - ball_pos_minus_goal
    my_dist_to_ball = torch.norm(my_pos_rel_ball)

    teammate_dist_to_ball = None
    if teammate is not None:
        teammate_pos_rel_goal = teammate['pos_rel_goal']
        teammate_pos_rel_ball = teammate_pos_rel_goal - ball_pos_minus_goal
        teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)

    # Decide alpha: if teammate is closer to the ball than we are -> take the offering role (move forward toward goal)
    alpha = alpha_base
    if teammate is not None and teammate_dist_to_ball is not None and my_dist_to_ball.item() > 0.0:
        if teammate_dist_to_ball.item() < my_dist_to_ball.item():
            alpha = offering_alpha
        else:
            alpha = alpha_base

    # Desired position between ball and opponent goal (in goal-relative coordinates)
    desired_rel_goal = (1.0 - alpha) * ball_pos_minus_goal.clone()

    # Lateral offset: find nearest defender and push the desired point laterally away from them
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
            v = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
            v_norm = torch.norm(v)
        perp = torch.tensor([-v[1], v[0]], dtype=dtype, device=device)
        perp_norm = torch.norm(perp)
        if perp_norm.item() > 1e-6:
            perp = perp / perp_norm
        else:
            perp = torch.zeros_like(perp)

        cand1 = desired_rel_goal + perp
        cand2 = desired_rel_goal - perp
        d1 = torch.norm(cand1 - nearest_adv['pos_rel_goal'])
        d2 = torch.norm(cand2 - nearest_adv['pos_rel_goal'])
        # choose the sign that increases distance from defender
        if d1 > d2:
            chosen_perp = perp
        else:
            chosen_perp = -perp

        ball_goal_dist = torch.norm(ball_pos_minus_goal)
        offset_mag = torch.clamp(perp_scale_factor * ball_goal_dist, 0.2, 1.0)
        desired_rel_goal = desired_rel_goal + chosen_perp * offset_mag

    # Defensive detection: determine whether ball is closer to our goal than to the opponent goal
    # approximate our goal relative to opponent goal (negative x)
    our_goal_rel_goal = torch.tensor([-8.0, 0.0], dtype=dtype, device=device)
    dist_ball_to_opp_goal = torch.norm(ball_pos_minus_goal)
    dist_ball_to_our_goal = torch.norm(our_goal_rel_goal - ball_pos_minus_goal)

    defensive = False
    if dist_ball_to_our_goal.item() < dist_ball_to_opp_goal.item():
        defensive = True

    # Also consider if any adversary is moving toward our goal (vector from adv to our goal)
    for adv in advs:
        adv_pos = adv['pos_rel_goal']
        adv_vel = adv['vel']
        vec_to_our_goal = our_goal_rel_goal - adv_pos
        toward_our_goal = torch.dot(adv_vel, vec_to_our_goal)
        # require a modest positive projection and that the adv is not already extremely far from our goal
        if toward_our_goal.item() > 0.25:
            defensive = True
            break

    if defensive:
        # Drop back to a central blocking position between ball and our goal; keep it central (y=0)
        desired_rel_goal = 0.5 * (ball_pos_minus_goal + our_goal_rel_goal)
        # centralize laterally
        desired_rel_goal = torch.stack([desired_rel_goal[0], torch.tensor(0.0, dtype=dtype, device=device)])

    # Teammate collision avoidance or presenting option
    if teammate is not None and teammate_dist_to_ball is not None:
        # If teammate is very close to ball -> increase distance to avoid collision
        if teammate_dist_to_ball.item() < contest_dist:
            away_dir = (agent_pos_minus_goal - ball_pos_minus_goal).clone()
            nd = torch.norm(away_dir)
            if nd.item() < 1e-6:
                away_dir = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
                nd = torch.norm(away_dir)
            away_dir = away_dir / nd
            desired_rel_goal = desired_rel_goal + away_dir * 0.8
        else:
            # Otherwise, present as a pass option: nudge slightly toward ball (small fraction)
            toward_ball = ball_pos_minus_goal - agent_pos_minus_goal
            nt = torch.norm(toward_ball)
            if nt.item() > 1e-6:
                nudge = 0.12 * (toward_ball / nt)
                desired_rel_goal = desired_rel_goal + nudge

    # Compute PD-style force toward desired position (both positions and velocities are in goal-relative coords)
    delta = desired_rel_goal - agent_pos_minus_goal
    force = kp * delta - kd * agent_vel

    # Clamp magnitude to environment limit
    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force * (max_force / (fmag + 1e-12))

    return force.view(1, 2)

import torch


def agent1_policy(obs):
    """
    Agent1 Reactive Support / Zone Maintainer tuned to coordinate with partner (agent0).

    Coordination heuristics added compared to a single reactive support policy:
    - When choosing a lateral offset, prefer the candidate that increases distance from BOTH the nearest defender and the teammate. This tends to place the two teammates on opposite sides of the ball->goal line (or at least avoid both choosing the same offset sign).
    - Use the same role-selection logic (offering vs base) based on which agent is closer to the ball, and avoid collision when teammate is in contest distance.

    Input: obs: 1D torch tensor (observation_dim,)
    Output: torch tensor shape (1,2) - force to apply (clamped to [-1,1] magnitude)
    """

    # Flatten observation
    if obs.dim() > 1:
        obs = obs.view(-1)
    total_len = obs.shape[0]

    # Known block sizes from observation_base
    pos_len = 2
    vel_len = 2
    adv_len = 16  # 2 adversaries * 8
    team_len = 8  # 1 teammate * 8

    # Compute rotation dim from total length
    obs_chunk_len = total_len - (pos_len + vel_len + adv_len + team_len)
    fixed_before_rot = 12
    rot_dim = max(0, obs_chunk_len - fixed_before_rot)

    idx = 0
    agent_force = obs[idx: idx + 2]; idx += 2
    agent_pos_minus_ball = obs[idx: idx + 2]; idx += 2
    agent_vel_minus_ball = obs[idx: idx + 2]; idx += 2
    ball_pos_minus_goal = obs[idx: idx + 2]; idx += 2
    ball_vel = obs[idx: idx + 2]; idx += 2
    ball_force = obs[idx: idx + 2]; idx += 2
    agent_rot = obs[idx: idx + rot_dim]; idx += rot_dim

    # agent pos relative to opponent goal and agent vel
    agent_pos_minus_goal = obs[idx: idx + pos_len]; idx += pos_len
    agent_vel = obs[idx: idx + vel_len]; idx += vel_len

    adversary_block = obs[idx: idx + adv_len]; idx += adv_len
    teammate_block = obs[idx: idx + team_len]; idx += team_len

    device = obs.device
    dtype = obs.dtype

    # Reconstruct adversaries (each block: agent_pos - adv_pos (2), agent_vel - adv_vel (2), adv_vel (2), adv_force (2))
    advs = []
    num_adv = adv_len // 8
    for i in range(num_adv):
        b = adversary_block[i * 8: (i + 1) * 8]
        agent_pos_minus_adv = b[0:2]
        agent_vel_minus_adv = b[2:4]
        adv_vel = b[4:6]
        adv_force = b[6:8]
        adv_pos_rel_goal = agent_pos_minus_goal - agent_pos_minus_adv
        advs.append({
            'pos_rel_goal': adv_pos_rel_goal,
            'vel': adv_vel,
            'agent_pos_minus_adv': agent_pos_minus_adv,
            'force': adv_force,
        })

    # Reconstruct teammate (single)
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
            'force': teammate_force,
        }

    # Parameters (tuned to match partner's style but with coordination heuristics)
    alpha_base = 0.35
    offering_alpha = 0.55
    contest_dist = 0.9
    perp_scale_factor = 0.35
    kp = 2.8
    kd = 0.9
    max_force = 1.0

    # distances to ball
    my_pos_rel_ball = agent_pos_minus_goal - ball_pos_minus_goal
    my_dist_to_ball = torch.norm(my_pos_rel_ball)

    teammate_dist_to_ball = None
    teammate_pos_rel_goal = None
    if teammate is not None:
        teammate_pos_rel_goal = teammate['pos_rel_goal']
        teammate_pos_rel_ball = teammate_pos_rel_goal - ball_pos_minus_goal
        teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)

    # Decide alpha: if teammate is closer to the ball than we are -> take the offering role (move forward toward goal)
    alpha = alpha_base
    if teammate is not None and teammate_dist_to_ball is not None and my_dist_to_ball.item() > 0.0:
        if teammate_dist_to_ball.item() < my_dist_to_ball.item():
            alpha = offering_alpha
        else:
            alpha = alpha_base

    # Desired position between ball and opponent goal (in goal-relative coordinates)
    desired_rel_goal = (1.0 - alpha) * ball_pos_minus_goal.clone()

    # Lateral offset: find nearest defender and push the desired point laterally away from them
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
            v = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
            v_norm = torch.norm(v)
        perp = torch.tensor([-v[1], v[0]], dtype=dtype, device=device)
        perp_norm = torch.norm(perp)
        if perp_norm.item() > 1e-6:
            perp = perp / perp_norm
        else:
            perp = torch.zeros_like(perp)

        cand1 = desired_rel_goal + perp
        cand2 = desired_rel_goal - perp

        # Evaluate candidates by combined distance to nearest defender and to teammate (if present).
        # We prefer the candidate that increases distance from both the defender and the teammate, which
        # tends to create opposing lateral offsets and reduce teammate collision.
        d1_def = torch.norm(cand1 - nearest_adv['pos_rel_goal'])
        d2_def = torch.norm(cand2 - nearest_adv['pos_rel_goal'])
        d1_team = torch.tensor(0.0, dtype=dtype, device=device)
        d2_team = torch.tensor(0.0, dtype=dtype, device=device)
        if teammate is not None:
            d1_team = torch.norm(cand1 - teammate_pos_rel_goal)
            d2_team = torch.norm(cand2 - teammate_pos_rel_goal)

        score1 = d1_def + d1_team
        score2 = d2_def + d2_team

        if score1 > score2:
            chosen_perp = perp
        else:
            chosen_perp = -perp

        ball_goal_dist = torch.norm(ball_pos_minus_goal)
        offset_mag = torch.clamp(perp_scale_factor * ball_goal_dist, 0.2, 1.0)
        desired_rel_goal = desired_rel_goal + chosen_perp * offset_mag

    # Defensive detection: determine whether ball is closer to our goal than to the opponent goal
    our_goal_rel_goal = torch.tensor([-8.0, 0.0], dtype=dtype, device=device)
    dist_ball_to_opp_goal = torch.norm(ball_pos_minus_goal)
    dist_ball_to_our_goal = torch.norm(our_goal_rel_goal - ball_pos_minus_goal)

    defensive = False
    if dist_ball_to_our_goal.item() < dist_ball_to_opp_goal.item():
        defensive = True

    # Also consider if any adversary is moving toward our goal
    for adv in advs:
        adv_pos = adv['pos_rel_goal']
        adv_vel = adv['vel']
        vec_to_our_goal = our_goal_rel_goal - adv_pos
        toward_our_goal = torch.dot(adv_vel, vec_to_our_goal)
        if toward_our_goal.item() > 0.25:
            defensive = True
            break

    if defensive:
        # Drop back to a central blocking position between ball and our goal; keep it central (y=0)
        desired_rel_goal = 0.5 * (ball_pos_minus_goal + our_goal_rel_goal)
        desired_rel_goal = torch.stack([desired_rel_goal[0], torch.tensor(0.0, dtype=dtype, device=device)])

    # Teammate collision avoidance or presenting option
    if teammate is not None and teammate_dist_to_ball is not None:
        # If teammate is very close to ball -> increase distance to avoid collision
        if teammate_dist_to_ball.item() < contest_dist:
            away_dir = (agent_pos_minus_goal - ball_pos_minus_goal).clone()
            nd = torch.norm(away_dir)
            if nd.item() < 1e-6:
                away_dir = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
                nd = torch.norm(away_dir)
            away_dir = away_dir / nd
            desired_rel_goal = desired_rel_goal + away_dir * 0.8
        else:
            # Otherwise, present as a pass option: nudge slightly toward ball (small fraction)
            toward_ball = ball_pos_minus_goal - agent_pos_minus_goal
            nt = torch.norm(toward_ball)
            if nt.item() > 1e-6:
                nudge = 0.12 * (toward_ball / nt)
                desired_rel_goal = desired_rel_goal + nudge

    # Compute PD-style force toward desired position
    delta = desired_rel_goal - agent_pos_minus_goal
    force = kp * delta - kd * agent_vel

    # Clamp magnitude to environment limit
    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force * (max_force / (fmag + 1e-12))

    return force.view(1, 2)