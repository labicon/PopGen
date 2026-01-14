import torch

def agent0_policy(obs):
    """
    Reactive Support / Zone Maintainer policy for agent0.

    Input: obs: 1D torch tensor (observation_dim,)
    Output: torch tensor shape (1,2) - force to apply (clamped to agent action range)

    Notes on fixes from prior attempt:
    - The environment enforces a maximum physical action magnitude of 1.0. We set
      max_force = 1.0 and choose gains so the commanded forces stay reasonable
      before final clamping.
    - All temporary tensors are created on the same device and dtype as obs.
    """

    # Ensure a flat 1D observation
    if obs.dim() > 1:
        obs = obs.view(-1)
    total_len = obs.shape[0]

    # Layout-derived block sizes
    pos_len = 2
    vel_len = 2
    adv_len = 16  # 2 adversaries * 8 features each
    team_len = 8  # 1 teammate * 8 features

    # Compute how many elements correspond to the initial chunk (obs fields + rot)
    obs_chunk_len = total_len - (pos_len + vel_len + adv_len + team_len)
    fixed_before_rot = 12  # known fixed fields before agent_rot
    rot_dim = max(0, obs_chunk_len - fixed_before_rot)

    idx = 0
    agent_force = obs[idx: idx + 2]; idx += 2
    agent_pos_minus_ball = obs[idx: idx + 2]; idx += 2
    agent_vel_minus_ball = obs[idx: idx + 2]; idx += 2
    ball_pos_minus_goal = obs[idx: idx + 2]; idx += 2
    ball_vel = obs[idx: idx + 2]; idx += 2
    ball_force = obs[idx: idx + 2]; idx += 2
    agent_rot = obs[idx: idx + rot_dim]; idx += rot_dim

    # pos and vel blocks
    agent_pos_minus_goal = obs[idx: idx + pos_len]; idx += pos_len
    agent_vel = obs[idx: idx + vel_len]; idx += vel_len

    # adversaries and teammates concatenated blocks
    adversary_block = obs[idx: idx + adv_len]; idx += adv_len
    teammate_block = obs[idx: idx + team_len]; idx += team_len

    device = obs.device
    dtype = obs.dtype

    # Reconstruct adversaries list (each 8-dim block)
    advs = []
    num_adv = adv_len // 8
    for i in range(num_adv):
        b = adversary_block[i * 8: (i + 1) * 8]
        agent_pos_minus_adv = b[0:2]
        agent_vel_minus_adv = b[2:4]
        adv_vel = b[4:6]
        adv_force = b[6:8]
        # adv_pos relative to opponent goal = agent_pos_rel_goal - (agent_pos - adv_pos)
        adv_pos_rel_goal = agent_pos_minus_goal - agent_pos_minus_adv
        advs.append({
            'pos_rel_goal': adv_pos_rel_goal,
            'vel': adv_vel,
            'agent_pos_minus_adv': agent_pos_minus_adv,
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
        }

    # Behaviour parameters (max_force set to agent's action limit)
    alpha_base = 0.35
    offering_alpha = 0.55
    contest_dist = 1.0
    perp_scale_factor = 0.35
    kp = 3.0   # positional gain
    kd = 0.8   # velocity damping
    max_force = 1.0  # environment action limit (fixed)

    # Basic desired position: interpolation from ball toward opponent goal (goal at origin in this frame)
    alpha = alpha_base
    if teammate is not None:
        teammate_pos_rel_goal = teammate['pos_rel_goal']
        teammate_pos_rel_ball = teammate_pos_rel_goal - ball_pos_minus_goal
        teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)
        if teammate_dist_to_ball.item() < contest_dist:
            # teammate is contesting the ball: offer forward by increasing interpolation toward goal
            alpha = offering_alpha

    desired_rel_goal = (1.0 - alpha) * ball_pos_minus_goal

    # Lateral offset to create a passing angle: choose perpendicular that moves away from nearest defender
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
        if d1 > d2:
            chosen_perp = perp
        else:
            chosen_perp = -perp

        ball_goal_dist = torch.norm(ball_pos_minus_goal)
        offset_mag = torch.clamp(perp_scale_factor * ball_goal_dist, 0.2, 1.0)
        desired_rel_goal = desired_rel_goal + chosen_perp * offset_mag

    # Defensive heuristic: if ball is on our half (heuristic) or defender advancing, drop back
    defensive = False
    # ball_pos_minus_goal[0] < 0 means ball is between us and opponent goal; negative beyond threshold -> our half
    if ball_pos_minus_goal.numel() >= 1 and ball_pos_minus_goal[0].item() < -0.5:
        defensive = True

    for adv in advs:
        adv_pos = adv['pos_rel_goal']
        adv_vel = adv['vel']
        toward_our_goal = torch.dot(adv_vel, -adv_pos)
        if toward_our_goal.item() > 0.5:
            defensive = True
            break

    if defensive:
        # approximate our goal relative to opponent goal and place central blocker between ball and our goal
        our_goal_rel_goal = torch.tensor([-8.0, 0.0], dtype=dtype, device=device)
        desired_rel_goal = 0.5 * (ball_pos_minus_goal + our_goal_rel_goal)

    # Teammate contest avoidance or offering: if teammate close to ball, move slightly away; else nudge to present pass
    if teammate is not None:
        teammate_pos_rel_goal = teammate['pos_rel_goal']
        teammate_pos_rel_ball = teammate_pos_rel_goal - ball_pos_minus_goal
        teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)
        agent_pos_rel_ball = agent_pos_minus_goal - ball_pos_minus_goal

        if teammate_dist_to_ball.item() < contest_dist:
            away_dir = agent_pos_rel_ball.clone()
            nd = torch.norm(away_dir)
            if nd.item() < 1e-6:
                away_dir = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
                nd = torch.norm(away_dir)
            away_dir = away_dir / nd
            desired_rel_goal = desired_rel_goal + away_dir * 0.6
        else:
            # small nudge toward presenting a pass: move a fraction toward the ball-goal line
            toward_ball = ball_pos_minus_goal - agent_pos_minus_goal
            nt = torch.norm(toward_ball)
            if nt.item() > 1e-6:
                # move a small fraction of the gap to appear as an option
                desired_rel_goal = desired_rel_goal - 0.10 * (desired_rel_goal - (agent_pos_minus_goal + 0.2 * toward_ball / nt))

    # Compute force toward desired with velocity damping
    delta = desired_rel_goal - agent_pos_minus_goal
    force = kp * delta - kd * agent_vel

    # Enforce action magnitude limit strictly
    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force * (max_force / (fmag + 1e-12))

    # Ensure output shape (1,2)
    return force.view(1, 2)

import torch


def agent1_policy(obs):
    """
    Agent 1 policy that coordinates with the given partner (agent0).

    Strategy / reasoning (implemented):
    - Parse the incoming observation in the same layout used by the environment and
      by the partner agent.
    - Reconstruct ball, adversary and teammate poses (relative to opponent goal) from
      the observation.
    - Predict the partner's desired zone position by re-implementing the partner's
      "Reactive Support / Zone Maintainer" decision logic but run from the partner's
      perspective (i.e. treating the teammate=us and agent=partner). This prediction
      uses the same heuristics: interpolation between ball and goal, lateral offset
      away from nearest defender, defensive drop-back, and offering/avoidance
      adjustments based on contest distance.
    - To ensure we meet the partner at the same landmark, we then move toward the
      partner's predicted desired position (in goal-relative coordinates). We apply
      a PD-like controller (kp, kd) and clamp the produced force to the environment
      action magnitude limit (max_force=1.0).

    This makes agent1 intentionally converge to the location partner prefers, which
    yields coordinated arrival at the same landmark while still reacting to the
    current ball/defender situation.
    """

    # Ensure a flat 1D observation
    if obs.dim() > 1:
        obs = obs.view(-1)
    total_len = obs.shape[0]

    # Layout-derived block sizes (match partner's expectations)
    pos_len = 2
    vel_len = 2
    adv_len = 16  # 2 adversaries * 8 features each
    team_len = 8  # 1 teammate * 8 features

    # Compute rot dimension similarly to partner parsing
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

    # pos and vel blocks (agent relative to goal and agent velocity)
    agent_pos_minus_goal = obs[idx: idx + pos_len]; idx += pos_len
    agent_vel = obs[idx: idx + vel_len]; idx += vel_len

    # adversaries and teammates concatenated blocks
    adversary_block = obs[idx: idx + adv_len]; idx += adv_len
    teammate_block = obs[idx: idx + team_len]; idx += team_len

    device = obs.device
    dtype = obs.dtype

    # Reconstruct adversaries list (each 8-dim block) -- compute adv_pos_rel_goal, adv_vel
    advs = []
    num_adv = adv_len // 8
    for i in range(num_adv):
        b = adversary_block[i * 8: (i + 1) * 8]
        agent_pos_minus_adv = b[0:2]
        agent_vel_minus_adv = b[2:4]
        adv_vel = b[4:6]
        adv_force = b[6:8]
        # adv_pos relative to opponent goal = agent_pos_minus_goal - (agent_pos - adv_pos)
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
        # teammate_pos_rel_goal = teammate_pos - goal = agent_pos_minus_goal - (agent_pos - teammate_pos)
        teammate_pos_rel_goal = agent_pos_minus_goal - agent_pos_minus_teammate
        teammate = {
            'pos_rel_goal': teammate_pos_rel_goal,
            'vel': teammate_vel,
            'agent_pos_minus_teammate': agent_pos_minus_teammate,
            'force': teammate_force,
        }

    # If teammate missing for some reason, fall back to a conservative behavior: move toward ball-goal line
    # But in our scenario teammate is expected to exist.

    # Helper: compute desired zone position using the partner's heuristic but parameterized by
    # a chosen "agent_pos_rel_goal" (so we can simulate the partner by passing teammate's pos_rel_goal)
    def compute_partner_like_desired(agent_pos_rel_goal, agent_vel_local, other_teammate_pos_rel_goal, other_teammate_vel):
        # Behavior parameters mirrored from partner implementation
        alpha_base = 0.35
        offering_alpha = 0.55
        contest_dist = 1.0
        perp_scale_factor = 0.35
        kp_local = 3.0
        kd_local = 0.8
        max_force_local = 1.0

        # Decide alpha based on whether partner's teammate is contesting the ball
        alpha = alpha_base
        if other_teammate_pos_rel_goal is not None:
            teammate_pos_rel_ball = other_teammate_pos_rel_goal - ball_pos_minus_goal
            teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)
            if teammate_dist_to_ball.item() < contest_dist:
                alpha = offering_alpha

        # Basic desired position (relative to opponent goal): interpolation from ball toward opponent goal
        desired_rel_goal = (1.0 - alpha) * ball_pos_minus_goal

        # Lateral offset away from nearest defender to create passing angle
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
            if d1 > d2:
                chosen_perp = perp
            else:
                chosen_perp = -perp

            ball_goal_dist = torch.norm(ball_pos_minus_goal)
            offset_mag = torch.clamp(perp_scale_factor * ball_goal_dist, 0.2, 1.0)
            desired_rel_goal = desired_rel_goal + chosen_perp * offset_mag

        # Defensive heuristic: if ball is on our half or a defender is advancing, drop back
        defensive = False
        if ball_pos_minus_goal.numel() >= 1 and ball_pos_minus_goal[0].item() < -0.5:
            defensive = True

        for adv in advs:
            adv_pos = adv['pos_rel_goal']
            adv_vel = adv['vel']
            toward_our_goal = torch.dot(adv_vel, -adv_pos)
            if toward_our_goal.item() > 0.5:
                defensive = True
                break

        if defensive:
            our_goal_rel_goal = torch.tensor([-8.0, 0.0], dtype=dtype, device=device)
            desired_rel_goal = 0.5 * (ball_pos_minus_goal + our_goal_rel_goal)

        # Teammate contest avoidance / offering adjustments (from partner's viewpoint)
        if other_teammate_pos_rel_goal is not None:
            teammate_pos_rel_ball = other_teammate_pos_rel_goal - ball_pos_minus_goal
            teammate_dist_to_ball = torch.norm(teammate_pos_rel_ball)
            agent_pos_rel_ball = agent_pos_rel_goal - ball_pos_minus_goal

            if teammate_dist_to_ball.item() < contest_dist:
                away_dir = agent_pos_rel_ball.clone()
                nd = torch.norm(away_dir)
                if nd.item() < 1e-6:
                    away_dir = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
                    nd = torch.norm(away_dir)
                away_dir = away_dir / nd
                desired_rel_goal = desired_rel_goal + away_dir * 0.6
            else:
                toward_ball = ball_pos_minus_goal - agent_pos_rel_goal
                nt = torch.norm(toward_ball)
                if nt.item() > 1e-6:
                    desired_rel_goal = desired_rel_goal - 0.10 * (desired_rel_goal - (agent_pos_rel_goal + 0.2 * toward_ball / nt))

        return desired_rel_goal

    # Simulate partner's desired position: partner is the teammate in our observation
    partner_desired_rel_goal = None
    if teammate is not None:
        # partner's own pos_rel_goal is teammate['pos_rel_goal']
        # partner's teammate (from partner POV) is us: our pos_rel_goal = agent_pos_minus_goal
        partner_desired_rel_goal = compute_partner_like_desired(
            agent_pos_rel_goal=teammate['pos_rel_goal'],
            agent_vel_local=teammate['vel'],
            other_teammate_pos_rel_goal=agent_pos_minus_goal,
            other_teammate_vel=agent_vel,
        )
    else:
        # No teammate info: fallback to a ball-goal interpolation desired
        alpha_fb = 0.35
        partner_desired_rel_goal = (1.0 - alpha_fb) * ball_pos_minus_goal

    # Move toward partner's predicted desired (so we meet at the same landmark)
    kp = 3.0
    kd = 0.8
    max_force = 1.0

    delta = partner_desired_rel_goal - agent_pos_minus_goal
    force = kp * delta - kd * agent_vel

    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force * (max_force / (fmag + 1e-12))

    return force.view(1, 2)