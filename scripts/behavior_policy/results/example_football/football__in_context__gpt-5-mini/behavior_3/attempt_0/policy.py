import torch

# Persistent internal state stored on the function object

def agent0_policy(obs: torch.Tensor):
    """
    Deceptive Dribbler & Shielding Playmaker policy.
    obs: 1D torch tensor (observation vector) as produced by observation_base.
    Returns: torch.Tensor of shape (1,2) (force applied).

    Internal state (persisted across calls):
      agent0_policy.hold_counter (float)
      agent0_policy.feint_phase (float)
    """

    # initialize persistent vars
    if not hasattr(agent0_policy, "hold_counter"):
        agent0_policy.hold_counter = 0.0
    if not hasattr(agent0_policy, "feint_phase"):
        agent0_policy.feint_phase = 0.0

    # Constants
    max_force = 1.0
    hold_distance = 0.45
    shield_offset = 0.25
    feint_amp = 0.12
    feint_freq = 2.0
    release_threshold = 3.0
    hold_counter_decay = 0.9
    role_margin = 0.2
    cover_distance = 0.8
    eps = 1e-6

    # helpers
    def length(v):
        return torch.sqrt((v * v).sum() + eps)

    def norm(v):
        ln = length(v)
        return v / ln

    def clamp_mag(v, m):
        ln = length(v)
        if ln <= m + 1e-8:
            return v
        return v * (m / (ln + 1e-12))

    def perp(v):
        # v is length-2 tensor
        return torch.stack((-v[1], v[0]))

    # ensure flat
    obs = obs.view(-1)
    total = obs.shape[0]

    # The observation structure in observation_base has 40 known dims plus agent_rot dims.
    # Compute agent_rot dimension robustly
    base_known = 40
    r = int(max(0, total - base_known))

    # Indexing following observation_base order
    i = 0
    agent_force = obs[i : i + 2]
    i += 2
    agent_pos_minus_ball = obs[i : i + 2]
    i += 2
    agent_vel_minus_ball_vel = obs[i : i + 2]
    i += 2
    ball_pos_minus_goal = obs[i : i + 2]
    i += 2
    ball_vel = obs[i : i + 2]
    i += 2
    ball_force = obs[i : i + 2]
    i += 2
    agent_rot = obs[i : i + r] if r > 0 else obs[i : i]
    i += r
    agent_pos_minus_goal = obs[i : i + 2]
    i += 2
    agent_vel = obs[i : i + 2]
    i += 2
    adversaries = obs[i : i + 16]
    i += 16
    teammates = obs[i : i + 8]
    # i += 8

    # useful vectors (all relative)
    # vec(ego, ball) = ball - ego = - (agent_pos - ball_pos)
    vec_ego_ball = -agent_pos_minus_ball
    d_ego_ball = length(vec_ego_ball)

    # teammate vectors
    a_minus_tm = teammates[0:2]
    # vec(ball, teammate) = teammate - ball = -(a - t) + (a - b)
    vec_ball_teammate = -a_minus_tm + agent_pos_minus_ball
    d_tm_ball = length(vec_ball_teammate)

    # approximate opponent direction (from our goal toward ball/opponent side)
    # ball_pos_minus_goal is ball - goal, so opp_dir_approx = norm(ball - goal)
    opp_dir_approx = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0])

    # parse adversaries (two adversaries, each 8 dims)
    advs = []  # list of dicts with adv vectors
    for k in range(2):
        seg = adversaries[k * 8 : (k + 1) * 8]
        a_minus_adv = seg[0:2]
        a_vel_minus_adv_vel = seg[2:4]
        adv_vel = seg[4:6]
        adv_force = seg[6:8]

        # adv_pos - ball_pos = -(a - adv) + (a - b)
        adv_minus_ball = -a_minus_adv + agent_pos_minus_ball
        # adv_pos - goal_pos = adv_minus_ball + ball_pos_minus_goal
        adv_minus_goal = adv_minus_ball + ball_pos_minus_goal
        advs.append({
            "a_minus_adv": a_minus_adv,
            "adv_minus_ball": adv_minus_ball,
            "adv_minus_goal": adv_minus_goal,
            "adv_vel": adv_vel,
            "adv_force": adv_force,
        })

    # find nearest adversary to ball
    dists_adv_ball = torch.stack([length(ad["adv_minus_ball"]) for ad in advs])
    idx_adv_ball = int(torch.argmin(dists_adv_ball).item())
    adv_near = advs[idx_adv_ball]
    d_adv_ball = dists_adv_ball[idx_adv_ball]

    # find adversary nearest our goal (for defensive fallback)
    dists_adv_goal = torch.stack([length(ad["adv_minus_goal"]) for ad in advs])
    idx_adv_goal = int(torch.argmin(dists_adv_goal).item())
    adv_near_goal = advs[idx_adv_goal]
    d_adv_goal = dists_adv_goal[idx_adv_goal]

    # Default outputs
    force = torch.zeros(2)
    burst = 1.0

    # Defensive fallback: if any adversary close to our goal, cover between that adv and our goal
    if d_adv_goal < cover_distance:
        # direction from our goal toward adversary = norm(adversary - goal)
        dir_to_adv = norm(adv_near_goal["adv_minus_goal"]) if length(adv_near_goal["adv_minus_goal"]) > eps else torch.tensor([0.0, 0.0])
        # target point is goal + dir * (cover_distance * 0.6), but we only have relative positions. Compute vector from ego to that target
        # agent_pos_minus_goal = a - goal => goal = a - (a - goal)
        # target - ego = (goal + dir * r) - ego = (goal - ego) + dir * r = - (agent_pos_minus_goal) + dir * r
        rdist = cover_distance * 0.6
        target_minus_ego = -agent_pos_minus_goal + dir_to_adv * rdist

        # simple PD controller toward target
        kp = 1.6
        kd = 0.7
        desired = kp * target_minus_ego - kd * agent_vel
        force = clamp_mag(desired, max_force)

        # reduce hold intent
        agent0_policy.hold_counter = agent0_policy.hold_counter * hold_counter_decay
        burst = 1.0

    else:
        # Determine if we can hold
        can_hold = (d_ego_ball <= hold_distance + 1e-9) and (d_ego_ball <= (d_tm_ball - role_margin + 1e-9))

        if can_hold:
            # Increase hold counter
            agent0_policy.hold_counter += 1.0

            # Determine protect direction
            to_teammate = vec_ball_teammate.clone()  # teammate - ball
            to_teammate_len = length(to_teammate)
            teammate_ahead = False
            if to_teammate_len > eps:
                teammate_ahead = (torch.dot(norm(to_teammate), opp_dir_approx) > 0.5)

            if teammate_ahead and to_teammate_len > eps:
                protect_dir = norm(to_teammate)
            else:
                # protect toward opposite of nearest adversary to ball (i.e., from ball toward adv then flip) - approximate as adv -> ball vector
                adv_vec = adv_near["adv_minus_ball"]
                if length(adv_vec) > eps:
                    # vector from ball toward adv is adv - ball = adv_vec; we want protect direction pointing away from adv (ball->protect)
                    protect_dir = norm(-adv_vec)
                else:
                    # fall back to opposite opp_dir_approx
                    protect_dir = -opp_dir_approx

            # shield point = ball + protect_dir * shield_offset
            shield_point_minus_ego = (-agent_pos_minus_ball) + protect_dir * shield_offset
            # feint oscillation perpendicular to protect_dir
            agent0_policy.feint_phase += feint_freq * 0.1
            phase = torch.tensor(agent0_policy.feint_phase)
            lateral = perp(protect_dir)
            if length(lateral) > eps:
                lateral = norm(lateral)
            else:
                lateral = torch.tensor([0.0, 0.0])
            feint = lateral * (feint_amp * torch.sin(phase))

            # if adversary committing, bias feint
            adv_vel = adv_near["adv_vel"]
            adv_vel_len = length(adv_vel)
            adv_motion_dir = norm(adv_vel) if adv_vel_len > eps else torch.tensor([0.0, 0.0])
            adv_committing = False
            adv_to_ball = adv_near["adv_minus_ball"]
            if length(adv_to_ball) > eps and adv_vel_len > eps:
                adv_committing = (torch.dot(adv_motion_dir, norm(adv_to_ball)) > 0.6)
            if adv_committing:
                feint = feint * 1.6 - adv_motion_dir * 0.05

            # target relative to ego
            target_minus_ego = shield_point_minus_ego + feint

            # Decide release conditions
            teammate_ready = (length(to_teammate) < 2.5) and teammate_ahead
            adv_retreating = False
            if adv_vel_len > eps and length(adv_to_ball) > eps:
                adv_retreating = (torch.dot(adv_motion_dir, norm(adv_to_ball)) < -0.2)

            if (agent0_policy.hold_counter >= release_threshold) and (teammate_ready or adv_retreating):
                # RELEASE/BURST
                if teammate_ready:
                    # compute teammate - goal vector to approximate forward
                    teammate_minus_goal = vec_ball_teammate + ball_pos_minus_goal
                    if length(teammate_minus_goal) > eps:
                        tm_forward = norm(teammate_minus_goal)
                    else:
                        tm_forward = opp_dir_approx
                    release_target_minus_ego = ( -agent_pos_minus_goal + (vec_ball_teammate + tm_forward * 0.4) )
                else:
                    # release away from our goal biased away from nearest adversary
                    # release_dir approx = norm(ball - goal) + 0.6 * away_from_adv
                    release_dir = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0])
                    away_from_adv = norm(adv_near["adv_minus_ball"]) if length(adv_near["adv_minus_ball"]) > eps else torch.tensor([0.0, 0.0])
                    combined = release_dir + 0.6 * away_from_adv
                    if length(combined) > eps:
                        release_dir = norm(combined)
                    release_target_minus_ego = -agent_pos_minus_ball + release_dir * 0.6

                burst = 1.4
                desired_dir = release_target_minus_ego
                if length(desired_dir) > eps:
                    desired_dir = norm(desired_dir)
                else:
                    desired_dir = torch.tensor([0.0, 0.0])

                base_force = desired_dir * (max_force * burst) - 0.6 * agent_vel
                force = clamp_mag(base_force, max_force * burst)

                # after release reduce hold intent
                agent0_policy.hold_counter = agent0_policy.hold_counter * 0.2
            else:
                # Normal holding motion (PD toward target)
                kp = 1.6
                kd = 0.7
                pos_err = target_minus_ego
                desired = kp * pos_err - kd * agent_vel
                force = clamp_mag(desired, max_force)

        else:
            # Not in holdable position
            agent0_policy.hold_counter = agent0_policy.hold_counter * hold_counter_decay

            # decide outlet/support or neutral reposition
            if d_tm_ball <= d_ego_ball - role_margin:
                # teammate has ball: become inside outlet between ball and goal, offset toward center
                ahead_point_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 0.9
                center_bias = 0.5
                # bias: subtract opp_dir_approx rotated by 90 deg scaled by center_bias
                bias = perp(opp_dir_approx) * center_bias
                target_minus_ego = ahead_point_minus_ego - bias
            else:
                # neutral reposition: short advanced support near ball but not contesting
                p = perp(opp_dir_approx)
                target_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 1.0 + p * 0.4

            kp2 = 1.2
            kd2 = 0.6
            pos_err = target_minus_ego
            desired = kp2 * pos_err - kd2 * agent_vel
            force = clamp_mag(desired, max_force)

    # final safety clamp to max_force * burst
    force = clamp_mag(force, max_force * burst)

    # ensure finite values
    force = torch.where(torch.isfinite(force), force, torch.zeros_like(force))

    return force.unsqueeze(0)

import torch

# Persistent internal state stored on the function object

def agent1_policy(obs: torch.Tensor):
    """
    Supportive Shielding / Outlet policy to coordinate with the partner (agent0).
    - When partner (teammate) is in a holding posture near the ball, this agent makes forward/off-center runs
      to present a passing window and to occupy defenders.
    - When this agent is clearly the better candidate for the ball it will approach and hold/release similarly
      (simple PD hold/burst) but with a bias toward becoming a clean outlet rather than overly contesting.
    - Defensive fallback: if an adversary is dangerously close to our goal, cover between that adversary and goal.

    obs: 1D torch tensor (observation vector) as produced by observation_base.
    Returns: torch.Tensor of shape (1,2) (force applied).

    Internal state (persisted across calls):
      agent1_policy.request_counter (float)  - small memory to bias forward runs when partner is holding
    """

    # initialize persistent vars
    if not hasattr(agent1_policy, "request_counter"):
        agent1_policy.request_counter = 0.0

    # Constants (tuned to coordinate with partner behavior)
    max_force = 1.0
    hold_distance = 0.45
    role_margin = 0.2
    cover_distance = 0.8
    kp_hold = 1.5
    kd_hold = 0.7
    kp_support = 1.2
    kd_support = 0.6
    eps = 1e-6

    # helpers
    def length(v):
        return torch.sqrt((v * v).sum() + eps)

    def norm(v):
        ln = length(v)
        return v / ln

    def clamp_mag(v, m):
        ln = length(v)
        if ln <= m + 1e-8:
            return v
        return v * (m / (ln + 1e-12))

    def perp(v):
        return torch.stack((-v[1], v[0]))

    # ensure flat
    obs = obs.view(-1)
    total = obs.shape[0]

    # parse observation robustly (same order as observation_base used by environment)
    base_known = 40
    r = int(max(0, total - base_known))

    i = 0
    agent_force = obs[i : i + 2]; i += 2
    agent_pos_minus_ball = obs[i : i + 2]; i += 2
    agent_vel_minus_ball_vel = obs[i : i + 2]; i += 2
    ball_pos_minus_goal = obs[i : i + 2]; i += 2
    ball_vel = obs[i : i + 2]; i += 2
    ball_force = obs[i : i + 2]; i += 2
    agent_rot = obs[i : i + r] if r > 0 else obs[i : i]; i += r
    agent_pos_minus_goal = obs[i : i + 2]; i += 2
    agent_vel = obs[i : i + 2]; i += 2
    adversaries = obs[i : i + 16]; i += 16
    teammates = obs[i : i + 8]

    # Useful vectors (relative quantities stored in obs):
    # vec(ego, ball) = ball - ego = - (agent_pos - ball_pos)
    vec_ego_ball = -agent_pos_minus_ball
    d_ego_ball = length(vec_ego_ball)

    # teammate data (this teammate is agent0)
    a_minus_tm = teammates[0:2]
    tm_vel_minus = teammates[2:4]
    tm_vel = teammates[4:6]
    tm_force = teammates[6:8]

    # vector ball -> teammate = teammate - ball = -(a - t) + (a - b)
    vec_ball_teammate = -a_minus_tm + agent_pos_minus_ball
    d_tm_ball = length(vec_ball_teammate)

    # approximate opponent direction: from our goal toward ball/opponent side
    opp_dir_approx = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0])

    # parse adversaries (2 adversaries, each 8 dims: agent_pos - adv_pos, agent_vel - adv_vel, adv_vel, adv_force)
    advs = []
    for k in range(2):
        seg = adversaries[k * 8 : (k + 1) * 8]
        a_minus_adv = seg[0:2]
        a_vel_minus_adv_vel = seg[2:4]
        adv_vel = seg[4:6]
        adv_force = seg[6:8]

        # adv_pos - ball_pos = -(a - adv) + (a - b)
        adv_minus_ball = -a_minus_adv + agent_pos_minus_ball
        # adv_pos - goal_pos = adv_minus_ball + ball_pos_minus_goal
        adv_minus_goal = adv_minus_ball + ball_pos_minus_goal
        advs.append({
            "adv_minus_ball": adv_minus_ball,
            "adv_minus_goal": adv_minus_goal,
            "adv_vel": adv_vel,
            "adv_force": adv_force,
        })

    dists_adv_ball = torch.stack([length(ad["adv_minus_ball"]) for ad in advs])
    idx_adv_ball = int(torch.argmin(dists_adv_ball).item())
    adv_near_ball = advs[idx_adv_ball]

    dists_adv_goal = torch.stack([length(ad["adv_minus_goal"]) for ad in advs])
    idx_adv_goal = int(torch.argmin(dists_adv_goal).item())
    adv_near_goal = advs[idx_adv_goal]

    # defensive fallback
    force = torch.zeros(2)
    burst = 1.0

    if dists_adv_goal[idx_adv_goal] < cover_distance:
        # cover between our goal and the nearest adversary to goal
        dir_to_adv = norm(adv_near_goal["adv_minus_goal"]) if length(adv_near_goal["adv_minus_goal"]) > eps else torch.tensor([0.0, 0.0])
        rdist = cover_distance * 0.6
        # target relative to ego = -agent_pos_minus_goal + dir_to_adv * rdist
        target_minus_ego = -agent_pos_minus_goal + dir_to_adv * rdist
        desired = kp_hold * target_minus_ego - kd_hold * agent_vel
        force = clamp_mag(desired, max_force)
        # decay request counter since we're forced defensive
        agent1_policy.request_counter *= 0.8

    else:
        # Decide team roles based on distances and teammate signals
        # Heuristic to detect whether teammate (agent0) is effectively holding: close to ball and applying some force or low velocity
        tm_force_mag = length(tm_force)
        tm_speed = length(tm_vel)
        teammate_holding = (d_tm_ball <= hold_distance + 1e-9 and d_tm_ball <= d_ego_ball - role_margin + 1e-9) or (
            (d_tm_ball <= hold_distance + 0.25) and (tm_force_mag > 1e-3) and (tm_speed < 0.6)
        )

        # Heuristic to detect teammate bursting (they are releasing): teammate moving away from ball toward opponent side
        tm_motion_dir = norm(tm_vel) if tm_speed > eps else torch.tensor([0.0, 0.0])
        tm_from_ball = vec_ball_teammate  # vector ball->teammate
        tm_from_ball_dir = norm(tm_from_ball) if length(tm_from_ball) > eps else torch.tensor([0.0, 0.0])
        teammate_releasing = False
        if tm_speed > 0.6 and (torch.dot(tm_motion_dir, tm_from_ball_dir) < -0.2 or torch.dot(tm_motion_dir, opp_dir_approx) > 0.3):
            teammate_releasing = True

        # If teammate is holding, increase our request_counter to push forward to be a passing target
        if teammate_holding:
            agent1_policy.request_counter += 1.0
        else:
            agent1_policy.request_counter *= 0.85

        # Cap request_counter
        if agent1_policy.request_counter > 6.0:
            agent1_policy.request_counter = 6.0

        # If teammate is holding, become an inside outlet / make forward runs
        if teammate_holding and not (d_ego_ball <= hold_distance and d_ego_ball <= d_tm_ball - role_margin):
            # target: slightly ahead of ball toward opponent, biased toward center by lateral offset
            center_bias = 0.5
            # ahead point relative to ego
            ahead_minus_ego = -agent_pos_minus_ball + opp_dir_approx * (0.9 + 0.15 * (agent1_policy.request_counter / 6.0))
            bias = perp(opp_dir_approx) * center_bias * (0.6 + 0.4 * (agent1_policy.request_counter / 6.0))
            # alternate lateral side based on request_counter parity to offer multiple passing lanes over time
            side = 1.0 if (int(agent1_policy.request_counter) % 2 == 0) else -1.0
            target_minus_ego = ahead_minus_ego - bias * side

            # soft PD toward target; if partner is about to release we try to get more aggressively ahead
            aggressive = 1.0 + 0.6 * float(teammate_releasing)
            desired = (kp_support * aggressive) * target_minus_ego - kd_support * agent_vel
            force = clamp_mag(desired, max_force)

        else:
            # Either we are the better candidate for the ball, or neutral reposition / support
            # If we are closer to ball (and clearly so), approach and attempt to hold (gentle hold to avoid overcommit)
            if d_ego_ball <= d_tm_ball - role_margin:
                # approach ball and try to position to hold and present to teammate
                # shield slightly toward opponent direction (so we face opponents) but remain pass-friendly
                shield_offset = 0.22
                protect_dir = opp_dir_approx if length(opp_dir_approx) > eps else torch.tensor([0.0, 0.0])
                shield_point_minus_ego = -agent_pos_minus_ball + protect_dir * shield_offset

                # small lateral oscillation to be dynamic but conservative
                lateral = perp(protect_dir)
                if length(lateral) > eps:
                    lateral = norm(lateral)
                else:
                    lateral = torch.tensor([0.0, 0.0])
                feint = lateral * 0.06 * torch.sin(torch.tensor(agent1_policy.request_counter * 0.4))
                target_minus_ego = shield_point_minus_ego + feint

                # PD controller toward target (slightly conservative burst if teammate-ready)
                desired = kp_hold * target_minus_ego - kd_hold * agent_vel
                force = clamp_mag(desired, max_force)

            else:
                # neutral support: position near ball but not contesting; prefer inside lanes
                p = perp(opp_dir_approx)
                target_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 1.0 + p * 0.35
                desired = kp_support * target_minus_ego - kd_support * agent_vel
                force = clamp_mag(desired, max_force)

        # If teammate is actively releasing (fast motion away from ball toward opponent), attempt to move to anticipated reception point
        if teammate_releasing:
            # Predict a short reception point ahead of teammate and try to reach it
            tm_ahead = vec_ball_teammate + tm_from_ball_dir * 0.4 if length(tm_from_ball_dir) > eps else vec_ball_teammate
            reception_minus_ego = -agent_pos_minus_ball + tm_ahead
            # blend with previous force to produce a more targeted move
            desired_rec = kp_hold * reception_minus_ego - kd_hold * agent_vel
            # stronger push if we are already moving and close to reception
            blend = 0.6
            force = clamp_mag((1.0 - blend) * force + blend * clamp_mag(desired_rec, max_force * 1.2), max_force * 1.2)

    # safety clamp
    force = clamp_mag(force, max_force * burst)
    force = torch.where(torch.isfinite(force), force, torch.zeros_like(force))

    return force.unsqueeze(0)