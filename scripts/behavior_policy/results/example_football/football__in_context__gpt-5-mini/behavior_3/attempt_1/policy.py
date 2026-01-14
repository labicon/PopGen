import torch

# Persistent internal state stored on the function object

def agent0_policy(obs: torch.Tensor):
    """
    Deceptive Dribbler & Shielding Playmaker policy (agent0).
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
    def length(v: torch.Tensor):
        return torch.sqrt((v * v).sum() + eps)

    def norm(v: torch.Tensor):
        ln = length(v)
        return v / (ln + 1e-12)

    def clamp_mag(v: torch.Tensor, m: float):
        ln = length(v)
        if ln <= m + 1e-8:
            return v
        return v * (m / (ln + 1e-12))

    def perp(v: torch.Tensor):
        return torch.stack((-v[1], v[0]))

    # ensure flat
    obs = obs.view(-1)
    total = obs.shape[0]

    # Parse observation robustly. observation_base concatenates several blocks. We follow the same order.
    # We allow the agent_rot size to vary; compute its size by subtracting known scalar block sizes from total.
    base_known = 40  # approx size of other fixed elements in observation_base (used previously)
    rot_size = int(max(0, total - base_known))

    i = 0
    agent_force = obs[i : i + 2]; i += 2
    agent_pos_minus_ball = obs[i : i + 2]; i += 2
    agent_vel_minus_ball_vel = obs[i : i + 2]; i += 2
    ball_pos_minus_goal = obs[i : i + 2]; i += 2
    ball_vel = obs[i : i + 2]; i += 2
    ball_force = obs[i : i + 2]; i += 2
    agent_rot = obs[i : i + rot_size] if rot_size > 0 else obs[i : i]; i += rot_size
    agent_pos_minus_goal = obs[i : i + 2]; i += 2
    agent_vel = obs[i : i + 2]; i += 2
    adversaries = obs[i : i + 16]; i += 16
    teammates = obs[i : i + 8]

    # Useful derived vectors
    # vec(ego, ball) = ball - ego = - (agent_pos - ball_pos)
    vec_ego_ball = -agent_pos_minus_ball
    d_ego_ball = length(vec_ego_ball)

    # teammate vectors (single teammate)
    a_minus_tm = teammates[0:2]
    tm_vel_minus = teammates[2:4]
    tm_vel = teammates[4:6]
    tm_force = teammates[6:8]

    # vec(ball, teammate) = teammate - ball = -(a - t) + (a - b)
    vec_ball_teammate = -a_minus_tm + agent_pos_minus_ball
    d_tm_ball = length(vec_ball_teammate)

    # approximate opponent direction: from our goal toward ball/opponent side
    opp_dir_approx = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)

    # parse adversaries (2 adversaries, each 8 dims as provided by observation_base)
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
            "a_minus_adv": a_minus_adv,
            "adv_minus_ball": adv_minus_ball,
            "adv_minus_goal": adv_minus_goal,
            "adv_vel": adv_vel,
            "adv_force": adv_force,
        })

    # distances
    dists_adv_ball = torch.stack([length(ad["adv_minus_ball"]) for ad in advs])
    idx_adv_ball = int(torch.argmin(dists_adv_ball).item())
    adv_near = advs[idx_adv_ball]
    d_adv_ball = dists_adv_ball[idx_adv_ball]

    dists_adv_goal = torch.stack([length(ad["adv_minus_goal"]) for ad in advs])
    idx_adv_goal = int(torch.argmin(dists_adv_goal).item())
    adv_near_goal = advs[idx_adv_goal]
    d_adv_goal = dists_adv_goal[idx_adv_goal]

    # Default outputs
    force = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    # Defensive fallback: if any adversary is dangerously close to our goal, cover
    if d_adv_goal < cover_distance:
        # direction from our goal toward adversary = norm(adversary - goal) = norm(adv_minus_goal)
        dir_to_adv = norm(adv_near_goal["adv_minus_goal"]) if length(adv_near_goal["adv_minus_goal"]) > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
        rdist = cover_distance * 0.6
        # target relative to ego: (goal + dir*rdist) - ego = -agent_pos_minus_goal + dir*rdist
        target_minus_ego = -agent_pos_minus_goal + dir_to_adv * rdist
        # PD controller toward target
        kp = 1.6
        kd = 0.7
        desired = kp * target_minus_ego - kd * agent_vel
        force = clamp_mag(desired, max_force)
        # decay hold counter
        agent0_policy.hold_counter = agent0_policy.hold_counter * hold_counter_decay

    else:
        # Decide if we can hold the ball
        can_hold = (d_ego_ball <= hold_distance + 1e-9) and (d_ego_ball <= (d_tm_ball - role_margin + 1e-9))

        if can_hold:
            # Increase hold counter
            agent0_policy.hold_counter += 1.0

            # Determine protect direction
            to_teammate = vec_ball_teammate.clone()
            to_teammate_len = length(to_teammate)
            teammate_ahead = False
            if to_teammate_len > eps:
                teammate_ahead = (torch.dot(norm(to_teammate), opp_dir_approx) > 0.5)

            if teammate_ahead and to_teammate_len > eps:
                protect_dir = norm(to_teammate)
            else:
                adv_vec = adv_near["adv_minus_ball"]
                if length(adv_vec) > eps:
                    # vector from ball toward adv is adv_vec; protect should be away from adv (ball -> protect)
                    protect_dir = norm(-adv_vec)
                else:
                    protect_dir = -opp_dir_approx

            # shield point relative to ego = (ball - ego) + protect_dir*shield_offset = -agent_pos_minus_ball + protect_dir*shield_offset
            shield_point_minus_ego = -agent_pos_minus_ball + protect_dir * shield_offset

            # feint oscillation
            agent0_policy.feint_phase += feint_freq * 0.1
            phase = torch.tensor(agent0_policy.feint_phase, dtype=obs.dtype, device=obs.device)
            lateral = perp(protect_dir)
            if length(lateral) > eps:
                lateral = norm(lateral)
            else:
                lateral = torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
            feint = lateral * (feint_amp * torch.sin(phase))

            # if adversary committing, bias feint
            adv_vel = adv_near["adv_vel"]
            adv_vel_len = length(adv_vel)
            adv_motion_dir = norm(adv_vel) if adv_vel_len > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
            adv_to_ball = adv_near["adv_minus_ball"]
            adv_committing = False
            if length(adv_to_ball) > eps and adv_vel_len > eps:
                adv_committing = (torch.dot(adv_motion_dir, norm(adv_to_ball)) > 0.6)
            if adv_committing:
                feint = feint * 1.6 - adv_motion_dir * 0.05

            target_minus_ego = shield_point_minus_ego + feint

            # Decide release conditions
            teammate_ready = (length(to_teammate) < 2.5) and teammate_ahead
            adv_retreating = False
            if adv_vel_len > eps and length(adv_to_ball) > eps:
                adv_retreating = (torch.dot(adv_motion_dir, norm(adv_to_ball)) < -0.2)

            if (agent0_policy.hold_counter >= release_threshold) and (teammate_ready or adv_retreating):
                # RELEASE/BURST: we still must not exceed max_force. Use a stronger desired direction but clamp to max_force.
                if teammate_ready:
                    # teammate_minus_goal approx = vec_ball_teammate + ball_pos_minus_goal
                    teammate_minus_goal = vec_ball_teammate + ball_pos_minus_goal
                    if length(teammate_minus_goal) > eps:
                        tm_forward = norm(teammate_minus_goal)
                    else:
                        tm_forward = opp_dir_approx
                    # release target relative to ego: (teammate - ego) + tm_forward*0.4 = (-agent_pos_minus_ball + vec_ball_teammate) + tm_forward*0.4
                    release_target_minus_ego = -agent_pos_minus_ball + (vec_ball_teammate + tm_forward * 0.4)
                else:
                    # release away from our goal biased away from adv
                    release_dir = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
                    away_from_adv = norm(adv_near["adv_minus_ball"]) if length(adv_near["adv_minus_ball"]) > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
                    combined = release_dir + 0.6 * away_from_adv
                    if length(combined) > eps:
                        release_dir = norm(combined)
                    release_target_minus_ego = -agent_pos_minus_ball + release_dir * 0.6

                desired_dir = release_target_minus_ego
                if length(desired_dir) > eps:
                    desired_dir = norm(desired_dir)
                else:
                    desired_dir = torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)

                # produce a decisive push direction but ensure we do not exceed max_force
                # We bias slightly more toward direction (equivalent to burst) but final clamp enforces allowed magnitude
                base_force = desired_dir * (max_force * 1.0) - 0.6 * agent_vel
                force = clamp_mag(base_force, max_force)

                # after release reduce hold intent
                agent0_policy.hold_counter = agent0_policy.hold_counter * 0.2

            else:
                # Normal holding motion: PD toward target while damping velocity
                kp = 1.6
                kd = 0.7
                pos_err = target_minus_ego
                desired = kp * pos_err - kd * agent_vel
                force = clamp_mag(desired, max_force)

        else:
            # We are not in a holdable position. Decay hold counter
            agent0_policy.hold_counter = agent0_policy.hold_counter * hold_counter_decay

            # If teammate is clearly closer to the ball, play outlet/support role
            if d_tm_ball <= d_ego_ball - role_margin:
                ahead_point_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 0.9
                center_bias = 0.5
                bias = perp(opp_dir_approx) * center_bias
                target_minus_ego = ahead_point_minus_ego - bias
            else:
                # Neutral reposition: short advanced/supporting position near ball but not contesting
                p = perp(opp_dir_approx)
                target_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 1.0 + p * 0.4

            kp2 = 1.2
            kd2 = 0.6
            pos_err = target_minus_ego
            desired = kp2 * pos_err - kd2 * agent_vel
            force = clamp_mag(desired, max_force)

    # Final safety clamp: ensure magnitude does not exceed max_force (environment requires actions within [-1,1] per axis magnitude)
    force = clamp_mag(force, max_force)
    force = torch.where(torch.isfinite(force), force, torch.zeros_like(force))

    return force.unsqueeze(0)

import torch

# Persistent internal state stored on the function object

def agent1_policy(obs: torch.Tensor):
    """
    Supportive Forward Runner & Secondary Dribbler policy (agent1).
    This policy coordinates with the provided Deceptive Dribbler & Shielding Playmaker partner.

    obs: 1D torch tensor (observation vector) as produced by observation_base.
    Returns: torch.Tensor of shape (1,2) (force applied).

    Internal state (persisted across calls):
      agent1_policy.hold_counter (float)  # for when this agent takes a dribble role
      agent1_policy.feint_phase (float)
      agent1_policy.request_counter (float)  # how long we've been requesting a pass / running
    """

    # initialize persistent vars
    if not hasattr(agent1_policy, "hold_counter"):
        agent1_policy.hold_counter = 0.0
    if not hasattr(agent1_policy, "feint_phase"):
        agent1_policy.feint_phase = 0.0
    if not hasattr(agent1_policy, "request_counter"):
        agent1_policy.request_counter = 0.0

    # Constants (aligned with partner for compatibility)
    max_force = 1.0
    hold_distance = 0.45
    shield_offset = 0.22
    feint_amp = 0.10
    feint_freq = 2.0
    release_threshold = 2.5  # slightly more willing to release as a secondary dribbler
    hold_counter_decay = 0.9
    role_margin = 0.2
    cover_distance = 0.8
    eps = 1e-6

    # helpers
    def length(v: torch.Tensor):
        return torch.sqrt((v * v).sum() + eps)

    def norm(v: torch.Tensor):
        ln = length(v)
        return v / (ln + 1e-12)

    def clamp_mag(v: torch.Tensor, m: float):
        ln = length(v)
        if ln <= m + 1e-8:
            return v
        return v * (m / (ln + 1e-12))

    def perp(v: torch.Tensor):
        return torch.stack((-v[1], v[0]))

    # ensure flat
    obs = obs.view(-1)
    total = obs.shape[0]

    # parse observation using the same layout used by partner
    # agent_force(2), agent_pos_minus_ball(2), agent_vel_minus_ball_vel(2),
    # ball_pos_minus_goal(2), ball_vel(2), ball_force(2), agent_rot(var),
    # agent_pos_minus_goal(2), agent_vel(2), adversaries(16), teammates(8)

    # estimate rot size by subtracting fixed known sizes
    # fixed parts besides rot: 2+2+2+2+2+2 + 2 +2 +16 +8 = 40 (rot is the remaining dims)
    fixed = 40
    rot_size = int(max(0, total - fixed))

    i = 0
    agent_force = obs[i: i + 2]; i += 2
    agent_pos_minus_ball = obs[i: i + 2]; i += 2
    agent_vel_minus_ball_vel = obs[i: i + 2]; i += 2
    ball_pos_minus_goal = obs[i: i + 2]; i += 2
    ball_vel = obs[i: i + 2]; i += 2
    ball_force = obs[i: i + 2]; i += 2
    agent_rot = obs[i: i + rot_size] if rot_size > 0 else obs[i: i]; i += rot_size
    agent_pos_minus_goal = obs[i: i + 2]; i += 2
    agent_vel = obs[i: i + 2]; i += 2
    adversaries = obs[i: i + 16]; i += 16
    teammates = obs[i: i + 8]

    # derived vectors
    # vec(ego, ball) = ball - ego = - (agent_pos - ball_pos)
    vec_ego_ball = -agent_pos_minus_ball
    d_ego_ball = length(vec_ego_ball)

    # teammate blocks (single teammate)
    a_minus_tm = teammates[0:2]
    tm_vel_minus = teammates[2:4]
    tm_vel = teammates[4:6]
    tm_force = teammates[6:8]

    # vec(ball, teammate) = teammate - ball = -(a - t) + (a - b)
    vec_ball_teammate = -a_minus_tm + agent_pos_minus_ball
    d_tm_ball = length(vec_ball_teammate)

    # opponent direction approx: from our goal toward ball/opponent side
    opp_dir_approx = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)

    # parse adversaries (two adversaries as in partner)
    advs = []
    for k in range(2):
        seg = adversaries[k * 8: (k + 1) * 8]
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

    # distances to ball and to goal for adversaries
    dists_adv_ball = torch.stack([length(ad["adv_minus_ball"]) for ad in advs])
    idx_adv_ball = int(torch.argmin(dists_adv_ball).item())
    adv_near = advs[idx_adv_ball]
    d_adv_ball = dists_adv_ball[idx_adv_ball]

    dists_adv_goal = torch.stack([length(ad["adv_minus_goal"]) for ad in advs])
    idx_adv_goal = int(torch.argmin(dists_adv_goal).item())
    adv_near_goal = advs[idx_adv_goal]
    d_adv_goal = dists_adv_goal[idx_adv_goal]

    # default force
    force = torch.zeros(2, dtype=obs.dtype, device=obs.device)

    # Defensive fallback: if any adversary is dangerously close to our goal, cover
    if d_adv_goal < cover_distance:
        dir_to_adv = norm(adv_near_goal["adv_minus_goal"]) if length(adv_near_goal["adv_minus_goal"]) > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
        rdist = cover_distance * 0.6
        # target relative to ego: (goal + dir*rdist) - ego = -agent_pos_minus_goal + dir*rdist
        target_minus_ego = -agent_pos_minus_goal + dir_to_adv * rdist
        kp = 1.6
        kd = 0.8
        desired = kp * target_minus_ego - kd * agent_vel
        force = clamp_mag(desired, max_force)
        # decay any hold intent
        agent1_policy.hold_counter = agent1_policy.hold_counter * hold_counter_decay
        agent1_policy.request_counter = agent1_policy.request_counter * 0.6
        return force.unsqueeze(0)

    # Determine if partner (teammate) is in a holding position (we treat them as primary dribbler if so)
    teammate_holding = (d_tm_ball <= hold_distance + 1e-9) and (d_tm_ball <= d_ego_ball - role_margin + 1e-9)

    # If teammate is holding, we become the forward runner/receiver: present a pass and attempt to time a run
    if teammate_holding:
        # increment request counter to indicate we're actively offering
        agent1_policy.request_counter += 1.0

        # Run ahead of the ball toward opponent goal, but offset slightly off-center to present a clean passing lane
        ahead_distance = 1.0 + 0.4 * (agent1_policy.request_counter > 3.0)
        ahead_point_minus_ego = -agent_pos_minus_ball + opp_dir_approx * ahead_distance

        # bias slightly toward inside (reduce extreme lateral) so we are an inside outlet
        center_bias = 0.45
        lateral = perp(opp_dir_approx)
        target_minus_ego = ahead_point_minus_ego - lateral * center_bias

        # Use a more aggressive controller to sprint into passing lanes
        kp_run = 2.2
        kd_run = 0.9
        desired = kp_run * target_minus_ego - kd_run * agent_vel
        # allow a small burst if we've been requesting for a while (to catch the pass)
        burst_factor = 1.2 if agent1_policy.request_counter > 4.0 else 1.0
        force = clamp_mag(desired * burst_factor, max_force)

        # if we get very close to the ideal receiving point, reduce request counter slowly (we're set)
        if length(target_minus_ego) < 0.25:
            agent1_policy.request_counter *= 0.8

        # do not try to contest the ball while teammate is holding; keep hold_counter low
        agent1_policy.hold_counter = agent1_policy.hold_counter * hold_counter_decay
        return force.unsqueeze(0)

    # If we are clearly closer to the ball than teammate, adopt a local dribble/hold strategy (secondary dribbler)
    can_hold = (d_ego_ball <= hold_distance + 1e-9) and (d_ego_ball <= (d_tm_ball - role_margin + 1e-9))

    if can_hold:
        agent1_policy.hold_counter += 1.0
        # prefer to protect toward teammate if teammate is ahead, else away from nearest adversary
        to_teammate = vec_ball_teammate.clone()
        to_teammate_len = length(to_teammate)
        teammate_ahead = False
        if to_teammate_len > eps:
            teammate_ahead = (torch.dot(norm(to_teammate), opp_dir_approx) > 0.5)

        if teammate_ahead and to_teammate_len > eps:
            protect_dir = norm(to_teammate)
        else:
            adv_vec = adv_near["adv_minus_ball"]
            if length(adv_vec) > eps:
                protect_dir = norm(-adv_vec)
            else:
                protect_dir = -opp_dir_approx

        shield_point_minus_ego = -agent_pos_minus_ball + protect_dir * shield_offset

        # feint oscillation (lower amplitude than partner to reduce conflict)
        agent1_policy.feint_phase += feint_freq * 0.1
        phase = torch.tensor(agent1_policy.feint_phase, dtype=obs.dtype, device=obs.device)
        lateral = perp(protect_dir)
        if length(lateral) > eps:
            lateral = norm(lateral)
        else:
            lateral = torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
        feint = lateral * (feint_amp * torch.sin(phase))

        # bias feint away if adversary committing
        adv_vel = adv_near["adv_vel"]
        adv_vel_len = length(adv_vel)
        adv_motion_dir = norm(adv_vel) if adv_vel_len > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
        adv_to_ball = adv_near["adv_minus_ball"]
        adv_committing = False
        if length(adv_to_ball) > eps and adv_vel_len > eps:
            adv_committing = (torch.dot(adv_motion_dir, norm(adv_to_ball)) > 0.6)
        if adv_committing:
            feint = feint * 1.4 - adv_motion_dir * 0.04

        target_minus_ego = shield_point_minus_ego + feint

        # decide release: if teammate is well-placed ahead or opponent retreats
        teammate_ready = (length(to_teammate) < 2.5) and teammate_ahead
        adv_retreating = False
        if adv_vel_len > eps and length(adv_to_ball) > eps:
            adv_retreating = (torch.dot(adv_motion_dir, norm(adv_to_ball)) < -0.2)

        if (agent1_policy.hold_counter >= release_threshold) and (teammate_ready or adv_retreating):
            # release toward teammate if ready otherwise push toward opponent side
            if teammate_ready:
                teammate_minus_goal = vec_ball_teammate + ball_pos_minus_goal
                if length(teammate_minus_goal) > eps:
                    tm_forward = norm(teammate_minus_goal)
                else:
                    tm_forward = opp_dir_approx
                release_target_minus_ego = -agent_pos_minus_ball + (vec_ball_teammate + tm_forward * 0.35)
            else:
                release_dir = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
                away_from_adv = norm(adv_near["adv_minus_ball"]) if length(adv_near["adv_minus_ball"]) > eps else torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
                combined = release_dir + 0.5 * away_from_adv
                if length(combined) > eps:
                    release_dir = norm(combined)
                release_target_minus_ego = -agent_pos_minus_ball + release_dir * 0.55

            desired_dir = release_target_minus_ego
            if length(desired_dir) > eps:
                desired_dir = norm(desired_dir)
            else:
                desired_dir = torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)

            base_force = desired_dir * (max_force * 1.0) - 0.6 * agent_vel
            force = clamp_mag(base_force, max_force)
            agent1_policy.hold_counter = agent1_policy.hold_counter * 0.2
            # after using release, make ourselves available as receiver
            agent1_policy.request_counter = max(agent1_policy.request_counter, 1.0)
            return force.unsqueeze(0)
        else:
            # PD toward holding shield point
            kp = 1.6
            kd = 0.7
            pos_err = target_minus_ego
            desired = kp * pos_err - kd * agent_vel
            force = clamp_mag(desired, max_force)
            return force.unsqueeze(0)

    # Otherwise we are not holding and teammate is not holding: take a supportive advanced position
    agent1_policy.hold_counter = agent1_policy.hold_counter * hold_counter_decay
    agent1_policy.request_counter = agent1_policy.request_counter * 0.95

    # If teammate is significantly closer, be an inside outlet (present passing lane closer to goal)
    if d_tm_ball <= d_ego_ball - role_margin:
        ahead_point_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 0.9
        center_bias = 0.5
        bias = perp(opp_dir_approx) * center_bias
        target_minus_ego = ahead_point_minus_ego - bias
    else:
        # neutral supporting position slightly ahead and to one side
        p = perp(opp_dir_approx)
        target_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 1.0 + p * 0.35

    kp2 = 1.2
    kd2 = 0.7
    pos_err = target_minus_ego
    desired = kp2 * pos_err - kd2 * agent_vel
    # small speed-up if we've requested a pass recently
    burst = 1.0 + 0.15 * (agent1_policy.request_counter > 2.0)
    force = clamp_mag(desired * burst, max_force)

    # final safety clamp and return
    force = clamp_mag(force, max_force)
    force = torch.where(torch.isfinite(force), force, torch.zeros_like(force))
    return force.unsqueeze(0)