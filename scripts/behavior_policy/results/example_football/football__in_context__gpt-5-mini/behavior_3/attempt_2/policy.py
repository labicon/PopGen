import torch

# Persistent internal state stored on the function object

def agent0_policy(obs: torch.Tensor):
    """
    Deceptive Dribbler & Shielding Playmaker (agent0).

    obs: 1D torch tensor (observation vector) as produced by observation_base.
    Returns: torch.Tensor of shape (1,2) (force applied).

    Persistent internal state on the function object:
      agent0_policy.hold_counter: float
      agent0_policy.feint_phase: float
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
    hold_counter_decay = 0.88
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

    # ensure flat vector
    obs = obs.view(-1)
    total = obs.shape[0]
    device = obs.device
    dtype = obs.dtype

    # observation layout (consistent with observation_base)
    # fixed parts besides rot: 2+2+2+2+2+2 + 2 +2 +16 +8 = 40
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

    # teammate block (single teammate)
    a_minus_tm = teammates[0:2]
    tm_vel_minus = teammates[2:4]
    tm_vel = teammates[4:6]
    tm_force = teammates[6:8]

    # vec(ball, teammate) = teammate - ball = -(a - t) + (a - b)
    vec_ball_teammate = -a_minus_tm + agent_pos_minus_ball
    d_tm_ball = length(vec_ball_teammate)

    # approx opponent direction: from our goal toward ball/opponent side
    opp_dir_approx = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)

    # parse two adversaries (each 8 dims: a_minus_adv, a_vel_minus_adv_vel, adv_vel, adv_force)
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

    # default
    force = torch.zeros(2, dtype=dtype, device=device)

    # Defensive fallback has highest priority
    if d_adv_goal < cover_distance:
        # cover between our goal and the nearest adversary to goal
        dir_to_adv = norm(adv_near_goal["adv_minus_goal"]) if length(adv_near_goal["adv_minus_goal"]) > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        rdist = cover_distance * 0.6
        # target minus ego = (goal + dir*rdist) - ego = -agent_pos_minus_goal + dir*rdist
        target_minus_ego = -agent_pos_minus_goal + dir_to_adv * rdist
        kp = 1.6
        kd = 0.7
        desired = kp * target_minus_ego - kd * agent_vel
        force = clamp_mag(desired, max_force)
        # decay hold intent
        agent0_policy.hold_counter *= hold_counter_decay

    else:
        # Decide if we can hold the ball
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
                adv_vec = adv_near["adv_minus_ball"]
                if length(adv_vec) > eps:
                    # choose protect direction away from nearest adversary (ball -> protect)
                    protect_dir = norm(-adv_vec)
                else:
                    protect_dir = -opp_dir_approx

            # shield point relative to ego = (ball - ego) + protect_dir * shield_offset
            shield_point_minus_ego = -agent_pos_minus_ball + protect_dir * shield_offset

            # feint oscillation: advance phase every step; use sin to compute lateral offset
            agent0_policy.feint_phase += feint_freq * 0.18
            phase = torch.tensor(agent0_policy.feint_phase, dtype=dtype, device=device)
            lateral = perp(protect_dir)
            if length(lateral) > eps:
                lateral = norm(lateral)
            else:
                lateral = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
            feint = lateral * (feint_amp * torch.sin(phase))

            # if adversary committing, bias feint to provoke
            adv_vel = adv_near["adv_vel"]
            adv_vel_len = length(adv_vel)
            adv_motion_dir = norm(adv_vel) if adv_vel_len > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)
            adv_to_ball = adv_near["adv_minus_ball"]
            adv_committing = False
            if length(adv_to_ball) > eps and adv_vel_len > eps:
                adv_committing = (torch.dot(adv_motion_dir, norm(adv_to_ball)) > 0.6)
            if adv_committing:
                # amplify lateral feint and bias slightly opposite motion to draw them
                feint = feint * 1.6 - adv_motion_dir * 0.05

            target_minus_ego = shield_point_minus_ego + feint

            # Decide release conditions
            teammate_ready = (length(to_teammate) < 2.5) and teammate_ahead
            adv_retreating = False
            if adv_vel_len > eps and length(adv_to_ball) > eps:
                adv_retreating = (torch.dot(adv_motion_dir, norm(adv_to_ball)) < -0.2)

            # If hold_counter high enough and conditions met -> release burst toward teammate or open space
            if (agent0_policy.hold_counter >= release_threshold) and (teammate_ready or adv_retreating):
                # prepare a directed nudge (burst). Environment restricts actions to magnitude <= max_force,
                # so we produce a decisive push clipped at max_force to remain valid.
                if teammate_ready:
                    teammate_minus_goal = vec_ball_teammate + ball_pos_minus_goal
                    if length(teammate_minus_goal) > eps:
                        tm_forward = norm(teammate_minus_goal)
                    else:
                        tm_forward = opp_dir_approx
                    # release target relative to ego
                    release_target_minus_ego = -agent_pos_minus_ball + (vec_ball_teammate + tm_forward * 0.4)
                else:
                    release_dir = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)
                    away_from_adv = norm(adv_near["adv_minus_ball"]) if length(adv_near["adv_minus_ball"]) > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)
                    combined = release_dir + 0.6 * away_from_adv
                    if length(combined) > eps:
                        release_dir = norm(combined)
                    release_target_minus_ego = -agent_pos_minus_ball + release_dir * 0.6

                desired_dir = release_target_minus_ego
                if length(desired_dir) > eps:
                    desired_dir = norm(desired_dir)
                else:
                    desired_dir = torch.tensor([0.0, 0.0], dtype=dtype, device=device)

                # create a crisp push toward desired_dir while damping current vel; clamp to allowed magnitude
                base_force = desired_dir * (max_force * 1.0) - 0.6 * agent_vel
                force = clamp_mag(base_force, max_force)

                # make hold intent decay strongly after release
                agent0_policy.hold_counter *= 0.22

            else:
                # Holding: PD controller toward shield+feint target, keep force within limits
                kp = 1.6
                kd = 0.7
                pos_err = target_minus_ego
                desired = kp * pos_err - kd * agent_vel
                force = clamp_mag(desired, max_force)

        else:
            # Not in holdable position
            agent0_policy.hold_counter *= hold_counter_decay

            # If teammate is closer, become inside outlet (in-front-of-ball and slightly biased toward center)
            if d_tm_ball <= d_ego_ball - role_margin:
                ahead_point_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 0.9
                center_bias = 0.5
                bias = perp(opp_dir_approx) * center_bias
                target_minus_ego = ahead_point_minus_ego - bias
            else:
                # neutral reposition: supportive point near ball but not contesting
                p = perp(opp_dir_approx)
                target_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 1.0 + p * 0.4

            kp2 = 1.2
            kd2 = 0.6
            pos_err = target_minus_ego
            desired = kp2 * pos_err - kd2 * agent_vel
            force = clamp_mag(desired, max_force)

    # Final safety clamp to environment limit
    force = clamp_mag(force, max_force)
    force = torch.where(torch.isfinite(force), force, torch.zeros_like(force))

    return force.unsqueeze(0)

import torch

# Persistent internal state stored on the function object

def agent1_policy(obs: torch.Tensor):
    """
    Supportive Deceptive Dribbler (agent1).

    This policy is designed to coordinate with the partner (agent0) by
    preferentially deferring holding to a partner who is nearer the ball and
    acting as a forward outlet/repositioner when the partner is holding. If
    agent1 is clearly the better candidate to hold, it will execute the
    shielding/feint/release behavior similar to the partner but with a
    slightly larger role_margin so both agents don't contest the ball.

    obs: 1D torch tensor (observation vector) as produced by observation_base.
    Returns: torch.Tensor of shape (1,2) (force applied).

    Persistent internal state on the function object:
      agent1_policy.hold_counter: float
      agent1_policy.feint_phase: float
    """

    # initialize persistent vars
    if not hasattr(agent1_policy, "hold_counter"):
        agent1_policy.hold_counter = 0.0
    if not hasattr(agent1_policy, "feint_phase"):
        agent1_policy.feint_phase = 0.0

    # Tuned constants (slightly more deferential/supporting than partner)
    max_force = 1.0
    hold_distance = 0.45
    shield_offset = 0.25
    feint_amp = 0.10
    feint_freq = 1.8
    release_threshold = 3.0
    hold_counter_decay = 0.9
    # increased so agent1 will more often act as outlet when partner is only slightly closer
    role_margin = 0.4
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

    # ensure flat vector
    obs = obs.view(-1)
    total = obs.shape[0]
    device = obs.device
    dtype = obs.dtype

    # parse observation layout (same layout as partner code expects)
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

    # teammate block (single teammate)
    a_minus_tm = teammates[0:2]
    tm_vel_minus = teammates[2:4]
    tm_vel = teammates[4:6]
    tm_force = teammates[6:8]

    # vec(ball, teammate) = teammate - ball = -(a - t) + (a - b)
    vec_ball_teammate = -a_minus_tm + agent_pos_minus_ball
    d_tm_ball = length(vec_ball_teammate)

    # approx opponent direction: from our goal toward ball/opponent side
    opp_dir_approx = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)

    # parse two adversaries
    advs = []
    for k in range(2):
        seg = adversaries[k * 8: (k + 1) * 8]
        a_minus_adv = seg[0:2]
        a_vel_minus_adv_vel = seg[2:4]
        adv_vel = seg[4:6]
        adv_force = seg[6:8]
        adv_minus_ball = -a_minus_adv + agent_pos_minus_ball
        adv_minus_goal = adv_minus_ball + ball_pos_minus_goal
        advs.append({
            "adv_minus_ball": adv_minus_ball,
            "adv_minus_goal": adv_minus_goal,
            "adv_vel": adv_vel,
            "adv_force": adv_force,
        })

    # distances to ball and goal
    dists_adv_ball = torch.stack([length(ad["adv_minus_ball"]) for ad in advs])
    idx_adv_ball = int(torch.argmin(dists_adv_ball).item())
    adv_near = advs[idx_adv_ball]
    d_adv_ball = dists_adv_ball[idx_adv_ball]

    dists_adv_goal = torch.stack([length(ad["adv_minus_goal"]) for ad in advs])
    idx_adv_goal = int(torch.argmin(dists_adv_goal).item())
    adv_near_goal = advs[idx_adv_goal]
    d_adv_goal = dists_adv_goal[idx_adv_goal]

    # default force
    force = torch.zeros(2, dtype=dtype, device=device)

    # Defensive fallback: highest priority
    if d_adv_goal < cover_distance:
        dir_to_adv = norm(adv_near_goal["adv_minus_goal"]) if length(adv_near_goal["adv_minus_goal"]) > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        rdist = cover_distance * 0.6
        target_minus_ego = -agent_pos_minus_goal + dir_to_adv * rdist
        kp = 1.6
        kd = 0.7
        desired = kp * target_minus_ego - kd * agent_vel
        force = clamp_mag(desired, max_force)
        agent1_policy.hold_counter *= hold_counter_decay
        return force.unsqueeze(0)

    # Determine if teammate likely holding (heuristic): teammate very close to ball and relatively slow
    teammate_speed = length(tm_vel)
    teammate_holding = (d_tm_ball <= (hold_distance + 1e-9)) and (teammate_speed < 0.35)

    # Decide if this agent can/should hold: use larger role_margin to favor being supportive
    can_hold = (d_ego_ball <= hold_distance + 1e-9) and (d_ego_ball <= (d_tm_ball - role_margin + 1e-9))

    # If teammate appears to be holding, prefer outlet/support role immediately
    if teammate_holding and not can_hold:
        agent1_policy.hold_counter *= hold_counter_decay
        # Become an inside outlet: position in front-of-ball toward opponent side, offset slightly for pass
        ahead_point_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 0.95
        center_bias = 0.55
        bias = perp(opp_dir_approx) * center_bias
        target_minus_ego = ahead_point_minus_ego - bias
        kp2 = 1.3
        kd2 = 0.65
        pos_err = target_minus_ego
        desired = kp2 * pos_err - kd2 * agent_vel
        force = clamp_mag(desired, max_force)
        return force.unsqueeze(0)

    # Otherwise behave similarly to shielding dribbler but defer more often
    if can_hold:
        agent1_policy.hold_counter += 1.0

        # Determine protect direction (toward teammate if teammate ahead, else away from nearest adversary)
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

        # shield point relative to ego
        shield_point_minus_ego = -agent_pos_minus_ball + protect_dir * shield_offset

        # feint oscillation
        agent1_policy.feint_phase += feint_freq * 0.18
        phase = torch.tensor(agent1_policy.feint_phase, dtype=dtype, device=device)
        lateral = perp(protect_dir)
        if length(lateral) > eps:
            lateral = norm(lateral)
        else:
            lateral = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        feint = lateral * (feint_amp * torch.sin(phase))

        # bias feint if adversary committing
        adv_vel = adv_near["adv_vel"]
        adv_vel_len = length(adv_vel)
        adv_motion_dir = norm(adv_vel) if adv_vel_len > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        adv_to_ball = adv_near["adv_minus_ball"]
        adv_committing = False
        if length(adv_to_ball) > eps and adv_vel_len > eps:
            adv_committing = (torch.dot(adv_motion_dir, norm(adv_to_ball)) > 0.6)
        if adv_committing:
            feint = feint * 1.5 - adv_motion_dir * 0.04

        target_minus_ego = shield_point_minus_ego + feint

        # Release conditions
        teammate_ready = (length(to_teammate) < 2.5) and teammate_ahead
        adv_retreating = False
        if adv_vel_len > eps and length(adv_to_ball) > eps:
            adv_retreating = (torch.dot(adv_motion_dir, norm(adv_to_ball)) < -0.2)

        if (agent1_policy.hold_counter >= release_threshold) and (teammate_ready or adv_retreating):
            # Release: prefer passing to teammate if ready
            if teammate_ready:
                teammate_minus_goal = vec_ball_teammate + ball_pos_minus_goal
                if length(teammate_minus_goal) > eps:
                    tm_forward = norm(teammate_minus_goal)
                else:
                    tm_forward = opp_dir_approx
                release_target_minus_ego = -agent_pos_minus_ball + (vec_ball_teammate + tm_forward * 0.4)
            else:
                release_dir = norm(ball_pos_minus_goal) if length(ball_pos_minus_goal) > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)
                away_from_adv = norm(adv_near["adv_minus_ball"]) if length(adv_near["adv_minus_ball"]) > eps else torch.tensor([0.0, 0.0], dtype=dtype, device=device)
                combined = release_dir + 0.6 * away_from_adv
                if length(combined) > eps:
                    release_dir = norm(combined)
                release_target_minus_ego = -agent_pos_minus_ball + release_dir * 0.6

            desired_dir = release_target_minus_ego
            if length(desired_dir) > eps:
                desired_dir = norm(desired_dir)
            else:
                desired_dir = torch.tensor([0.0, 0.0], dtype=dtype, device=device)

            base_force = desired_dir * (max_force * 1.0) - 0.6 * agent_vel
            force = clamp_mag(base_force, max_force)
            agent1_policy.hold_counter *= 0.22
            return force.unsqueeze(0)

        else:
            kp = 1.6
            kd = 0.7
            pos_err = target_minus_ego
            desired = kp * pos_err - kd * agent_vel
            force = clamp_mag(desired, max_force)
            return force.unsqueeze(0)

    else:
        # Not in holdable position: act as supporting outlet or reposition
        agent1_policy.hold_counter *= hold_counter_decay

        # If teammate is closer to ball, actively become an inside outlet
        if d_tm_ball <= d_ego_ball - (role_margin * 0.5):
            ahead_point_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 0.95
            center_bias = 0.6
            bias = perp(opp_dir_approx) * center_bias
            target_minus_ego = ahead_point_minus_ego - bias
        else:
            # neutral supportive reposition near ball but not contesting
            p = perp(opp_dir_approx)
            target_minus_ego = -agent_pos_minus_ball + opp_dir_approx * 1.05 + p * 0.45

        kp2 = 1.25
        kd2 = 0.62
        pos_err = target_minus_ego
        desired = kp2 * pos_err - kd2 * agent_vel
        force = clamp_mag(desired, max_force)
        return force.unsqueeze(0)


# end of agent1_policy