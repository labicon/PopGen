import torch

def agent0_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    High-Tempo Press-and-Break Specialist policy (agent 0).
    Expects obs to be a 1D torch tensor in the environment's concatenated format.
    Returns action as torch.tensor of shape (1,2) representing applied force.
    """

    # --- Helpers ---
    def length(v):
        return torch.norm(v) + 1e-8

    def norm(v):
        n = torch.norm(v)
        if n.item() <= 1e-8:
            return torch.zeros_like(v)
        return v / n

    def clamp_mag(v, m):
        mag = torch.norm(v)
        if mag.item() <= m:
            return v
        return v / mag * m

    def dot(a, b):
        return (a * b).sum()

    # --- Parameters ---
    max_force = 1.2
    burst_force_multiplier = 1.7
    press_range = 1.2
    control_radius = 0.35
    prefer_pass_distance = 3.0
    defend_threshold = 1.3
    role_margin = 0.15
    damping = 0.6

    # --- Unpack observation (assumes rotation length == 1) ---
    # Layout (by construction in observation_base):
    # [0:2]   agent_force
    # [2:4]   agent_pos - ball_pos
    # [4:6]   agent_vel - ball_vel
    # [6:8]   ball_pos - goal_pos
    # [8:10]  ball_vel
    # [10:12] ball_force
    # [12:13] agent_rot
    # [13:15] agent_pos - goal_pos
    # [15:17] agent_vel
    # [17:33] adversaries (2 adversaries * 8 entries)
    # [33:41] teammates (1 teammate * 8 entries)

    # Ensure 1D
    o = obs.view(-1)

    # Basic slices
    agent_force = o[0:2]
    ap_b = o[2:4]        # agent_pos - ball_pos
    av_b = o[4:6]       # agent_vel - ball_vel (not used directly)
    bp_g = o[6:8]       # ball_pos - goal_pos
    ball_vel = o[8:10]
    ball_force = o[10:12]
    # agent_rot = o[12:13]
    ap_g = o[13:15]     # agent_pos - goal_pos
    agent_vel = o[15:17]

    adv_block = o[17:33].reshape(2, 8)
    tm_block = o[33:41].reshape(8)

    # Convert to agent-centric coordinates (positions relative to agent)
    # ball_rel = ball_pos - agent_pos = - (agent_pos - ball_pos)
    ball = -ap_b
    # teammate position relative to agent: teammate_pos - agent_pos = - (agent_pos - teammate_pos)
    teammate = -tm_block[0:2]

    # teammate velocity is provided
    tvel = tm_block[4:6]

    # adversaries positions relative to agent and their velocities
    adv_rels = -adv_block[:, 0:2]        # shape (2,2)
    adv_vels = adv_block[:, 4:6]

    # goal relative to agent: goal_pos - agent_pos = - (agent_pos - goal_pos)
    goal_rel = -ap_g

    # Distances
    d_ego_ball = length(ball)
    # teammate to ball (vector ball->teammate = teammate - ball)
    to_teammate = teammate - ball
    d_tm_ball = length(to_teammate)

    # adversary distances to ball
    adv_pos_minus_ball = adv_rels - ball  # adv_pos - ball_pos (in agent frame)
    d_adv_ball = torch.norm(adv_pos_minus_ball, dim=-1)
    idx_adv_ball = torch.argmin(d_adv_ball)
    adv_near_ball_rel = adv_rels[idx_adv_ball]
    adv_near_ball_vel = adv_vels[idx_adv_ball]
    d_adv_ball_min = d_adv_ball[idx_adv_ball]

    # adversary nearest to our goal
    # adv_pos - goal_pos = adv_rel - goal_rel
    adv_pos_minus_goal = adv_rels - goal_rel
    d_adv_goal = torch.norm(adv_pos_minus_goal, dim=-1)
    idx_adv_goal = torch.argmin(d_adv_goal)
    adv_near_goal_rel = adv_rels[idx_adv_goal]
    d_adv_goal_min = d_adv_goal[idx_adv_goal]

    # Default target (relative to agent) and burst
    target_rel = torch.zeros(2)
    burst = 1.0

    # Defensive override
    if d_adv_goal_min.item() < defend_threshold:
        # place between adversary and our goal, offset outward from goal
        # direction from goal toward adversary: adv - goal = adv_pos_minus_goal[idx]
        dir_to_adv = adv_pos_minus_goal[idx_adv_goal]
        dir_to_adv = norm(dir_to_adv)
        offset = defend_threshold * 0.6
        # We want target = goal + dir_to_adv * offset; in agent frame: (goal_rel + dir*offset)
        target_rel = goal_rel + dir_to_adv * offset
        # but pos_err = target - agent_pos ; here target_rel already is goal_pos - agent_pos + ... so good
        burst = 1.0
    else:
        # Evaluate press eligibility
        will_press = (d_adv_ball_min.item() < d_ego_ball + 0.05) and (
            (d_ego_ball <= d_tm_ball - role_margin) or (d_ego_ball < press_range)
        )

        if will_press and d_ego_ball <= press_range:
            # PRESS PHASE: approach a point slightly offset from the ball to intercept
            adv_motion = adv_near_ball_vel
            if length(adv_motion).item() <= 1e-6:
                # fallback: use vector from our goal to ball (ball - goal)
                # ball_pos - goal_pos in agent frame = (ball - goal_rel)
                adv_motion = ball - goal_rel
            escape_dir = norm(adv_motion)
            approach_dir = -escape_dir
            if length(approach_dir).item() <= 1e-6:
                approach_point_rel = ball.clone()
            else:
                approach_point_rel = ball + approach_dir * 0.25
            target_rel = approach_point_rel
            burst = burst_force_multiplier

        elif (d_ego_ball <= control_radius) or (d_ego_ball < d_tm_ball - role_margin):
            # CONTROL / BREAK PHASE
            # opp_dir approximated as ball_pos - goal_pos = (ball - goal_rel)
            opp_dir = norm(ball - goal_rel)

            # Determine if teammate is ahead (relative to opponent goal)
            to_teammate = teammate - ball  # vec(ball, teammate)
            to_teammate_norm = norm(to_teammate) if length(to_teammate).item() > 1e-6 else torch.zeros(2)
            teammate_ahead = dot(to_teammate_norm, -opp_dir) > 0.5
            teammate_close_forward = (length(to_teammate).item() < prefer_pass_distance) and teammate_ahead

            if teammate_close_forward:
                # PASS-ORIENTED BREAK: aim slightly in front of teammate (toward opponent)
                # direction from teammate toward opponent goal ~ (teammate - goal)
                teammate_to_goal = teammate - goal_rel
                dir_forward = norm(teammate_to_goal)
                if length(dir_forward).item() <= 1e-6:
                    pass_target_rel = teammate.clone()
                else:
                    pass_target_rel = teammate + dir_forward * 0.4
                target_rel = pass_target_rel
                burst = burst_force_multiplier
            else:
                # DIRECT BREAK: push toward opponent goal (away from our goal)
                base_break = norm(ball - goal_rel)
                # offset slightly away from nearest adversary to ball
                away_from_adv = norm(ball - adv_near_ball_rel) if d_adv_ball_min.item() > 1e-6 else torch.zeros(2)
                mixed = base_break + 0.6 * away_from_adv
                break_dir = norm(mixed)
                target_rel = ball + break_dir * 0.6
                burst = burst_force_multiplier
        else:
            # REPOSITION / READY PHASE: take an advanced central ready spot relative to ball
            opp_dir = norm(ball - goal_rel)
            perp = torch.tensor([-opp_dir[1], opp_dir[0]])
            center_bias = 0.5
            # choose sign to push toward center: reduce lateral extremes by pushing opposite sign of perp
            s0 = torch.sign(perp[0])
            s1 = torch.sign(perp[1])
            s0 = s0 if s0.item() != 0.0 else torch.tensor(1.0)
            s1 = s1 if s1.item() != 0.0 else torch.tensor(1.0)
            # position slightly forward and biased to center
            target_rel = ball + opp_dir * 1.0 + perp * (center_bias * -torch.stack([s0, s1]))
            # For safety, if that produced NaNs or zeros, fallback
            if torch.isnan(target_rel).any():
                target_rel = ball + opp_dir * 1.0
            burst = 1.0

    # Low-level PD-like steering toward target_rel (target position relative to agent)
    pos_err = target_rel  # because agent is origin in our agent-centric variables
    desired_dir = norm(pos_err) if length(pos_err).item() > 1e-6 else torch.zeros(2)
    mag_scale = max_force * (burst if burst > 1.0 else 1.0)
    base_force = desired_dir * mag_scale
    force = base_force - damping * agent_vel
    # clamp magnitude
    max_allowed = max_force * burst_force_multiplier
    force = clamp_mag(force, max_allowed)

    # Small jitter when not bursting to reduce predictability
    if float(burst) == 1.0:
        perp = torch.tensor([-force[1], force[0]])
        if length(perp).item() > 1e-6:
            perp_j = norm(perp) * 0.03
            # random sign small variability
            rand_sign = (torch.rand(1).item() * 2.0 - 1.0)
            perp_j = perp_j * rand_sign
            force = force + perp_j
            force = clamp_mag(force, max_allowed)

    action = force.view(1, 2)
    return action

import torch

def agent1_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    Supportive High-Tempo Press-and-Break Specialist policy (agent 1).
    Coordinates with a high-tempo press-and-break partner by:
      - Avoiding redundant simultaneous presses (respecting role_margin)
      - Moving into forward receiving space when partner presses (be the pass/receive target)
      - Taking defensive cover when an adversary threatens our goal
      - Pressing or breaking when appropriate and not duplicating partner's press

    Expects obs to be a 1D torch tensor in the environment's concatenated format.
    Returns action as torch.tensor of shape (1,2) representing applied force.
    """

    # --- Helpers ---
    def length(v: torch.Tensor):
        return torch.norm(v) + 1e-8

    def norm(v: torch.Tensor):
        n = torch.norm(v)
        if n.item() <= 1e-8:
            return torch.zeros_like(v)
        return v / n

    def clamp_mag(v: torch.Tensor, m: float):
        mag = torch.norm(v)
        if mag.item() <= m:
            return v
        return v / mag * m

    def dot(a: torch.Tensor, b: torch.Tensor):
        return (a * b).sum()

    # --- Parameters (aligned with partner) ---
    max_force = 1.2
    burst_force_multiplier = 1.7
    press_range = 1.2
    control_radius = 0.35
    prefer_pass_distance = 3.0
    defend_threshold = 1.3
    role_margin = 0.15
    damping = 0.6

    o = obs.view(-1)

    # --- Unpack observation (same layout as partner expects) ---
    agent_force = o[0:2]
    ap_b = o[2:4]        # agent_pos - ball_pos
    av_b = o[4:6]
    bp_g = o[6:8]       # ball_pos - goal_pos
    ball_vel = o[8:10]
    ball_force = o[10:12]
    # agent_rot = o[12:13]
    ap_g = o[13:15]     # agent_pos - goal_pos
    agent_vel = o[15:17]

    adv_block = o[17:33].reshape(2, 8)
    tm_block = o[33:41].reshape(8)

    # Convert to agent-centric coordinates (positions relative to this agent)
    ball = -ap_b
    teammate = -tm_block[0:2]
    tvel = tm_block[4:6]

    adv_rels = -adv_block[:, 0:2]        # adversary positions relative to this agent
    adv_vels = adv_block[:, 4:6]

    goal_rel = -ap_g

    # Distances and basic relations
    d_ego_ball = length(ball)
    to_teammate = teammate - ball  # teammate_pos - ball_pos
    d_tm_ball = length(to_teammate)

    adv_pos_minus_ball = adv_rels - ball  # adv_pos - ball_pos
    d_adv_ball = torch.norm(adv_pos_minus_ball, dim=-1)
    idx_adv_ball = torch.argmin(d_adv_ball)
    d_adv_ball_min = d_adv_ball[idx_adv_ball]
    adv_near_ball_rel = adv_rels[idx_adv_ball]
    adv_near_ball_vel = adv_vels[idx_adv_ball]

    adv_pos_minus_goal = adv_rels - goal_rel
    d_adv_goal = torch.norm(adv_pos_minus_goal, dim=-1)
    idx_adv_goal = torch.argmin(d_adv_goal)
    d_adv_goal_min = d_adv_goal[idx_adv_goal]
    adv_near_goal_rel = adv_rels[idx_adv_goal]

    # Determine if teammate (agent0) is likely to attempt a press.
    # We approximate teammate's distance to ball by d_tm_ball and relative closeness to adversary near ball.
    # Teammate is represented in this agent's frame; comparisons remain valid.
    will_teammate_press = False
    # nearest adversary to ball distance already computed (shared for both agents)
    if (d_adv_ball_min.item() < d_tm_ball + 0.05) and (
        (d_tm_ball <= d_ego_ball - role_margin) or (d_tm_ball < press_range)
    ):
        will_teammate_press = True

    # Determine if self should press (but avoid if teammate is clearly better)
    will_self_press = False
    if (d_adv_ball_min.item() < d_ego_ball + 0.05) and (
        (d_ego_ball <= d_tm_ball - role_margin) or (d_ego_ball < press_range)
    ):
        will_self_press = True

    # Default target relative to this agent
    target_rel = torch.zeros(2)
    burst = 1.0

    # Defensive override: highest priority
    if d_adv_goal_min.item() < defend_threshold:
        # place between that adversary and our goal, offset outward from goal
        dir_to_adv = adv_pos_minus_goal[idx_adv_goal]
        dir_to_adv = norm(dir_to_adv)
        offset = defend_threshold * 0.6
        target_rel = goal_rel + dir_to_adv * offset
        burst = 1.0

    else:
        # If teammate appears to be committing to a press, adopt a supporting forward receiver role
        if will_teammate_press and (not will_self_press):
            # Move into forward space ahead of the ball towards the opponent goal to be a pass target.
            # Opponent direction approximated as ball - our_goal
            opp_dir = norm(ball - goal_rel)
            # lateral perpendicular
            perp = torch.tensor([-opp_dir[1], opp_dir[0]])
            # choose lateral sign to create separation from teammate: push to side opposite to teammate relative to ball
            side_dot = dot(perp, teammate - ball)
            side_sign = -1.0 if side_dot.item() >= 0.0 else 1.0
            lateral = perp * (0.4 * side_sign)
            forward_dist = 1.2
            target_rel = ball + opp_dir * forward_dist + lateral
            burst = 1.0  # be ready but not burst so partner can do the aggressive move

        elif will_self_press and (not will_teammate_press):
            # PRESS PHASE for self: approach ball with an intercept angle like partner
            adv_motion = adv_near_ball_vel
            if length(adv_motion).item() <= 1e-6:
                adv_motion = ball - goal_rel
            escape_dir = norm(adv_motion)
            approach_dir = -escape_dir
            if length(approach_dir).item() <= 1e-6:
                approach_point_rel = ball.clone()
            else:
                approach_point_rel = ball + approach_dir * 0.25
            target_rel = approach_point_rel
            burst = burst_force_multiplier

        elif (d_ego_ball <= control_radius) or (d_ego_ball < d_tm_ball - role_margin):
            # CONTROL / BREAK PHASE for self: prefer passing to teammate if they are a forward option,
            # otherwise break toward opponent goal but bias away from nearby opponent
            opp_dir = norm(ball - goal_rel)
            to_teammate = teammate - ball
            to_teammate_norm = norm(to_teammate) if length(to_teammate).item() > 1e-6 else torch.zeros(2)
            teammate_ahead = dot(to_teammate_norm, -opp_dir) > 0.5
            teammate_close_forward = (length(to_teammate).item() < prefer_pass_distance) and teammate_ahead

            if teammate_close_forward:
                # aim slightly in front of teammate
                teammate_to_goal = teammate - goal_rel
                dir_forward = norm(teammate_to_goal)
                if length(dir_forward).item() <= 1e-6:
                    pass_target_rel = teammate.clone()
                else:
                    pass_target_rel = teammate + dir_forward * 0.4
                target_rel = pass_target_rel
                burst = burst_force_multiplier
            else:
                base_break = norm(ball - goal_rel)
                away_from_adv = norm(ball - adv_near_ball_rel) if d_adv_ball_min.item() > 1e-6 else torch.zeros(2)
                mixed = base_break + 0.6 * away_from_adv
                break_dir = norm(mixed)
                target_rel = ball + break_dir * 0.6
                burst = burst_force_multiplier

        else:
            # REPOSITION / READY PHASE: adopt an advanced central supporting position relative to ball.
            opp_dir = norm(ball - goal_rel)
            perp = torch.tensor([-opp_dir[1], opp_dir[0]])
            center_bias = 0.5
            # bias toward center by pushing opposite sign of perp components
            s0 = torch.sign(perp[0])
            s1 = torch.sign(perp[1])
            s0 = s0 if s0.item() != 0.0 else torch.tensor(1.0)
            s1 = s1 if s1.item() != 0.0 else torch.tensor(1.0)
            target_rel = ball + opp_dir * 1.0 + perp * (center_bias * -torch.stack([s0, s1]))
            if torch.isnan(target_rel).any():
                target_rel = ball + opp_dir * 1.0
            burst = 1.0

    # Low-level PD-like steering toward target_rel
    pos_err = target_rel
    desired_dir = norm(pos_err) if length(pos_err).item() > 1e-6 else torch.zeros(2)
    mag_scale = max_force * (burst if burst > 1.0 else 1.0)
    base_force = desired_dir * mag_scale
    force = base_force - damping * agent_vel
    max_allowed = max_force * burst_force_multiplier
    force = clamp_mag(force, max_allowed)

    # Small jitter when not bursting to reduce predictability
    if float(burst) == 1.0:
        perp = torch.tensor([-force[1], force[0]])
        if length(perp).item() > 1e-6:
            perp_j = norm(perp) * 0.03
            rand_sign = (torch.rand(1).item() * 2.0 - 1.0)
            perp_j = perp_j * rand_sign
            force = force + perp_j
            force = clamp_mag(force, max_allowed)

    action = force.view(1, 2)
    return action