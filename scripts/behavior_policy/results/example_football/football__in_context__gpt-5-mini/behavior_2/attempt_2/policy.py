import torch

def agent0_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    High-Tempo Press-and-Break Specialist policy (agent 0).
    Expects obs to be a 1D torch tensor in the environment's concatenated format.
    Returns action as torch.tensor of shape (1,2) representing applied force.

    Notes on fixes from prior feedback:
    - Ensures final action is clipped into environment action bounds ([-1, 1] per component).
    - Keeps robust parsing similar to previous working version to avoid shape mismatches.
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

    # --- Parameters (adjusted to respect env action limits) ---
    # environment action limit per component appears to be 1.0, so ensure final action lies in [-1,1]
    max_force = 1.0
    burst_force_multiplier = 1.6
    press_range = 1.2
    control_radius = 0.35
    prefer_pass_distance = 3.0
    defend_threshold = 1.3
    role_margin = 0.15
    damping = 0.6

    # Ensure 1D
    o = obs.view(-1)
    device = o.device
    dtype = o.dtype

    # Safe-list length check: if the vector is short, pad with zeros to avoid indexing errors
    n = o.numel()
    if n < 41:
        pad = o.new_zeros(41 - n)
        o = torch.cat([o, pad], dim=0)

    # --- Unpack observation (assumes standard layout produced by observation_base) ---
    # Layout as used previously:
    # [0:2]   agent_force
    # [2:4]   agent_pos - ball_pos
    # [4:6]   agent_vel - ball_vel
    # [6:8]   ball_pos - goal_pos
    # [8:10]  ball_vel
    # [10:12] ball_force
    # [12:13] agent_rot (assumed length 1)
    # [13:15] agent_pos - goal_pos
    # [15:17] agent_vel
    # [17:33] adversaries (2 adversaries * 8 entries)
    # [33:41] teammates (1 teammate * 8 entries)

    agent_force = o[0:2]
    ap_b = o[2:4]        # agent_pos - ball_pos
    av_b = o[4:6]        # agent_vel - ball_vel
    bp_g = o[6:8]        # ball_pos - goal_pos
    ball_vel = o[8:10]
    ball_force = o[10:12]
    # agent_rot = o[12:13]
    ap_g = o[13:15]      # agent_pos - goal_pos
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

    # adversary positions relative to agent and their velocities
    adv_rels = -adv_block[:, 0:2]        # shape (2,2)
    adv_vels = adv_block[:, 4:6]

    # goal relative to agent: goal_pos - agent_pos = - (agent_pos - goal_pos)
    goal_rel = -ap_g

    # Distances
    d_ego_ball = length(ball)
    # teammate to ball
    to_teammate = teammate - ball
    d_tm_ball = length(to_teammate)

    # adversary distances to ball
    adv_pos_minus_ball = adv_rels - ball.unsqueeze(0)  # adv_pos - ball_pos
    d_adv_ball = torch.norm(adv_pos_minus_ball, dim=-1)
    idx_adv_ball = int(torch.argmin(d_adv_ball).item())
    adv_near_ball_rel = adv_rels[idx_adv_ball]
    adv_near_ball_vel = adv_vels[idx_adv_ball]
    d_adv_ball_min = d_adv_ball[idx_adv_ball]

    # adversary nearest to our goal
    adv_pos_minus_goal = adv_rels - goal_rel.unsqueeze(0)
    d_adv_goal = torch.norm(adv_pos_minus_goal, dim=-1)
    idx_adv_goal = int(torch.argmin(d_adv_goal).item())
    adv_near_goal_rel = adv_rels[idx_adv_goal]
    d_adv_goal_min = d_adv_goal[idx_adv_goal]

    # Default target (relative to agent) and burst
    target_rel = torch.zeros(2, device=device, dtype=dtype)
    burst = 1.0

    # Defensive override
    if d_adv_goal_min.item() < defend_threshold:
        # place between adversary and our goal, offset outward from goal
        dir_to_adv = adv_pos_minus_goal[idx_adv_goal]
        dir_to_adv = norm(dir_to_adv)
        offset = defend_threshold * 0.6
        target_rel = goal_rel + dir_to_adv * offset
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
            opp_dir = norm(ball - goal_rel)

            # Determine if teammate is ahead (relative to opponent goal)
            to_teammate = teammate - ball  # vec(ball, teammate)
            to_teammate_norm = norm(to_teammate) if length(to_teammate).item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
            teammate_ahead = dot(to_teammate_norm, -opp_dir) > 0.5
            teammate_close_forward = (length(to_teammate).item() < prefer_pass_distance) and teammate_ahead

            if teammate_close_forward:
                # PASS-ORIENTED BREAK: aim slightly in front of teammate (toward opponent)
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
                away_from_adv = norm(ball - adv_near_ball_rel) if d_adv_ball_min.item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
                mixed = base_break + 0.6 * away_from_adv
                break_dir = norm(mixed)
                target_rel = ball + break_dir * 0.6
                burst = burst_force_multiplier

        else:
            # REPOSITION / READY PHASE: advanced, central ready position
            opp_dir = norm(ball - goal_rel)
            perp = torch.tensor([-opp_dir[1], opp_dir[0]], device=device, dtype=dtype)
            center_bias = 0.5
            s0 = torch.sign(perp[0])
            s1 = torch.sign(perp[1])
            s0 = s0 if s0.item() != 0.0 else torch.tensor(1.0, device=device, dtype=dtype)
            s1 = s1 if s1.item() != 0.0 else torch.tensor(1.0, device=device, dtype=dtype)
            lateral = perp * (center_bias * -torch.stack([s0, s1]))
            target_rel = ball + opp_dir * 1.0 + lateral
            if torch.isnan(target_rel).any():
                target_rel = ball + opp_dir * 1.0
            burst = 1.0

    # Low-level PD-like steering toward target_rel (agent is origin)
    pos_err = target_rel
    desired_dir = norm(pos_err) if length(pos_err).item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
    mag_scale = max_force * (burst if burst > 1.0 else 1.0)
    base_force = desired_dir * mag_scale
    force = base_force - damping * agent_vel

    # Clamp overall magnitude to environment limit
    env_max = 1.0
    force = clamp_mag(force, env_max)

    # Small jitter when not bursting to reduce predictability
    if float(burst) == 1.0:
        perp = torch.tensor([-force[1], force[0]], device=device, dtype=dtype)
        if length(perp).item() > 1e-6:
            perp_j = norm(perp) * 0.03
            # small random sign
            rand_sign = (torch.rand(1, device=device).item() * 2.0 - 1.0)
            perp_j = perp_j * rand_sign
            force = force + perp_j
            force = clamp_mag(force, env_max)

    # Final per-component clamp to ensure we never exceed actuator bounds
    force = torch.clamp(force, -env_max, env_max)

    action = force.view(1, 2)
    return action

import torch

def agent1_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    Agent 1 policy (support-oriented High-Tempo Press-and-Break Specialist).

    This policy is designed to coordinate with the partner (agent0) by:
    - Deferring an aggressive press when the teammate (partner) is clearly closer or actively bursting,
      to avoid redundant simultaneous presses.
    - Adopting a receiver / forward-support role when the teammate is pressing (moving into pass space
      ahead of the ball toward the opponent goal).
    - Keeping the same defensive override as the partner: if an adversary is dangerously near our goal,
      move to interpose.
    - If agent1 itself is the appropriate agent to press (closer to the ball and not superseded by the
      teammate), it will perform a press similar to the partner, including burst behavior.

    Expects the same observation layout produced by observation_base, and returns a torch.Tensor of
    shape (1,2) for the applied force.
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

    # --- Parameters ---
    max_force = 1.0
    burst_force_multiplier = 1.55
    press_range = 1.2
    control_radius = 0.35
    prefer_pass_distance = 3.0
    defend_threshold = 1.3
    role_margin = 0.15
    damping = 0.6

    # Ensure 1D
    o = obs.view(-1)
    device = o.device
    dtype = o.dtype

    # Safe pad like partner
    n = o.numel()
    if n < 41:
        pad = o.new_zeros(41 - n)
        o = torch.cat([o, pad], dim=0)

    # Unpack observation indices consistent with observation_base
    agent_force = o[0:2]
    ap_b = o[2:4]        # agent_pos - ball_pos
    av_b = o[4:6]        # agent_vel - ball_vel
    bp_g = o[6:8]        # ball_pos - goal_pos
    ball_vel = o[8:10]
    ball_force = o[10:12]
    # agent_rot = o[12:13]
    ap_g = o[13:15]      # agent_pos - goal_pos
    agent_vel = o[15:17]

    adv_block = o[17:33].reshape(2, 8)
    tm_block = o[33:41].reshape(8)

    # Convert to agent-centric coordinates (positions relative to agent)
    ball = -ap_b
    teammate = -tm_block[0:2]
    tvel = tm_block[4:6]
    teammate_force = tm_block[6:8]

    adv_rels = -adv_block[:, 0:2]
    adv_vels = adv_block[:, 4:6]
    adv_forces = adv_block[:, 6:8]

    goal_rel = -ap_g

    # Distances
    d_ego_ball = length(ball)
    to_teammate = teammate - ball
    d_tm_ball = length(to_teammate)

    # adversary distances to ball
    adv_pos_minus_ball = adv_rels - ball.unsqueeze(0)
    d_adv_ball = torch.norm(adv_pos_minus_ball, dim=-1)
    idx_adv_ball = int(torch.argmin(d_adv_ball).item())
    adv_near_ball_rel = adv_rels[idx_adv_ball]
    adv_near_ball_vel = adv_vels[idx_adv_ball]
    d_adv_ball_min = d_adv_ball[idx_adv_ball]

    # adversary nearest to our goal
    adv_pos_minus_goal = adv_rels - goal_rel.unsqueeze(0)
    d_adv_goal = torch.norm(adv_pos_minus_goal, dim=-1)
    idx_adv_goal = int(torch.argmin(d_adv_goal).item())
    adv_near_goal_rel = adv_rels[idx_adv_goal]
    d_adv_goal_min = d_adv_goal[idx_adv_goal]

    # teammate activity indicators
    teammate_force_mag = length(teammate_force)
    teammate_vel_mag = length(tvel)

    # Determine if teammate is actively pressing/bursting toward the ball
    teammate_looks_pressing = (teammate_force_mag.item() > 0.6 and d_tm_ball.item() < press_range + 0.3) or (
        teammate_vel_mag.item() > 0.8 and d_tm_ball.item() < press_range + 0.6
    )

    # Default target_rel and burst
    target_rel = torch.zeros(2, device=device, dtype=dtype)
    burst = 1.0

    # Defensive override (highest priority)
    if d_adv_goal_min.item() < defend_threshold:
        dir_to_adv = adv_pos_minus_goal[idx_adv_goal]
        dir_to_adv = norm(dir_to_adv)
        offset = defend_threshold * 0.6
        target_rel = goal_rel + dir_to_adv * offset
        burst = 1.0

    else:
        # If teammate is actively pressing and is closer, adopt receiver/support role
        if teammate_looks_pressing and (d_tm_ball.item() + role_margin < d_ego_ball.item() or (teammate_force_mag.item() > 0.9 and d_tm_ball.item() < press_range + 0.1)):
            # Move to a forward receiving lane: ahead of the ball toward opponent (away from our goal)
            opp_dir = norm(ball - goal_rel)
            # lateral bias: choose side away from nearest adversary to ball to be more open
            away_adv = norm(ball - adv_near_ball_rel) if d_adv_ball_min.item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
            perp = torch.tensor([-opp_dir[1], opp_dir[0]], device=device, dtype=dtype)
            # Decide sign of lateral offset by checking which side is further from nearest adversary
            side_choice = torch.sign(dot(perp, ball - adv_near_ball_rel))
            lateral_off = perp * (0.5 * side_choice)
            # Put target a bit ahead of the ball to receive a forward pass
            target_rel = ball + opp_dir * 1.2 + lateral_off * 0.6
            burst = 1.0  # repositioning, not burst from receiver

        else:
            # Evaluate press eligibility for this agent
            will_press = (d_adv_ball_min.item() < d_ego_ball + 0.05) and (
                (d_ego_ball <= d_tm_ball - role_margin) or (d_ego_ball < press_range)
            )

            if will_press and d_ego_ball <= press_range:
                # PRESS PHASE: approach point slightly offset from ball to intercept (similar to partner)
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
                # CONTROL / BREAK PHASE: if we are effectively touching or closest, decide the counter
                opp_dir = norm(ball - goal_rel)

                # Determine if teammate is ahead relative to opponent goal
                to_teammate = teammate - ball
                to_teammate_norm = norm(to_teammate) if length(to_teammate).item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
                teammate_ahead = dot(to_teammate_norm, -opp_dir) > 0.5
                teammate_close_forward = (length(to_teammate).item() < prefer_pass_distance) and teammate_ahead

                if teammate_close_forward:
                    # PASS-ORIENTED BREAK: place ball slightly in front of teammate toward opponent
                    teammate_to_goal = teammate - goal_rel
                    dir_forward = norm(teammate_to_goal)
                    if length(dir_forward).item() <= 1e-6:
                        pass_target_rel = teammate.clone()
                    else:
                        pass_target_rel = teammate + dir_forward * 0.4
                    target_rel = pass_target_rel
                    burst = burst_force_multiplier
                else:
                    # DIRECT BREAK: push toward opponent goal slightly away from nearest adversary
                    base_break = norm(ball - goal_rel)
                    away_from_adv = norm(ball - adv_near_ball_rel) if d_adv_ball_min.item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
                    mixed = base_break + 0.6 * away_from_adv
                    break_dir = norm(mixed)
                    target_rel = ball + break_dir * 0.6
                    burst = burst_force_multiplier

            else:
                # REPOSITION / READY PHASE: take an advanced, central ready position to be able to press quickly
                opp_dir = norm(ball - goal_rel)
                perp = torch.tensor([-opp_dir[1], opp_dir[0]], device=device, dtype=dtype)
                center_bias = 0.5
                s0 = torch.sign(perp[0])
                s1 = torch.sign(perp[1])
                s0 = s0 if s0.item() != 0.0 else torch.tensor(1.0, device=device, dtype=dtype)
                s1 = s1 if s1.item() != 0.0 else torch.tensor(1.0, device=device, dtype=dtype)
                lateral = perp * (center_bias * -torch.stack([s0, s1]))
                target_rel = ball + opp_dir * 1.0 + lateral
                if torch.isnan(target_rel).any():
                    target_rel = ball + opp_dir * 1.0
                burst = 1.0

    # Low-level PD-like steering toward target_rel (agent is origin)
    pos_err = target_rel
    desired_dir = norm(pos_err) if length(pos_err).item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
    mag_scale = max_force * (burst if burst > 1.0 else 1.0)
    base_force = desired_dir * mag_scale
    force = base_force - damping * agent_vel

    # If we are in receiver role (teammate pressing) give a bit more urgency to reach the pass-lane
    if teammate_looks_pressing and (d_tm_ball.item() + role_margin < d_ego_ball.item() or teammate_force_mag.item() > 0.9):
        # small boost toward target to get into the pass lane faster
        force = force + desired_dir * 0.15

    # Clamp overall magnitude to environment limit
    env_max = 1.0
    force = clamp_mag(force, env_max)

    # Small jitter when not bursting to reduce predictability
    if float(burst) == 1.0:
        perp = torch.tensor([-force[1], force[0]], device=device, dtype=dtype)
        if length(perp).item() > 1e-6:
            perp_j = norm(perp) * 0.03
            rand_sign = (torch.rand(1, device=device).item() * 2.0 - 1.0)
            perp_j = perp_j * rand_sign
            force = force + perp_j
            force = clamp_mag(force, env_max)

    # Final per-component clamp
    force = torch.clamp(force, -env_max, env_max)

    action = force.view(1, 2)
    return action