import torch

def agent0_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    High-Tempo Press-and-Break Specialist policy (agent 0).
    Robust parsing of observation vector to avoid shape-mismatch errors when some
    parts of the observation are missing or have unexpected lengths.

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

    # Safe slice helper: returns slice if available, otherwise zeros of requested length
    def safe_slice(base: torch.Tensor, start: int, length_req: int):
        if base.numel() <= start:
            return base.new_zeros(length_req)
        avail = base.numel() - start
        take = min(length_req, avail)
        out = base[start : start + take]
        if take < length_req:
            pad = base.new_zeros(length_req - take)
            out = torch.cat([out, pad], dim=0)
        return out

    # Ensure 1D
    o = obs.view(-1)
    n = o.numel()
    device = o.device
    dtype = o.dtype

    # Expected block sizes (from observation_base): teammates:8, adversaries:16 (2*8)
    TEAM_LEN = 8
    ADV_LEN = 16

    # Compute base length (remaining elements at front)
    base_len = max(0, n - TEAM_LEN - ADV_LEN)
    base = o[:base_len] if base_len > 0 else o.new_zeros(0)

    # Extract adversary block and teammate block robustly from the end
    adv_block_raw = o[base_len : base_len + ADV_LEN] if n >= base_len + 1 else o.new_zeros(0)
    if adv_block_raw.numel() < ADV_LEN:
        adv_block = torch.cat([adv_block_raw, o.new_zeros(ADV_LEN - adv_block_raw.numel())], dim=0)
    else:
        adv_block = adv_block_raw

    tm_block_raw_start = base_len + ADV_LEN
    tm_block_raw = o[tm_block_raw_start : tm_block_raw_start + TEAM_LEN] if n >= tm_block_raw_start + 1 else o.new_zeros(0)
    if tm_block_raw.numel() < TEAM_LEN:
        tm_block = torch.cat([tm_block_raw, o.new_zeros(TEAM_LEN - tm_block_raw.numel())], dim=0)
    else:
        tm_block = tm_block_raw

    # Now parse base fields with safe slicing (base is variable length)
    # Intended layout for base (when present):
    # [0:2]   agent_force
    # [2:4]   agent_pos - ball_pos
    # [4:6]   agent_vel - ball_vel
    # [6:8]   ball_pos - goal_pos
    # [8:10]  ball_vel
    # [10:12] ball_force
    # [12:13] agent_rot (1)
    # [13:15] agent_pos - goal_pos
    # [15:17] agent_vel

    a_force = safe_slice(base, 0, 2)
    ap_b = safe_slice(base, 2, 2)
    av_b = safe_slice(base, 4, 2)
    bp_g = safe_slice(base, 6, 2)
    ball_vel = safe_slice(base, 8, 2)
    ball_force = safe_slice(base, 10, 2)
    agent_rot = safe_slice(base, 12, 1)
    ap_g = safe_slice(base, 13, 2)
    agent_vel = safe_slice(base, 15, 2)

    # Convert blocks into matrices/vectors with right shapes
    adv_block = adv_block.view(-1, 8)
    # If fewer than 2 adversaries present, pad to 2x8
    if adv_block.shape[0] < 2:
        pad_rows = 2 - adv_block.shape[0]
        adv_block = torch.cat([adv_block, o.new_zeros(pad_rows, 8)], dim=0)

    tm_block = tm_block.view(-1)[:8]

    # Convert to agent-centric coordinates (positions relative to agent)
    # ball_rel = ball_pos - agent_pos = - (agent_pos - ball_pos)
    ball = -ap_b

    # teammate position relative to agent: teammate_pos - agent_pos = - (agent_pos - teammate_pos)
    teammate = -tm_block[0:2]
    tvel = tm_block[4:6]

    # adversaries positions relative to agent and their velocities
    adv_rels = -adv_block[:, 0:2]        # shape (2,2)
    adv_vels = adv_block[:, 4:6]

    # goal relative to agent: goal_pos - agent_pos = - (agent_pos - goal_pos)
    goal_rel = -ap_g

    # --- Parameters ---
    max_force = 1.2
    burst_force_multiplier = 1.7
    press_range = 1.2
    control_radius = 0.35
    prefer_pass_distance = 3.0
    defend_threshold = 1.3
    role_margin = 0.15
    damping = 0.6

    # Distances
    d_ego_ball = length(ball)
    to_teammate = teammate - ball
    d_tm_ball = length(to_teammate)

    # adversary distances to ball
    adv_pos_minus_ball = adv_rels - ball.unsqueeze(0)
    d_adv_ball = torch.norm(adv_pos_minus_ball, dim=-1)
    # safe argmin even if zeros: pick first min
    idx_adv_ball = int(torch.argmin(d_adv_ball).item()) if d_adv_ball.numel() > 0 else 0
    adv_near_ball_rel = adv_rels[idx_adv_ball]
    adv_near_ball_vel = adv_vels[idx_adv_ball]
    d_adv_ball_min = d_adv_ball[idx_adv_ball] if d_adv_ball.numel() > 0 else torch.tensor(1e6, device=device, dtype=dtype)

    # adversary nearest to our goal
    adv_pos_minus_goal = adv_rels - goal_rel.unsqueeze(0)
    d_adv_goal = torch.norm(adv_pos_minus_goal, dim=-1)
    idx_adv_goal = int(torch.argmin(d_adv_goal).item()) if d_adv_goal.numel() > 0 else 0
    adv_near_goal_rel = adv_rels[idx_adv_goal]
    d_adv_goal_min = d_adv_goal[idx_adv_goal] if d_adv_goal.numel() > 0 else torch.tensor(1e6, device=device, dtype=dtype)

    # Default target (relative to agent) and burst
    target_rel = o.new_zeros(2)
    burst = 1.0

    # Defensive override
    if d_adv_goal_min.item() < defend_threshold:
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
            # PRESS PHASE
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
            to_teammate = teammate - ball
            to_teammate_norm = norm(to_teammate) if length(to_teammate).item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
            teammate_ahead = dot(to_teammate_norm, -opp_dir) > 0.5
            teammate_close_forward = (length(to_teammate).item() < prefer_pass_distance) and teammate_ahead

            if teammate_close_forward:
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
                away_from_adv = norm(ball - adv_near_ball_rel) if d_adv_ball_min.item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
                mixed = base_break + 0.6 * away_from_adv
                break_dir = norm(mixed)
                target_rel = ball + break_dir * 0.6
                burst = burst_force_multiplier
        else:
            # REPOSITION / READY PHASE
            opp_dir = norm(ball - goal_rel)
            perp = torch.tensor([-opp_dir[1], opp_dir[0]], device=device, dtype=dtype)
            center_bias = 0.5
            s0 = torch.sign(perp[0])
            s1 = torch.sign(perp[1])
            s0 = s0 if s0.item() != 0.0 else torch.tensor(1.0, device=device, dtype=dtype)
            s1 = s1 if s1.item() != 0.0 else torch.tensor(1.0, device=device, dtype=dtype)
            lateral = perp * (center_bias * -torch.stack([s0, s1]))
            # Ensure lateral has correct shape: torch.stack -> (2,), perp*(...) broadcasts
            target_rel = ball + opp_dir * 1.0 + lateral
            if torch.isnan(target_rel).any():
                target_rel = ball + opp_dir * 1.0
            burst = 1.0

    # Low-level PD-like steering toward target_rel (agent is at origin)
    pos_err = target_rel
    desired_dir = norm(pos_err) if length(pos_err).item() > 1e-6 else o.new_zeros(2)
    mag_scale = max_force * (burst if burst > 1.0 else 1.0)
    base_force = desired_dir * mag_scale
    force = base_force - damping * agent_vel
    max_allowed = max_force * burst_force_multiplier
    force = clamp_mag(force, max_allowed)

    # Small jitter when not bursting
    if float(burst) == 1.0:
        perp = torch.stack([-force[1], force[0]])
        if length(perp).item() > 1e-6:
            perp_j = norm(perp) * 0.03
            rand_sign = (torch.rand(1, device=device).item() * 2.0 - 1.0)
            perp_j = perp_j * rand_sign
            force = force + perp_j
            force = clamp_mag(force, max_allowed)

    action = force.view(1, 2)
    return action

import torch

def agent1_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    Cooperative partner to a High-Tempo Press-and-Break Specialist (agent0).

    Strategy summary (brief):
    - Parse the environment observation robustly (same layout as partner agent).
    - Detect when the partner (teammate seen in the observation) is executing a high-tempo burst
      (press or break) by inspecting their force/velocity.
    - When the partner bursts toward the ball, adopt a receiving/support position ahead of the partner
      (toward the opponent direction) so both agents can meet the same landmark (the ball / pass target).
    - Otherwise, if this agent is closer to the ball, take the press/control role; if not, take a
      ready/reposition role to present as a forward/central support point.
    - Low-level control uses a PD-like steering with damping and magnitude clamping similar to the partner.

    Returns: torch.tensor of shape (1,2) representing agent force.
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

    # Safe slice helper
    def safe_slice(base: torch.Tensor, start: int, length_req: int):
        if base.numel() <= start:
            return base.new_zeros(length_req)
        avail = base.numel() - start
        take = min(length_req, avail)
        out = base[start : start + take]
        if take < length_req:
            pad = base.new_zeros(length_req - take)
            out = torch.cat([out, pad], dim=0)
        return out

    # Ensure 1D
    o = obs.view(-1)
    n = o.numel()
    device = o.device
    dtype = o.dtype

    # Expected block sizes (from observation_base): teammates:8, adversaries:16 (2*8)
    TEAM_LEN = 8
    ADV_LEN = 16

    base_len = max(0, n - TEAM_LEN - ADV_LEN)
    base = o[:base_len] if base_len > 0 else o.new_zeros(0)

    adv_block_raw = o[base_len : base_len + ADV_LEN] if n >= base_len + 1 else o.new_zeros(0)
    if adv_block_raw.numel() < ADV_LEN:
        adv_block = torch.cat([adv_block_raw, o.new_zeros(ADV_LEN - adv_block_raw.numel())], dim=0)
    else:
        adv_block = adv_block_raw

    tm_block_raw_start = base_len + ADV_LEN
    tm_block_raw = o[tm_block_raw_start : tm_block_raw_start + TEAM_LEN] if n >= tm_block_raw_start + 1 else o.new_zeros(0)
    if tm_block_raw.numel() < TEAM_LEN:
        tm_block = torch.cat([tm_block_raw, o.new_zeros(TEAM_LEN - tm_block_raw.numel())], dim=0)
    else:
        tm_block = tm_block_raw

    # Parse base fields (best-effort mapping to observation_base layout)
    a_force = safe_slice(base, 0, 2)
    ap_b = safe_slice(base, 2, 2)    # agent_pos - ball_pos
    av_b = safe_slice(base, 4, 2)    # agent_vel - ball_vel
    bp_g = safe_slice(base, 6, 2)    # ball_pos - goal_pos
    ball_vel = safe_slice(base, 8, 2)
    ball_force = safe_slice(base, 10, 2)
    agent_rot = safe_slice(base, 12, 1)
    ap_g = safe_slice(base, 13, 2)   # agent_pos - goal_pos
    agent_vel = safe_slice(base, 15, 2)

    # Convert blocks into appropriate shapes
    # adversaries: shape (2,8) expected
    adv_block = adv_block.view(-1, 8)
    if adv_block.shape[0] < 2:
        pad_rows = 2 - adv_block.shape[0]
        adv_block = torch.cat([adv_block, o.new_zeros(pad_rows, 8)], dim=0)

    # teammate block is 8-long, interpret accordingly
    tm_block = tm_block.view(-1)[:8]

    # Convert to agent-centric coordinates (agent at origin)
    # ball = ball_pos - agent_pos = - (agent_pos - ball_pos)
    ball = -ap_b

    # teammate position relative to agent: teammate_pos - agent_pos = - (agent_pos - teammate_pos)
    teammate = -tm_block[0:2]
    tvel = tm_block[4:6]
    tforce = tm_block[6:8]

    # adversaries positions relative to agent
    adv_rels = -adv_block[:, 0:2]
    adv_vels = adv_block[:, 4:6]

    # goal relative to agent: goal_pos - agent_pos = - (agent_pos - goal_pos)
    goal_rel = -ap_g

    # --- Parameters (aligned with partner) ---
    max_force = 1.2
    burst_force_multiplier = 1.7
    press_range = 1.2
    control_radius = 0.35
    prefer_pass_distance = 3.0
    defend_threshold = 1.3
    role_margin = 0.15
    damping = 0.6

    # Distances
    d_ego_ball = length(ball)
    d_tm_ball = length(teammate - ball)  # teammate distance to ball

    # adversary distances to ball
    adv_pos_minus_ball = adv_rels - ball.unsqueeze(0)
    d_adv_ball = torch.norm(adv_pos_minus_ball, dim=-1)
    if d_adv_ball.numel() > 0:
        idx_adv_ball = int(torch.argmin(d_adv_ball).item())
        d_adv_ball_min = d_adv_ball[idx_adv_ball]
        adv_near_ball_rel = adv_rels[idx_adv_ball]
        adv_near_ball_vel = adv_vels[idx_adv_ball]
    else:
        idx_adv_ball = 0
        d_adv_ball_min = torch.tensor(1e6, device=device, dtype=dtype)
        adv_near_ball_rel = adv_rels.new_zeros(2)
        adv_near_ball_vel = adv_vels.new_zeros(2)

    # adversary nearest to our goal
    adv_pos_minus_goal = adv_rels - goal_rel.unsqueeze(0)
    d_adv_goal = torch.norm(adv_pos_minus_goal, dim=-1)
    if d_adv_goal.numel() > 0:
        idx_adv_goal = int(torch.argmin(d_adv_goal).item())
        d_adv_goal_min = d_adv_goal[idx_adv_goal]
        adv_near_goal_rel = adv_rels[idx_adv_goal]
    else:
        d_adv_goal_min = torch.tensor(1e6, device=device, dtype=dtype)
        adv_near_goal_rel = adv_rels.new_zeros(2)

    # Detect teammate (partner) burst intent: check teammate applied force magnitude
    teammate_force_mag = length(tforce)
    teammate_vel_mag = length(tvel)
    partner_burst = (teammate_force_mag.item() > 1.2) or (teammate_vel_mag.item() > 1.0)

    # Default target (relative to agent) and burst for this agent
    target_rel = o.new_zeros(2)
    burst = 1.0

    # Defensive override: if opponent dangerously close to our goal, help cover
    if d_adv_goal_min.item() < defend_threshold:
        dir_to_adv = adv_pos_minus_goal[idx_adv_goal]
        dir_to_adv = norm(dir_to_adv)
        offset = defend_threshold * 0.6
        target_rel = goal_rel + dir_to_adv * offset
        burst = 1.0
    else:
        # If partner is bursting near the ball, play support/receiver role
        partner_near_ball = d_tm_ball.item() < (press_range + 0.3)
        if partner_burst and partner_near_ball:
            # Present ahead of partner toward the opponent direction (away from our goal)
            # Compute forward direction from partner toward opponent direction
            partner_to_goal = teammate - goal_rel
            dir_forward = norm(partner_to_goal) if length(partner_to_goal).item() > 1e-6 else norm(ball - goal_rel)
            # place a support point slightly in front of partner (toward opponent)
            support_offset = 0.7
            target_rel = teammate + dir_forward * support_offset
            burst = burst_force_multiplier  # move quickly into the receiving lane
        else:
            # No immediate partner burst: decide whether to press (if we are closer) or to be ready support
            # Determine if this agent should take press/control role
            will_press = (d_adv_ball_min.item() < d_ego_ball + 0.05) and (
                (d_ego_ball <= d_tm_ball - role_margin) or (d_ego_ball < press_range)
            )

            if will_press and d_ego_ball <= press_range:
                # PRESS: approach the ball to contest
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
                # CONTROL / BREAK: if we have influence, attempt quick counter or pass to partner
                opp_dir = norm(ball - goal_rel)
                to_teammate = teammate - ball
                to_teammate_norm = norm(to_teammate) if length(to_teammate).item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
                teammate_ahead = dot(to_teammate_norm, -opp_dir) > 0.5
                teammate_close_forward = (length(to_teammate).item() < prefer_pass_distance) and teammate_ahead

                if teammate_close_forward:
                    # Prefer passing to partner: place ball in front of them
                    teammate_to_goal = teammate - goal_rel
                    dir_forward = norm(teammate_to_goal)
                    if length(dir_forward).item() <= 1e-6:
                        pass_target_rel = teammate.clone()
                    else:
                        pass_target_rel = teammate + dir_forward * 0.4
                    target_rel = pass_target_rel
                    burst = burst_force_multiplier
                else:
                    # Direct break toward opponent space, but offset slightly away from nearest adversary
                    base_break = norm(ball - goal_rel)
                    away_from_adv = norm(ball - adv_near_ball_rel) if d_adv_ball_min.item() > 1e-6 else torch.zeros(2, device=device, dtype=dtype)
                    mixed = base_break + 0.6 * away_from_adv
                    break_dir = norm(mixed)
                    target_rel = ball + break_dir * 0.6
                    burst = burst_force_multiplier
            else:
                # READY / POSITION: take an advanced central support spot relative to ball
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

    # Low-level PD-like steering toward target_rel (agent at origin)
    pos_err = target_rel
    desired_dir = norm(pos_err) if length(pos_err).item() > 1e-6 else o.new_zeros(2)
    mag_scale = max_force * (burst if burst > 1.0 else 1.0)
    base_force = desired_dir * mag_scale
    force = base_force - damping * agent_vel
    max_allowed = max_force * burst_force_multiplier
    force = clamp_mag(force, max_allowed)

    # Small jitter when not bursting to remain less predictable and keep lateral motion
    if float(burst) == 1.0:
        perp = torch.stack([-force[1], force[0]])
        if length(perp).item() > 1e-6:
            perp_j = norm(perp) * 0.03
            rand_sign = (torch.rand(1, device=device).item() * 2.0 - 1.0)
            perp_j = perp_j * rand_sign
            force = force + perp_j
            force = clamp_mag(force, max_allowed)

    action = force.view(1, 2)
    return action