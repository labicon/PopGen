import torch

def agent0_policy(obs):
    """
    Wide Decoy & Overlap Runner (agent0)
    obs: 1D torch.Tensor as produced by observation_base
    returns: torch.Tensor of shape (1,2)
    """
    eps = 1e-6
    # Behavior constants
    max_force = 1.0
    kp = 1.8
    kd = 0.6
    side_offset = 1.0
    close_margin = 0.25
    defend_threshold = 1.4
    overlap_speed_threshold = 0.5
    cut_in_bias = 0.6

    o = obs.view(-1)
    device = o.device
    dtype = o.dtype

    # --- Slice observation according to observation_base concatenation ---
    # Indices derived from observation_base ordering in the task description
    # [0:2]   agent_force (2)
    # [2:4]   agent_pos - ball_pos (2)
    # [4:6]   agent_vel - ball_vel (2)
    # [6:8]   ball_pos - goal_pos (2)
    # [8:10]  ball_vel (2)
    # [10:12] ball_force (2)
    # [12:13] agent_rot (1)
    # [13:15] agent_pos - goal_pos (2)
    # [15:17] agent_vel (2)
    # [17:33] adversaries (2 adversaries * 8 each = 16)
    # [33:41] teammates (1 teammate * 8 = 8)

    agent_pos_minus_ball = o[2:4]
    # agent_vel_minus_ball = o[4:6]
    ball_pos_minus_goal = o[6:8]
    # ball_vel = o[8:10]
    # ball_force = o[10:12]
    # agent_rot = o[12:13]
    agent_pos_minus_goal = o[13:15]
    agent_vel = o[15:17]

    adv_block = o[17:33].view(-1)
    adv1 = adv_block[0:8]
    adv2 = adv_block[8:16]

    team_block = o[33:41].view(-1)
    agentpos_minus_teammate = team_block[0:2]
    teammate_vel = team_block[4:6]

    # Reconstruct goal-centric positions (goal at origin):
    agent_pos_g = agent_pos_minus_goal
    ball_pos_g = ball_pos_minus_goal
    # teammate pos (goal-centric): agent_pos - (agent_pos - teammate_pos)
    teammate_pos_g = agent_pos_g - agentpos_minus_teammate

    # adversary positions in goal-centric frame
    adv1_agentpos_minus = adv1[0:2]
    adv2_agentpos_minus = adv2[0:2]
    adv1_pos_g = agent_pos_g - adv1_agentpos_minus
    adv2_pos_g = agent_pos_g - adv2_agentpos_minus
    adv_positions = torch.stack([adv1_pos_g, adv2_pos_g], dim=0)

    def length(v):
        return torch.sqrt((v * v).sum(dim=-1) + eps)

    # Distances to ball (use goal-centric coords: vector from agent to ball = ball_pos_g - agent_pos_g)
    d_ego_ball = length(ball_pos_g - agent_pos_g)
    d_tm_ball = length(ball_pos_g - teammate_pos_g)

    # Defensive override: if an adversary is too close to our goal, intercept between them and the goal
    adv_to_goal_dists = length(adv_positions)  # adv_pos_g is adv - goal
    adv_min_val, adv_min_idx = torch.min(adv_to_goal_dists, dim=0)
    if adv_min_idx.item() == 0:
        nearest_adv_pos_g = adv1_pos_g
    else:
        nearest_adv_pos_g = adv2_pos_g

    target = None
    defensive_mode = False

    if adv_min_val.item() < defend_threshold:
        # put agent on line between goal(0) and adversary at an intercept distance
        defensive_mode = True
        dir_g_to_adv = nearest_adv_pos_g / (length(nearest_adv_pos_g) + eps)
        target = dir_g_to_adv * (defend_threshold * 0.6)
    else:
        # Decide whether ego is clearly the best to go to the ball
        ego_closer_flag = (d_ego_ball + close_margin) < d_tm_ball

        # compute approximate opponent direction (from our goal toward ball)
        approx_opp_dir = ball_pos_g / (length(ball_pos_g) + eps)
        perp = torch.tensor([-approx_opp_dir[1], approx_opp_dir[0]], dtype=dtype, device=device)

        if ego_closer_flag:
            # Attacker mode: approach the ball but bias toward cutting-in along a lateral side
            # bias uses approx_opp_dir (direction toward ball/opponent side)
            forward_dir = approx_opp_dir

            # choose side by checking which lateral side is more open relative to nearest adversary to the ball
            adv_dists_to_ball = length(adv_positions - ball_pos_g.unsqueeze(0))
            minadv_idx = torch.argmin(adv_dists_to_ball)
            nearest_adv_to_ball = adv_positions[minadv_idx]
            vec_ball_to_adv = nearest_adv_to_ball - ball_pos_g
            # side_sign positive means put perp in same orientation as perp, else negative
            side_sign = 1.0 if (torch.dot(vec_ball_to_adv, perp) < 0.0) else -1.0

            # target is a little ahead of the ball toward opponent plus a lateral bias that leans to cut-in
            target = ball_pos_g + forward_dir * 0.3 + perp * (side_sign * (-cut_in_bias))
        else:
            # Teammate has better access: play wide / decoy and optionally perform timed overlap
            candidate1 = ball_pos_g + approx_opp_dir * 1.0 + perp * side_offset
            candidate2 = ball_pos_g + approx_opp_dir * 1.0 - perp * side_offset

            # choose flank that maximizes min distance to adversaries
            d1 = torch.min(length(adv_positions - candidate1.unsqueeze(0)))
            d2 = torch.min(length(adv_positions - candidate2.unsqueeze(0)))
            flank = candidate1 if d1 > d2 else candidate2

            # compute teammate forwardness relative to opponent direction
            tvel_len = length(teammate_vel)
            if tvel_len.item() > 1e-6:
                tvel_norm = teammate_vel / (tvel_len + eps)
            else:
                tvel_norm = torch.zeros_like(teammate_vel)
            teammate_forwardness = torch.dot(tvel_norm, approx_opp_dir)

            # Overlap conditions: teammate is committing forward and agent is in supporting range
            # Additionally ensure teammate is actually ahead (moving toward opp goal) and not extremely far
            teammate_ahead_metric = torch.dot(teammate_pos_g - ball_pos_g, approx_opp_dir)
            if (teammate_forwardness > overlap_speed_threshold) and (teammate_ahead_metric > -0.2):
                # run beyond the teammate to provide an overlap/through option
                target = teammate_pos_g + approx_opp_dir * 1.2 + perp * (side_offset * 0.5)
            else:
                # default: hold the flank to create width and pull defenders wide
                # ensure flank keeps a lateral buffer from the ball
                target = flank

    # PD controller toward target (in goal-centric frame)
    # Guard if target was somehow not set (should not happen), fall back to holding current position
    if target is None:
        desired_force = -kd * agent_vel
    else:
        pos_err = target - agent_pos_g
        desired_force = kp * pos_err - kd * agent_vel

    # clamp magnitude to max_force
    mag = length(desired_force)
    if mag.item() > max_force:
        desired_force = desired_force * (max_force / (mag + eps))

    force = desired_force

    # Add a small jitter when not in attacker mode (keeps movement dynamic) and not defensive
    jitter_mag = 0.02 * (1.0 if (d_ego_ball.item() > 0.5) else 0.0)
    if (not defensive_mode) and (not ((d_ego_ball + close_margin) < d_tm_ball)):
        # jitter perpendicular to current force (if force is near zero, jitter along perp of approx_opp_dir)
        f_perp = torch.tensor([-force[1], force[0]], dtype=dtype, device=device)
        fperp_len = length(f_perp)
        if fperp_len.item() > 1e-6:
            fperp_norm = f_perp / (fperp_len + eps)
        else:
            # fallback jitter direction
            approx_dir = (ball_pos_g - agent_pos_g)
            approx_dir_len = length(approx_dir)
            if approx_dir_len.item() > 1e-6:
                approx_dir_n = approx_dir / (approx_dir_len + eps)
                fperp_norm = torch.tensor([-approx_dir_n[1], approx_dir_n[0]], dtype=dtype, device=device)
            else:
                fperp_norm = torch.zeros(2, dtype=dtype, device=device)
        force = force + fperp_norm * jitter_mag

    # final clamp to ensure obey max_force
    mag2 = length(force)
    if mag2.item() > max_force:
        force = force * (max_force / (mag2 + eps))

    return force.view(1, 2)

import torch

def agent1_policy(obs):
    """
    Complementary Wide Decoy / Central Support policy for Agent 1.
    - Coordinates with the described partner (Wide Decoy & Overlap Runner) by favoring
      more central/through options and by reacting to the partner's forward commitment.
    - Uses the same observation slicing as the partner policy.

    obs: 1D torch.Tensor (as produced by observation_base)
    returns: torch.Tensor of shape (1,2)
    """
    eps = 1e-6
    # Behavior constants tuned to be complementary (more central/through runs)
    max_force = 1.0
    kp = 2.0
    kd = 0.7
    side_offset = 0.7
    close_margin = 0.15
    defend_threshold = 1.4
    overlap_speed_threshold = 0.45
    cut_in_bias = 0.35

    o = obs.view(-1)
    device = o.device
    dtype = o.dtype

    # --- Slice observation according to observation_base ordering ---
    # [0:2]   agent_force (2)
    # [2:4]   agent_pos - ball_pos (2)
    # [4:6]   agent_vel - ball_vel (2)
    # [6:8]   ball_pos - goal_pos (2)
    # [8:10]  ball_vel (2)
    # [10:12] ball_force (2)
    # [12:13] agent_rot (1)
    # [13:15] agent_pos - goal_pos (2)
    # [15:17] agent_vel (2)
    # [17:33] adversaries (2 adversaries * 8 each = 16)
    # [33:41] teammates (1 teammate * 8 = 8)

    agent_pos_minus_ball = o[2:4]
    ball_pos_minus_goal = o[6:8]
    agent_pos_minus_goal = o[13:15]
    agent_vel = o[15:17]

    adv_block = o[17:33].view(-1)
    adv1 = adv_block[0:8]
    adv2 = adv_block[8:16]

    team_block = o[33:41].view(-1)
    agentpos_minus_teammate = team_block[0:2]
    teammate_vel = team_block[4:6]

    # reconstruct positions in goal-centric frame (goal at origin)
    agent_pos_g = agent_pos_minus_goal
    ball_pos_g = ball_pos_minus_goal
    teammate_pos_g = agent_pos_g - agentpos_minus_teammate

    adv1_agentpos_minus = adv1[0:2]
    adv2_agentpos_minus = adv2[0:2]
    adv1_pos_g = agent_pos_g - adv1_agentpos_minus
    adv2_pos_g = agent_pos_g - adv2_agentpos_minus
    adv_positions = torch.stack([adv1_pos_g, adv2_pos_g], dim=0)

    def length(v):
        return torch.sqrt((v * v).sum(dim=-1) + eps)

    # distances to ball
    d_ego_ball = length(ball_pos_g - agent_pos_g)
    d_tm_ball = length(ball_pos_g - teammate_pos_g)

    # defensive override: nearest adversary to our goal (goal at origin in this frame)
    adv_to_goal_dists = length(adv_positions)
    adv_min_val, adv_min_idx = torch.min(adv_to_goal_dists, dim=0)
    if adv_min_idx.item() == 0:
        nearest_adv_pos_g = adv1_pos_g
    else:
        nearest_adv_pos_g = adv2_pos_g

    target = None
    defensive_mode = False

    if adv_min_val.item() < defend_threshold:
        # intercept between that adversary and our goal (goal at 0)
        defensive_mode = True
        dir_g_to_adv = nearest_adv_pos_g / (length(nearest_adv_pos_g) + eps)
        target = dir_g_to_adv * (defend_threshold * 0.6)
    else:
        # determine if ego should attack the ball or play support
        ego_closer_flag = (d_ego_ball + close_margin) < d_tm_ball

        # approximate opponent direction: from our goal through the ball toward opponents
        approx_opp_dir = ball_pos_g / (length(ball_pos_g) + eps)
        perp = torch.tensor([-approx_opp_dir[1], approx_opp_dir[0]], dtype=dtype, device=device)

        if ego_closer_flag:
            # Attacker mode (Agent1 chooses a more central/through-biased approach):
            # move to the ball but prefer a slightly more direct route toward opponent goal
            forward_dir = approx_opp_dir

            # bias toward center: choose lateral sign that reduces absolute lateral offset (y) in goal-centric frame
            # If moving to center is ambiguous, choose side away from nearest adversary to ball
            adv_dists_to_ball = length(adv_positions - ball_pos_g.unsqueeze(0))
            minadv_idx = torch.argmin(adv_dists_to_ball)
            nearest_adv_to_ball = adv_positions[minadv_idx]

            # prefer side that reduces |y| (center) if possible
            candidate_target_center_plus = ball_pos_g + forward_dir * 0.35 + perp * cut_in_bias
            candidate_target_center_minus = ball_pos_g + forward_dir * 0.35 - perp * cut_in_bias
            if torch.abs(candidate_target_center_plus[1]) < torch.abs(candidate_target_center_minus[1]):
                chosen_center = candidate_target_center_plus
            else:
                chosen_center = candidate_target_center_minus

            # If nearest adversary is blocking chosen center, nudge away from that adversary
            if length(nearest_adv_to_ball - chosen_center) < 0.6:
                # push slightly opposite to vector from chosen_center to adversary
                away_dir = (chosen_center - nearest_adv_to_ball)
                if length(away_dir).item() > 1e-6:
                    away_dir = away_dir / (length(away_dir) + eps)
                    chosen_center = chosen_center + away_dir * 0.25

            target = chosen_center
        else:
            # Teammate has better access: play central support (not wide) to complement partner's wide behavior.
            candidate_central = ball_pos_g + approx_opp_dir * 0.9

            # bias toward center lane to increase through/pass options
            center_bias = -0.4 * torch.sign(candidate_central[1])
            # if already near center, keep it small
            if torch.abs(candidate_central[1]) < 0.2:
                center_bias = -0.2 * torch.sign(candidate_central[1])
            candidate_central = candidate_central + torch.tensor([0.0, center_bias], dtype=dtype, device=device)

            # compute flank positions but with reduced lateral distance (partner likely takes the wide flank)
            candidate1 = ball_pos_g + approx_opp_dir * 0.9 + perp * side_offset
            candidate2 = ball_pos_g + approx_opp_dir * 0.9 - perp * side_offset

            # choose flank that is less likely to conflict with partner by preferring the side closer to center
            # (i.e., smaller |y|) to leave extreme width to partner
            flank = candidate1 if (torch.abs(candidate1[1]) < torch.abs(candidate2[1])) else candidate2

            # teammate forwardness to detect when the partner (agent0) is committing forward
            tvel_len = length(teammate_vel)
            if tvel_len.item() > 1e-6:
                tvel_norm = teammate_vel / (tvel_len + eps)
            else:
                tvel_norm = torch.zeros_like(teammate_vel)
            teammate_forwardness = torch.dot(tvel_norm, approx_opp_dir)

            # If teammate is committing forward, run a central through/overlap line ahead of them
            teammate_ahead_metric = torch.dot(teammate_pos_g - ball_pos_g, approx_opp_dir)
            if (teammate_forwardness > overlap_speed_threshold) and (teammate_ahead_metric > -0.4):
                # run centrally beyond the teammate to offer a through or inside pass
                # keep a smaller lateral offset so partner can still go wide
                target = teammate_pos_g + approx_opp_dir * 1.1 + (-perp) * (side_offset * 0.25)
            else:
                # default: hold a central/support position (not extreme flank)
                # prefer candidate_central but ensure we are not colliding with nearest adversary
                dflank1 = torch.min(length(adv_positions - candidate1.unsqueeze(0)))
                dflank2 = torch.min(length(adv_positions - candidate2.unsqueeze(0)))
                # if both flanks are quite contested, prefer central
                if (dflank1 < 0.5) and (dflank2 < 0.5):
                    target = candidate_central
                else:
                    # prefer flank but closer to center (to complement partner)
                    target = flank

    # PD controller toward target (goal-centric frame)
    if target is None:
        desired_force = -kd * agent_vel
    else:
        pos_err = target - agent_pos_g
        desired_force = kp * pos_err - kd * agent_vel

    # clamp magnitude to max_force
    def clamp_mag(v, m):
        mag = length(v)
        if mag.item() > m:
            return v * (m / (mag + eps))
        return v

    force = clamp_mag(desired_force, max_force)

    # Add a modest jitter to avoid locking against partner when not defensive
    jitter_mag = 0.018 * (1.0 if (d_ego_ball.item() > 0.5) else 0.0)
    if (not defensive_mode):
        f_perp = torch.tensor([-force[1], force[0]], dtype=dtype, device=device)
        fperp_len = length(f_perp)
        if fperp_len.item() > 1e-6:
            fperp_norm = f_perp / (fperp_len + eps)
            force = force + fperp_norm * jitter_mag

    # final clamp
    force = clamp_mag(force, max_force)

    return force.view(1, 2)