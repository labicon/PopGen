import torch

def agent0_policy(obs):
    """
    Wide Decoy & Overlap Runner policy.
    obs: 1D torch.Tensor (observation vector produced by observation_base concatenation)
    returns: torch.Tensor of shape (1,2)
    """
    eps = 1e-6
    # Constants
    max_force = 1.0
    kp = 1.8
    kd = 0.6
    side_offset = 1.0
    close_margin = 0.25
    defend_threshold = 1.4
    overlap_speed_threshold = 0.5
    cut_in_bias = 0.6

    # Slicing indices (based on observation construction order)
    # obs breakdown (lengths):
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

    # Ensure obs is 1D
    o = obs.view(-1)

    agent_pos_minus_ball = o[2:4]
    agent_vel_minus_ball = o[4:6]
    ball_pos_minus_goal = o[6:8]
    ball_vel = o[8:10]
    # ball_force = o[10:12]
    # agent_rot = o[12:13]
    agent_pos_minus_goal = o[13:15]
    agent_vel = o[15:17]

    adv_block = o[17:33].view(-1)  # length 16 -> two adversaries
    # For each adversary: [agent_pos - adv_pos (2), agent_vel - adv_vel (2), adv_vel (2), adv_force (2)]
    adv1 = adv_block[0:8]
    adv2 = adv_block[8:16]

    team_block = o[33:41].view(-1)  # one teammate: length 8
    # teammate layout: [agent_pos - teammate_pos (2), agent_vel - teammate_vel (2), teammate_vel (2), teammate_force (2)]
    agentpos_minus_teammate = team_block[0:2]
    # agentvel_minus_teammatevel = team_block[2:4]
    teammate_vel = team_block[4:6]

    # Compute positions in goal-centric frame (goal at origin)
    # agent_pos_g = agent_pos - goal_pos
    agent_pos_g = agent_pos_minus_goal
    # ball_pos_g = ball_pos - goal_pos
    ball_pos_g = ball_pos_minus_goal

    # adversary positions in goal frame: adv_pos_g = agent_pos_g - (agent_pos - adv_pos)
    adv1_agentpos_minus = adv1[0:2]
    adv2_agentpos_minus = adv2[0:2]
    adv1_pos_g = agent_pos_g - adv1_agentpos_minus
    adv2_pos_g = agent_pos_g - adv2_agentpos_minus
    adv_positions = torch.stack([adv1_pos_g, adv2_pos_g], dim=0)

    # teammate position in goal frame: teammate_pos_g = agent_pos_g - (agent_pos - teammate_pos)
    teammate_pos_g = agent_pos_g - agentpos_minus_teammate

    # Distances
    def length(v):
        return torch.sqrt((v * v).sum(dim=-1) + eps)

    # d_ego_ball = ||agent - ball||
    d_ego_ball = length(-agent_pos_minus_ball)
    # d_tm_ball = ||teammate - ball||  (teammate_pos_g - ball_pos_g)
    d_tm_ball = length(teammate_pos_g - ball_pos_g)

    # Defensive override: nearest adversary to our goal (goal at origin)
    adv_to_goal_dists = length(adv_positions)  # since adv_pos_g is adv - goal
    adv_min_val, adv_min_idx = torch.min(adv_to_goal_dists, dim=0)
    if adv_min_idx.item() == 0:
        nearest_adv_pos_g = adv1_pos_g
    else:
        nearest_adv_pos_g = adv2_pos_g

    # target in goal-centric frame
    target = None

    if adv_min_val.item() < defend_threshold:
        # place between goal (0) and adversary, a bit out for interception
        dir_g_to_adv = nearest_adv_pos_g / (length(nearest_adv_pos_g) + eps)
        target = dir_g_to_adv * (defend_threshold * 0.6)
    else:
        # Decide closer to ball
        if d_ego_ball <= d_tm_ball - close_margin:
            # Attacker mode: approach ball but bias laterally (cut in / sideline)
            # point ahead = ball + (ball - goal)*0.2 => in goal frame: ball_pos_g * 1.2
            ahead_point = ball_pos_g * 1.2
            forward_dir = ahead_point - agent_pos_g
            fd_norm = forward_dir / (length(forward_dir) + eps)
            perp = torch.tensor([-fd_norm[1], fd_norm[0]], dtype=o.dtype, device=o.device)

            # Choose side sign based on nearest adversary to ball
            # find adv closest to ball: compare ||adv_pos - ball_pos||
            adv_dists_to_ball = length(adv_positions - ball_pos_g.unsqueeze(0))
            minadv_idx = torch.argmin(adv_dists_to_ball)
            if minadv_idx.item() == 0:
                adv_near_ball = adv1_pos_g
            else:
                adv_near_ball = adv2_pos_g
            vec_ball_to_adv = adv_near_ball - ball_pos_g
            side_sign = 1.0 if (torch.dot(vec_ball_to_adv, perp) < 0) else -1.0

            target = ball_pos_g + fd_norm * 0.3 + perp * (side_sign * (-cut_in_bias))
        else:
            # Teammate has better access: play wide / decoy or overlap
            approx_opp_dir = ball_pos_g / (length(ball_pos_g) + eps)
            perp = torch.tensor([-approx_opp_dir[1], approx_opp_dir[0]], dtype=o.dtype, device=o.device)

            candidate1 = ball_pos_g + approx_opp_dir * 1.0 + perp * side_offset
            candidate2 = ball_pos_g + approx_opp_dir * 1.0 - perp * side_offset

            # compute min distance from candidates to adversaries
            d1 = torch.min(length(adv_positions - candidate1.unsqueeze(0)))
            d2 = torch.min(length(adv_positions - candidate2.unsqueeze(0)))
            flank = candidate1 if d1 > d2 else candidate2

            # teammate forwardness: dot(norm(tvel), approx_opp_dir)
            tvel_norm = teammate_vel / (length(teammate_vel) + eps)
            teammate_forwardness = torch.dot(tvel_norm, approx_opp_dir)
            if teammate_forwardness > overlap_speed_threshold:
                target = teammate_pos_g + approx_opp_dir * 1.2 + perp * (side_offset * 0.5)
            else:
                target = flank

    # PD controller toward target (in goal frame)
    pos_err = target - agent_pos_g
    desired_force = kp * pos_err - kd * agent_vel

    # clamp magnitude
    mag = length(desired_force)
    scale = torch.minimum(torch.tensor(1.0, dtype=o.dtype, device=o.device), max_force / (mag + eps))
    force = desired_force * scale

    # jitter application: small perpendicular jitter to avoid determinism
    jitter_mag = 0.02 * (1.0 if d_ego_ball.item() > 0.5 else 0.0)
    f_perp = torch.tensor([-force[1], force[0]], dtype=o.dtype, device=o.device)
    fperp_norm = f_perp / (length(f_perp) + eps)
    force = force + fperp_norm * jitter_mag

    # final clamp to be safe
    mag2 = length(force)
    scale2 = torch.minimum(torch.tensor(1.0, dtype=o.dtype, device=o.device), max_force / (mag2 + eps))
    force = force * scale2

    action = force.view(1, 2)
    return action

import torch

def agent1_policy(obs):
    """
    Complementary partner policy to coordinate with the provided "Wide Decoy & Overlap Runner".
    This policy adopts a more central/support/primary-attacker role so the pair reliably reach the
    same landmark (ball/attack corridor) together. It:
      - Takes responsibility to approach the ball when it is clearly closer than the teammate.
      - When the teammate is better positioned (teammate closer), it becomes a close central support
        (a short passing option or a holding position), or remains behind the ball to let the partner
        overlap when the partner is the one running wide.
      - Enacts a defensive interception when an adversary is dangerously close to our goal.

    Input: obs (1D torch tensor) as produced by observation_base
    Output: torch tensor of shape (1,2) representing applied force
    """
    eps = 1e-6
    # Controller & behavior constants
    max_force = 1.0
    kp = 2.0
    kd = 0.7
    close_margin = 0.25
    defend_threshold = 1.4
    overlap_speed_threshold = 0.5
    support_backoff = 0.4  # how much to sit behind the ball when supporting
    central_offset = 0.6

    # Ensure 1D view
    o = obs.view(-1)

    # Indices follow the observation construction used by the partner policy
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
    agent_vel_minus_ball = o[4:6]
    ball_pos_minus_goal = o[6:8]
    ball_vel = o[8:10]
    agent_pos_minus_goal = o[13:15]
    agent_vel = o[15:17]

    adv_block = o[17:33].view(-1)
    adv1 = adv_block[0:8]
    adv2 = adv_block[8:16]

    team_block = o[33:41].view(-1)
    agentpos_minus_teammate = team_block[0:2]
    teammate_vel = team_block[4:6]

    # Reconstruct positions in goal-centric frame (goal at origin)
    agent_pos_g = agent_pos_minus_goal
    ball_pos_g = ball_pos_minus_goal

    adv1_agentpos_minus = adv1[0:2]
    adv2_agentpos_minus = adv2[0:2]
    adv1_pos_g = agent_pos_g - adv1_agentpos_minus
    adv2_pos_g = agent_pos_g - adv2_agentpos_minus
    adv_positions = torch.stack([adv1_pos_g, adv2_pos_g], dim=0)

    teammate_pos_g = agent_pos_g - agentpos_minus_teammate

    def length(v):
        return torch.sqrt((v * v).sum(dim=-1) + eps)

    # Distances to ball
    d_ego_ball = length(-agent_pos_minus_ball)
    d_tm_ball = length(teammate_pos_g - ball_pos_g)

    # Defensive override: nearest adversary to our goal
    adv_to_goal_dists = length(adv_positions)  # adv pos are in goal-centric frame
    adv_min_val, adv_min_idx = torch.min(adv_to_goal_dists, dim=0)
    if adv_min_idx.item() == 0:
        nearest_adv_pos_g = adv1_pos_g
    else:
        nearest_adv_pos_g = adv2_pos_g

    # Target in goal-centric frame
    target = None

    if adv_min_val.item() < defend_threshold:
        # Intercept between goal (origin) and the threatening adversary
        dir_g_to_adv = nearest_adv_pos_g / (length(nearest_adv_pos_g) + eps)
        target = dir_g_to_adv * (defend_threshold * 0.6)
    else:
        # Compute approximate opponent-forward direction through ball (away from our goal)
        approx_opp_dir = ball_pos_g / (length(ball_pos_g) + eps)

        # teammate forwardness relative to opponent goal
        tvel_norm = teammate_vel / (length(teammate_vel) + eps)
        teammate_forwardness = torch.dot(tvel_norm, approx_opp_dir)

        if d_ego_ball <= d_tm_ball - close_margin:
            # Ego is the better candidate to take the ball: be decisive and approach the ball
            # Bias movement toward opponent direction (to be the primary attacker)
            target = ball_pos_g + approx_opp_dir * 0.35
        else:
            # Teammate is better positioned. We should support centrally and be a reliable pass option.
            if teammate_forwardness > overlap_speed_threshold:
                # Teammate appears to be advancing: stay slightly behind the ball to be a safe receiver
                target = ball_pos_g - approx_opp_dir * support_backoff
            else:
                # Default supporting location: a compact central support between ball and goal but
                # slightly offset to create a passing lane with the partner who will play wide.
                perp = torch.tensor([-approx_opp_dir[1], approx_opp_dir[0]], dtype=o.dtype, device=o.device)
                # choose side sign relative to teammate so we stay compact for quick combinations
                side_sign = 1.0 if (torch.dot(ball_pos_g - teammate_pos_g, perp) >= 0) else -1.0
                target = ball_pos_g + approx_opp_dir * 0.18 + perp * (-central_offset * side_sign)

    # PD controller toward target (goal-centric frame)
    pos_err = target - agent_pos_g
    desired_force = kp * pos_err - kd * agent_vel

    # Clamp magnitude to max_force
    mag = length(desired_force)
    scale = torch.minimum(torch.tensor(1.0, dtype=o.dtype, device=o.device), max_force / (mag + eps))
    force = desired_force * scale

    # small perpendicular jitter when not extremely close to the ball to avoid deadlocks
    jitter_mag = 0.015 * (1.0 if d_ego_ball.item() > 0.5 else 0.0)
    f_perp = torch.tensor([-force[1], force[0]], dtype=o.dtype, device=o.device)
    fperp_norm = f_perp / (length(f_perp) + eps)
    force = force + fperp_norm * jitter_mag

    # final clamp
    mag2 = length(force)
    scale2 = torch.minimum(torch.tensor(1.0, dtype=o.dtype, device=o.device), max_force / (mag2 + eps))
    force = force * scale2

    return force.view(1, 2)