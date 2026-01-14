import torch

def agent0_policy(obs):
    """
    Proactive Space Creator / Lead Runner policy for agent0.

    Observation parsing assumptions (inferred from environment.observation_base):
      - There are two adversaries and one teammate.
      - Final observation is a 1D tensor formed by concatenating these blocks in order:
          [obs_block, pos, vel, adversaries_block, teammates_block]
      - obs_block = [agent_force (2), agent_pos - ball_pos (2), agent_vel - ball_vel (2),
                     ball_pos - goal_pos (2), ball_vel (2), ball_force (2), agent_rot (Nrot)]
        where agent_rot size is deduced from remaining length.
      - pos = [agent_pos - goal_pos] (2)
      - vel = [agent_vel] (2)
      - adversaries_block = 2 * [agent_pos - adv_pos (2), agent_vel - adv_vel (2), adv_vel (2), adv_force (2)] => 16
      - teammates_block  = 1 * [agent_pos - teammate_pos (2), agent_vel - teammate_vel (2), teammate_vel (2), teammate_force (2)] => 8

    Behavior summary implemented:
      - Predict teammate attack direction from teammate_vel.
      - Compute lead_point ahead of teammate: teammate_pos + normalize(teammate_vel) * lead_distance.
      - Try lateral offsets (left/right/none) perpendicular to teammate direction to maximize min distance to defenders.
      - If defenders appear to be converging on teammate, compute a point behind the defender line instead.
      - If chosen lead_point is likely blocked (closest defender closer to lead_point than ego), perform a distractor run (wide or deep).
      - Use ball and teammate velocity to estimate a pass window. If pass likely, accelerate toward lead_point; otherwise perform a milder distractor run.
      - Apply short-range repulsive avoidance from defenders.

    Returns:
      torch.Tensor of shape (1,2) giving force vector.
    """

    # Ensure obs is 1D
    if obs.dim() > 1:
        obs = obs.view(-1)
    x = obs
    n = x.numel()

    # Known block sizes
    teammates_len = 8
    adversaries_len = 16
    vel_len = 2  # agent_vel in "vel"
    pos_len = 2  # agent_pos - goal_pos in "pos"

    # Basic sanity: if observation shorter than expected, fallback to simple behavior
    min_expected = teammates_len + adversaries_len + vel_len + pos_len + 12  # 12 = sum of fixed parts in obs_block before agent_rot
    if n < min_expected:
        # fallback: simple chase ball direction (if available) or no force
        # try to take last two entries as a direction
        fallback = torch.zeros(2, dtype=obs.dtype, device=obs.device)
        if n >= 2:
            fallback = x[-2:]
            if torch.norm(fallback) < 1e-4:
                fallback = fallback + torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
        return fallback.unsqueeze(0)

    # Parse from the tail backwards
    idx = n
    teammates_block = x[idx - teammates_len: idx].clone(); idx -= teammates_len
    adversaries_block = x[idx - adversaries_len: idx].clone(); idx -= adversaries_len
    agent_vel_block = x[idx - vel_len: idx].clone(); idx -= vel_len
    agent_pos_minus_goal = x[idx - pos_len: idx].clone(); idx -= pos_len
    obs_main = x[:idx].clone()  # contains obs_block

    # Parse obs_main into known pieces. First six fixed 2D items = 12 dims
    fixed_obs_len = 12
    if obs_main.numel() < fixed_obs_len:
        # unexpected, fallback
        return torch.zeros(1, 2, dtype=obs.dtype, device=obs.device)
    agent_force = obs_main[0:2]
    agent_pos_minus_ball = obs_main[2:4]
    agent_vel_minus_ball_vel = obs_main[4:6]
    ball_pos_minus_goal = obs_main[6:8]
    ball_vel = obs_main[8:10]
    ball_force = obs_main[10:12]
    agent_rot = obs_main[12: ]  # may be empty or size 1/2

    # Helper lambdas
    def norm(v):
        return torch.norm(v)

    def safe_normalize(v, eps=1e-6):
        vnorm = norm(v)
        if vnorm < eps:
            return torch.zeros_like(v)
        return v / vnorm

    # Extract teammate data from teammates_block
    # teammates_block = [agent_pos - teammate_pos (2), agent_vel - teammate_vel (2), teammate_vel (2), teammate_force (2)]
    teammate_agentpos_minus_pos = teammates_block[0:2]
    teammate_agentvel_minus_vel = teammates_block[2:4]
    teammate_vel = teammates_block[4:6]
    teammate_force = teammates_block[6:8]

    # teammate_pos relative to agent: teammate_pos_rel = teammate_pos - agent_pos = - (agent_pos - teammate_pos)
    teammate_pos_rel = -teammate_agentpos_minus_pos

    # Extract adversary data (two adversaries concatenated)
    advs = []  # list of dicts with pos_rel (relative to agent), vel, force
    for i in range(2):
        base = i * 8
        agentpos_minus_advpos = adversaries_block[base: base + 2]
        agentvel_minus_advvel = adversaries_block[base + 2: base + 4]
        adv_vel = adversaries_block[base + 4: base + 6]
        adv_force = adversaries_block[base + 6: base + 8]
        adv_pos_rel = -agentpos_minus_advpos
        advs.append({
            "pos_rel": adv_pos_rel,
            "vel": adv_vel,
            "force": adv_force,
        })

    # Ball position relative to agent: ball_pos_rel = ball_pos - agent_pos = - (agent_pos - ball_pos)
    ball_pos_rel = -agent_pos_minus_ball

    # Agent position relative to goal is agent_pos_minus_goal; goal relative to agent is -agent_pos_minus_goal
    goal_pos_rel = -agent_pos_minus_goal

    # Core parameters
    lead_distance = 1.5  # how far ahead of teammate to lead (tunable)
    lead_lateral = 0.8  # lateral offset to try
    behind_distance = 1.2
    avoid_radius = 0.6
    avoid_strength = 1.0
    max_force = 1.0
    pass_time_threshold = 1.2  # seconds
    prediction_horizon = 1.0

    # Compute teammate attack direction
    teammate_speed = norm(teammate_vel)
    teammate_dir = safe_normalize(teammate_vel)
    if teammate_speed < 1e-3:
        # If teammate is stationary, use vector from teammate to goal (in agent frame) as proxy
        teammate_to_goal = ( (goal_pos_rel) - teammate_pos_rel )
        teammate_dir = safe_normalize(teammate_to_goal)
        if norm(teammate_dir) < 1e-3:
            # fallback direction: toward goal
            teammate_dir = safe_normalize(-agent_pos_minus_goal)

    # Compute candidate lead_point relative to agent
    lead_point_rel = teammate_pos_rel + teammate_dir * lead_distance

    # Evaluate lateral offsets to maximize distance from nearest defender
    perp = torch.tensor([-teammate_dir[1], teammate_dir[0]], dtype=obs.dtype, device=obs.device)
    perp = safe_normalize(perp)
    lateral_offsets = torch.tensor([-lead_lateral, 0.0, lead_lateral], dtype=obs.dtype, device=obs.device)

    def min_defender_distance_to(point_rel):
        dists = []
        for adv in advs:
            d = norm(point_rel - adv["pos_rel"])
            dists.append(d)
        return torch.min(torch.stack(dists))

    best_point = lead_point_rel
    best_min_dist = min_defender_distance_to(lead_point_rel)
    for offs in lateral_offsets:
        cand = lead_point_rel + perp * offs
        d = min_defender_distance_to(cand)
        if d > best_min_dist:
            best_min_dist = d
            best_point = cand

    chosen_lead = best_point.clone()

    # Detect if adversaries are converging on the teammate
    converging_count = 0
    for adv in advs:
        rel = teammate_pos_rel - adv["pos_rel"]  # vector from adv to teammate (in agent frame)
        dist = norm(rel) + 1e-6
        approach_speed = torch.dot(adv["vel"], safe_normalize(rel))  # positive if adv vel points toward teammate
        # normalize by magnitude to get relative tendency
        if approach_speed > 0.45:  # tunable threshold
            converging_count += 1

    if converging_count >= 1:
        # defenders converging: compute line through defenders and try to cut behind it
        adv_pos0 = advs[0]["pos_rel"]
        adv_pos1 = advs[1]["pos_rel"]
        line_vec = adv_pos1 - adv_pos0
        line_mid = (adv_pos0 + adv_pos1) * 0.5
        # perpendicular to defender line
        line_perp = torch.tensor([-line_vec[1], line_vec[0]], dtype=obs.dtype, device=obs.device)
        line_perp = safe_normalize(line_perp)
        # choose side of perp that is in direction of teammate attack (so we run behind their line in attack direction)
        sign = 1.0 if torch.dot(line_perp, teammate_dir) > 0 else -1.0
        behind_point = line_mid + line_perp * (behind_distance * sign)
        # prefer behind_point if it's reasonably far from defenders
        if min_defender_distance_to(behind_point) > best_min_dist * 0.9:
            chosen_lead = behind_point

    # If the predicted channel is likely blocked (closest defender nearer to chosen lead than ego), revert to drawing defenders
    ego_dist_to_lead = norm(chosen_lead)
    closest_def_dist = min_defender_distance_to(chosen_lead)
    blocked = (closest_def_dist + 0.35) < ego_dist_to_lead  # buffer
    if blocked:
        # choose distractor: either wide (larger lateral) or deep (farther along dir), whichever increases distance to defenders
        wide_point = teammate_pos_rel + perp * (lead_lateral * 2.0) + teammate_dir * (lead_distance * 0.5)
        deep_point = teammate_pos_rel + teammate_dir * (lead_distance * 2.0)
        if min_defender_distance_to(wide_point) > min_defender_distance_to(deep_point):
            chosen_lead = wide_point
        else:
            chosen_lead = deep_point

    # Timing: predict if a pass window exists (ball will arrive near chosen_lead soon)
    ball_speed = norm(ball_vel)
    dist_ball_to_lead = norm(chosen_lead - ball_pos_rel)
    ball_time = 1e6
    if ball_speed > 1e-3:
        ball_time = dist_ball_to_lead / (ball_speed + 1e-6)
    # Also check teammate alignment with chosen_lead
    teammate_to_lead = chosen_lead - teammate_pos_rel
    teammate_alignment = torch.dot(safe_normalize(teammate_vel), safe_normalize(teammate_to_lead)) if norm(teammate_to_lead) > 1e-6 else 0.0

    # Decide acceleration multiplier based on pass likelihood
    accel_mult = 0.6  # default distractor pace
    if (ball_time < pass_time_threshold and teammate_alignment > 0.3) or (not blocked and teammate_alignment > 0.5):
        accel_mult = 1.5  # accelerate toward lead_point

    # Desired vector toward chosen lead (in agent frame)
    desired = chosen_lead.clone()
    desired_dir = safe_normalize(desired)
    desired_force = desired_dir * accel_mult

    # Short-range repulsive avoidance from defenders
    repulsion = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    for adv in advs:
        rel = adv["pos_rel"] - torch.zeros_like(adv["pos_rel"])  # adv_pos relative to agent
        d = norm(rel)
        if d < avoid_radius and d > 1e-6:
            # push away from defender (i.e., negative of vector from agent to defender)
            push_dir = -safe_normalize(rel)
            strength = avoid_strength * (1.0 - (d / avoid_radius))
            repulsion += push_dir * strength

    # Combine desired force and repulsion
    force = desired_force + repulsion

    # If desired magnitude is very small, provide a mild lateral motion to stay active
    if norm(force) < 1e-3:
        force = teammate_dir * 0.2 + perp * 0.1

    # Clip to max_force
    f_norm = norm(force)
    if f_norm > max_force:
        force = force / (f_norm + 1e-9) * max_force

    # Return as (1,2) tensor
    return force.reshape(1, 2)

import torch

def agent1_policy(obs):
    """
    Complementary Proactive Space Creator / Lead Runner policy for agent1.

    Coordination reasoning implemented in this policy:
      - Be predictable: maintain a clear attack direction (toward goal) when teammate (agent0)
        expects a consistent teammate_velocity to predict a lead point.
      - Avoid occupying the exact lead point partner would run into: if partner's predicted
        lead_point is near our chosen lead, shift laterally to be complementary (create space)
        rather than collide.
      - If partner appears to be the primary target of a pass (ball and alignment timing),
        adopt a distractor role (run wide or deep) to pull defenders away.
      - If defenders are converging on either of us, try to cut behind their line to open a through channel.
      - Use short-range avoidance from defenders and from the teammate to avoid clustering.

    The observation parsing mirrors the environment.observation_base layout used by the
    partner's policy. Returns a torch tensor of shape (1,2) giving the desired force.
    """

    # Ensure obs is 1D
    if obs.dim() > 1:
        obs = obs.view(-1)
    x = obs
    n = x.numel()

    # Known block sizes (same assumptions as partner)
    teammates_len = 8
    adversaries_len = 16
    vel_len = 2  # agent_vel in "vel"
    pos_len = 2  # agent_pos - goal_pos in "pos"

    # Basic sanity fallback
    min_expected = teammates_len + adversaries_len + vel_len + pos_len + 12
    if n < min_expected:
        fallback = torch.zeros(2, dtype=obs.dtype, device=obs.device)
        if n >= 2:
            fallback = x[-2:]
            if torch.norm(fallback) < 1e-4:
                fallback = fallback + torch.tensor([0.0, 0.0], dtype=obs.dtype, device=obs.device)
        return fallback.unsqueeze(0)

    # Parse observation from tail
    idx = n
    teammates_block = x[idx - teammates_len: idx].clone(); idx -= teammates_len
    adversaries_block = x[idx - adversaries_len: idx].clone(); idx -= adversaries_len
    agent_vel_block = x[idx - vel_len: idx].clone(); idx -= vel_len
    agent_pos_minus_goal = x[idx - pos_len: idx].clone(); idx -= pos_len
    obs_main = x[:idx].clone()

    # Parse obs_main fixed parts
    fixed_obs_len = 12
    if obs_main.numel() < fixed_obs_len:
        return torch.zeros(1, 2, dtype=obs.dtype, device=obs.device)
    agent_force = obs_main[0:2]
    agent_pos_minus_ball = obs_main[2:4]
    agent_vel_minus_ball_vel = obs_main[4:6]
    ball_pos_minus_goal = obs_main[6:8]
    ball_vel = obs_main[8:10]
    ball_force = obs_main[10:12]
    agent_rot = obs_main[12: ]

    # Helpers
    def norm(v):
        return torch.norm(v)

    def safe_normalize(v, eps=1e-6):
        vnorm = norm(v)
        if vnorm < eps:
            return torch.zeros_like(v)
        return v / vnorm

    # Extract teammate (the partner agent) info
    # teammates_block = [agent_pos - teammate_pos (2), agent_vel - teammate_vel (2), teammate_vel (2), teammate_force (2)]
    teammate_agentpos_minus_pos = teammates_block[0:2]
    teammate_agentvel_minus_vel = teammates_block[2:4]
    teammate_vel = teammates_block[4:6]
    teammate_force = teammates_block[6:8]

    # teammate_pos relative to agent: teammate_pos_rel = teammate_pos - agent_pos = - (agent_pos - teammate_pos)
    teammate_pos_rel = -teammate_agentpos_minus_pos

    # Extract adversaries
    advs = []
    for i in range(2):
        base = i * 8
        agentpos_minus_advpos = adversaries_block[base: base + 2]
        agentvel_minus_advvel = adversaries_block[base + 2: base + 4]
        adv_vel = adversaries_block[base + 4: base + 6]
        adv_force = adversaries_block[base + 6: base + 8]
        adv_pos_rel = -agentpos_minus_advpos
        advs.append({
            "pos_rel": adv_pos_rel,
            "vel": adv_vel,
            "force": adv_force,
        })

    # Ball and goal relative positions in agent frame
    ball_pos_rel = -agent_pos_minus_ball
    goal_pos_rel = -agent_pos_minus_goal

    # Core tunable params (similar to partner but slightly biased to avoid collisions)
    lead_distance = 1.5
    lead_lateral = 0.8
    behind_distance = 1.2
    avoid_radius = 0.6
    avoid_strength = 1.0
    teammate_avoid_strength = 0.9
    max_force = 1.0
    pass_time_threshold = 1.2
    prediction_horizon = 1.0

    # Compute our default attack direction (toward goal) as a stable baseline
    to_goal = safe_normalize(-agent_pos_minus_goal)

    # Compute teammate (partner) attack direction from their velocity
    teammate_speed = norm(teammate_vel)
    teammate_dir = safe_normalize(teammate_vel)
    if teammate_speed < 1e-3:
        teammate_to_goal = (goal_pos_rel - teammate_pos_rel)
        teammate_dir = safe_normalize(teammate_to_goal)
        if norm(teammate_dir) < 1e-3:
            teammate_dir = to_goal

    # Compute candidate lead_point relative to our agent (we position ourselves in relation to our agent frame)
    # Here we consider moving to create/exploit space relative to ourselves, but we also compute partner's expected lead.
    my_teammate_dir = teammate_dir  # partner direction
    my_lead_point_rel = teammate_pos_rel + my_teammate_dir * lead_distance

    # Also compute our own preferred lead based on our own velocity (attempt to be proactive too)
    my_speed = norm(agent_vel_block)
    my_dir = safe_normalize(agent_vel_block) if my_speed > 1e-3 else to_goal
    my_candidate_lead = my_dir * lead_distance + torch.zeros_like(my_dir) + ( -agent_pos_minus_goal * 0.0 )
    # The above is a short heuristic: lead ahead of our current direction relative to ourselves

    # Evaluate lateral offsets to maximize distance from nearest defender
    perp = torch.tensor([-my_dir[1], my_dir[0]], dtype=obs.dtype, device=obs.device)
    perp = safe_normalize(perp)
    lateral_offsets = torch.tensor([-lead_lateral, 0.0, lead_lateral], dtype=obs.dtype, device=obs.device)

    def min_defender_distance_to(point_rel):
        dists = []
        for adv in advs:
            d = norm(point_rel - adv["pos_rel"])
            dists.append(d)
        return torch.min(torch.stack(dists))

    # Choose best offset for our candidate lead
    best_point = my_candidate_lead
    best_min_dist = min_defender_distance_to(best_point)
    for offs in lateral_offsets:
        cand = my_candidate_lead + perp * offs
        d = min_defender_distance_to(cand)
        if d > best_min_dist:
            best_min_dist = d
            best_point = cand

    chosen_lead = best_point.clone()

    # Detect if adversaries are converging on either our teammate or us; if so, compute behind-point strategy
    def count_converging(target_pos_rel):
        cnt = 0
        for adv in advs:
            rel = target_pos_rel - adv["pos_rel"]
            dist = norm(rel) + 1e-6
            approach_speed = torch.dot(adv["vel"], safe_normalize(rel))
            if approach_speed > 0.45:
                cnt += 1
        return cnt

    conv_on_teammate = count_converging(teammate_pos_rel)
    conv_on_me = count_converging(-agent_pos_minus_ball * 0.0 + torch.zeros(2, dtype=obs.dtype, device=obs.device))

    if conv_on_teammate >= 1:
        adv_pos0 = advs[0]["pos_rel"]
        adv_pos1 = advs[1]["pos_rel"]
        line_vec = adv_pos1 - adv_pos0
        line_mid = (adv_pos0 + adv_pos1) * 0.5
        line_perp = torch.tensor([-line_vec[1], line_vec[0]], dtype=obs.dtype, device=obs.device)
        line_perp = safe_normalize(line_perp)
        sign = 1.0 if torch.dot(line_perp, my_teammate_dir) > 0 else -1.0
        behind_point = line_mid + line_perp * (behind_distance * sign)
        if min_defender_distance_to(behind_point) > best_min_dist * 0.9:
            chosen_lead = behind_point

    # If our chosen lead is likely blocked by defenders (they're closer than us), switch to distractor
    ego_dist_to_lead = norm(chosen_lead)
    closest_def_dist = min_defender_distance_to(chosen_lead)
    blocked = (closest_def_dist + 0.35) < ego_dist_to_lead
    if blocked:
        wide_point = teammate_pos_rel + perp * (lead_lateral * 2.0) + my_teammate_dir * (lead_distance * 0.5)
        deep_point = teammate_pos_rel + my_teammate_dir * (lead_distance * 2.0)
        if min_defender_distance_to(wide_point) > min_defender_distance_to(deep_point):
            chosen_lead = wide_point
        else:
            chosen_lead = deep_point

    # Predict partner's intended lead based on their observed velocity (so we can avoid conflicting)
    partner_predicted_lead = teammate_pos_rel + my_teammate_dir * lead_distance
    partner_dist_to_lead = norm(partner_predicted_lead - chosen_lead)

    # If partner likely to target the same spot (and partner seems aligned and ball timing favors them), move to complement
    # Estimate ball time to partner_predicted_lead
    ball_speed = norm(ball_vel)
    dist_ball_to_partner_lead = norm(partner_predicted_lead - ball_pos_rel)
    ball_time = 1e6
    if ball_speed > 1e-3:
        ball_time = dist_ball_to_partner_lead / (ball_speed + 1e-6)

    partner_alignment = torch.dot(safe_normalize(teammate_vel), safe_normalize(partner_predicted_lead - teammate_pos_rel)) if norm(partner_predicted_lead - teammate_pos_rel) > 1e-6 else 0.0

    # If partner likely to run to that spot (ball_time small and alignment good) and partner_predicted_lead is close to our chosen lead,
    # then shift laterally away to avoid conflict (be complementary). If partner unlikely, we may try to occupy it.
    if partner_dist_to_lead < 0.8 and (ball_time < pass_time_threshold and partner_alignment > 0.3 or partner_alignment > 0.6):
        # Shift chosen_lead laterally away from partner_predicted_lead
        away_dir = safe_normalize(chosen_lead - partner_predicted_lead)
        if norm(away_dir) < 1e-3:
            # pick perpendicular
            away_dir = perp
        chosen_lead = chosen_lead + away_dir * 0.9

    # Decide acceleration: if pass window predicted for partner, we adopt distractor pace; if pass likely to us (ball near our chosen lead), accelerate
    dist_ball_to_my_lead = norm(chosen_lead - ball_pos_rel)
    ball_time_to_my_lead = 1e6
    if ball_speed > 1e-3:
        ball_time_to_my_lead = dist_ball_to_my_lead / (ball_speed + 1e-6)

    teammate_to_my_lead = chosen_lead - teammate_pos_rel
    teammate_alignment_to_my_lead = torch.dot(safe_normalize(teammate_vel), safe_normalize(teammate_to_my_lead)) if norm(teammate_to_my_lead) > 1e-6 else 0.0

    accel_mult = 0.6
    # accelerate if ball will arrive soon to our spot or if partner is not clearly the receiver
    if (ball_time_to_my_lead < pass_time_threshold and teammate_alignment_to_my_lead > 0.3) or (not blocked and teammate_alignment_to_my_lead > 0.5):
        accel_mult = 1.3
    # if partner likely to be receiver, keep moderate distractor pace
    if (ball_time < pass_time_threshold and partner_alignment > 0.5):
        accel_mult = 0.5

    # Desired vector toward chosen lead (in agent frame)
    desired = chosen_lead.clone()
    desired_dir = safe_normalize(desired)
    desired_force = desired_dir * accel_mult

    # Short-range repulsive avoidance from defenders
    repulsion = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    for adv in advs:
        rel = adv["pos_rel"]
        d = norm(rel)
        if d < avoid_radius and d > 1e-6:
            push_dir = -safe_normalize(rel)
            strength = avoid_strength * (1.0 - (d / avoid_radius))
            repulsion += push_dir * strength

    # Avoid crowding teammate: if teammate is near our chosen lead, push away a bit
    teammate_rel = teammate_pos_rel
    d_teammate = norm(teammate_rel - chosen_lead)
    teammate_rep = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    if d_teammate < 0.9 and d_teammate > 1e-6:
        teammate_rep_dir = safe_normalize(chosen_lead - teammate_rel)
        teammate_rep += teammate_rep_dir * (teammate_avoid_strength * (1.0 - (d_teammate / 0.9)))

    # Combine
    force = desired_force + repulsion + teammate_rep

    # If extremely small, nudge toward goal to remain predictable
    if norm(force) < 1e-3:
        force = to_goal * 0.2 + perp * 0.1

    # Clip
    f_norm = norm(force)
    if f_norm > max_force:
        force = force / (f_norm + 1e-9) * max_force

    return force.reshape(1, 2)