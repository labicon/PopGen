import torch

def agent0_policy(obs):
    """
    Anticipatory Playmaker with Shadowing Support (agent0)

    Robust parsing of the provided observation vector (constructed by observation_base):
      final vector = [ obs['obs'] , obs['pos'] , obs['vel'] , obs['adversaries'] , obs['teammates'] ]
    We slice from the end (teammate/adversaries/pos/vel have fixed sizes) to avoid brittle
    assumptions about the size of agent_rot in obs['obs'].

    Returns: torch.Tensor of shape (1,2) representing the applied force.
    """
    eps = 1e-6

    x = obs.flatten()
    L = x.shape[0]

    # fixed block sizes (per observation_base construction)
    teammate_len = 8           # one teammate: [agent_pos - teammate_pos(2), agent_vel - teammate_vel(2), teammate_vel(2), teammate_force(2)]
    num_adversaries = 2
    adv_len = 8 * num_adversaries
    vel_len = 2                # obs['vel'] -> agent_vel (2)
    pos_len = 2                # obs['pos'] -> agent_pos - goal_pos (2)

    min_needed = teammate_len + adv_len + vel_len + pos_len
    if L < min_needed:
        # malformed observation, safe fallback
        return torch.zeros((1, 2), dtype=x.dtype, device=x.device)

    # slice from the end to be robust to variable agent_rot or extra obs entries
    idx = L
    teammate_block = x[idx - teammate_len: idx].clone(); idx -= teammate_len
    adv_block = x[idx - adv_len: idx].clone(); idx -= adv_len
    vel_block = x[idx - vel_len: idx].clone(); idx -= vel_len
    pos_block = x[idx - pos_len: idx].clone(); idx -= pos_len
    main = x[:idx].clone()  # the remaining prefix contains obs['obs'] (variable tail: agent_rot etc.)

    # parse main (first part has fixed entries in order)
    # expected min size for main is at least 12 (2 each for: agent_force, agent_pos-ball, agent_vel-ball_vel,
    # ball_pos-goal_pos, ball_vel, ball_force)
    if main.shape[0] < 12:
        return torch.zeros((1, 2), dtype=x.dtype, device=x.device)

    agent_force = main[0:2]
    a_pos_minus_ball = main[2:4]        # agent_pos - ball_pos
    a_vel_minus_ball_vel = main[4:6]    # agent_vel - ball_vel
    ball_minus_goal = main[6:8]         # ball_pos - goal_pos
    ball_vel = main[8:10]
    ball_force = main[10:12]
    # agent_rot = main[12:]

    # pos_block is agent_pos - goal_pos
    agent_pos_minus_goal = pos_block[0:2]
    # vel_block is agent_vel (absolute)
    agent_vel = vel_block[0:2]

    # teammate parsing
    teammate_rel_pos = teammate_block[0:2]   # agent_pos - teammate_pos
    teammate_vel_rel = teammate_block[2:4]
    teammate_vel = teammate_block[4:6]
    teammate_force = teammate_block[6:8]

    # adversary parsing (each 8-long chunk)
    adv_rel_list = []          # agent_pos - adversary_pos
    adv_agent_vel_diff_list = []
    adv_vel_true_list = []
    adv_force_list = []
    for i in range(0, adv_len, 8):
        adv_rel_list.append(adv_block[i : i + 2])
        adv_agent_vel_diff_list.append(adv_block[i + 2 : i + 4])
        adv_vel_true_list.append(adv_block[i + 4 : i + 6])
        adv_force_list.append(adv_block[i + 6 : i + 8])

    # helper functions
    def vec_norm(v):
        n = torch.norm(v)
        if n.item() < eps:
            return torch.zeros_like(v)
        return v / (n + eps)

    def length(v):
        return torch.norm(v)

    # convert to agent-centered coords
    # r_ball = ball_pos - agent_pos
    r_ball = -a_pos_minus_ball
    # r_goal = goal_pos - agent_pos  (pos_block stores agent_pos - goal_pos)
    r_goal = -agent_pos_minus_goal
    # teammate position (relative to agent): teammate_pos - agent_pos
    r_teammate = -teammate_rel_pos
    # adversary positions relative to agent
    r_advs = [(-r).clone() for r in adv_rel_list]

    # Distances
    d_ego_ball = length(a_pos_minus_ball)  # |agent_pos - ball_pos|
    teammate_pos_minus_ball = r_teammate - r_ball  # teammate_pos - ball_pos
    d_tm_ball = length(teammate_pos_minus_ball)

    # nearest adversary to our goal (threat)
    adv_dist_to_goal = []
    for r_adv in r_advs:
        adv_to_goal = r_adv - r_goal  # adversary_pos - goal_pos
        adv_dist_to_goal.append(length(adv_to_goal))
    adv_dist_to_goal = torch.stack(adv_dist_to_goal)
    nearest_adv_idx = int(torch.argmin(adv_dist_to_goal).item())

    # parameters (tunable)
    max_force = 1.0
    attack_lead = 0.9
    block_offset = 0.6
    defend_distance = 0.8
    role_margin = 0.2
    jitter = 0.05
    damping = 0.5
    field_threat_threshold = 1.5

    # target_rel is in agent frame: vector from agent to desired position
    target_rel = torch.zeros(2, dtype=x.dtype, device=x.device)

    # Defensive override: cover if an adversary is dangerously close to our goal
    if adv_dist_to_goal[nearest_adv_idx].item() < field_threat_threshold:
        r_adv = r_advs[nearest_adv_idx]
        # direction from goal to adversary in agent frame: (adversary_pos - goal_pos) = r_adv - r_goal
        direction = vec_norm(r_adv - r_goal)
        # target in agent frame: position at defend_distance along that line from goal
        target_rel = r_goal + direction * defend_distance
    else:
        # decide roles based on distances to ball
        if (d_ego_ball + role_margin) < d_tm_ball:
            # attacker: go to a lead point in front of the ball towards the opponent goal
            # approximate opponent goal direction using ball_minus_goal (ball_pos - goal_pos)
            opp_dir = vec_norm(ball_minus_goal)
            # target (agent frame) = r_ball + opp_dir * attack_lead
            target_rel = r_ball + opp_dir * attack_lead
        else:
            # support/shadow: cut passing lane / shadow most dangerous adversary
            adv_dist_to_ball = []
            for r_adv in r_advs:
                adv_dist_to_ball.append(length(r_adv - r_ball))
            adv_dist_to_ball = torch.stack(adv_dist_to_ball)
            adv_to_ball_idx = int(torch.argmin(adv_dist_to_ball).item())
            r_danger = r_advs[adv_to_ball_idx]

            midpoint = 0.5 * (r_ball + r_teammate)
            cut_dir = vec_norm(r_danger - midpoint)
            target_rel = midpoint + cut_dir * block_offset

    # small perpendicular jitter to reduce predictability
    to_target = target_rel
    if torch.norm(to_target) < eps:
        # if target is essentially on top of agent, nudge a bit perpendicular to agent_vel if available
        if torch.norm(agent_vel) > eps:
            perp = torch.tensor([-agent_vel[1], agent_vel[0]], dtype=x.dtype, device=x.device)
            perp = vec_norm(perp) * jitter
        else:
            perp = torch.tensor([0.0, 0.0], dtype=x.dtype, device=x.device)
    else:
        perp = torch.tensor([-to_target[1], to_target[0]], dtype=x.dtype, device=x.device)
        perp = vec_norm(perp) * jitter

    desired = to_target + perp
    # desired direction (unit) to convert to preferred heading
    n = torch.norm(desired)
    if n.item() > eps:
        desired_dir = desired / (n + eps)
    else:
        desired_dir = torch.zeros_like(desired)

    # Force = push in desired_dir minus damping proportional to current velocity
    force = desired_dir * max_force - (damping * agent_vel)

    # clamp magnitude
    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force / (fmag + eps) * max_force

    action = force.unsqueeze(0)
    return action

import torch

def agent1_policy(obs):
    """
    Coordinating Anticipatory Playmaker (agent1)

    This policy implements a role-switching playmaker similar to the partner but
    with small coordination adjustments to complement it:
    - Be more conservative claiming the ball (larger role_margin) so we don't both
      try to attack. If the teammate is committed (moving toward the ball), yield.
    - When supporting and the teammate looks like attacking, move to a receiving
      offset (lateral to the ball-teammate line) to be a better pass target.
    - Keep the same defensive override: cover any adversary threatening our goal.

    The observation slicing is robust to variable-sized "main" prefix like the
    partner implementation.
    Returns: torch.Tensor of shape (1,2)
    """
    eps = 1e-6

    x = obs.flatten()
    L = x.shape[0]

    # fixed block sizes (matching observation_base):
    teammate_len = 8
    num_adversaries = 2
    adv_len = 8 * num_adversaries
    vel_len = 2
    pos_len = 2

    min_needed = teammate_len + adv_len + vel_len + pos_len
    if L < min_needed:
        return torch.zeros((1, 2), dtype=x.dtype, device=x.device)

    # slice from end to be robust
    idx = L
    teammate_block = x[idx - teammate_len: idx].clone(); idx -= teammate_len
    adv_block = x[idx - adv_len: idx].clone(); idx -= adv_len
    vel_block = x[idx - vel_len: idx].clone(); idx -= vel_len
    pos_block = x[idx - pos_len: idx].clone(); idx -= pos_len
    main = x[:idx].clone()

    # basic sanity
    if main.shape[0] < 12:
        return torch.zeros((1, 2), dtype=x.dtype, device=x.device)

    agent_force = main[0:2]
    a_pos_minus_ball = main[2:4]
    a_vel_minus_ball_vel = main[4:6]
    ball_minus_goal = main[6:8]
    ball_vel = main[8:10]
    ball_force = main[10:12]

    agent_pos_minus_goal = pos_block[0:2]
    agent_vel = vel_block[0:2]

    teammate_rel_pos = teammate_block[0:2]
    teammate_vel_rel = teammate_block[2:4]
    teammate_vel = teammate_block[4:6]
    teammate_force = teammate_block[6:8]

    # adversary parsing
    r_advs = []
    adv_rel_list = []
    for i in range(0, adv_len, 8):
        adv_rel = adv_block[i : i + 2]
        adv_rel_list.append(adv_rel)
        # convert to adversary position relative to agent: r_adv = -adv_rel
        r_advs.append((-adv_rel).clone())

    def vec_norm(v):
        n = torch.norm(v)
        if n.item() < eps:
            return torch.zeros_like(v)
        return v / (n + eps)

    def length(v):
        return torch.norm(v)

    # convert to agent frame useful vectors
    r_ball = -a_pos_minus_ball            # ball_pos - agent_pos
    r_goal = -agent_pos_minus_goal        # goal_pos - agent_pos
    r_teammate = -teammate_rel_pos        # teammate_pos - agent_pos

    # distances
    d_ego_ball = length(a_pos_minus_ball)
    teammate_pos_minus_ball = r_teammate - r_ball
    d_tm_ball = length(teammate_pos_minus_ball)

    # nearest adversary to our goal (threat)
    adv_dist_to_goal = []
    for r_adv in r_advs:
        adv_to_goal = r_adv - r_goal  # adversary_pos - goal_pos
        adv_dist_to_goal.append(length(adv_to_goal))
    adv_dist_to_goal = torch.stack(adv_dist_to_goal)
    nearest_adv_idx = int(torch.argmin(adv_dist_to_goal).item())

    # parameters (tuned for complementary play)
    max_force = 1.0
    attack_lead = 0.95
    block_offset = 0.6
    defend_distance = 0.9
    role_margin = 0.35    # be more conservative claiming attack
    jitter = 0.06
    damping = 0.6
    field_threat_threshold = 1.5

    target_rel = torch.zeros(2, dtype=x.dtype, device=x.device)

    # Defensive override: cover if any adversary is dangerously close to our goal
    if adv_dist_to_goal[nearest_adv_idx].item() < field_threat_threshold:
        r_adv = r_advs[nearest_adv_idx]
        direction = vec_norm(r_adv - r_goal)
        target_rel = r_goal + direction * defend_distance
    else:
        # compute teammate commitment: is teammate moving toward the ball?
        # teammate_to_ball = ball_pos - teammate_pos in agent frame = r_ball - r_teammate
        teammate_to_ball = r_ball - r_teammate
        # projection of teammate_vel onto teammate_to_ball
        proj = 0.0
        denom = torch.norm(teammate_to_ball)
        if denom.item() > eps:
            proj = torch.dot(teammate_vel, teammate_to_ball) / (denom + eps)
        # normalized commitment (positive = moving toward ball)
        teammate_commitment = torch.clamp(proj / (1.0 + torch.abs(proj)), min=-1.0, max=1.0).item()

        # If teammate is effectively committed to the ball, yield the attacker role unless we are much closer.
        teammate_is_committed = teammate_commitment > 0.15

        # Role decision: attack only if clearly closer and teammate not committed
        if ((d_ego_ball + role_margin) < d_tm_ball) and not teammate_is_committed:
            # attacker: lead point ahead of ball toward opponent goal
            opp_dir = vec_norm(ball_minus_goal)
            target_rel = r_ball + opp_dir * attack_lead
        else:
            # support role: either cut passing lane or become a lateral receiving target
            # find adversary closest to the ball (most dangerous)
            adv_dist_to_ball = []
            for r_adv in r_advs:
                adv_dist_to_ball.append(length(r_adv - r_ball))
            adv_dist_to_ball = torch.stack(adv_dist_to_ball)
            adv_to_ball_idx = int(torch.argmin(adv_dist_to_ball).item())
            r_danger = r_advs[adv_to_ball_idx]

            # midpoint between ball and teammate (in agent frame)
            midpoint = 0.5 * (r_ball + r_teammate)

            # If teammate appears to be attacking (committed), position laterally for a pass
            if teammate_is_committed:
                # direction from teammate to ball (in agent frame)
                t2b = r_ball - r_teammate
                if torch.norm(t2b) < eps:
                    lateral = torch.tensor([0.0, 0.0], dtype=x.dtype, device=x.device)
                else:
                    perp = torch.tensor([-t2b[1], t2b[0]], dtype=x.dtype, device=x.device)
                    # choose sign of perp based on agent's current position to avoid colliding with teammate:
                    # prefer the side that places agent further from teammate
                    # compare dot((agent_pos - teammate_pos), perp) sign; agent_pos - teammate_pos = -teammate_rel_pos
                    side = torch.dot(-teammate_rel_pos, perp)
                    if side.item() < 0:
                        perp = -perp
                    lateral = vec_norm(perp) * block_offset
                target_rel = midpoint + lateral
            else:
                # defensive cut toward the dangerous adversary from the midpoint
                cut_dir = vec_norm(r_danger - midpoint)
                target_rel = midpoint + cut_dir * block_offset

    # small perpendicular jitter to reduce predictability
    to_target = target_rel
    if torch.norm(to_target) < eps:
        if torch.norm(agent_vel) > eps:
            perp_j = torch.tensor([-agent_vel[1], agent_vel[0]], dtype=x.dtype, device=x.device)
            perp_j = vec_norm(perp_j) * jitter
        else:
            perp_j = torch.tensor([0.0, 0.0], dtype=x.dtype, device=x.device)
    else:
        perp_j = torch.tensor([-to_target[1], to_target[0]], dtype=x.dtype, device=x.device)
        perp_j = vec_norm(perp_j) * jitter

    desired = to_target + perp_j
    n = torch.norm(desired)
    if n.item() > eps:
        desired_dir = desired / (n + eps)
    else:
        desired_dir = torch.zeros_like(desired)

    force = desired_dir * max_force - (damping * agent_vel)

    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force / (fmag + eps) * max_force

    return force.unsqueeze(0)