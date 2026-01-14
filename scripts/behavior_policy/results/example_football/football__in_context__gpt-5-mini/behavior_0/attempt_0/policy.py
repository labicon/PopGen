import torch

def agent0_policy(obs):
    """
    Anticipatory Playmaker with Shadowing Support

    Observation parsing notes (based on observation_base ordering):
      - The final observation vector layout (concatenated) is:
        [ main_obs | pos (2) | vel (2) | adversaries (8*num_adv) | teammates (8) ]
      - main_obs contains (in order):
        agent_force(2), agent_pos - ball_pos(2), agent_vel - ball_vel(2),
        ball_pos - goal_pos(2), ball_vel(2), ball_force(2), agent_rot(?)

      We extract necessary quantities by slicing relative blocks. We treat the agent as
      the origin for position vectors: r_ball = ball_pos - agent_pos = - (agent_pos - ball_pos),
      r_teammate = teammate_pos - agent_pos = - (agent_pos - teammate_pos), etc.

    Returns:
      torch.Tensor of shape (1,2) representing force to apply.
    """

    eps = 1e-6

    x = obs.flatten()
    L = x.shape[0]
    # basic expected trailing sizes
    team_len = 8
    adv_len = 16  # two adversaries * 8
    vel_len = 2
    pos_len = 2
    tail_len = team_len + adv_len + vel_len + pos_len
    if L <= tail_len + 12:  # minimal safety check
        # fallback: do nothing
        return torch.zeros((1, 2), dtype=x.dtype, device=x.device)

    main = x[: (L - tail_len)]
    pos_block = x[(L - tail_len) : (L - tail_len + pos_len)]
    vel_block = x[(L - tail_len + pos_len) : (L - tail_len + pos_len + vel_len)]
    adv_block = x[(L - tail_len + pos_len + vel_len) : (L - team_len)]
    team_block = x[-team_len:]

    # parse main block (length is >=12; agent_rot may add extra dims)
    # indices:
    # 0:2 agent_force
    # 2:4 agent_pos - ball_pos
    # 4:6 agent_vel - ball_vel
    # 6:8 ball_pos - goal_pos
    # 8:10 ball_vel
    # 10:12 ball_force
    agent_force = main[0:2]
    a_pos_minus_ball = main[2:4]  # agent_pos - ball_pos
    a_vel_minus_ball_vel = main[4:6]
    ball_minus_goal = main[6:8]  # ball_pos - goal_pos
    ball_vel = main[8:10]
    ball_force = main[10:12]
    # agent_rot = main[12:]

    agent_pos_minus_goal = pos_block[0:2]
    agent_vel = vel_block[0:2]

    # teammate block: [agent_pos - teammate_pos, agent_vel - teammate_vel, teammate_vel, teammate_force]
    teammate_rel_pos = team_block[0:2]  # agent_pos - teammate_pos
    teammate_vel_rel = team_block[2:4]
    teammate_vel = team_block[4:6]
    teammate_force = team_block[6:8]

    # adversary blocks: for each adversary: [agent_pos - adversary_pos, agent_vel - adversary_vel, adversary_vel, adversary_force]
    adv_rel_list = []
    adv_vel_list = []
    adv_force_list = []
    for i in range(0, adv_len, 8):
        adv_rel_list.append(adv_block[i : i + 2])
        adv_vel_list.append(adv_block[i + 2 : i + 4])
        adv_vel_list.append(adv_block[i + 4 : i + 6])  # note: this corresponds to adversary_vel slot layout
        # careful: according to construction, it's [agent_pos - adversary_pos, agent_vel - adversary_vel, adversary_vel, adversary_force]
        # so indices are: 0:2 adv_rel_pos, 2:4 adv_agent_vel_diff, 4:6 adversary_vel, 6:8 adversary_force
        # We already appended a 2-length slice for 2:4 into adv_vel_list; the next 4:6 is adversary_vel
        # The code above appended both as separate list entries; we'll reindex correctly below.
    # Re-parse properly
    adv_rel_list = []
    adv_agent_vel_diff_list = []
    adv_vel_true_list = []
    adv_force_list = []
    for i in range(0, adv_len, 8):
        adv_rel_list.append(adv_block[i : i + 2])
        adv_agent_vel_diff_list.append(adv_block[i + 2 : i + 4])
        adv_vel_true_list.append(adv_block[i + 4 : i + 6])
        adv_force_list.append(adv_block[i + 6 : i + 8])

    # helper norms
    def vec_norm(v):
        n = torch.norm(v)
        if n.item() < eps:
            return v * 0.0
        return v / (n + eps)

    def length(v):
        return torch.norm(v)

    # Convert to agent-centered coordinates (agent at origin)
    r_ball = -a_pos_minus_ball  # ball_pos - agent_pos
    r_goal = -agent_pos_minus_goal  # goal_pos - agent_pos
    r_teammate = -teammate_rel_pos
    r_advs = [ -r for r in adv_rel_list ]  # adversary_pos - agent_pos

    # Distances
    d_ego_ball = length(a_pos_minus_ball)  # same as length(r_ball)
    # teammate distance to ball: teammate_pos - ball_pos = r_teammate - r_ball
    teammate_pos_minus_ball = r_teammate - r_ball
    d_tm_ball = length(teammate_pos_minus_ball)

    # find nearest adversary to goal (threat to our goal)
    adv_dist_to_goal = []
    for r_adv in r_advs:
        adv_to_goal = r_adv - r_goal  # adversary_pos - goal_pos
        adv_dist_to_goal.append(length(adv_to_goal))
    adv_dist_to_goal = torch.stack(adv_dist_to_goal)
    nearest_adv_idx = int(torch.argmin(adv_dist_to_goal).item())

    # threat threshold (field units)
    field_threat_threshold = 1.5

    # constants
    max_force = 1.0
    attack_lead = 0.9
    block_offset = 0.6
    defend_distance = 0.8
    role_margin = 0.2
    jitter = 0.05
    damping = 0.5

    target_rel = torch.zeros(2, dtype=x.dtype, device=x.device)

    # Defensive override: adversary close to our goal
    if adv_dist_to_goal[nearest_adv_idx] < field_threat_threshold:
        r_adv = r_advs[nearest_adv_idx]
        # direction from goal to adversary in agent frame: (adversary - goal) = r_adv - r_goal
        direction = vec_norm(r_adv - r_goal)
        # target in agent frame = r_goal + direction * defend_distance
        target_rel = r_goal + direction * defend_distance
    else:
        # Decide roles
        if (d_ego_ball + role_margin) < d_tm_ball:
            # Attacker role
            # Opponent goal direction approximated by ball_pos - goal_pos (this is invariant)
            opp_dir = vec_norm(ball_minus_goal)
            # target in world = ball_pos + opp_dir * attack_lead
            # in agent frame: r_ball + opp_dir * attack_lead
            target_rel = r_ball + opp_dir * attack_lead
        else:
            # Support / shadow role
            # find adversary nearest to ball (most dangerous)
            adv_dist_to_ball = []
            for r_adv in r_advs:
                # adv_pos - ball_pos = r_adv - r_ball
                adv_dist_to_ball.append(length(r_adv - r_ball))
            adv_dist_to_ball = torch.stack(adv_dist_to_ball)
            adv_to_ball_idx = int(torch.argmin(adv_dist_to_ball).item())
            r_danger = r_advs[adv_to_ball_idx]

            midpoint = 0.5 * (r_ball + r_teammate)
            cut_dir = vec_norm(r_danger - midpoint)
            target_rel = midpoint + cut_dir * block_offset

    # add small perpendicular jitter to avoid predictability
    to_target = target_rel
    # if target is extremely close to agent, no meaningful direction; keep small jitter
    if torch.norm(to_target) < eps:
        perp = torch.tensor([0.0, 0.0], dtype=x.dtype, device=x.device)
    else:
        perp = torch.tensor([-to_target[1], to_target[0]], dtype=x.dtype, device=x.device)
        perp = vec_norm(perp) * jitter

    desired = to_target + perp
    desired_dir = desired
    n = torch.norm(desired_dir)
    if n.item() > eps:
        desired_dir = desired_dir / (n + eps)
    else:
        desired_dir = desired_dir * 0.0

    # damping from current velocity (agent_vel is absolute velocity)
    force = desired_dir * max_force - (damping * agent_vel)

    # clamp
    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force / (fmag + eps) * max_force

    action = force.unsqueeze(0)  # shape (1,2)
    return action

import torch

def agent1_policy(obs):
    """
    Coordinating Anticipatory Playmaker / Support

    This policy implements the same high-level roles as the partner (attacker vs support/defend)
    but explicitly infers the partner's likely role from teammate-ball distances and
    yields a complementary behavior: when the teammate (partner) is clearly closer to the ball
    we take the support/shadowing role; when we are clearly closer we take the attacking lead.
    In ambiguous cases we prefer support to avoid both agents charging the ball at once.

    The observation parsing mirrors the partner's observation layout.
    Returns a torch.Tensor of shape (1,2).
    """

    eps = 1e-6

    x = obs.flatten()
    L = x.shape[0]

    # expected trailing sizes (based on observation_base construction)
    team_len = 8
    adv_len = 16  # two adversaries * 8
    vel_len = 2
    pos_len = 2
    tail_len = team_len + adv_len + vel_len + pos_len

    if L <= tail_len + 12:
        # fallback: do nothing
        return torch.zeros((1, 2), dtype=x.dtype, device=x.device)

    main = x[: (L - tail_len)]
    pos_block = x[(L - tail_len) : (L - tail_len + pos_len)]
    vel_block = x[(L - tail_len + pos_len) : (L - tail_len + pos_len + vel_len)]
    adv_block = x[(L - tail_len + pos_len + vel_len) : (L - team_len)]
    team_block = x[-team_len:]

    # parse main block indices (consistent with partner parsing)
    # 0:2 agent_force
    # 2:4 agent_pos - ball_pos
    # 4:6 agent_vel - ball_vel
    # 6:8 ball_pos - goal_pos
    # 8:10 ball_vel
    # 10:12 ball_force
    agent_force = main[0:2]
    a_pos_minus_ball = main[2:4]
    a_vel_minus_ball_vel = main[4:6]
    ball_minus_goal = main[6:8]
    ball_vel = main[8:10]
    ball_force = main[10:12]

    agent_pos_minus_goal = pos_block[0:2]
    agent_vel = vel_block[0:2]

    # teammate block: [agent_pos - teammate_pos, agent_vel - teammate_vel, teammate_vel, teammate_force]
    teammate_rel_pos = team_block[0:2]  # agent_pos - teammate_pos
    teammate_vel_rel = team_block[2:4]
    teammate_vel = team_block[4:6]
    teammate_force = team_block[6:8]

    # adversary blocks parsing
    adv_rel_list = []
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
            return v * 0.0
        return v / (n + eps)

    def length(v):
        return torch.norm(v)

    # Convert to agent-centered coordinates (agent at origin)
    r_ball = -a_pos_minus_ball  # ball_pos - agent_pos
    r_goal = -agent_pos_minus_goal  # goal_pos - agent_pos
    r_teammate = -teammate_rel_pos
    r_advs = [ -r for r in adv_rel_list ]  # adversary_pos - agent_pos

    # Distances
    d_ego_ball = length(a_pos_minus_ball)  # distance agent -> ball
    # teammate_pos_minus_ball = teammate_pos - ball_pos = r_teammate - r_ball
    teammate_pos_minus_ball = r_teammate - r_ball
    d_tm_ball = length(teammate_pos_minus_ball)

    # find nearest adversary to goal (threat to our goal)
    adv_dist_to_goal = []
    for r_adv in r_advs:
        adv_to_goal = r_adv - r_goal  # adversary_pos - goal_pos
        adv_dist_to_goal.append(length(adv_to_goal))
    adv_dist_to_goal = torch.stack(adv_dist_to_goal)
    nearest_adv_idx = int(torch.argmin(adv_dist_to_goal).item())

    # constants (kept aligned with partner where reasonable)
    max_force = 1.0
    attack_lead = 0.9
    block_offset = 0.6
    defend_distance = 0.8
    role_margin = 0.2
    jitter = 0.05
    damping = 0.5
    field_threat_threshold = 1.5

    target_rel = torch.zeros(2, dtype=x.dtype, device=x.device)

    # Defensive override: adversary close to our goal
    if adv_dist_to_goal[nearest_adv_idx] < field_threat_threshold:
        r_adv = r_advs[nearest_adv_idx]
        direction = vec_norm(r_adv - r_goal)
        target_rel = r_goal + direction * defend_distance
    else:
        # infer teammate (partner) role and pick complementary action
        # If we are clearly closer than teammate -> attacker
        # If teammate clearly closer -> support
        # If ambiguous -> prefer support to avoid both charging
        we_attack = (d_ego_ball + role_margin) < d_tm_ball
        teammate_attacks = (d_tm_ball + role_margin) < d_ego_ball

        if we_attack and not teammate_attacks:
            # Attacker role
            opp_dir = vec_norm(ball_minus_goal)
            target_rel = r_ball + opp_dir * attack_lead
        else:
            # Support / shadow role
            # find adversary nearest to ball (most dangerous)
            adv_dist_to_ball = []
            for r_adv in r_advs:
                adv_dist_to_ball.append(length(r_adv - r_ball))
            adv_dist_to_ball = torch.stack(adv_dist_to_ball)
            adv_to_ball_idx = int(torch.argmin(adv_dist_to_ball).item())
            r_danger = r_advs[adv_to_ball_idx]

            midpoint = 0.5 * (r_ball + r_teammate)
            cut_dir = vec_norm(r_danger - midpoint)
            target_rel = midpoint + cut_dir * block_offset

    # Add small perpendicular jitter to avoid predictability
    to_target = target_rel
    if torch.norm(to_target) < eps:
        perp = torch.tensor([0.0, 0.0], dtype=x.dtype, device=x.device)
    else:
        perp = torch.tensor([-to_target[1], to_target[0]], dtype=x.dtype, device=x.device)
        perp = vec_norm(perp) * jitter

    desired = to_target + perp
    n = torch.norm(desired)
    if n.item() > eps:
        desired_dir = desired / (n + eps)
    else:
        desired_dir = desired * 0.0

    # damping from current velocity
    force = desired_dir * max_force - (damping * agent_vel)

    # clamp
    fmag = torch.norm(force)
    if fmag.item() > max_force:
        force = force / (fmag + eps) * max_force

    action = force.unsqueeze(0)
    return action