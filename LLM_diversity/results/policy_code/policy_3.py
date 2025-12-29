def agent1_policy(obs):
    d = obs.shape[0]
    agent_pos = obs[0:2]
    agent_vel = obs[2:4]

    remaining = d - 4
    if remaining < 2:
        return agent_pos.new_zeros((1, 2))

    # Assume one other agent if possible; otherwise treat as none
    L = max(int((remaining - 2) // 2), 0)
    if 4 + 2 * L + 2 <= d:
        other_present = True
    else:
        L = max(int(remaining // 2), 0)
        other_present = False

    if L == 0:
        return agent_pos.new_zeros((1, 2))

    landmarks_rel = obs[4:4 + 2 * L]
    if other_present:
        other_rel = obs[4 + 2 * L:4 + 2 * L + 2]
    else:
        other_rel = agent_pos.new_zeros(2)

    other_pos = agent_pos + other_rel
    leader = 0.5 * (agent_pos + other_pos + (agent_pos - other_pos).abs())

    best_i = 0
    best_primary = None
    best_secondary = None
    best_lex = None
    eps = 1e-12

    for i in range(L):
        rel = landmarks_rel[2 * i:2 * i + 2]
        ego_d2 = float((rel * rel).sum())
        lpos = agent_pos + rel
        diff_leader = lpos - leader
        leader_d2 = float((diff_leader * diff_leader).sum())
        lex_tuple = (float(lpos[0]), float(lpos[1]))

        if best_primary is None:
            best_primary = ego_d2
            best_secondary = leader_d2
            best_lex = lex_tuple
            best_i = i
        else:
            if ego_d2 < best_primary - eps:
                best_primary = ego_d2
                best_secondary = leader_d2
                best_lex = lex_tuple
                best_i = i
            elif abs(ego_d2 - best_primary) <= eps:
                if leader_d2 < best_secondary - eps:
                    best_primary = ego_d2
                    best_secondary = leader_d2
                    best_lex = lex_tuple
                    best_i = i
                elif abs(leader_d2 - best_secondary) <= eps:
                    if lex_tuple < best_lex:
                        best_primary = ego_d2
                        best_secondary = leader_d2
                        best_lex = lex_tuple
                        best_i = i
                    elif lex_tuple == best_lex and i < best_i:
                        best_primary = ego_d2
                        best_secondary = leader_d2
                        best_lex = lex_tuple
                        best_i = i

    target_rel = landmarks_rel[2 * best_i:2 * best_i + 2]
    kp = 1.0
    kd = 0.3
    force = kp * target_rel - kd * agent_vel

    norm = force.norm(p=2)
    max_force = 1.0
    if norm > max_force:
        force = force * (max_force / (norm + 1e-8))

    return force.view(1, 2)