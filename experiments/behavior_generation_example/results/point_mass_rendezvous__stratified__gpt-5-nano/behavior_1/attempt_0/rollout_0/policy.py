def agent0_policy(obs):
    import torch

    # obs layout: [ego_pos(2), ego_vel(2), landmarks_delta(8 for 4 landmarks), other_delta(2)]
    ego_pos = obs[0:2]
    ego_vel = obs[2:4]

    deltas = []
    for i in range(4):
        deltas.append(obs[4 + 2*i : 4 + 2*i + 2])

    other_delta = obs[-2:]

    # Distances to each landmark for ego and for other
    dist_E = [torch.norm(d) for d in deltas]          # dist(ego, Li)
    dist_O = [torch.norm(other_delta - d) for d in deltas]  # dist(other, Li)

    # Choose i* that minimizes dist_E[i] + dist_O[i]
    min_i = 0
    min_score = None
    for i in range(4):
        score = dist_E[i] + dist_O[i]
        val = float(score.item())
        if (min_score is None) or (val < min_score):
            min_score = val
            min_i = i

    delta_star = deltas[min_i]  # Li - ego_pos
    dist_E_star = dist_E[min_i]  # dist(E, Li)
    dist_O_star = dist_O[min_i]  # dist(O, Li)

    # Synchrony parameters
    tau = 0.5  # tolerance in seconds

    # Estimate times to arrival
    v_E_mag = torch.norm(ego_vel)
    if v_E_mag.item() < 1e-6:
        v_E_mag = torch.tensor(1e-6)
    t_E = dist_E_star / v_E_mag  # time for Ego to arrive (approx)

    # Assume other agent speed magnitude ~ 1.0 m/s for estimation
    t_O = dist_O_star / 1.0  # time for Other to arrive (approx)

    # Convert to Python floats for comparison
    t_Ef = float(t_E.item()) if isinstance(t_E, torch.Tensor) else float(t_E)
    t_Of = float(t_O) if isinstance(t_O, float) else float(t_O.item()) if isinstance(t_O, torch.Tensor) else float(t_O)

    # Determine speed scaling based on synchrony
    if abs(t_Of - t_Ef) <= tau:
        s = 1.0
    elif t_Of < t_Ef - tau:
        # Other would arrive earlier; ego slows down
        s = 0.5
    elif t_Of > t_Ef + tau:
        # Other would arrive later; ego speeds up
        s = 1.5
    else:
        s = 1.0

    magnitude = max(0.0, min(2.0, s))  # clamp speed factor

    delta_norm = torch.norm(delta_star)
    if delta_norm.item() < 1e-6:
        action = torch.zeros((1, 2))
    else:
        action_dir = delta_star / (delta_norm + 1e-6)
        action_vec = action_dir * magnitude
        action = action_vec.view(1, 2)
        action = torch.clamp(action, -1.0, 1.0)

    return action

def agent1_policy(obs):
    import torch

    # obs layout: [agent1_pos(2), agent1_vel(2), landmarks_delta(8 for 4 landmarks), other_delta(2)]
    p1 = obs[0:2]
    v1 = obs[2:4]

    deltas = []
    for i in range(4):
        deltas.append(obs[4 + 2*i : 4 + 2*i + 2])

    other_delta = obs[-2:]  # ego_pos - agent1_pos

    dist_E = []
    dist_O = []
    for i in range(4):
        d = deltas[i]  # Li - p1
        dist_E_i = torch.norm(d - other_delta)  # |Li - p0| where p0 is ego
        dist_O_i = torch.norm(other_delta - d)  # |p0 - Li|
        dist_E.append(dist_E_i)
        dist_O.append(dist_O_i)

    # Choose i* that minimizes dist_E[i] + dist_O[i]
    min_i = 0
    min_score = None
    for i in range(4):
        score = dist_E[i] + dist_O[i]
        val = float(score.item())
        if (min_score is None) or (val < min_score):
            min_score = val
            min_i = i

    delta_star = deltas[min_i]  # Li - p1
    dist_E_star = dist_E[min_i]
    dist_O_star = dist_O[min_i]

    tau = 0.5  # tolerance in seconds

    # Estimate arrival times
    # Ego velocity is not directly observable to agent1; assume ~1.0 m/s
    t_E = dist_E_star / 1.0  # time for Ego to arrive (approx)
    t_O = dist_O_star / 1.0  # time for Other (ego) to arrive (approx)

    t_Ef = float(t_E.item()) if isinstance(t_E, torch.Tensor) else float(t_E)
    t_Of = float(t_O) if isinstance(t_O, float) else float(t_O) if isinstance(t_O, torch.Tensor) else float(t_O)

    # Determine speed scaling based on synchrony
    if abs(t_Of - t_Ef) <= tau:
        s = 1.0
    elif t_Of < t_Ef - tau:
        # Other would arrive earlier; ego slows down (from follower's perspective)
        s = 0.5
    elif t_Of > t_Ef + tau:
        # Other would arrive later; ego speeds up
        s = 1.5
    else:
        s = 1.0

    magnitude = max(0.0, min(2.0, s))  # clamp speed factor

    delta_norm = torch.norm(delta_star)
    if delta_norm.item() < 1e-6:
        action = torch.zeros((1, 2))
    else:
        action_dir = delta_star / (delta_norm + 1e-6)
        action_vec = action_dir * magnitude
        action = action_vec.view(1, 2)
        action = torch.clamp(action, -1.0, 1.0)

    return action