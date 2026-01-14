def agent0_policy(obs):
    import torch

    if obs is None:
        return torch.zeros(1, 2, dtype=torch.float32)

    o = obs
    if o.dim() != 1:
        o = o.view(-1)

    if o.numel() < 40:
        return torch.zeros(1, 2, dtype=o.dtype)

    # Parse key components from observation
    ball_dir = -o[2:4]          # vector from agent to ball
    teammate_rel = o[32:34]     # agent_pos - teammate_pos
    v = -ball_dir - teammate_rel  # desired ball-to-teammate direction proxy

    # Baseline action: push toward desired direction
    vnorm = torch.norm(v)
    if vnorm > 1e-6:
        action = (v / vnorm) * 0.5
    else:
        action = torch.zeros(2, dtype=o.dtype, device=o.device)

    # Adversary gating: slightly shift arc away if adversaries are nearby
    adv = o[16:32]  # adversaries block
    adv1 = adv[0:2]
    adv2 = adv[8:10]
    for sub in [adv1, adv2]:
        if torch.norm(sub) < 1.0:
            if torch.norm(sub) > 1e-6:
                d = sub / sub.norm()
                perp = torch.tensor([-d[1], d[0]], dtype=o.dtype, device=o.device)
            else:
                perp = torch.zeros(2, dtype=o.dtype, device=o.device)
            action = action + 0.3 * perp
            break

    # Ball near red goal: avoid interference
    ball_to_goal = o[6:8]  # ball_pos - goal_pos
    if torch.norm(ball_to_goal) < 0.5:
        action = torch.zeros(2, dtype=o.dtype, device=o.device)

    # Clip to valid range
    if torch.norm(action) > 1.0:
        action = action / torch.norm(action)

    return action.view(1, 2)

def agent1_policy(obs):
    import torch

    if obs is None:
        return torch.zeros(1, 2, dtype=torch.float32)

    o = obs
    if o.dim() != 1:
        o = o.view(-1)

    if o.numel() < 40:
        return torch.zeros(1, 2, dtype=o.dtype)

    # Parse key components from observation (robust to environment specifics)
    ball_dir = -o[2:4]          # vector from agent to ball
    teammate_rel = o[32:34]     # agent_pos - teammate_pos
    v = -ball_dir - teammate_rel  # desired ball-to-teammate direction proxy

    # Baseline action: push toward desired direction with a milder intensity
    vnorm = torch.norm(v)
    if vnorm > 1e-6:
        action = (v / vnorm) * 0.3
    else:
        action = torch.zeros(2, dtype=o.dtype, device=o.device)

    # Adversary gating: slightly shift arc away if adversaries are nearby
    adv = o[16:32]  # adversaries block
    adv1 = adv[0:2]
    adv2 = adv[8:10]
    for sub in [adv1, adv2]:
        if torch.norm(sub) < 1.0:
            if torch.norm(sub) > 1e-6:
                d = sub / sub.norm()
                perp = torch.tensor([-d[1], d[0]], dtype=o.dtype, device=o.device)
            else:
                perp = torch.zeros(2, dtype=o.dtype, device=o.device)
            action = action - 0.25 * perp
            break

    # Ball near red goal: avoid interference
    ball_to_goal = o[6:8]  # ball_pos - goal_pos
    if torch.norm(ball_to_goal) < 0.5:
        action = torch.zeros(2, dtype=o.dtype, device=o.device)

    # Clip to valid range
    if torch.norm(action) > 1.0:
        action = action / torch.norm(action)

    return action.view(1, 2)