import torch

def agent0_policy(obs):
    try:
        o = obs
        if not isinstance(o, torch.Tensor):
            o = torch.tensor(obs, dtype=torch.float32)

        # Ensure 1D vector
        if o.dim() > 1:
            o = o.view(-1)

        def safe_slice(t, a, b):
            if isinstance(t, torch.Tensor) and t.numel() >= b:
                return t[a:b]
            elif isinstance(t, torch.Tensor) and t.numel() > a:
                return t[a:t.numel()]
            else:
                return torch.tensor([], dtype=torch.float32)

        def first_valid_slice(pairs):
            for (a, b) in pairs:
                s = safe_slice(o, a, b)
                if s.numel() >= 2:
                    return s
            return torch.tensor([], dtype=torch.float32)

        teammate_diff = first_valid_slice([(32, 34), (34, 36)])
        adv1_diff = first_valid_slice([(16, 18), (18, 20)])
        adv2_diff = first_valid_slice([(24, 26), (26, 28)])
        pos_to_goal = first_valid_slice([(12, 14), (14, 16)])

        action = torch.zeros(2, dtype=torch.float32)

        # Pass-oriented positioning: move towards teammate and create a lane
        if teammate_diff.numel() >= 2:
            d = torch.norm(teammate_diff)
            if d > 1e-5:
                action += (-teammate_diff) / d * 0.45
                dx = teammate_diff[0]
                dy = teammate_diff[1]
                perp = torch.stack([-dy, dx], dim=0)
                if perp.numel() >= 2:
                    d_perp = torch.norm(perp)
                    if d_perp > 1e-5:
                        action += (perp / d_perp) * 0.25

        # Maintain spacing by drifting away from adversaries when close
        for adv_diff in [adv1_diff, adv2_diff]:
            if adv_diff.numel() >= 2:
                d = torch.norm(adv_diff)
                if d > 1e-5:
                    weight = max(0.0, 0.55 - d)
                    if weight > 0:
                        action += (adv_diff / d) * weight

        # Slight forward progression toward goal when possible
        if pos_to_goal.numel() >= 2:
            goal_dir = -pos_to_goal
            d_goal = torch.norm(goal_dir)
            if d_goal > 1e-5:
                action += (goal_dir / d_goal) * 0.08

        action = torch.clamp(action, -1.0, 1.0)
        return action.view(1, 2)
    except Exception:
        return torch.zeros((1, 2))

import torch

def agent1_policy(obs):
    try:
        o = obs
        if not isinstance(o, torch.Tensor):
            o = torch.tensor(obs, dtype=torch.float32)

        # Ensure 1D vector
        if o.dim() > 1:
            o = o.view(-1)

        def safe_slice(t, a, b):
            if isinstance(t, torch.Tensor) and t.numel() >= b:
                return t[a:b]
            elif isinstance(t, torch.Tensor) and t.numel() > a:
                return t[a:t.numel()]
            else:
                return torch.tensor([], dtype=torch.float32)

        def first_valid_slice(pairs):
            for (a, b) in pairs:
                s = safe_slice(o, a, b)
                if s.numel() >= 2:
                    return s
            return torch.tensor([], dtype=torch.float32)

        teammate_diff = first_valid_slice([(32, 34), (34, 36)])
        adv1_diff = first_valid_slice([(16, 18), (18, 20)])
        adv2_diff = first_valid_slice([(24, 26), (26, 28)])
        pos_to_goal = first_valid_slice([(12, 14), (14, 16)])

        action = torch.zeros(2, dtype=torch.float32)

        # Pass-oriented positioning: move towards teammate and create a lane
        if teammate_diff.numel() >= 2:
            d = torch.norm(teammate_diff)
            if d > 1e-5:
                action += (-teammate_diff) / d * 0.5
                dx = teammate_diff[0]
                dy = teammate_diff[1]
                perp = torch.stack([-dy, dx], dim=0)
                if perp.numel() >= 2:
                    d_perp = torch.norm(perp)
                    if d_perp > 1e-5:
                        action += (perp / d_perp) * 0.25

        # Maintain spacing by drifting away from adversaries when close
        for adv_diff in [adv1_diff, adv2_diff]:
            if adv_diff.numel() >= 2:
                d = torch.norm(adv_diff)
                if d > 1e-5:
                    weight = max(0.0, 0.5 - d)
                    if weight > 0:
                        action += (adv_diff / d) * weight

        # Slight forward progression toward goal when possible
        if pos_to_goal.numel() >= 2:
            goal_dir = -pos_to_goal
            d_goal = torch.norm(goal_dir)
            if d_goal > 1e-5:
                action += (goal_dir / d_goal) * 0.1

        action = torch.clamp(action, -1.0, 1.0)
        return action.view(1, 2)
    except Exception:
        return torch.zeros((1, 2))