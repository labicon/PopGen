import torch

def agent0_policy(obs):
    try:
        o = obs
        if not isinstance(o, torch.Tensor):
            o = torch.tensor(obs, dtype=torch.float32)

        if o.dim() == 2 and o.shape[0] == 1:
            o = o.view(-1)

        n = o.numel()

        def safe_norm(v):
            if v.numel() == 0:
                return torch.tensor(0.0, dtype=torch.float32)
            return torch.norm(v)

        def slice_range(a, b):
            if n >= b:
                return o[a:b]
            elif n > a:
                return o[a:n]
            else:
                return torch.tensor([], dtype=torch.float32)

        action = torch.zeros(2, dtype=torch.float32)

        # Teammate diff: agent_pos - teammate_pos
        teammate_diff = slice_range(34, 36)
        if teammate_diff.numel() >= 2:
            d = safe_norm(teammate_diff)
            if d > 1e-5:
                action += (-teammate_diff) / d * 0.4
                dx = teammate_diff[0]
                dy = teammate_diff[1]
                perp = torch.stack([-dy, dx], dim=0)
                if perp.numel() >= 2:
                    d_perp = safe_norm(perp)
                    if d_perp > 1e-5:
                        action += (perp / d_perp) * 0.2

        # Adversaries: use positional diffs to create spacing
        adv1_diff = slice_range(18, 20)
        adv2_diff = slice_range(26, 28)
        for adv_diff in [adv1_diff, adv2_diff]:
            if adv_diff.numel() >= 2:
                d = safe_norm(adv_diff)
                if d > 1e-5:
                    weight = max(0.0, 0.5 - d)
                    if weight > 0:
                        action += (adv_diff / d) * weight

        # Progression toward goal: agent_pos - goal_pos
        pos_to_goal = slice_range(14, 16)
        if pos_to_goal.numel() >= 2:
            goal_dir = -pos_to_goal
            d_goal = safe_norm(goal_dir)
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

        if o.dim() == 2 and o.shape[0] == 1:
            o = o.view(-1)

        n = o.numel()

        def safe_norm(v):
            if v.numel() == 0:
                return torch.tensor(0.0, dtype=torch.float32)
            return torch.norm(v)

        def slice_range(a, b):
            if n >= b:
                return o[a:b]
            elif n > a:
                return o[a:n]
            else:
                return torch.tensor([], dtype=torch.float32)

        action = torch.zeros(2, dtype=torch.float32)

        # Teammate diff: agent_pos - teammate_pos
        teammate_diff = slice_range(34, 36)
        if teammate_diff.numel() >= 2:
            d = safe_norm(teammate_diff)
            if d > 1e-5:
                # Slightly different weighting to encourage complementary positioning
                action += (-teammate_diff) / d * 0.25
                dx = teammate_diff[0]
                dy = teammate_diff[1]
                perp = torch.stack([-dy, dx], dim=0)
                if perp.numel() >= 2:
                    d_perp = safe_norm(perp)
                    if d_perp > 1e-5:
                        action += (perp / d_perp) * 0.15

        # Adversaries: use positional diffs to create spacing
        adv1_diff = slice_range(18, 20)
        adv2_diff = slice_range(26, 28)
        for adv_diff in [adv1_diff, adv2_diff]:
            if adv_diff.numel() >= 2:
                d = safe_norm(adv_diff)
                if d > 1e-5:
                    weight = max(0.0, 0.55 - d)
                    if weight > 0:
                        action += (adv_diff / d) * weight

        # Progression toward goal: agent_pos - goal_pos
        pos_to_goal = slice_range(14, 16)
        if pos_to_goal.numel() >= 2:
            goal_dir = -pos_to_goal
            d_goal = safe_norm(goal_dir)
            if d_goal > 1e-5:
                action += (goal_dir / d_goal) * 0.05

        # Small repulsion from the teammate when too close to avoid stacking
        if teammate_diff.numel() >= 2:
            d = safe_norm(teammate_diff)
            if d < 0.4 and d > 1e-5:
                rep = teammate_diff / d
                action += rep * 0.2  # push away a bit

        action = torch.clamp(action, -1.0, 1.0)
        return action.view(1, 2)
    except Exception:
        return torch.zeros((1, 2))