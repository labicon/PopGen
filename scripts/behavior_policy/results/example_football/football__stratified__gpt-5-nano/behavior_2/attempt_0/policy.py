import torch

def agent0_policy(obs):
    # Pass-First Deliberate Organizer policy
    try:
        o = obs
        if not isinstance(o, torch.Tensor):
            o = torch.tensor(obs, dtype=torch.float32)

        # Fallback if observation is smaller than expected
        if o.shape[0] < 40:
            # Try a simple goal-oriented movement if possible
            if o.shape[0] >= 14:
                pos_to_goal = o[12:14]
                dir_to_goal = -pos_to_goal
                d = torch.norm(dir_to_goal)
                if d > 0:
                    a = (dir_to_goal / d) * 0.2
                else:
                    a = torch.zeros(2)
                return a.view(1, 2)
            else:
                return torch.zeros((1, 2))

        # Parse meaningful slices from the observation
        # teammate: first 2 dims of teammate_diff corresponds to agent_pos - teammate_pos
        teammate_diff = o[32:34]
        # adversaries: first 2 dims of each adversary_diff correspond to agent_pos - adversary_pos
        adv1_diff = o[16:18]
        adv2_diff = o[24:26]
        # goal direction: agent_pos - goal_pos
        pos_to_goal = o[12:14]
        goal_dir = -pos_to_goal

        action = torch.zeros(2, dtype=torch.float32)

        # Pass attempt: move toward teammate (approach to reduce distance)
        d_teammate = torch.norm(teammate_diff)
        if d_teammate > 1e-5:
            action += (-teammate_diff) / d_teammate * 0.6

        # Maintain spacing: repel away from adversaries proportional to closeness
        for adv_diff in [adv1_diff, adv2_diff]:
            d = torch.norm(adv_diff)
            if d > 1e-5:
                weight = max(0.0, 0.6 - d)  # stronger when closer
                action += (adv_diff / d) * weight

        # Slow progression toward goal
        d_goal = torch.norm(goal_dir)
        if d_goal > 1e-5:
            action += (goal_dir / d_goal) * 0.1

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

        # Fallback if observation is smaller than expected
        if o.shape[0] < 40:
            if o.shape[0] >= 14:
                pos_to_goal = o[12:14]
                dir_to_goal = -pos_to_goal
                d = torch.norm(dir_to_goal)
                if d > 0:
                    a = (dir_to_goal / d) * 0.2
                else:
                    a = torch.zeros(2)
                return a.view(1, 2)
            else:
                return torch.zeros((1, 2))

        # Parse meaningful slices from the observation
        # teammate: first 2 dims of teammate_diff corresponds to agent_pos - teammate_pos
        teammate_diff = o[32:34]
        # adversaries: first 2 dims of each adversary_diff correspond to agent_pos - adversary_pos
        adv1_diff = o[16:18]
        adv2_diff = o[24:26]
        # goal direction: agent_pos - goal_pos
        pos_to_goal = o[12:14]
        goal_dir = -pos_to_goal

        action = torch.zeros(2, dtype=torch.float32)

        # Move toward teammate and create a passing lane by slight lateral offset
        d_teammate = torch.norm(teammate_diff)
        if d_teammate > 1e-5:
            action += (-teammate_diff) / d_teammate * 0.4

            dx = teammate_diff[0]
            dy = teammate_diff[1]
            perp = torch.stack([-dy, dx], dim=0)
            d_perp = torch.norm(perp)
            if d_perp > 1e-5:
                action += (perp / d_perp) * 0.2

        # Repel from adversaries when close
        for adv_diff in [adv1_diff, adv2_diff]:
            d = torch.norm(adv_diff)
            if d > 1e-5:
                weight = max(0.0, 0.5 - d)
                action += (adv_diff / d) * weight

        # Slight progression toward goal
        d_goal = torch.norm(goal_dir)
        if d_goal > 1e-5:
            action += (goal_dir / d_goal) * 0.08

        action = torch.clamp(action, -1.0, 1.0)
        return action.view(1, 2)

    except Exception:
        return torch.zeros((1, 2))