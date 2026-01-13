import torch

def agent0_policy(obs: torch.Tensor) -> torch.Tensor:
    # Ensure we have a batch dimension
    if obs.dim() == 1:
        o = obs.unsqueeze(0)
    else:
        o = obs

    batch = o.shape[0]
    device = o.device
    dtype = o.dtype

    # Fallback if observation is smaller than expected
    if o.shape[1] < 12:
        action = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype) * 0.5
        return action

    # Extract likely structured features from the first 12 dims
    # Based on observation_base: 
    # [0:2] agent_force, [2:4] agent_pos - ball_pos, [4:6] agent_vel - ball_vel,
    # [6:8] ball_pos - goal_pos, [8:10] ball_vel, [10:12] ball_force
    rel_agent_ball = o[:, 2:4]      # agent_pos - ball_pos
    ball_to_goal = o[:, 6:8]        # ball_pos - goal_pos

    # Distances and directions
    dist = torch.norm(rel_agent_ball, dim=1, keepdim=True)
    dir_to_ball = -rel_agent_ball / (dist + 1e-6)

    norm_to_goal = torch.norm(ball_to_goal, dim=1, keepdim=True)
    dir_to_goal = -ball_to_goal / (norm_to_goal + 1e-6)

    # If ball is close, push toward goal to finish opportunistically
    close_mask = (dist < 0.6).float()  # shape (batch, 1)

    final_dir = dir_to_goal * close_mask + dir_to_ball * (1.0 - close_mask)

    # Magnitude: stronger when ball is far (to close distance), taper when close
    mag = 0.9 + 0.3 * close_mask  # shape (batch, 1)
    action = final_dir * mag  # shape (batch, 2)

    # Clamp to valid action range
    action = torch.clamp(action, -1.0, 1.0)

    return action[:1, :]

import torch

def agent1_policy(obs: torch.Tensor) -> torch.Tensor:
    # Ensure batch dimension
    if obs.dim() == 1:
        o = obs.unsqueeze(0)
    else:
        o = obs

    device = o.device
    dtype = o.dtype

    # Fallback if observation is smaller than expected
    if o.shape[1] < 12:
        action = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)
        return action

    # Compute direction towards the landmark (goal) using available debug features
    # Based on observation_base ordering:
    # [0:2] agent_force, [2:4] agent_pos - ball_pos, [4:6] agent_vel - ball_vel,
    # [6:8] ball_pos - goal_pos, [8:10] ball_vel, [10:12] ball_force
    rel_agent_ball = o[:, 2:4]      # agent_pos - ball_pos
    ball_to_goal = o[:, 6:8]        # ball_pos - goal_pos

    # Direction from agent to goal
    vec_to_goal = -(rel_agent_ball + ball_to_goal)
    dist = torch.norm(vec_to_goal, dim=1, keepdim=True)
    dir_to_goal = vec_to_goal / (dist + 1e-6)

    # Magnitude controlled to have a strong push towards the goal
    mag = 0.9
    action = dir_to_goal * mag

    # Clamp to valid action range
    action = torch.clamp(action, -1.0, 1.0)

    return action[:1, :]