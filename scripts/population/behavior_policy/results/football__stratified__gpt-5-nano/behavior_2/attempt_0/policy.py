import torch

def agent0_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    Playmaker Strategist policy (robust heuristic).
    Inputs:
        obs: 1D tensor representing the environment observation. The exact layout
             is environment-specific; this policy uses simple, robust statistics
             from the observation vector to generate a 2D movement command.
    Output:
        A tensor of shape (1, 2) representing the force to apply (x, y).
    Strategy:
        - Derive a directional bias from large-scale trends in the observation vector.
        - Use the sum of a small leading window and a trailing window to generate
          two components (dx, dy) that encourage drifting toward open spaces and
          better spacing for passes.
        - Clip to [-1, 1] implicitly via tanh to keep forces modest.
    """
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    # Ensure 1D vector
    if obs.dim() != 1:
        obs = obs.view(-1)

    n = obs.numel()
    if n == 0:
        return torch.zeros((1, 2), dtype=obs.dtype, device=obs.device)

    # Use a small leading window and a trailing window to compute directional cues
    m = min(4, n)
    front = obs[:m]
    back = obs[-m:]

    sum_front = front.sum()
    sum_back = back.sum()

    # Derive two directional components via tanh to keep outputs in [-1, 1]
    dx = torch.tanh(sum_front)
    dy = torch.tanh(sum_back)

    # Add a tiny bias if the observation is very inactive to encourage exploration
    if dx.abs() < 1e-3 and dy.abs() < 1e-3:
        dx = torch.tensor(0.1, dtype=dx.dtype, device=dx.device)
        dy = torch.tensor(0.0, dtype=dy.dtype, device=dy.device)

    action = torch.stack([dx, dy], dim=0).unsqueeze(0)
    return action
}

import torch

def agent1_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    Complementary policy to the partner's Playmaker Strategist.
    Inputs:
        obs: 1D tensor representing the environment observation. The exact layout is
             environment-specific; this policy uses robust statistics from the observation
             vector to generate a 2D movement command.
    Output:
        A tensor of shape (1, 2) representing the force to apply (x, y).
    Strategy:
        - Derive directional cues using a different aggregation of the observation to
          bias movement toward open space and to support teammate passing opportunities.
        - Use a combination of the leading and trailing windows to produce dx, dy that
          differ from the Playmaker Strategist's bias, promoting diversified coordination.
        - Clip to [-1, 1] via tanh to keep forces modest; small exploration bias when very inactive.
    """
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    # Ensure 1D vector
    if obs.dim() != 1:
        obs = obs.view(-1)

    n = obs.numel()
    if n == 0:
        return torch.zeros((1, 2), dtype=obs.dtype, device=obs.device)

    # Compute directional cues using a slightly different scheme than agent0
    m = min(6, n)
    front = obs[:m]
    back = obs[-m:]

    sum_front = front.sum()
    sum_back = back.sum()

    dx = torch.tanh(sum_front - sum_back)
    dy = torch.tanh(sum_front + sum_back)

    # Gentle exploration bias when very inactive
    if dx.abs() < 1e-3 and dy.abs() < 1e-3:
        dx = torch.tensor(0.08, dtype=dx.dtype, device=dx.device)
        dy = torch.tensor(-0.04, dtype=dy.dtype, device=dy.device)

    action = torch.stack([dx, dy], dim=0).unsqueeze(0)
    return action