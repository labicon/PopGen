import torch

def agent0_policy(obs):
    device = obs.device
    dtype = obs.dtype

    # Simple phase-based decoy strategy: oscillate between forward decoy and trailing decoy
    s = obs.sum()
    forward_mode = (s > 0)

    if forward_mode:
        base = torch.tensor([0.55, 0.25], dtype=dtype, device=device)  # forward-offset decoy
    else:
        base = torch.tensor([-0.25, 0.60], dtype=dtype, device=device)  # trailing decoy

    # Small rhythmic modulation to simulate timing cues
    t = torch.sin(s * 0.7)
    mod = torch.stack([t, -t], dim=0).unsqueeze(0)  # shape (1, 2)
    action = base.view(1, 2) + 0.12 * mod

    action = torch.clamp(action, -1.0, 1.0)
    return action

import torch

def agent1_policy(obs):
    device = obs.device
    dtype = obs.dtype

    s = obs.sum()
    forward_mode = (s > 0)

    if forward_mode:
        base = torch.tensor([0.55, 0.25], dtype=dtype, device=device)  # forward decoy alignment
    else:
        base = torch.tensor([-0.25, 0.60], dtype=dtype, device=device)  # trailing decoy alignment

    t = torch.sin(s * 0.7)
    mod = torch.stack([t, -t], dim=0).unsqueeze(0)  # shape (1, 2)
    action = base.view(1, 2) + 0.12 * mod

    action = torch.clamp(action, -1.0, 1.0)
    return action