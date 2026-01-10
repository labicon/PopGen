import torch

def agent0_policy(obs):
    # Parse observation
    pos = obs[0:2]
    vel = obs[2:4]
    
    delta_Li = []
    base = 4
    for i in range(4):
        delta = obs[base + i*2 : base + i*2 + 2]
        delta_Li.append(delta)
    delta_other = obs[12:14]
    
    # Distances
    dist_ego = [torch.norm(dl) for dl in delta_Li]
    dist_other_to_Li = [torch.norm(dl - delta_other) for dl in delta_Li]
    
    # Egalitarian distance average to each landmark
    d_i = [(dist_ego[i] + dist_other_to_Li[i]) / 2 for i in range(4)]
    d_tensor = torch.stack(d_i)
    i_star = int(torch.argmin(d_tensor))
    
    delta_star = delta_Li[i_star]
    dist_ego_star = dist_ego[i_star]
    dist_other_star = dist_other_to_Li[i_star]
    
    # Synchrony parameters
    v_ref = torch.clamp(torch.norm(vel), min=1e-3)
    t_star = torch.max(dist_ego_star, dist_other_star) / v_ref
    if t_star <= 0:
        t_star = torch.tensor(1.0, dtype=obs.dtype, device=obs.device)
    
    # Desired velocity magnitude toward the chosen landmark
    if dist_ego_star <= 1e-6:
        vel_desired = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    else:
        vel_target_mag = v_ref * dist_ego_star / (torch.max(dist_ego_star, dist_other_star) + 1e-6)
        vel_desired = (delta_star / dist_ego_star) * vel_target_mag
    
    # Simple PD-like control to reach the target velocity (proxy for synchronized arrival)
    a = vel_desired - vel
    gain = 2.0
    force = gain * a
    
    # Clamp force to avoid excessive values
    max_mag = 30.0
    mag = torch.norm(force)
    if mag > max_mag:
        force = force / mag * max_mag
    
    return force.view(1, 2)

import torch

def agent1_policy(obs):
    # Parse observation
    pos = obs[0:2]
    vel = obs[2:4]
    
    delta_Li = []
    base = 4
    for i in range(4):
        delta = obs[base + i*2 : base + i*2 + 2]
        delta_Li.append(delta)
    delta_other = obs[12:14]
    
    # Distances
    dist_ego = [torch.norm(dl) for dl in delta_Li]
    dist_other_to_Li = [torch.norm(dl - delta_other) for dl in delta_Li]
    
    # Egalitarian distance average to each landmark
    d_i = [(dist_ego[i] + dist_other_to_Li[i]) / 2 for i in range(4)]
    d_tensor = torch.stack(d_i)
    i_star = int(torch.argmin(d_tensor))
    
    delta_star = delta_Li[i_star]
    dist_ego_star = dist_ego[i_star]
    dist_other_star = dist_other_to_Li[i_star]
    
    # Synchrony parameters
    v_ref = torch.clamp(torch.norm(vel), min=1e-3)
    t_star = torch.max(dist_ego_star, dist_other_star) / v_ref
    if t_star <= 0:
        t_star = torch.tensor(1.0, dtype=obs.dtype, device=obs.device)
    
    # Desired velocity magnitude toward the chosen landmark
    if dist_ego_star <= 1e-6:
        vel_desired = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    else:
        vel_target_mag = v_ref * dist_ego_star / (torch.max(dist_ego_star, dist_other_star) + 1e-6)
        vel_desired = (delta_star / dist_ego_star) * vel_target_mag
    
    # PD-like control to reach the target velocity
    a = vel_desired - vel
    gain = 2.0
    force = gain * a
    
    # Clamp force to avoid excessive values
    max_mag = 30.0
    mag = torch.norm(force)
    if mag > max_mag:
        force = force / mag * max_mag
    
    return force.view(1, 2)