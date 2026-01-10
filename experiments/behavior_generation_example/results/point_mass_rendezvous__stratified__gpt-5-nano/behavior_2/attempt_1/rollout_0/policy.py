import torch

def agent0_policy(obs):
    # Parse observation
    # pos and vel are not strictly needed for computation beyond current velocity
    pos = obs[0:2]
    vel = obs[2:4]
    
    # Deltas to landmarks Li (Li - ego_pos)
    delta_Li = [obs[4 + i*2 : 6 + i*2] for i in range(4)]
    
    # Relative position to the other agent
    delta_other = obs[12:14]
    
    # Distances from ego to each landmark
    dist_ego = [torch.norm(dl) for dl in delta_Li]
    
    # Distances from other agent to each landmark
    dist_other_to_Li = [torch.norm(dl - delta_other) for dl in delta_Li]
    
    # Egalitarian average distance to each landmark
    d_i = [(dist_ego[i] + dist_other_to_Li[i]) / 2 for i in range(4)]
    d_tensor = torch.stack(d_i)
    i_star = int(torch.argmin(d_tensor).item())
    
    delta_star = delta_Li[i_star]
    dist_ego_star = dist_ego[i_star]
    dist_other_star = dist_other_to_Li[i_star]
    
    # Synchrony parameters: use a fixed cruising speed to enforce strict synchrony
    v_ref = torch.tensor(2.0, dtype=obs.dtype, device=obs.device)
    t_star = torch.max(dist_ego_star, dist_other_star) / (v_ref + 1e-6)
    if t_star <= 0:
        t_star = torch.tensor(1.0, dtype=obs.dtype, device=obs.device)
    
    # Desired velocity toward the chosen landmark
    if dist_ego_star <= 1e-6:
        vel_desired = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    else:
        vel_target_mag = v_ref * dist_ego_star / (torch.max(dist_ego_star, dist_other_star) + 1e-6)
        vel_desired = (delta_star / dist_ego_star) * vel_target_mag
    
    # PD-like control to reach the target velocity
    a = vel_desired - vel
    gain = 1.5
    force = gain * a
    
    # Clamp force to avoid excessive values
    max_mag = 25.0
    mag = torch.norm(force)
    if mag > max_mag:
        force = force / mag * max_mag
    
    return force.view(1, 2)

def agent1_policy(obs):
    import torch

    # Parse observation
    pos = obs[0:2]
    vel = obs[2:4]
    
    # Deltas to landmarks Li (Li - ego_pos)
    delta_Li = [obs[4 + i*2 : 6 + i*2] for i in range(4)]
    
    # Relative position to the other agent
    delta_other = obs[12:14]
    
    # Distances from ego to each landmark
    dist_ego = [torch.norm(dl) for dl in delta_Li]
    
    # Distances from other agent to each landmark
    dist_other_to_Li = [torch.norm(dl - delta_other) for dl in delta_Li]
    
    # Egalitarian average distance to each landmark
    d_i = [(dist_ego[i] + dist_other_to_Li[i]) / 2 for i in range(4)]
    d_tensor = torch.stack(d_i)
    i_star = int(torch.argmin(d_tensor).item())
    
    delta_star = delta_Li[i_star]
    dist_ego_star = dist_ego[i_star]
    dist_other_star = dist_other_to_Li[i_star]
    
    # Synchrony parameters: use a fixed cruising speed to enforce strict synchrony
    v_ref = torch.tensor(2.0, dtype=obs.dtype, device=obs.device)
    t_star = torch.max(dist_ego_star, dist_other_star) / (v_ref + 1e-6)
    if t_star <= 0:
        t_star = torch.tensor(1.0, dtype=obs.dtype, device=obs.device)
    
    # Desired velocity toward the chosen landmark
    if dist_ego_star <= 1e-6:
        vel_desired = torch.zeros(2, dtype=obs.dtype, device=obs.device)
    else:
        vel_target_mag = v_ref * dist_ego_star / (torch.max(dist_ego_star, dist_other_star) + 1e-6)
        vel_desired = (delta_star / dist_ego_star) * vel_target_mag
    
    # PD-like control to reach the target velocity
    a = vel_desired - vel
    gain = 1.5
    force = gain * a
    
    # Clamp force to avoid excessive values
    max_mag = 25.0
    mag = torch.norm(force)
    if mag > max_mag:
        force = force / mag * max_mag
    
    return force.view(1, 2)