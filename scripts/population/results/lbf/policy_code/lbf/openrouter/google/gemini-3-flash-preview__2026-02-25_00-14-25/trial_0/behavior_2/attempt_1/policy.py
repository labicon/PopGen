import torch

def policy(obs):
    # Observation space structure:
    # Foods: [F_r, F_c, F_l] * num_foods
    # Agents: [A_r, A_c, A_l] * num_agents
    # For the provided LBF environment: 1 food, 2 agents
    
    num_foods = 1
    num_agents = 2
    
    # Parsing observation
    # obs shape: (9,) for 1 food and 2 agents
    f_r, f_c, f_l = obs[0], obs[1], obs[2]
    ego_r, ego_c, ego_l = obs[3], obs[4], obs[5]
    tm_r, tm_c, tm_l = obs[6], obs[7], obs[8]
    
    # If food is already collected, do nothing
    if f_l <= 0:
        return torch.tensor(0)

    # Calculate Manhattan distance to food
    dist_r = ego_r - f_r
    dist_c = ego_c - f_c
    abs_dist = torch.abs(dist_r) + torch.abs(dist_c)

    # If adjacent, check for teammate to coordinate LOAD
    if abs_dist == 1:
        tm_dist_r = tm_r - f_r
        tm_dist_c = tm_c - f_c
        tm_abs_dist = torch.abs(tm_dist_r) + torch.abs(tm_dist_c)
        
        # Teammate is also adjacent: LOAD together
        if tm_abs_dist == 1:
            return torch.tensor(5)
        else:
            # Anchor behavior: wait at the adjacent cell
            return torch.tensor(0)

    # Navigation logic: Move toward the food until adjacent
    # We prioritize getting to an adjacent cell (distance == 1)
    # If distance > 1, we move greedily towards food coordinates
    if torch.abs(dist_r) > 0 and not (torch.abs(dist_r) == 1 and dist_c == 0):
        if dist_r > 0:
            return torch.tensor(1) # Move NORTH
        else:
            return torch.tensor(2) # Move SOUTH
    
    if torch.abs(dist_c) > 0 and not (torch.abs(dist_c) == 1 and dist_r == 0):
        if dist_c > 0:
            return torch.tensor(3) # Move WEST
        else:
            return torch.tensor(4) # Move EAST

    return torch.tensor(0)