import torch

def policy(obs):
    # Observation constants based on task description
    # 1 food, 2 agents (ego and teammate)
    # Obs format: [F_r, F_c, F_l, A1_r, A1_c, A1_l, A2_r, A2_c, A2_l]
    num_foods = 1
    num_agents = 2
    
    # Reshape obs to identify components
    # Since problem says num_foods*3 and num_agents*3, we slice accordingly
    foods = obs[0:3].view(-1, 3)
    ego_agent = obs[3:6]
    teammate = obs[6:9]
    
    # 1. Identify high-level food (if multiple existed, we would find max level)
    # In this case, there is 1 food. Target: foods[0]
    target_food = foods[0]
    f_r, f_c, f_l = target_food[0], target_food[1], target_food[2]
    a_r, a_c = ego_agent[0], ego_agent[1]
    
    # If food is already collected (level 0), do nothing
    if f_l == 0:
        return torch.tensor(0)

    # 2. Check distance to food (Manhattan distance)
    dist = torch.abs(a_r - f_r) + torch.abs(a_c - f_c)
    
    # If adjacent, check if teammate is also adjacent and food is loadable
    if dist == 1:
        t_r, t_c = teammate[0], teammate[1]
        t_dist = torch.abs(t_r - f_r) + torch.abs(t_c - f_c)
        
        # If teammate is also adjacent, attempt synchronization
        if t_dist == 1:
            return torch.tensor(5) # LOAD
        else:
            return torch.tensor(0) # Wait/Anchor
            
    # 3. Pathfinding to the nearest adjacent cell
    # Target cells: (f_r-1, f_c), (f_r+1, f_c), (f_r, f_c-1), (f_r, f_c+1)
    target_r, target_c = f_r, f_c
    
    # Simple greedy movement towards the food coordinates
    if a_r < f_r - 1:
        return torch.tensor(2) # SOUTH
    elif a_r > f_r + 1:
        return torch.tensor(1) # NORTH
    elif a_c < f_c - 1:
        return torch.tensor(4) # EAST
    elif a_c > f_c + 1:
        return torch.tensor(3) # WEST
    
    # If we are aligned on one axis but not yet adjacent
    if a_r == f_r:
        return torch.tensor(4) if a_c < f_c else torch.tensor(3)
    if a_c == f_c:
        return torch.tensor(2) if a_r < f_r else torch.tensor(1)

    return torch.tensor(0)