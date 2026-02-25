import torch

def policy(obs):
    # Observation space consists of num_foods * 3 followed by num_agents * 3
    # Dimensions: 8x8 grid. num_foods = 2, num_agents = 2 (standard observation layout)
    # Each entry: [row, col, level]
    num_foods = (obs.shape[0] - 2 * 3) // 3
    
    # 1. Identify the highest-level food
    target_food_idx = -1
    max_level = -1.0
    
    for i in range(num_foods):
        f_level = obs[i * 3 + 2]
        if f_level > max_level:
            max_level = f_level
            target_food_idx = i
            
    if target_food_idx == -1:
        return torch.tensor(0)

    f_r, f_c = obs[target_food_idx * 3], obs[target_food_idx * 3 + 1]
    
    # 2. Identify Ego Agent position
    ego_idx = num_foods * 3
    e_r, e_c, e_l = obs[ego_idx], obs[ego_idx + 1], obs[ego_idx + 2]

    # 3. Identify Teammate position
    tm_idx = ego_idx + 3
    tm_r, tm_c, tm_l = obs[tm_idx], obs[tm_idx + 1], obs[tm_idx + 2]

    # 4. Calculate distances
    dr = f_r - e_r
    dc = f_c - e_c
    dist = torch.abs(dr) + torch.abs(dc)

    # 5. Check if we should LOAD (only if both agents are adjacent and food exists)
    tm_dr = f_r - tm_r
    tm_dc = f_c - tm_c
    tm_dist = torch.abs(tm_dr) + torch.abs(tm_dc)

    if dist == 1 and tm_dist == 1 and max_level > 0:
        return torch.tensor(5)

    # 6. Anchoring/Navigation logic
    # Goal: Get to distance == 1 and stay there.
    if dist == 1:
        return torch.tensor(0) # Wait for teammate at the high-value food

    # Navigation: Move toward the food greedily. 
    # Simple pathfinding: reduce distance on the axis that is furthest.
    if torch.abs(dr) > torch.abs(dc):
        return torch.tensor(2 if dr > 0 else 1)
    elif torch.abs(dc) > 0:
        return torch.tensor(4 if dc > 0 else 3)
    else:
        return torch.tensor(2 if dr > 0 else 1)