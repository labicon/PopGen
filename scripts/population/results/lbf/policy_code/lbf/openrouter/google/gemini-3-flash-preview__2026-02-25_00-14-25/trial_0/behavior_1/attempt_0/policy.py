import torch

def policy(obs):
    # obs layout: [Food_row1, Food_col1, Food_lvl1, ..., Agent_row1, Agent_col1, Agent_lvl1, ...]
    # Assume 2 agents and let's find the number of foods
    # Observation total size = 3 * num_foods + 3 * num_agents
    num_agents = 2
    num_foods = (obs.shape[0] - (num_agents * 3)) // 3

    foods = []
    for i in range(num_foods):
        foods.append({
            'r': obs[3*i], 
            'c': obs[3*i+1], 
            'v': obs[3*i+2]
        })

    # Target agent (Teammate) is Agent 1, Ego is Agent 0
    ego_r, ego_c, ego_lvl = obs[3*num_foods], obs[3*num_foods+1], obs[3*num_foods+2]
    tea_r, tea_c, tea_lvl = obs[3*num_foods+3], obs[3*num_foods+4], obs[3*num_foods+5]

    # Identify which food the teammate is targeting (the closest one to teammate)
    target_food = None
    min_dist = 999
    for f in foods:
        if f['v'] > 0: # Check if food still exists
            dist = abs(f['r'] - tea_r) + abs(f['c'] - tea_c)
            if dist < min_dist:
                min_dist = dist
                target_food = f

    # If no food, just move toward teammate
    if target_food is None:
        tr, tc = tea_r, tea_c
    else:
        tr, tc = target_food['r'], target_food['c']

    # If ego is adjacent to target food, and teammate is also adjacent, LOAD
    if target_food is not None:
        dist_ego_food = abs(ego_r - tr) + abs(ego_c - tc)
        dist_tea_food = abs(tea_r - tr) + abs(tea_c - tc)
        if dist_ego_food == 1 and dist_tea_food == 1:
            return torch.tensor(5)
        
    # Navigation logic to get adjacent to target (tr, tc)
    # We prioritize vertical alignment then horizontal
    if ego_r < tr - 1:
        return torch.tensor(2) # SOUTH
    elif ego_r > tr + 1:
        return torch.tensor(1) # NORTH
    elif ego_c < tc - 1:
        return torch.tensor(4) # EAST
    elif ego_c > tc + 1:
        return torch.tensor(3) # WEST
    
    # If in range of target cell but not adjacent yet
    if ego_r == tr:
        if ego_c < tc: return torch.tensor(4) if abs(ego_c-tc) > 1 else torch.tensor(5) # Adjacent check handled above
        if ego_c > tc: return torch.tensor(3)
    if ego_c == tc:
        if ego_r < tr: return torch.tensor(2)
        if ego_r > tr: return torch.tensor(1)

    return torch.tensor(0)