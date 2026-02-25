import torch

def policy(obs):
    # Observation space: food (3*num_foods) + agents (3*num_agents)
    num_agents = 2
    num_foods = (obs.shape[0] - (num_agents * 3)) // 3

    foods = []
    for i in range(num_foods):
        foods.append({
            'r': obs[3*i].item(), 
            'c': obs[3*i+1].item(), 
            'v': obs[3*i+2].item()
        })

    ego_r, ego_c, ego_lvl = obs[3*num_foods].item(), obs[3*num_foods+1].item(), obs[3*num_foods+2].item()
    tea_r, tea_c, tea_lvl = obs[3*num_foods+3].item(), obs[3*num_foods+4].item(), obs[3*num_foods+5].item()

    # Dedicated Supporter: Identify the food the teammate is currently closest to
    target_food = None
    min_dist_tea = 999
    for f in foods:
        if f['v'] > 0:
            dist = abs(f['r'] - tea_r) + abs(f['c'] - tea_c)
            if dist < min_dist_tea:
                min_dist_tea = dist
                target_food = f

    # If teammate is adjacent to food, ego must also get adjacent and LOAD
    if target_food is not None:
        fr, fc = target_food['r'], target_food['c']
        dist_ego_food = abs(ego_r - fr) + abs(ego_c - fc)
        dist_tea_food = abs(tea_r - fr) + abs(tea_c - fc)

        # If both are adjacent, synchronize the LOAD action
        if dist_ego_food == 1 and dist_tea_food <= 1:
            return torch.tensor(5)

        # Otherwise, maneuver to the closest empty neighbor of the target food
        # Candidates are up, down, left, right of food
        candidates = [(fr-1, fc), (fr+1, fc), (fr, fc-1), (fr, fc+1)]
        best_spot = None
        min_dist_to_spot = 999
        
        for (sr, sc) in candidates:
            # Ensure the spot is within grid and not occupied by teammate
            if 0 <= sr < 8 and 0 <= sc < 8 and not (sr == tea_r and sc == tea_c):
                d = abs(ego_r - sr) + abs(ego_c - sc)
                if d < min_dist_to_spot:
                    min_dist_to_spot = d
                    best_spot = (sr, sc)

        if best_spot:
            sr, sc = best_spot
            if ego_r < sr: return torch.tensor(2) # SOUTH
            if ego_r > sr: return torch.tensor(1) # NORTH
            if ego_c < sc: return torch.tensor(4) # EAST
            if ego_c > sc: return torch.tensor(3) # WEST

    # If no target food or teammate far from food, simply follow teammate closely
    if ego_r < tea_r - 1: return torch.tensor(2)
    if ego_r > tea_r + 1: return torch.tensor(1)
    if ego_c < tea_c - 1: return torch.tensor(4)
    if ego_c > tea_c + 1: return torch.tensor(3)

    return torch.tensor(0)