import torch

def policy(obs):
    # Grid bounds and map
    num_agents = 2
    num_foods = (obs.shape[0] - (num_agents * 3)) // 3

    foods = []
    for i in range(num_foods):
        fr, fc, fv = obs[3*i].item(), obs[3*i+1].item(), obs[3*i+2].item()
        if fv > 0:
            foods.append({'r': fr, 'c': fc, 'v': fv})

    # Ego agent data (Agent 0)
    ego_r = obs[3*num_foods].item()
    ego_c = obs[3*num_foods+1].item()
    ego_lvl = obs[3*num_foods+2].item()

    # Teammate data (Agent 1)
    tea_r = obs[3*num_foods+3].item()
    tea_c = obs[3*num_foods+4].item()
    tea_lvl = obs[3*num_foods+5].item()

    # Identify which food the teammate is currently targeting
    target_food = None
    min_dist_tea = 999
    for f in foods:
        dist = abs(f['r'] - tea_r) + abs(f['c'] - tea_c)
        if dist < min_dist_tea:
            min_dist_tea = dist
            target_food = f

    # 1. Coordinate loading if teammate is adjacent to its target
    if target_food is not None and min_dist_tea <= 1:
        fr, fc = target_food['r'], target_food['c']
        dist_ego_food = abs(ego_r - fr) + abs(ego_c - fc)
        
        if dist_ego_food == 1:
            return torch.tensor(5) # Ready to LOAD together
        
        # Move to any adjacent cell of that food that isn't the teammate's current cell
        adj = [(fr-1, fc), (fr+1, fc), (fr, fc-1), (fr, fc+1)]
        best_spot = None
        d_min = 999
        for (sr, sc) in adj:
            if 0 <= sr < 8 and 0 <= sc < 8 and not (sr == tea_r and sc == tea_c):
                d = abs(ego_r - sr) + abs(ego_c - sc)
                if d < d_min:
                    d_min = d
                    best_spot = (sr, sc)
        
        if best_spot:
            sr, sc = best_spot
            if ego_r < sr: return torch.tensor(2)
            if ego_r > sr: return torch.tensor(1)
            if ego_c < sc: return torch.tensor(4)
            if ego_c > sc: return torch.tensor(3)

    # 2. Dedicated Supporter: If not loading, stay in close proximity to the teammate (Shadowing)
    # We define proximity as being within 1 tile of the teammate at all times
    tr, tc = tea_r, tea_c
    if abs(ego_r - tr) + abs(ego_c - tc) > 1:
        if ego_r < tr: return torch.tensor(2) # South
        if ego_r > tr: return torch.tensor(1) # North
        if ego_c < tc: return torch.tensor(4) # East
        if ego_c > tc: return torch.tensor(3) # West

    return torch.tensor(0)