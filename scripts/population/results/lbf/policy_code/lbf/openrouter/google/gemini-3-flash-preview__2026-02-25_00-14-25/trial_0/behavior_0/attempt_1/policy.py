import torch

def policy(obs):
    # Level-Based Foraging (8x8 grid)
    # Observation: num_foods * [r, c, L] then num_agents * [r, c, L]
    # Based on the prompt, we typically have 2 agents and 2-3 foods.
    # Total elements in obs: num_foods*3 + num_agents*3
    # Observation length is likely 12, 15, or 18 if 2 agents.
    num_agents = 2
    num_foods = (len(obs) - (num_agents * 3)) // 3

    foods = []
    for i in range(num_foods):
        r, c, lvl = obs[3*i].item(), obs[3*i+1].item(), obs[3*i+2].item()
        # Level 0 usually implies food already collected
        if lvl > 0:
            foods.append({'r': r, 'c': c, 'lvl': lvl})

    offset = num_foods * 3
    ego_r = obs[offset].item()
    ego_c = obs[offset + 1].item()
    ego_lvl = obs[offset + 2].item()

    # Filter foods the agent can collect alone
    solo_foods = [f for f in foods if f['lvl'] <= ego_lvl]

    if not solo_foods:
        return torch.tensor(0)

    # Find the nearest soloable food
    target = min(solo_foods, key=lambda f: abs(f['r'] - ego_r) + abs(f['c'] - ego_c))
    
    dr = target['r'] - ego_r
    dc = target['c'] - ego_c
    dist = abs(dr) + abs(dc)

    # If adjacent, load it
    if dist == 1:
        return torch.tensor(5)

    # Navigation: Prioritize row movement then column
    if dr != 0:
        return torch.tensor(1 if dr < 0 else 2)
    if dc != 0:
        return torch.tensor(3 if dc < 0 else 4)

    return torch.tensor(0)