import torch

def policy(obs):
    # Extract indices based on LBF observation format (8x8 grid world)
    # num_foods = (len(obs) - num_agents*3) // 3
    # Assuming 2 agents as per visual description (blue and white)
    num_agents = 2
    num_foods = (len(obs) - (num_agents * 3)) // 3

    foods = []
    for i in range(num_foods):
        foods.append({
            'r': obs[3*i].item(),
            'c': obs[3*i+1].item(),
            'level': obs[3*i+2].item()
        })

    # Ego agent is usually the first agent in its own observation slice
    offset = num_foods * 3
    ego_r = obs[offset].item()
    ego_c = obs[offset + 1].item()
    ego_level = obs[offset + 2].item()

    # Filter foods that can be collected solo
    solo_foods = [f for f in foods if 0 < f['level'] <= ego_level]

    if not solo_foods:
        return torch.tensor(0)  # NONE if no suitable food found

    # Pick the closest solo food
    target = min(solo_foods, key=lambda f: abs(f['r'] - ego_r) + abs(f['c'] - ego_c))

    dr = target['r'] - ego_r
    dc = target['c'] - ego_c
    dist = abs(dr) + abs(dc)

    # If adjacent to food, load it
    if dist == 1:
        return torch.tensor(5)
    
    # Navigation
    if dr < 0:
        return torch.tensor(1) # NORTH
    elif dr > 0:
        return torch.tensor(2) # SOUTH
    elif dc < 0:
        return torch.tensor(3) # WEST
    elif dc > 0:
        return torch.tensor(4) # EAST

    return torch.tensor(0)