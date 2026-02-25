import torch

def policy(obs):
    # Observation space structure:
    # Foods: [Food_row, Food_col, Food_level] for each food
    # Agents: [Agent_row, Agent_col, Agent_level] for each agent
    
    # Determining num_agents from the environment (defaulting 2 if unspecified, but calculated based on array size)
    # Assuming 2 agents as per standard LBF scenarios given in feedback.
    num_agents = 2
    num_foods = (len(obs) - (num_agents * 3)) // 3

    # Extract Ego Agent info (Agent 0)
    offset = num_foods * 3
    ego_r = obs[offset].item()
    ego_c = obs[offset + 1].item()
    ego_lvl = obs[offset + 2].item()

    # Extract all active foods
    foods = []
    for i in range(num_foods):
        fr, fc, flvl = obs[3*i].item(), obs[3*i+1].item(), obs[3*i+2].item()
        if flvl > 0: # Check if food still exists
            foods.append({'r': fr, 'c': fc, 'lvl': flvl})

    # Filter foods that the agent can collect by itself (Independent Scavenger)
    manageable_foods = [f for f in foods if f['lvl'] <= ego_lvl]

    # If no manageable foods found, the agent maintains its role and doesn't interfere with others
    # but for the sake of task completion in varied environments, we pick the closest reachable one.
    target_foods = manageable_foods if len(manageable_foods) > 0 else foods

    if not target_foods:
        return torch.tensor(0)

    # Find the nearest food from the target set
    target = min(target_foods, key=lambda f: abs(f['r'] - ego_r) + abs(f['c'] - ego_c))
    
    dr = target['r'] - ego_r
    dc = target['c'] - ego_c
    abs_dist = abs(dr) + abs(dc)

    # Behavior: If adjacent to food, load it
    if abs_dist == 1:
        return torch.tensor(5)

    # Navigation logic: Move towards the target food
    # We use a simple Manhattan distance movement
    if dr != 0:
        return torch.tensor(1 if dr < 0 else 2) # 1: North, 2: South
    if dc != 0:
        return torch.tensor(3 if dc < 0 else 4) # 3: West, 4: East

    return torch.tensor(0)