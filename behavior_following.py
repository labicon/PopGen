import time
import torch
from vmas import make_env
from pyglet.window import key

def test_behavior_following():
    # Create environment
    # We use 1 environment for manual control
    env = make_env(
        scenario="point_mass_rendezvous",
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        n_agents=2,
        mode="PMR-C",
    )
    
    # Reset environment
    obs = env.reset()
    
    # Initialize rendering to create the viewer
    env.render(mode="human")
    
    print("Running with policies...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Construct actions
            action_0 = agent1_policy(obs[0])
            action_1 = agent2_policy(obs[1])
            
            actions = [action_0, action_1]
            
            # Step environment
            obs, rews, dones, info = env.step(actions)
            
            # Render the environment
            env.render(mode="human")
            
            # Check for done
            if dones.any():
                print("Goal reached! Resetting...")
                obs = env.reset()
            
            # Control frame rate
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    # finally:
    #     env.close()

# def agent1_policy(obs):
#     return torch.tensor([[0.0, 0.0]])

def agent2_policy(obs):
    return torch.tensor([[0.0, 0.0]])


# Generated Policy
def agent1_policy(obs):
    # Ensure 1D tensor
    x = obs.view(-1)

    # Extract agent velocity and first landmark relative position
    vel = x[2:4] if x.numel() >= 4 else torch.zeros(2, device=x.device, dtype=x.dtype)
    rel_to_landmark0 = x[4:6] if x.numel() >= 6 else torch.zeros(2, device=x.device, dtype=x.dtype)

    # PD control towards landmark 0
    kp = 2.0
    kd = 0.5
    u = kp * rel_to_landmark0 - kd * vel

    # Limit force magnitude
    max_force = 1.0
    norm = torch.norm(u)
    if norm > max_force:
        u = u / (norm + 1e-8) * max_force

    return u.view(1, 2)


if __name__ == "__main__":
    test_behavior_following()