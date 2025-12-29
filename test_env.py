import time
import torch
from vmas import make_env
from pyglet.window import key

def test_manual_control():
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
    env.reset()
    
    # Initialize rendering to create the viewer
    env.render(mode="human")
    
    # Access the viewer window from the environment
    # If env is wrapped, use env.unwrapped.viewer
    # If env is not wrapped, use env.viewer
    if hasattr(env, "unwrapped"):
        viewer = env.unwrapped.viewer
    else:
        viewer = env.viewer
    
    if viewer is None:
        print("Viewer not initialized. Make sure you have a display.")
        return

    # Dictionary to track key states
    keys = {
        "w": False, "a": False, "s": False, "d": False,
        "up": False, "down": False, "left": False, "right": False
    }
    
    # Key press handler
    def on_key_press(symbol, modifiers):
        if symbol == key.W: keys["w"] = True
        elif symbol == key.S: keys["s"] = True
        elif symbol == key.A: keys["a"] = True
        elif symbol == key.D: keys["d"] = True
        elif symbol == key.UP: keys["up"] = True
        elif symbol == key.DOWN: keys["down"] = True
        elif symbol == key.LEFT: keys["left"] = True
        elif symbol == key.RIGHT: keys["right"] = True

    # Key release handler
    def on_key_release(symbol, modifiers):
        if symbol == key.W: keys["w"] = False
        elif symbol == key.S: keys["s"] = False
        elif symbol == key.A: keys["a"] = False
        elif symbol == key.D: keys["d"] = False
        elif symbol == key.UP: keys["up"] = False
        elif symbol == key.DOWN: keys["down"] = False
        elif symbol == key.LEFT: keys["left"] = False
        elif symbol == key.RIGHT: keys["right"] = False

    # Attach handlers to the viewer window
    viewer.window.push_handlers(on_key_press, on_key_release)
    
    print("Controls:")
    print("Agent 0: W, A, S, D")
    print("Agent 1: Arrows")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Construct actions
            # Action shape: (num_envs, action_dim)
            # For point mass, action is force (2D)
            
            # Agent 0 actions (WASD)
            action_0 = torch.zeros(1, 2)
            if keys["w"]: action_0[0, 1] += 1.0
            if keys["s"]: action_0[0, 1] -= 1.0
            if keys["a"]: action_0[0, 0] -= 1.0
            if keys["d"]: action_0[0, 0] += 1.0
            
            # Agent 1 actions (Arrows)
            action_1 = torch.zeros(1, 2)
            if keys["up"]: action_1[0, 1] += 1.0
            if keys["down"]: action_1[0, 1] -= 1.0
            if keys["left"]: action_1[0, 0] -= 1.0
            if keys["right"]: action_1[0, 0] += 1.0
            
            actions = [action_0, action_1]
            
            # Step environment
            obs, rews, dones, info = env.step(actions)
            
            # Render the environment
            env.render(mode="human")
            
            # Check for done
            if dones.any():
                print("Goal reached! Resetting...")
                env.reset()
            
            # Control frame rate
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        env.close()

if __name__ == "__main__":
    test_manual_control()
