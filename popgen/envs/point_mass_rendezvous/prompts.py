ENVIRONMENT_DESCRIPTION = """
Two agents must coordinate to navigate to the same landmark simultaneously.
There are 4 landmarks available, representing different potential solutions.
4 landmarks are located in a 2D space at fixed positions.
Each landmark is represented by its (x, y) coordinates.
The agents must agree on which landmark to go to without explicit communication.

There are two agents: Agent 0 (Red) and Agent 1 (Blue).

When you want to describe agent behaviors as code, these are the accessible variables:
- `ego_agent_position`: A tuple (x, y) representing the position of ego agent.
- `other_agent_position`: A tuple (x, y) representing the position of other agent.
- `landmark_positions`: A list of tuples [(x, y), ...] representing the positions of the landmarks.
- `ego_agent_velocity`: A tuple (vx, vy) representing the velocity of ego agent.
"""

OBSERVATION_FORMAT = """
def observation(self, agent: Agent):
    # Agent pos, vel
    obs = [agent.state.pos, agent.state.vel]

    # Landmarks relative pos
    # In here, we have 4 landmarks
    for landmark in self.world.landmarks:
        obs.append(landmark.state.pos - agent.state.pos)

    # Other agents relative pos
    # In here, we have only 1 other agent
    for a in self.world.agents:
        if a != agent:
            obs.append(a.state.pos - agent.state.pos)

    return torch.cat(obs, dim=-1)
"""

EGO_VISUAL_DESCRIPTION = "The ego agent is represented as the red point mass."
