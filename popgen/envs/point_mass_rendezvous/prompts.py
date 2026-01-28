ENVIRONMENT_DESCRIPTION = """Two agents must coordinate to navigate to the same landmark simultaneously.
There are 4 landmarks available, representing different potential solutions.
4 landmarks are located in a 2D space at fixed positions.
Each landmark is represented by its (x, y) coordinates.
The agents must agree on which landmark to go to without explicit communication.

There are two agents: Agent 0 (Red) and Agent 1 (Blue).

When you want to describe agent behaviors as code, these are the accessible variables:
- `ego_agent_position`: A tuple (x, y) representing the position of ego agent.
- `other_agent_position`: A tuple (x, y) representing the position of other agent.
- `landmark_positions`: A list of tuples [(x, y), ...] representing the positions of the landmarks.
- `ego_agent_velocity`: A tuple (vx, vy) representing the velocity of ego agent."""

ACTION_FORMAT = """The action should be a torch tensor of shape (1, 2) representing the force applied to the agent.
For example, to move the agent to the right with a force of 1.0 and upwards with a force of 0.5, you would return `torch.tensor([[1.0, 0.5]])`"""

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

EGO_VISUAL_DESCRIPTION = """The landmarks are represented as green circles located at fixed positions in a 2D space.
The ego agent is represented as the red point mass.
The other blue point mass is the teammate agent."""

REWARD_FORMAT = """
def reward_fn(self, agent: Agent):

    reward = torch.zeros(self.world.batch_dim, device=self.world.device)

    # You are designing reward for 'agent' in 'world'
    # You can access to list of agents and landmarks via world.agents and world.landmarks
    agents = self.world.agents
    landmarks = self.world.landmarks

    # You can access to agent's state via agent.state
    # agent.state.pos gives you the position of the agent in torch tensor shape (batch_dim, 2)
    # agent.state.vel gives you the velocity of the agent in torch tensor shape (batch_dim, 2)

    # You can access to landmark's state via landmark.state
    # landmark.state.pos gives you the position of the landmark in torch tensor shape (batch_dim, 2)

    # Example reward give +1 if both agents are on the same landmark
    for landmark in landmarks:
        # Check if all agents are on this landmark
        agents_on_landmark = torch.stack(
            [
                torch.linalg.vector_norm(
                    a.state.pos - landmark.state.pos, dim=1
                ) < a.shape.radius + landmark.shape.radius # shape.radius gives you the radius of the agent/landmark in scalar
                for a in agents
            ],
            dim=1,
        ).all(dim=1)
        reward[agents_on_landmark] += 1.0

    return reward
"""
