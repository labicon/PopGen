ENVIRONMENT_DESCRIPTION = """You are playing a 2v2 football (soccer) game in a simulated environment.
You are in the blue team which consist of two agents. You must coordinate with a teammate to score a ball in football.
You cannot communicate with your teammate explicitly.
Opponent red team also consist of 2 agents.
Blue team should score a goal by getting the ball into the red team's goal area while preventing the red team from scoring in your goal area.

There are two agents for blue team: Agent 0 and Agent 1.

When you want to describe agent behaviors as code, these are the accessible variables:
- `ego_agent_position`: A tuple (x, y) representing the position of ego agent.
- `ego_agent_velocity`: A tuple (vx, vy) representing the velocity of ego agent.
- `teammate_position`: A tuple (x, y) representing the position of teammate agent.
- `teammate_velocity`: A tuple (vx, vy) representing the velocity of teammate agent.
- `adversary_positions`: A list of tuples [(x, y), ...] representing the positions of adversary agents.
- `adversary_velocities`: A list of tuples [(vx, vy), ...] representing the velocities of adversary agents.
- `ball_position`: A tuple (x, y) representing the position of the ball.
- `goal_positions`: A tuple (x, y) representing the goal position for the blue team.

The action should be a torch tensor of shape (1, 2) representing the force applied to the agent.
Therefore, you can only move the agent, and do not have direct access to kick the ball."""

ACTION_TYPE = "Continuous"

ACTION_FORMAT = """The action should be a torch tensor of shape (1, 2) representing the force applied to the agent.
For example, to move the agent to the right with a force of 1.0 and upwards with a force of 0.5, you would return `torch.tensor([[1.0, 0.5]])`"""

OBSERVATION_FORMAT = """
def observation(
    self,
    agent: Agent,
    blue=True,
    env_index=Ellipsis,
):
    assert agent in self.blue_agents

    my_team, other_team = (self.blue_agents, self.red_agents)
    goal_pos = self.right_goal_pos

    actual_adversary_poses = []
    actual_adversary_forces = []
    actual_adversary_vels = []
    for a in other_team: # There are two adversaries
        actual_adversary_poses.append(a.state.pos[env_index])
        actual_adversary_vels.append(a.state.vel[env_index])
        actual_adversary_forces.append(a.state.force[env_index])

    actual_teammate_poses = []
    actual_teammate_forces = []
    actual_teammate_vels = []
    for a in my_team: # There two agent in the same team. Therefore, ego agent has one teammate.
        if a != agent:
            actual_teammate_poses.append(a.state.pos[env_index])
            actual_teammate_vels.append(a.state.vel[env_index])
            actual_teammate_forces.append(a.state.force[env_index])

    obs = self.observation_base(
        agent.state.pos[env_index],
        agent.state.rot[env_index],
        agent.state.vel[env_index],
        agent.state.force[env_index],
        goal_pos=goal_pos,
        ball_pos=self.ball.state.pos[env_index],
        ball_vel=self.ball.state.vel[env_index],
        ball_force=self.ball.state.force[env_index],
        adversary_poses=actual_adversary_poses,
        adversary_forces=actual_adversary_forces,
        adversary_vels=actual_adversary_vels,
        teammate_poses=actual_teammate_poses,
        teammate_forces=actual_teammate_forces,
        teammate_vels=actual_teammate_vels,
        blue=blue,
    )
    return obs

def observation_base(
    self,
    agent_pos,
    agent_rot,
    agent_vel,
    agent_force,
    teammate_poses,
    teammate_forces,
    teammate_vels,
    adversary_poses,
    adversary_forces,
    adversary_vels,
    ball_pos,
    ball_vel,
    ball_force,
    goal_pos,
    blue: bool,
):
    # Make all inputs same batch size (this is needed when this function is called for rendering
    input = [
        agent_pos,
        agent_rot,
        agent_vel,
        agent_force,
        ball_pos,
        ball_vel,
        ball_force,
        goal_pos,
        teammate_poses,
        teammate_forces,
        teammate_vels,
        adversary_poses,
        adversary_forces,
        adversary_vels,
    ]
    for o in input:
        if isinstance(o, Tensor) and len(o.shape) > 1:
            batch_dim = o.shape[0]
            break
    for j in range(len(input)):
        if isinstance(input[j], Tensor):
            if len(input[j].shape) == 1:
                input[j] = input[j].unsqueeze(0).expand(batch_dim, *input[j].shape)
            input[j] = input[j].clone()

        else:
            o = input[j]
            for i in range(len(o)):
                if len(o[i].shape) == 1:
                    o[i] = o[i].unsqueeze(0).expand(batch_dim, *o[i].shape)
                o[i] = o[i].clone()

    (
        agent_pos,
        agent_rot,
        agent_vel,
        agent_force,
        ball_pos,
        ball_vel,
        ball_force,
        goal_pos,
        teammate_poses,
        teammate_forces,
        teammate_vels,
        adversary_poses,
        adversary_forces,
        adversary_vels,
    ) = input
    #  End rendering code

    obs = {
        "obs": [
            agent_force,
            agent_pos - ball_pos,
            agent_vel - ball_vel,
            ball_pos - goal_pos,
            ball_vel,
            ball_force,
        ],
        "pos": [agent_pos - goal_pos],
        "vel": [agent_vel],
    }
    obs["obs"].append(agent_rot)


    obs["adversaries"] = []
    for adversary_pos, adversary_force, adversary_vel in zip(
        adversary_poses, adversary_forces, adversary_vels
    ):
        obs["adversaries"].append(
            torch.cat(
                [
                    agent_pos - adversary_pos,
                    agent_vel - adversary_vel,
                    adversary_vel,
                    adversary_force,
                ],
                dim=-1,
            )
        )
    obs["adversaries"] = [
        torch.cat(obs["adversaries"], dim=-1)
    ]


    obs["teammates"] = []
    for teammate_pos, teammate_force, teammate_vel in zip(
        teammate_poses, teammate_forces, teammate_vels
    ):
        obs["teammates"].append(
            torch.cat(
                [
                    agent_pos - teammate_pos,
                    agent_vel - teammate_vel,
                    teammate_vel,
                    teammate_force,
                ],
                dim=-1,
            )
        )
    obs["teammates"] = [
        torch.cat(obs["teammates"], dim=-1)
    ]

    for key, value in obs.items():
        obs[key] = torch.cat(value, dim=-1)

    return torch.cat(list(obs.values()), dim=-1)
"""

EGO_VISUAL_DESCRIPTION = """Ego team is located in the left side of the football field, wearing blue jerseys.
Opponent team is located in the right side of the football field, wearing red jerseys.
The ego agent is represented as the blue football player with the tag #1."""

REWARD_FORMAT = """
def reward_fn(self, agent: Agent):

    reward = torch.zeros(self.world.batch_dim, device=self.world.device)

    # Sparse Reward
    over_right_line = (
        self.ball.state.pos[:, X] > self.pitch_length / 2 + self.ball_size / 2
    )
    over_left_line = (
        self.ball.state.pos[:, X] < -self.pitch_length / 2 - self.ball_size / 2
    )
    goal_mask = (self.ball.state.pos[:, Y] <= self.goal_size / 2) * (
        self.ball.state.pos[:, Y] >= -self.goal_size / 2
    )
    blue_score = over_right_line * goal_mask
    red_score = over_left_line * goal_mask
    self._sparse_reward_blue = (
        self.scoring_reward * blue_score - self.scoring_reward * red_score
    )

    self._done = blue_score | red_score

    reward = self._sparse_reward_blue + self._dense_reward_blue

    return reward
"""
