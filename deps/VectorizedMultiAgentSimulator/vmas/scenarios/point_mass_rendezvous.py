#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import math
import torch
from typing import Dict

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    """
    Point Mass Rendevous Scenario.

    The goal of this environment is for the two agents to navigate to a landmark together.
    There are M = 4 landmarks, and we consider each landmark as a solution in this environment.

    Modes:
    - PMR-C: Landmarks are distributed evenly on the circumference of a circle.
    - PMR-L: Landmarks are placed on a line.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 2)
        self.n_landmarks = kwargs.pop("n_landmarks", 4)
        self.mode = kwargs.pop("mode", "PMR-C")  # PMR-C or PMR-L
        self.landmark_radius = kwargs.pop("landmark_radius", 0.05)
        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.shared_reward = kwargs.pop("shared_reward", True)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world
        world = World(batch_dim, device, x_semidim=2, y_semidim=2)

        # Add agents
        for i in range(self.n_agents):
            if i == 0:
                agent = Agent(
                    name=f"agent_{i}",
                    collide=True,
                    shape=Sphere(radius=self.agent_radius),
                    color=Color.RED,
                )
            else:
                agent = Agent(
                    name=f"agent_{i}",
                    collide=True,
                    shape=Sphere(radius=self.agent_radius),
                    color=Color.BLUE,
                )
            world.add_agent(agent)

        # Add landmarks
        for i in range(self.n_landmarks):
            landmark = Landmark(
                name=f"landmark_{i}",
                collide=False,
                shape=Sphere(radius=self.landmark_radius),
                color=Color.GREEN,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        # Reset agents to origin with small noise
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(-0.3, 0.3),
                batch_index=env_index,
            )

        # Reset landmarks based on mode
        if self.mode == "PMR-C":
            # Distribute on circle
            radius = 1.0
            for i, landmark in enumerate(self.world.landmarks):
                angle = 2 * math.pi * i / self.n_landmarks
                pos = torch.tensor(
                    [radius * math.cos(angle), radius * math.sin(angle)],
                    device=self.world.device,
                )
                pos = pos.unsqueeze(0).expand(
                    (1 if env_index is not None else self.world.batch_dim), -1
                )
                landmark.set_pos(pos, batch_index=env_index)

        elif self.mode == "PMR-L":
            # Distribute on line
            # Should be centered on (0,1), step size 0.5
            step = 0.5
            for i, landmark in enumerate(self.world.landmarks):
                pos = torch.tensor([(-0.5 * (self.n_landmarks - 1) * step) + i * step, 1], device=self.world.device)
                pos = pos.unsqueeze(0).expand(
                    (1 if env_index is not None else self.world.batch_dim), -1
                )
                landmark.set_pos(pos, batch_index=env_index)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device)

            for landmark in self.world.landmarks:
                # Check if all agents are on this landmark
                agents_on_landmark = torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos - landmark.state.pos, dim=1
                        )
                        < a.shape.radius + landmark.shape.radius
                        for a in self.world.agents
                    ],
                    dim=1,
                ).all(dim=1)

                self.rew[agents_on_landmark] += 1.0

        return self.rew

    def observation(self, agent: Agent):
        # Agent pos, vel
        obs = [agent.state.pos, agent.state.vel]

        # Landmarks relative pos
        for landmark in self.world.landmarks:
            obs.append(landmark.state.pos - agent.state.pos)

        # Other agents relative pos
        for a in self.world.agents:
            if a != agent:
                obs.append(a.state.pos - agent.state.pos)

        return torch.cat(obs, dim=-1)

    def done(self):
        any_landmark_captured = torch.zeros(
            self.world.batch_dim, dtype=torch.bool, device=self.world.device
        )
        for landmark in self.world.landmarks:
            agents_on_landmark = torch.stack(
                [
                    torch.linalg.vector_norm(a.state.pos - landmark.state.pos, dim=1)
                    < a.shape.radius + landmark.shape.radius
                    for a in self.world.agents
                ],
                dim=1,
            ).all(dim=1)
            any_landmark_captured = any_landmark_captured | agents_on_landmark
        return any_landmark_captured

if __name__ == "__main__":
    render_interactively(
        __file__, control_two_agents=True, n_agents=2, mode="PMR-L"
    )
