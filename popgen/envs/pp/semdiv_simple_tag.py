"""SEMDIV-style predator-prey VMAS scenario.

Extends VMAS MPE simple_tag with two prey classes:
- stags: require multiple simultaneous predator collisions to count as capture
- rabbits: require fewer predator collisions (default one)
"""

from __future__ import annotations

import math

import torch
from vmas import render_interactively
from vmas.scenarios.mpe.simple_tag import Scenario as BaseSimpleTagScenario
from vmas.simulator.core import Agent
from vmas.simulator.utils import Color


class Scenario(BaseSimpleTagScenario):
    """Simple-tag variant supporting prey-specific capture requirements."""

    @staticmethod
    def _safe_normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True).clamp_min(eps)
        return vec / norm

    def _make_prey_action_script(self, style_index: int):
        angle = 2.0 * math.pi * ((int(style_index) % 11) / 11.0)
        style_bias = (math.cos(angle), math.sin(angle))

        def action_script(agent: Agent, world) -> None:
            predators = self.adversaries()
            peer_prey = [other for other in self.good_agents() if other is not agent]

            if predators:
                predator_pos = torch.stack(
                    [pred.state.pos - agent.state.pos for pred in predators],
                    dim=1,
                )
                pred_dist = torch.linalg.vector_norm(predator_pos, dim=-1, keepdim=True).clamp_min(1e-6)
                flee = (-(predator_pos / pred_dist**2)).sum(dim=1)
                nearest_pred_idx = pred_dist.squeeze(-1).argmin(dim=-1)
                nearest_pred = predator_pos[torch.arange(world.batch_dim, device=world.device), nearest_pred_idx]
            else:
                flee = torch.zeros(world.batch_dim, world.dim_p, device=world.device, dtype=torch.float32)
                nearest_pred = torch.zeros_like(flee)

            spread = torch.zeros_like(flee)
            if peer_prey:
                peer_pos = torch.stack([peer.state.pos - agent.state.pos for peer in peer_prey], dim=1)
                peer_dist = torch.linalg.vector_norm(peer_pos, dim=-1, keepdim=True).clamp_min(1e-6)
                spread = (-(peer_pos / peer_dist**2) * (peer_dist < 0.25)).sum(dim=1)

            landmark_term = torch.zeros_like(flee)
            if self.world.landmarks:
                landmarks = torch.stack(
                    [landmark.state.pos - agent.state.pos for landmark in self.world.landmarks],
                    dim=1,
                )
                pred_dir = self._safe_normalize(nearest_pred)
                landmark_dir = self._safe_normalize(landmarks)
                score = (landmark_dir * pred_dir.unsqueeze(1)).sum(dim=-1)
                landmark_idx = score.argmin(dim=-1)
                landmark_target = landmarks[torch.arange(world.batch_dim, device=world.device), landmark_idx]
                landmark_term = self._safe_normalize(landmark_target)

            style = torch.tensor(style_bias, device=world.device, dtype=torch.float32).unsqueeze(0).expand(world.batch_dim, -1)
            control = (
                0.75 * self._safe_normalize(flee + 1e-8)
                + 0.2 * self._safe_normalize(spread + landmark_term + 1e-8)
                + 0.05 * style
            )
            control = self._safe_normalize(control).clamp(-agent.u_range, agent.u_range)
            agent.action.u = control * agent.action.u_multiplier_tensor.unsqueeze(0).expand_as(control)

        return action_script

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_stags = int(kwargs.pop("num_stags", 2))
        num_rabbits = kwargs.pop("num_rabbits", None)
        num_good_agents = kwargs.pop("num_good_agents", None)
        self.ai_good_agents = bool(kwargs.pop("ai_good_agents", True))

        if num_good_agents is None:
            if num_rabbits is None:
                num_rabbits = 0
            num_good_agents = num_stags + int(num_rabbits)
        else:
            num_good_agents = int(num_good_agents)
            if num_rabbits is None:
                num_rabbits = num_good_agents - num_stags

        num_rabbits = int(num_rabbits)
        if num_stags < 0 or num_rabbits < 0:
            raise ValueError(
                f"num_stags and num_rabbits must be non-negative, got {num_stags=} {num_rabbits=}."
            )
        if num_stags + num_rabbits != num_good_agents:
            raise ValueError(
                f"Expected num_good_agents == num_stags + num_rabbits, got "
                f"{num_good_agents=} and {num_stags + num_rabbits=}."
            )

        self.num_stags = num_stags
        self.num_rabbits = num_rabbits
        self.stag_capture_requirement = int(kwargs.pop("stag_capture_requirement", 2))
        self.rabbit_capture_requirement = int(kwargs.pop("rabbit_capture_requirement", 1))
        self.stag_capture_reward = float(kwargs.pop("stag_capture_reward", 2.0))
        self.rabbit_capture_reward = float(kwargs.pop("rabbit_capture_reward", 0.5))
        self.stag_caught_penalty = float(kwargs.pop("stag_caught_penalty", 1.0))
        self.rabbit_caught_penalty = float(kwargs.pop("rabbit_caught_penalty", 0.5))
        self.solo_stag_attempt_penalty = float(kwargs.pop("solo_stag_attempt_penalty", 0.01))
        self.remove_caught_prey = bool(kwargs.pop("remove_caught_prey", True))
        self.stags_to_end = int(kwargs.pop("stags_to_end", 1))
        self.rabbits_to_end = int(kwargs.pop("rabbits_to_end", 2))

        world = super().make_world(
            batch_dim=batch_dim,
            device=device,
            num_good_agents=num_good_agents,
            **kwargs,
        )

        prey_agents = [agent for agent in world.agents if not agent.adversary]
        if len(prey_agents) != num_good_agents:
            raise RuntimeError(
                f"Expected {num_good_agents} prey agents, found {len(prey_agents)}."
            )

        self._prey_is_stag: list[bool] = []
        self._prey_capture_requirement: list[int] = []
        self._prey_capture_reward: list[float] = []
        self._prey_caught_penalty: list[float] = []

        for idx, prey in enumerate(prey_agents):
            is_stag = idx < self.num_stags
            self._prey_is_stag.append(is_stag)
            self._prey_capture_requirement.append(
                self.stag_capture_requirement if is_stag else self.rabbit_capture_requirement
            )
            self._prey_capture_reward.append(
                self.stag_capture_reward if is_stag else self.rabbit_capture_reward
            )
            self._prey_caught_penalty.append(
                self.stag_caught_penalty if is_stag else self.rabbit_caught_penalty
            )

            prey.color = Color.YELLOW if is_stag else Color.GREEN
            if self.ai_good_agents:
                prey._action_script = self._make_prey_action_script(style_index=idx)

        self._prey_caught = torch.zeros(
            (batch_dim, len(prey_agents)),
            device=device,
            dtype=torch.bool,
        )
        self._captured_this_step = torch.zeros_like(self._prey_caught)
        self._stag_captures = torch.zeros(batch_dim, device=device, dtype=torch.int64)
        self._rabbit_captures = torch.zeros(batch_dim, device=device, dtype=torch.int64)
        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self._team_attempt_penalty = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        return world

    def reset_world_at(self, env_index: int = None):
        super().reset_world_at(env_index=env_index)
        if not hasattr(self, "_prey_caught"):
            return
        if env_index is None:
            self._prey_caught.zero_()
            self._captured_this_step.zero_()
            self._stag_captures.zero_()
            self._rabbit_captures.zero_()
            self._done.zero_()
            self._team_attempt_penalty.zero_()
        else:
            self._prey_caught[env_index].fill_(False)
            self._captured_this_step[env_index].fill_(False)
            self._stag_captures[env_index] = 0
            self._rabbit_captures[env_index] = 0
            self._done[env_index] = False
            self._team_attempt_penalty[env_index] = 0.0
        for prey in self.good_agents():
            if prey._render is not None:
                if env_index is None:
                    prey.reset_render()
                else:
                    prey._render[env_index] = True

    def post_step(self):
        if not hasattr(self, "_prey_caught"):
            return
        for j, prey in enumerate(self.good_agents()):
            caught = self._prey_caught[:, j]
            if prey._render is None:
                prey.reset_render()
            prey._render[:] = ~caught
            if caught.any():
                prey.state.pos[caught, 0] = self.bound * 10.0
                prey.state.pos[caught, 1] = self.bound * 10.0
                prey.state.vel[caught] = 0.0

    def _compute_capture_mask(self) -> torch.Tensor:
        prey_agents = self.good_agents()
        adversaries = self.adversaries()
        if not prey_agents or not adversaries:
            return torch.zeros(
                (self.world.batch_dim, len(prey_agents)),
                device=self.world.device,
                dtype=torch.bool,
            )

        captured_cols = []
        for j, prey in enumerate(prey_agents):
            if prey.collide:
                collisions = torch.stack(
                    [self.is_collision(prey, adv) for adv in adversaries],
                    dim=-1,
                )
                n_colliding = collisions.sum(dim=-1)
            else:
                n_colliding = torch.zeros(
                    self.world.batch_dim, device=self.world.device, dtype=torch.int64
                )

            req = int(self._prey_capture_requirement[j])
            newly_captured = n_colliding >= req
            if hasattr(self, "_prey_caught"):
                newly_captured = newly_captured & (~self._prey_caught[:, j])
            captured_cols.append(newly_captured)

        return torch.stack(captured_cols, dim=-1)

    def _respawn_or_remove_captured_prey(self, captured_mask: torch.Tensor) -> None:
        prey_agents = self.good_agents()
        for j, prey in enumerate(prey_agents):
            cap = captured_mask[:, j]
            if not cap.any():
                continue
            if self.respawn_at_catch:
                new_pos = torch.zeros(
                    (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(-self.bound, self.bound)
                prey.state.pos[cap] = new_pos[cap]
                prey.state.vel[cap] = 0.0
            elif self.remove_caught_prey:
                despawn = torch.zeros(
                    (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                )
                despawn[:, 0] = self.bound * 10.0
                despawn[:, 1] = self.bound * 10.0
                prey.state.pos[cap] = despawn[cap]
                prey.state.vel[cap] = 0.0
                if prey._render is None:
                    prey.reset_render()
                prey._render[cap] = False

    def _compute_solo_stag_attempts(self) -> torch.Tensor:
        """Count per-env solo stag attempts (exactly one predator colliding with a stag)."""
        if self.num_stags <= 0:
            return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.int64)

        prey_agents = self.good_agents()
        adversaries = self.adversaries()
        if not prey_agents or not adversaries:
            return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.int64)

        attempts = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.int64)
        for j in range(min(self.num_stags, len(prey_agents))):
            prey = prey_agents[j]
            req = int(self._prey_capture_requirement[j])
            if req <= 1:
                continue
            collisions = torch.stack([self.is_collision(prey, adv) for adv in adversaries], dim=-1)
            n_colliding = collisions.sum(dim=-1)
            attempts += (n_colliding == 1).to(torch.int64)
        return attempts

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            self._captured_this_step = self._compute_capture_mask()
            solo_stag_attempts = self._compute_solo_stag_attempts()
            self._team_attempt_penalty = solo_stag_attempts.to(torch.float32) * float(
                self.solo_stag_attempt_penalty
            )

            cap_long = self._captured_this_step.to(torch.int64)
            if self.num_stags > 0:
                self._stag_captures += cap_long[:, : self.num_stags].sum(dim=-1)
            if self.num_rabbits > 0:
                self._rabbit_captures += cap_long[:, self.num_stags :].sum(dim=-1)
            self._done = (self._stag_captures >= self.stags_to_end) | (
                self._rabbit_captures >= self.rabbits_to_end
            )

            for a in self.world.agents:
                a.rew = self.adversary_reward(a) if a.adversary else self.agent_reward(a)

            adversaries = self.adversaries()
            if adversaries:
                per_adv_penalty = self._team_attempt_penalty / float(len(adversaries))
                for adv in adversaries:
                    adv.rew = adv.rew - per_adv_penalty

            self.agents_rew = torch.stack([a.rew for a in self.good_agents()], dim=-1).sum(-1)
            self.adverary_rew = torch.stack([a.rew for a in self.adversaries()], dim=-1).sum(-1)

            self._respawn_or_remove_captured_prey(self._captured_this_step)
            if not self.respawn_at_catch:
                self._prey_caught |= self._captured_this_step

        if agent.adversary:
            if self.adversaries_share_rew:
                return self.adverary_rew
            return agent.rew
        if self.agents_share_rew:
            return self.agents_rew
        return agent.rew

    def agent_reward(self, agent: Agent):
        rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
        adversaries = self.adversaries()
        if self.shape_agent_rew:
            for adv in adversaries:
                rew += 0.1 * torch.linalg.vector_norm(agent.state.pos - adv.state.pos, dim=-1)

        prey_agents = self.good_agents()
        try:
            prey_idx = prey_agents.index(agent)
        except ValueError:
            return rew

        captured = self._captured_this_step[:, prey_idx]
        penalty = float(self._prey_caught_penalty[prey_idx])
        rew[captured] -= penalty
        return rew

    def adversary_reward(self, agent: Agent):
        rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
        prey_agents = self.good_agents()
        adversaries = self.adversaries()

        if self.shape_adversary_rew and prey_agents:
            rew -= (
                0.1
                * torch.min(
                    torch.stack(
                        [torch.linalg.vector_norm(prey.state.pos - agent.state.pos, dim=-1) for prey in prey_agents],
                        dim=-1,
                    ),
                    dim=-1,
                )[0]
            )

        if not agent.collide:
            return rew

        for j, prey in enumerate(prey_agents):
            coll = self.is_collision(prey, agent)
            captured = self._captured_this_step[:, j]
            all_collisions = torch.stack([self.is_collision(prey, adv) for adv in adversaries], dim=-1)
            n_colliding = all_collisions.sum(dim=-1).to(torch.float32).clamp_min(1.0)
            reward_amt = float(self._prey_capture_reward[j])
            rew[coll & captured] += reward_amt / n_colliding[coll & captured]
        return rew

    def done(self) -> torch.Tensor:
        return self._done


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
