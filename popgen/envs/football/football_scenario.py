"""Custom football scenario with improved agent-to-ball and opponent-pressure rewards.

Changes from the stock VMAS football scenario:
- `reward_all_agent_to_ball`: removed `ball_moving` suppression so agents are
  incentivised to chase a loose ball rather than going passive once it rolls.
- `reward_opponent_to_ball` (new): potential-based penalty when the opponent
  team moves closer to the ball; small reward when they move away.  Factor is
  kept well below `pos_shaping_factor_ball_goal` so it does not dominate.
- `red_standoff_radius` (new, default 0.0 = off): a red agent stops applying force
  while a blue agent is marking it (i.e. is within this distance of it).  This keeps
  the red AI from barging through a marker into the ball's contact zone, which
  otherwise makes possession -- and therefore the reward -- very noisy.  It also
  gives the defensive role real value: marking an opponent neutralises it.
"""

import torch
from vmas.scenarios import football as football_scenario


class Scenario(football_scenario.Scenario):
    def init_params(self, **kwargs):
        self.pos_shaping_factor_opponent_ball = kwargs.pop(
            "pos_shaping_factor_opponent_ball", 0.1
        )
        # Optional (low, high) ranges for the red AI's strength params. When set,
        # a fresh value is sampled from the range on every episode reset instead
        # of using the fixed ai_strength / ai_decision_strength / ai_precision_strength.
        # Sampling happens once per reset_controllers call, so all parallel envs
        # share the same sampled strength until the next reset.
        self.ai_strength_range = kwargs.pop("ai_strength_range", None)
        self.ai_decision_strength_range = kwargs.pop("ai_decision_strength_range", None)
        self.ai_precision_strength_range = kwargs.pop("ai_precision_strength_range", None)
        # When > 0, a red agent applies zero force in envs where a blue agent is
        # within this distance of it (i.e. is marking it).  0.0 reproduces stock
        # VMAS behaviour.
        self.red_standoff_radius = kwargs.pop("red_standoff_radius", 0.0)
        # Dense per-step reward for holding red agents in a marked state.  0.0 = off.
        # Unlike `reward_opponent_to_ball` this is NOT potential-based, so it pays every
        # step a red agent is marked and is therefore farmable by design: an agent can
        # collect it by sitting on an opponent and ignoring the ball.  Whether that is
        # a defensive role or a degenerate one depends entirely on the weight: too high
        # and both agents mark instead of attacking.  See rl_cfg/football.yaml for the
        # calibrated value and the measurements behind it.
        self.neutralize_reward = kwargs.pop("neutralize_reward", 0.0)
        neutralize_radius = kwargs.pop("neutralize_radius", None)
        self.neutralize_radius = (
            neutralize_radius
            if neutralize_radius is not None
            else (self.red_standoff_radius if self.red_standoff_radius > 0 else 0.15)
        )
        super().init_params(**kwargs)

    def init_agents(self, world):
        super().init_agents(world)
        # `super().init_agents` binds `self.red_controller.run` as each red agent's
        # action_script, so the standoff wrapper has to be swapped in afterwards.
        if self.red_standoff_radius > 0 and self.ai_red_agents:
            for agent in self.red_agents:
                # `action_script` is a read-only property over `_action_script`.
                agent._action_script = self._red_run_with_standoff

    def _red_run_with_standoff(self, agent, world):
        """Run the red AI, then zero its force in envs where a blue agent is marking it."""
        self.red_controller.run(agent, world)

        blue_dists = torch.stack(
            [
                torch.linalg.vector_norm(a.state.pos - agent.state.pos, dim=-1)
                for a in self.blue_agents
            ],
            dim=-1,
        )
        marked = blue_dists.min(dim=-1)[0] < self.red_standoff_radius
        agent.action.u = torch.where(marked.unsqueeze(-1), 0.0, agent.action.u)

    @staticmethod
    def _sample_strength(strength_range):
        low, high = strength_range
        return low + (high - low) * torch.rand(()).item()

    def reset_controllers(self, env_index=None):
        if self.ai_red_agents and self.red_controller is not None:
            if self.ai_strength_range is not None:
                speed_strength = self._sample_strength(self.ai_strength_range)
                self.red_controller.speed_strength = speed_strength**2
                self.red_controller.dribble_speed = 0.16 + 0.16 * speed_strength
            if self.ai_decision_strength_range is not None:
                self.red_controller.decision_strength = self._sample_strength(
                    self.ai_decision_strength_range
                )
            if self.ai_precision_strength_range is not None:
                self.red_controller.precision_strength = self._sample_strength(
                    self.ai_precision_strength_range
                )
        super().reset_controllers(env_index)

    def init_ball(self, world):
        super().init_ball(world)
        ball = world.ball
        ball.pos_shaping_opponent_blue = ball.pos_rew_blue.clone()
        ball.pos_shaping_opponent_red = ball.pos_rew_blue.clone()
        ball.pos_rew_opponent_blue = ball.pos_rew_blue.clone()
        ball.pos_rew_opponent_red = ball.pos_rew_blue.clone()
        self._opponent_rew_blue = torch.zeros(world.batch_dim, device=world.device)
        self._opponent_rew_red = torch.zeros(world.batch_dim, device=world.device)
        self._neutralize_rew_blue = torch.zeros(world.batch_dim, device=world.device)

    def reset_ball(self, env_index=None):
        super().reset_ball(env_index)
        if env_index is None:
            if not self.ai_blue_agents:
                self.ball.pos_shaping_opponent_blue = (
                    self.get_closest_agent_to_ball(self.red_agents, env_index=None)
                    * self.pos_shaping_factor_opponent_ball
                )
            if not self.ai_red_agents:
                self.ball.pos_shaping_opponent_red = (
                    self.get_closest_agent_to_ball(self.blue_agents, env_index=None)
                    * self.pos_shaping_factor_opponent_ball
                )
        else:
            if not self.ai_blue_agents:
                self.ball.pos_shaping_opponent_blue[env_index] = (
                    self.get_closest_agent_to_ball(self.red_agents, env_index=env_index)
                    * self.pos_shaping_factor_opponent_ball
                )
            if not self.ai_red_agents:
                self.ball.pos_shaping_opponent_red[env_index] = (
                    self.get_closest_agent_to_ball(self.blue_agents, env_index=env_index)
                    * self.pos_shaping_factor_opponent_ball
                )

    def reward_all_agent_to_ball(self, blue: bool):
        min_dist_to_ball = self.get_closest_agent_to_ball(
            team=self.blue_agents if blue else self.red_agents, env_index=None
        )
        if blue:
            self.min_agent_dist_to_ball_blue = min_dist_to_ball
        else:
            self.min_agent_dist_to_ball_red = min_dist_to_ball
        pos_shaping = min_dist_to_ball * self.pos_shaping_factor_agent_ball

        # Only suppress when agent is already touching the ball; removed ball_moving
        # suppression so agents chase loose balls rather than going passive.
        agent_close_to_ball = min_dist_to_ball < self.distance_to_ball_trigger

        if blue:
            self.ball.pos_rew_agent_blue = torch.where(
                agent_close_to_ball,
                0.0,
                self.ball.pos_shaping_agent_blue - pos_shaping,
            )
            self.ball.pos_shaping_agent_blue = pos_shaping
            return self.ball.pos_rew_agent_blue
        else:
            self.ball.pos_rew_agent_red = torch.where(
                agent_close_to_ball,
                0.0,
                self.ball.pos_shaping_agent_red - pos_shaping,
            )
            self.ball.pos_shaping_agent_red = pos_shaping
            return self.ball.pos_rew_agent_red

    def reward_opponent_to_ball(self, blue: bool):
        opponent_team = self.red_agents if blue else self.blue_agents
        min_dist_to_ball = self.get_closest_agent_to_ball(
            team=opponent_team, env_index=None
        )
        pos_shaping = min_dist_to_ball * self.pos_shaping_factor_opponent_ball

        # curr - prev: negative (penalty) when opponent moves closer, positive when farther.
        if blue:
            self.ball.pos_rew_opponent_blue = (
                pos_shaping - self.ball.pos_shaping_opponent_blue
            )
            self.ball.pos_shaping_opponent_blue = pos_shaping
            return self.ball.pos_rew_opponent_blue
        else:
            self.ball.pos_rew_opponent_red = (
                pos_shaping - self.ball.pos_shaping_opponent_red
            )
            self.ball.pos_shaping_opponent_red = pos_shaping
            return self.ball.pos_rew_opponent_red

    def reward_neutralize_opponents(self):
        """Dense per-step reward for each red agent currently held in a marked state."""
        blue_pos = torch.stack([a.state.pos for a in self.blue_agents], dim=1)  # (B, n_blue, 2)
        red_pos = torch.stack([a.state.pos for a in self.red_agents], dim=1)  # (B, n_red, 2)
        dist = torch.linalg.vector_norm(
            blue_pos.unsqueeze(2) - red_pos.unsqueeze(1), dim=-1
        )  # (B, n_blue, n_red)
        marked = (dist.min(dim=1)[0] < self.neutralize_radius).float()  # (B, n_red)
        # Mean over red agents keeps the scale independent of n_red_agents.
        return self.neutralize_reward * marked.mean(dim=-1)

    def reward(self, agent):
        # Compute opponent penalty once per step (same cadence as other dense rewards).
        if agent is None or agent == self.world.agents[0]:
            if self.dense_reward and agent is not None:
                if not self.ai_blue_agents:
                    self._opponent_rew_blue = self.reward_opponent_to_ball(blue=True)
                if not self.ai_red_agents:
                    self._opponent_rew_red = self.reward_opponent_to_ball(blue=False)
            if self.neutralize_reward > 0 and agent is not None:
                self._neutralize_rew_blue = self.reward_neutralize_opponents()

        # super() uses MRO, so reward_all_agent_to_ball resolves to our override above.
        base = super().reward(agent)

        if agent is None:
            return base

        blue = agent in self.blue_agents
        rew = base + (self._opponent_rew_blue if blue else self._opponent_rew_red)
        if blue and self.neutralize_reward > 0:
            rew = rew + self._neutralize_rew_blue
        return rew

    def info(self, agent):
        info = super().info(agent)
        blue = agent in self.blue_agents
        info["opponent_ball_pos_rew"] = (
            self.ball.pos_rew_opponent_blue if blue else self.ball.pos_rew_opponent_red
        )
        return info
