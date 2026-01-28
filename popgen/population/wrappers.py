import torch
import torch.nn as nn


class PolicyCodeAgentWrapper(nn.Module):
    """Takes a deterministic function from observations to actions."""

    def __init__(self, act_fn):
        super().__init__()
        self.act_fn = act_fn

    def forward(self, obs):
        """MAPPO like forward function."""
        # torch rl returns (num_envs, num_agents, obs_dim) shaped obs
        obs = obs["observation"][0]  # Extract single env, single agent
        acts = []
        for a_obs in obs:
            act = self.act_fn(a_obs)
            acts.append(act)
        act = torch.cat(acts, dim=0).unsqueeze(0)  # (1, num_agents, action_dim)
        return act
