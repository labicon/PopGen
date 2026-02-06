from pathlib import Path

from benchmarl.environments import VmasTask
from vmas.scenarios import football as football_scenario

from popgen.envs.vmas_common import rollout_vmas_env, train_vmas_env

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"


def get_state_football(env):
    state = {}

    blue_team, red_team = env.scenario.blue_agents, env.scenario.red_agents
    goal_pos = env.scenario.right_goal_pos

    def _get_agent_state(agent):
        pos = agent.state.pos[0].cpu().numpy().tolist()
        vel = agent.state.vel[0].cpu().numpy().tolist()
        action = agent.action.u[0].cpu().numpy().tolist()
        return {
            "action": [round(x, 2) for x in action],
            "pos": [round(x, 2) for x in pos],
            "vel": [round(x, 2) for x in vel],
        }

    # Ego team agents
    for agent in blue_team:
        state[agent.name] = _get_agent_state(agent)

    # Adversary team agents
    for agent in red_team:
        state[agent.name] = _get_agent_state(agent)

    # Goal position
    env_state = {}
    env_state["goal_pos"] = [round(x, 2) for x in goal_pos.cpu().numpy().tolist()]

    # Ball state
    ball_pos = env.scenario.ball.state.pos[0].cpu().numpy().tolist()
    ball_vel = env.scenario.ball.state.vel[0].cpu().numpy().tolist()
    env_state["ball_pos"] = [round(x, 2) for x in ball_pos]
    env_state["ball_vel"] = [round(x, 2) for x in ball_vel]

    state["env"] = env_state

    return state


def train_fn(reward_fn: callable, output_dir: str, **kwargs):
    """
    Trains agents using BenchMARL with a custom reward function.

    Args:
        reward_fn: Callable taking (agent, world) and returning reward tensor.
        output_dir: Directory to save results.
    """
    return train_vmas_env(reward_fn=reward_fn, output_dir=output_dir, scenario_module=football_scenario, scenario_name="football", vmas_task=VmasTask.FOOTBALL, rl_cfg_dir=RL_CFG_DIR, **kwargs)


def rollout_fn(policy, env=None, **kwargs):
    """
    Rollout the policy loaded from the checkpoint in save_dir.
    Uses the experiment's test_env to ensure correct observation structure.
    """
    # if env is not defined, load default football environment
    if env is None:
        env = VmasTask.FOOTBALL.get_from_yaml(path=str(RL_CFG_DIR / "football.yaml")).get_env_fun(
            num_envs=1,
            continuous_actions=True,
            seed=0,
            device="cpu",
        )()

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 200  # Default max steps for football

    return rollout_vmas_env(policy=policy, env=env, get_state_fn=get_state_football, **kwargs)
