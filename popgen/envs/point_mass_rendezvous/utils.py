from pathlib import Path

from benchmarl.environments import VmasTask
from vmas.scenarios import point_mass_rendezvous as point_mass_rendezvous_scenario

from popgen.envs.vmas_common import rollout_vmas_env, train_vmas_env

# Path to local RL config directory
RL_CFG_DIR = Path(__file__).parent / "rl_cfg"


def get_state_point_mass_rendezvous(env):
    state = {}

    def _get_agent_state(agent):
        pos = agent.state.pos[0].cpu().numpy().tolist()
        vel = agent.state.vel[0].cpu().numpy().tolist()
        action = agent.action.u[0].cpu().numpy().tolist()
        return {
            "action": [round(x, 2) for x in action],
            "pos": [round(x, 2) for x in pos],
            "vel": [round(x, 2) for x in vel],
        }

    for i, agent in enumerate(env.world.agents):
        state[agent.name] = _get_agent_state(agent)

    for i, landmark in enumerate(env.world.landmarks):
        pos = landmark.state.pos[0].cpu().numpy().tolist()
        state[f"landmark{i}_pos"] = [round(x, 2) for x in pos]

    return state


def train_fn(reward_fn: callable, output_dir: str, **kwargs):
    """
    Trains agents using BenchMARL with a custom reward function.

    Args:
        reward_fn: Callable taking (agent, world) and returning reward tensor.
        output_dir: Directory to save results.
    """
    return train_vmas_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        scenario_module=point_mass_rendezvous_scenario,
        scenario_name="point_mass_rendezvous",
        vmas_task=VmasTask.POINT_MASS_RENDEZVOUS,
        rl_cfg_dir=RL_CFG_DIR,
        **kwargs,
    )


def rollout_fn(policy, env=None, **kwargs):
    """
    Rollout the policy loaded from the checkpoint in save_dir.
    Uses the experiment's test_env to ensure correct observation structure.
    """
    # if env is not defined, load default pm_rdv environment
    if env is None:
        env = VmasTask.POINT_MASS_RENDEZVOUS.get_from_yaml(path=str(RL_CFG_DIR / "point_mass_rendezvous.yaml")).get_env_fun(
            num_envs=1,
            continuous_actions=True,
            seed=0,
            device="cpu",
        )()

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 100  # Default max steps for point_mass_rendezvous

    return rollout_vmas_env(policy=policy, env=env, get_state_fn=get_state_point_mass_rendezvous, **kwargs)
