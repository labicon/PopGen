import torch

from popgen.agents.policy_code_agent import PolicyCodeAgent
from popgen.agents.router import MultiPolicyRouter
from popgen.envs.lbf.utils import ScriptedPolicy, create_env


def test_policy_code_agent_returns_discrete_action_for_batch_of_one():
    agent = PolicyCodeAgent(lambda _: 4)

    out = agent(torch.zeros(1, 12))

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1,)
    assert out.item() == 4


def test_policy_code_agent_preserves_scalar_discrete_actions_for_batched_obs():
    agent = PolicyCodeAgent(lambda obs: torch.tensor(4, dtype=torch.long))

    out = agent(torch.zeros(3, 12))

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3,)
    assert out.tolist() == [4, 4, 4]


def test_policy_code_agent_keeps_vector_actions_for_continuous_outputs():
    agent = PolicyCodeAgent(lambda obs: torch.tensor([0.5, -0.25]))

    out = agent(torch.zeros(2, 4))

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2)
    assert torch.allclose(out, torch.tensor([[0.5, -0.25], [0.5, -0.25]]))


def test_ego_teammate_router_routes_discrete_actions_to_grouped_env():
    # create_env applies _build_lbf_group_map → {"ego": ["agent_0"], "teammate": ["agent_1"]}
    env = create_env(num_envs=1, continuous_actions=False, device="cpu")
    input_td = env.reset()
    ego = PolicyCodeAgent(lambda obs: torch.tensor(4, dtype=torch.long))
    teammate = ScriptedPolicy(env)
    wrapped = MultiPolicyRouter(policies={"ego": [ego], "teammate": [teammate]})

    out = wrapped(input_td)

    assert out["ego", "action"].shape == (1,)
    assert out["teammate", "action"].shape == (1,)
    env.step(out)
