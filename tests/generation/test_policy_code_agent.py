import torch

from popgen.agents.policy_code_agent import PolicyCodeAgent
from popgen.agents.scripted_partner import MultiPolicyWrapper
from popgen.envs.lbf.utils import ScriptedPolicy, create_env


def test_policy_code_agent_preserves_scalar_discrete_actions_for_single_obs():
    agent = PolicyCodeAgent(lambda obs: 4)

    out = agent(torch.zeros(12))

    assert isinstance(out, torch.Tensor)
    assert out.dim() == 0
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


def test_multi_policy_wrapper_preserves_scalar_discrete_actions_for_flat_agents_group():
    env = create_env(num_envs=1, continuous_actions=False, device="cpu")
    input_td = env.reset()
    ego = PolicyCodeAgent(lambda obs: torch.tensor(4, dtype=torch.long))
    teammate = ScriptedPolicy(env)
    wrapped = MultiPolicyWrapper([ego, teammate])

    out = wrapped(input_td)

    assert out["agents", "action"].shape == (2,)
    env.step(out)
