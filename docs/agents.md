# Agents

Agent wrappers live in `popgen/agents/`. Each agent is an `nn.Module` that maps observations to actions. The `MultiPolicyRouter` combines multiple agents for multi-agent rollouts.

## Agent Types

### PolicyCodeAgent

Wraps a deterministic function from observations to actions. Typically used for LLM-generated policy code.

```python
from popgen.agents import PolicyCodeAgent

agent = PolicyCodeAgent(act_fn=lambda obs: torch.tensor(4, dtype=torch.long))
```

- **Input**: `(batch, obs_dim)` observation tensor.
- **Output**: `(batch,)` for discrete actions or `(batch, action_dim)` for continuous.
- The `act_fn` callable is applied per-sample; return values (scalars, lists, NumPy arrays, tensors) are normalised to a uniform output tensor.

### LLMAgent

Uses an LLM (via LiteLLM) to generate actions from observations. Supports multi-step planning and trajectory history.

```python
from popgen.agents import LLMAgent

agent = LLMAgent(
    behavior="cooperative forager",
    model_config=model_config,
    environment_description="Grid-world where agents collect food.",
    action_type=MyAction,
    horizon_length=5,
)
```

- **Input**: `(batch, obs_dim)` observation tensor.
- **Output**: `(batch,)` for discrete actions or `(batch, action_dim)` for continuous.
- When `horizon_length > 1`, the LLM generates a trajectory of actions which are cached and replayed over multiple `forward()` calls, reducing API calls.
- `include_trajectory_history=True` feeds the previous trajectory back into the prompt for strategic coherence.

### ScriptedPolicy

Heuristic policies defined per-environment in `popgen/envs/<env_name>/utils.py`. There is no shared base class; each environment implements its own `ScriptedPolicy(env)` as an `nn.Module`.

```python
from popgen.envs.lbf.utils import ScriptedPolicy, create_env

env = create_env(num_envs=1)
policy = ScriptedPolicy(env)
```

- **Input**: `(batch, obs_dim)` observation tensor.
- **Output**: `(batch,)` discrete action tensor.
- Used as teammate policies during rollouts and RL training.

## Routing

### MultiPolicyRouter

Combines per-group agent policies for multi-agent rollouts over a grouped TensorDict.

The conventional groups used across this repository are `"ego"` (the agent being trained or evaluated) and `"teammate"` (partner agents), but any group name is accepted.

Each group's policy value determines how it is routed:

- **Tensor group** (`list[nn.Module]`) -- one policy per agent. Each receives a `(batch, obs_dim)` observation slice and returns a per-agent action.
- **RL group** (`TensorDictModuleBase`) -- a single TensorDict-based policy (e.g. from BenchMARL) that handles all agents in the group at once.

```python
from popgen.agents import MultiPolicyRouter

# All tensor-only (e.g. scripted/LLM agents for both groups)
router = MultiPolicyRouter(policies={
    "ego": [ego_policy],
    "teammate": [teammate_policy_1, teammate_policy_2],
})

# Mixed: RL ego + scripted teammates
router = MultiPolicyRouter(policies={
    "ego": rl_td_policy,                        # TensorDictModuleBase
    "teammate": [scripted_1, scripted_2],        # list[nn.Module]
})

output_td = router(input_td)
```

Groups present in the TensorDict but absent from `policies` are ignored. Groups present in `policies` but absent from the TensorDict are skipped.

### TeammatePolicyCallback

BenchMARL callback that wires an RL ego policy together with scripted teammate policies during training. It wraps the experiment's ego policy with a `MultiPolicyRouter` so that both collection and evaluation use the RL ego alongside tensor-only teammates.

```python
from popgen.agents import TeammatePolicyCallback

callback = TeammatePolicyCallback(
    teammate_policies=[ScriptedPolicy(env)],
    trainable_group={"ego": ["agent_0"]},
)
```
