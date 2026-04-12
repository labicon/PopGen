# Overcooked Environment

A popgen wrapper for the [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) cooperative cooking environment, integrated with BenchMARL for MARL training.

---

## Overview

Two agents share a kitchen and must cooperate to prepare and deliver soups. Each successful delivery earns a sparse reward (+20). The task requires dividing labour: one agent may fetch ingredients while the other manages the pot, then both must coordinate to serve completed soups.

---

## Public API

```python
from popgen.envs.overcooked import (
    create_env,          # instantiate the environment
    train_fn,            # train with a custom reward function
    rollout_fn,          # run a rollout with a policy
    serialize_state_fn,  # enrich rollout states with structured fields
    ScriptedPolicy,      # default heuristic teammate (GreedyHumanModel)
    ActionType,          # Pydantic model for structured action output
    ENVIRONMENT_DESCRIPTION,  # natural-language description for LLM agents
    REWARD_API_REFERENCE,     # reward function API reference string
    REWARD_EXAMPLE,           # example reward function
)
```

---

## Environment Configuration

Configuration lives in `rl_cfg/`:

| File | Purpose |
|---|---|
| `rl_cfg/overcooked.yaml` | Task config: layout and episode length |
| `rl_cfg/base_experiment.yaml` | BenchMARL experiment settings (devices, batch sizes, training length) |
| `rl_cfg/ippo.yaml` | IPPO algorithm hyperparameters |
| `rl_cfg/mlp.yaml` | MLP policy architecture |

### Task config (`rl_cfg/overcooked.yaml`)

```yaml
layout_name: large_room     # kitchen layout
horizon: 400                # max steps per episode
```

`ENVIRONMENT_DESCRIPTION` in [utils.py](/home/kang/zeroshot_coordination_dev/popgen/envs/overcooked/utils.py) is written for the currently configured layout. If you change `layout_name`, update that natural-language description as well so LLM-generated policies are not given stale item positions, room size, or task geometry. That prompt is intentionally selective: it emphasizes fixed layout/workspace facts, pot state, wall adjacency, teammate delta, and ego position, and it does not enumerate every auxiliary observation field even when those fields are still present at runtime.

**Available layouts** (from `deps/overcooked_ai/src/overcooked_ai_py/data/layouts/`):

Common layouts used in research:
- `cramped_room` — small 5×4 kitchen, good default
- `asymmetric_advantages` — agents have different optimal roles
- `coordination_ring` — requires explicit role division
- `forced_coordination` — agents must cooperate to complete any order
- `counter_circuit` — shared counter is the only staging area
- `bottleneck` — narrow passage forces sequential movement

Any `.layout` file in the layouts directory is a valid `layout_name`.

### Experiment config (`rl_cfg/base_experiment.yaml`)

Key settings to tune for production training:

```yaml
# Devices
sampling_device: "cuda"
train_device: "cuda"

# Training length — currently set to 1 iter for quick smoke tests;
# set max_n_iters: null and increase max_n_frames for real training runs
max_n_iters: 1
max_n_frames: null

# On-policy collection
on_policy_collected_frames_per_batch: 800   # frames per iteration
on_policy_n_envs_per_worker: 4              # parallel envs
on_policy_n_minibatch_iters: 20
on_policy_minibatch_size: 512
```

For a real training run, a typical starting point:
```yaml
max_n_iters: null
max_n_frames: 2_000_000
on_policy_collected_frames_per_batch: 12800
on_policy_n_envs_per_worker: 8
```

---

## Observation Space

The upstream `OvercookedEnv.step(...)` does **not** return a feature vector. It returns:

```python
(next_state, timestep_sparse_reward, done, env_info)
```

where `next_state` is an `OvercookedState` object containing:
- `next_state.players` — tuple of `PlayerState` objects
- `next_state.objects` — map from grid position to `ObjectState`
- `next_state.bonus_orders` / `next_state.all_orders`
- `next_state.timestep`

This popgen wrapper converts that raw state into the policy observation by calling `OvercookedEnv.featurize_state_mdp(next_state)`.
For most layouts, each agent receives the upstream **96-dimensional float32 feature vector**. The features are player-centric:

```
[player_i_features]  +  [other_player_features]  +  [dist_to_others]  +  [player_i_position]
```

**Player features** (46 values per player when `num_pots=2`):
- `orientation` — 4-element one-hot (N/S/E/W)
- `held_object` — 4-element one-hot in the order `(onion, soup, dish, tomato)`; all zeros means holding nothing
- Distance `(dx, dy)` to closest: onion dispenser, tomato dispenser, dish dispenser, soup, serving station, empty counter
- Closest soup ingredient counts (num onions, num tomatoes)
- For each of the 2 nearest pots: exists, is_empty, is_full, is_cooking, is_ready, num_onions, num_tomatoes, cook_time_remaining, `(dx, dy)`
- `wall` — 4-element boolean (wall present in each cardinal direction)

**Trailing fields:**
- `dist_to_other_players` — `(dx, dy)` per other player
- `player_position` — `(x, y)` absolute grid position (last 2 elements)

The obs dim is determined at wrapper construction time from an initial `featurize_state_mdp` call.

### `large_room` compact observation

The wrapper now applies a layout-specific projection for `large_room`. Instead of exposing the full 96-dim upstream vector, it returns a **38-dimensional float32 feature vector** per agent:

- keep `[0:10]` for orientation, held object, and closest onion
- drop `[10:12]` tomato-dispenser distance because `large_room` has no tomato dispenser
- keep `[12:22]` for dish, soup, serving station, and empty counter
- keep `[22:32]` for the single meaningful pot block
- drop `[32:42]` because `large_room` has no second pot
- keep `[42:46]` for wall flags
- drop `[46:92]` because the duplicated teammate feature block is unused in this wrapper
- keep `[92:96]` for teammate-relative position and ego absolute position

That compact `large_room` order matches the index description in [utils.py](/home/kang/zeroshot_coordination_dev/popgen/envs/overcooked/utils.py).

For LLM policy generation, the prompt in `utils.py` does **not** spell out all of those compact fields. In particular, it avoids highlighting the auxiliary object-delta features (`closest_onion`, `closest_dish`, `closest_soup`, `closest_serving`, `empty_counter`) because those details tended to distract generated policies. The runtime observation still contains them; the prompt simply steers the model toward using fixed map knowledge plus a smaller set of core signals.

Wall adjacency is especially important in `large_room`:
- `wall_north`, `wall_south`, `wall_east`, `wall_west` indicate whether the immediately adjacent tile in that direction is non-walkable on this step.
- A non-walkable adjacent tile can be a counter, pot, dispenser, serving station, or the map boundary.
- These flags are mainly for movement planning. If an object is directly adjacent, a policy may still intentionally choose the direction toward that object once to face it before using `INTERACT`.

### Layout examples

For `cramped_room.layout`, the wrapper still produces the full 96-dim vector per agent. For `large_room.layout`, it now produces the compact 38-dim vector described above. In both of these layouts:
- there is exactly 1 pot, so the `pot1_*` block is all zeros
- there are no tomato dispensers, so `closest_tomato_*` is `(0, 0)`
- the initial held-object bits are all zero because agents start empty-handed

Example: `cramped_room.layout` start-state observation for `agent_0`
- Layout facts: one pot at `(2, 0)`, onion dispensers at `(0, 1)` and `(4, 1)`, dish dispenser at `(1, 3)`, serving station at `(3, 3)`, players start at `(1, 2)` and `(3, 1)`
- Non-zero ego features are: `orientation=N`, `closest_onion=(-1, -1)`, `closest_dish=(0, 1)`, `closest_serving=(2, 1)`, `pot0_exists=1`, `pot0_is_empty=1`, `pot0=(1, -2)`, `wall_south=1`, `wall_west=1`
- The final 4 values are `dist_to_other=(2, -1)` and `player_position=(1, 2)`

Example: `large_room.layout` start-state observation for `agent_0`
- Layout facts: one pot at `(3, 0)`, onion dispensers at `(0, 1)` and `(6, 1)`, dish dispenser at `(1, 6)`, serving station at `(5, 6)`, players start at `(1, 5)` and `(5, 1)`
- In the compact wrapper observation, the non-zero features are: `orientation=N`, `closest_onion=(-1, -4)`, `closest_dish=(0, 1)`, `closest_serving=(4, 1)`, `pot0_exists=1`, `pot0_is_empty=1`, `pot0=(2, -5)`, `wall_south=1`, `wall_west=1`
- The final 4 compact values are `dist_to_other=(4, -4)` and `player_position=(1, 5)`
- The LLM prompt for this layout intentionally focuses on the pot/workspace coordinates, wall flags, teammate delta, and ego position rather than listing all of those auxiliary deltas explicitly.

---

## Action Space

Discrete, 6 actions:

| Index | Name | Effect |
|---|---|---|
| 0 | NORTH | Move up |
| 1 | SOUTH | Move down |
| 2 | EAST | Move right |
| 3 | WEST | Move left |
| 4 | STAY | No movement |
| 5 | INTERACT | Pick up / put down / serve (context-dependent) |

INTERACT behaviour depends on what is in front of the agent:
- Facing a dispenser → picks up an ingredient or dish
- Facing a pot with space → adds held ingredient
- Facing a pot with a ready soup and holding a dish → picks up the filled dish
- Facing a serving station while holding a filled dish → delivers (earns reward)
- Facing an empty counter → places held object; facing a counter with object → picks it up

---

## Reward

In the upstream env, soup delivery reward is tracked per agent in `mdp_infos["sparse_reward_by_agent"]`, and `OvercookedEnv.step(...)` returns their sum as `timestep_sparse_reward`. In this wrapper, that scalar team reward is written to each group entry unchanged, so a delivery produces reward `20.0` for both `ego` and `teammate` tensors at that timestep. All other timesteps return `0`.

Custom dense reward shaping is supported via `train_fn`:

```python
def my_reward(env, agent_index):
    """
    env          — the OvercookedEnv instance
    agent_index  — 0 (ego) or 1 (teammate)
    Returns a scalar float reward for this agent at this step.
    """
    player = env.state.players[agent_index]
    reward = 0.0
    if player.held_object is not None and player.held_object.name == "onion":
        reward += 0.05
    return reward

train_fn(my_reward, output_dir="/tmp/my_run")
```

See `REWARD_API_REFERENCE` (exported from this package) for the full list of accessible state fields.

---

## Agent Groups

Agents are split into two groups following the popgen ego/teammate convention:

| Group | Agent name | Role |
|---|---|---|
| `ego` | `agent_0` | The learning agent |
| `teammate` | `agent_1` | Scripted or frozen partner |

TensorDict keys follow `(group_name, key)`, e.g. `("ego", "observation")`, `("teammate", "action")`.

---

## Class Reference

| Class / function | Location | Description |
|---|---|---|
| `OvercookedBenchWrapper` | `overcooked_benchmarl.py` | Main TorchRL `EnvBase` subclass. Wraps `OvercookedEnv` and exposes per-group TensorDict interface. |
| `OvercookedTask` | `overcooked_benchmarl.py` | BenchMARL `Task` enum with a single member `OVERCOOKED`. |
| `OvercookedClass` | `overcooked_benchmarl.py` | BenchMARL `TaskClass` — constructs the wrapper from task config, implements all required spec methods. |
| `_MdpGenerator` | `overcooked_benchmarl.py` | Picklable callable that produces `OvercookedGridworld` instances (replaces the unpicklable `from_mdp` lambda). |
| `create_env` | `utils.py` | Convenience function; builds an `OvercookedBenchWrapper` from the yaml task config. |
| `train_fn` | `utils.py` | Sets up a BenchMARL IPPO experiment with reward injection and a scripted teammate. |
| `rollout_fn` | `utils.py` | Runs a single rollout, collecting states and rendered frames. |
| `ScriptedPolicy` | `utils.py` | `nn.Module` teammate using `GreedyHumanModel` (requires MLAM; cached after first use). |

### `OvercookedBenchWrapper` constructor

```python
OvercookedBenchWrapper(
    layout_name="cramped_room",  # kitchen layout name
    horizon=400,                 # episode length
    device="cpu",                # torch device
    group_map=None,              # defaults to ego/teammate split
    seed=None,
)
```

Key attributes after construction:
- `env.oc_env` — the underlying `OvercookedEnv` instance
- `env.oc_env.state` — current `OvercookedState`
- `env.oc_env.mdp` — the `OvercookedGridworld` MDP
- `env._obs_dim` — observation dimension (`38` for `large_room`, `96` for current full-observation layouts)

---

## Upstream source

The core environment lives at:

```
deps/overcooked_ai/src/overcooked_ai_py/
├── mdp/
│   ├── overcooked_env.py   # OvercookedEnv
│   ├── overcooked_mdp.py   # OvercookedGridworld, featurize_state
│   └── actions.py          # Action, Direction
├── agents/
│   └── agent.py            # GreedyHumanModel and other agents
└── data/
    ├── layouts/            # .layout files for each kitchen
    └── planners/           # cached MLAM pickle files (auto-generated)
```

### First-run note

`ScriptedPolicy` initialises `GreedyHumanModel`, which triggers `MediumLevelActionManager` computation the first time a new layout is used. This is slow (~5–10 min) but the result is cached as a pickle in `deps/overcooked_ai/src/overcooked_ai_py/data/planners/`. Subsequent runs load from cache and are fast. The `cramped_room` MLAM is already pre-cached.
