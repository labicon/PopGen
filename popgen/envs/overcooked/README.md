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

`ENVIRONMENT_DESCRIPTION` in [utils.py](./utils.py) is written for the currently configured layout. If you change `layout_name`, update that natural-language description as well so LLM-generated policies are not given stale item positions, room size, or task geometry. That prompt is intentionally selective: it emphasizes fixed layout/workspace facts, pot state, wall adjacency, teammate delta, and ego position, and it does not enumerate every auxiliary observation field even when those fields are still present at runtime.

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

That compact `large_room` order matches the index description in [utils.py](./utils.py).

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

### Default reward (sparse + upstream shaping)

Each call to `OvercookedBenchWrapper._step` computes the per-agent reward as:

```python
reward_by_agent[i] = sparse_r_by_agent[i] + shaped_r_by_agent[i]
```

Both values come from the info dict returned by `OvercookedEnv.step(...)`.

**Sparse component (`sparse_r_by_agent`)** — soup delivery only:

| Event | Reward |
|---|---|
| Soup delivered to serving station | `+20.0` per delivery |
| All other steps | `0.0` |

**Shaped component (`shaped_r_by_agent`)** — dense progress signals built into the upstream overcooked-ai library, defined in `BASE_REW_SHAPING_PARAMS` in `overcooked_mdp.py`. These are active by default because `_MdpGenerator` calls `OvercookedGridworld.from_layout_name(...)` without overriding `rew_shaping_params`, so the library defaults apply:

| Event | Reward | Condition |
|---|---|---|
| Place ingredient in pot (`PLACEMENT_IN_POT_REW`) | `+3.0` | Per onion (or tomato) successfully added to a pot that is not yet full |
| Pick up dish (`DISH_PICKUP_REWARD`) | `+3.0` | Only when the pickup is _useful_ (a pot is cooking or ready) |
| Pick up soup from pot (`SOUP_PICKUP_REWARD`) | `+5.0` | On the step the agent takes a finished soup out of the pot |
| Distance-based signals (`DISH_DISP_DISTANCE_REW`, `POT_DISTANCE_REW`, `SOUP_DISTANCE_REW`) | `0.0` | Disabled by default |

These upstream shaped rewards are given independently to whichever agent performs the action — they are not shared between agents.

Sparse reward alone is very difficult to learn from in `large_room` because the delivery pipeline is long (pick onion → fill pot × 3 → start cooking → wait → pick dish → serve) and the agent only sees the `+20` signal at the very end. The upstream shaping above mitigates this by rewarding intermediate steps, but may not be sufficient or well-calibrated for a specific training objective.

### Custom dense reward shaping

Custom dense reward shaping is supported via `train_fn`. When a `reward_fn` is provided, it **completely replaces** `sparse_r_by_agent + shaped_r_by_agent` — the upstream shaping rewards described above are no longer applied. The function is called once per agent per environment step and must return a scalar `float`.

```python
def compute_reward(env, agent_index):
    # env          — OvercookedEnv instance
    # agent_index  — 0 (ego) or 1 (teammate)
    ...
    return reward  # float

train_fn(compute_reward, output_dir="/tmp/my_run")
```

#### Key state signals for `large_room`

| Expression | Type | Description |
|---|---|---|
| `env.state.players[agent_index]` | `PlayerState` | Ego or teammate state |
| `player.held_object` | `ObjectState \| None` | `None` when empty-handed; `.name` is `"onion"`, `"dish"`, or `"soup"` |
| `player.position` | `(x, y)` tuple | Agent's current grid position |
| `player.orientation` | `(dx, dy)` tuple | Facing direction |
| `env.mdp.get_pot_states(env.state)` | `dict` | Pot states grouped by category (see below) |
| `env.game_stats["useful_onion_pickup"][agent_index]` | `int` | Cumulative useful onion pickups this episode |
| `env.game_stats["dish_pickup"][agent_index]` | `int` | Cumulative dish pickups |
| `env.game_stats["soup_pickup"][agent_index]` | `int` | Cumulative soups picked up from pot |
| `env.game_stats["cumulative_sparse_rewards_by_agent"][agent_index]` | `float` | Cumulative deliveries × 20 |

`env.mdp.get_pot_states(env.state)` returns a dict with keys:
- `"empty"` — list of pot positions that are empty
- `"onion"` — list of positions of pots currently filling with onions
- `"cooking"` — list of positions of pots actively cooking
- `"ready"` — list of positions of pots with finished soup

In `large_room` there is exactly one pot at `(3, 0)`, so at most one entry appears across these lists.

#### Example: progress-shaping reward

This reward gives dense per-step signals for each stage of the pipeline. Reward components are sized to reflect relative progress toward delivery.

```python
def compute_reward(env, agent_index):
    reward = 0.0
    player = env.state.players[agent_index]
    held = player.held_object

    # Stage 1 – holding an onion is progress toward filling the pot
    if held is not None and held.name == "onion":
        reward += 0.1

    # Stage 2 – holding a dish means soup is (or will be) ready
    if held is not None and held.name == "dish":
        reward += 0.2

    # Stage 3 – holding soup means delivery is imminent
    if held is not None and held.name == "soup":
        reward += 0.5

    # Stage 4 – delivery (sparse bonus on top of the above)
    cumulative = env.game_stats["cumulative_sparse_rewards_by_agent"]
    if cumulative[agent_index] > 0:
        reward += 1.0

    return reward
```

#### Example: event-delta reward

A cleaner alternative counts discrete events rather than held-object state, avoiding the "stay holding onion forever" failure mode:

```python
def compute_reward(env, agent_index):
    reward = 0.0
    stats = env.game_stats

    # One-time bonuses on the step an event first occurs
    # (compare against previous step's count via a closure or global)
    onion_pickups = stats["useful_onion_pickup"][agent_index]
    dish_pickups  = stats["dish_pickup"][agent_index]
    soup_pickups  = stats["soup_pickup"][agent_index]
    deliveries    = stats["cumulative_sparse_rewards_by_agent"][agent_index] / 20.0

    # Use deltas: attach these to a stateful wrapper that stores prev counts
    # (see REWARD_API_REFERENCE for field definitions)
    return reward
```

Because `game_stats` values are cumulative totals, to reward individual events you need to compare the current count against the previous step. A common pattern is to carry previous values in a `nonlocal` closure or a simple wrapper class.

#### Design tips for `large_room`

- **Reward the critical bottleneck.** The hardest step is filling the pot to 3 onions. A small per-onion bonus (`+0.1` per useful pickup) dramatically speeds up early learning.
- **Avoid rewarding idle holding.** Giving `+0.1` every step the agent holds an onion incentivises picking up onions and never putting them down. Use event deltas or time the bonus to the moment of pot interaction instead.
- **Dish pickup timing.** Rewarding dish pickup before the pot is cooking can cause the agent to grab a dish and park near the pot indefinitely. Gate the dish bonus on `pot["is_cooking"] or pot["is_ready"]`.
- **Keep magnitudes proportional.** A delivery is worth `20.0` sparse. Dense bonuses that sum to more than `20.0` before a delivery can cause the agent to optimise the shaping signal rather than actually score.
- **Per-agent vs. shared reward.** `train_fn` calls `compute_reward(env, 0)` and `compute_reward(env, 1)` separately. You can return asymmetric rewards to specialise roles (e.g., only reward `agent_0` for onion pickups and `agent_1` for soup delivery).

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
