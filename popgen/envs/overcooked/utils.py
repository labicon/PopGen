import inspect
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from pydantic import BaseModel, Field
from tensordict import TensorDictBase
from torchrl.envs import EnvBase, TransformedEnv

from overcooked_ai_py.mdp.actions import Action, Direction

from popgen.envs.overcooked.overcooked_benchmarl import NUM_ACTIONS, OvercookedBenchWrapper, OvercookedTask
from popgen.envs.utils import apply_rollout_seed, apply_train_kwargs, load_algorithm_config, load_model_config, resolve_cfg_dir

# Suppress pygame display for headless rendering
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ---------------------------------------------------------------------------- #
#                              Environment Config                              #
# ---------------------------------------------------------------------------- #

RL_CFG_DIR = Path(__file__).parent / "rl_cfg"
_DIRECTION_NAMES = ("NORTH", "SOUTH", "EAST", "WEST")
_ACTION_NAMES = {
    Action.ACTION_TO_INDEX[Direction.NORTH]: "NORTH",
    Action.ACTION_TO_INDEX[Direction.SOUTH]: "SOUTH",
    Action.ACTION_TO_INDEX[Direction.EAST]: "EAST",
    Action.ACTION_TO_INDEX[Direction.WEST]: "WEST",
    Action.ACTION_TO_INDEX[Action.STAY]: "STAY",
    Action.ACTION_TO_INDEX[Action.INTERACT]: "INTERACT",
}
_LARGE_ROOM_LAYOUT_INFO = {
    "layout_name": "large_room",
    "room_size": [7, 7],
    "pot_locations": [[3, 0]],
    "pot_interaction_tiles": [[3, 1]],
    "onion_dispenser_locations": [[0, 1], [6, 1]],
    "dish_dispenser_locations": [[1, 6]],
    "serving_station_locations": [[5, 6]],
    "counter_locations": [[2, 0], [4, 0], [0, 5], [6, 5], [2, 6], [3, 6], [4, 6]],
    "tomato_dispenser_locations": [],
}

ENVIRONMENT_DESCRIPTION = """
Environment: Overcooked

This is a cooperative cooking environment where two agents work together in a kitchen to prepare and deliver soups.

--------------------------------------------------
Objective
--------------------------------------------------

Agents must cooperate to:
1. Pick up ingredients (onions) from dispensers.
2. Place ingredients into pots.
3. If a pot has 3 onions but is not cooking, an empty-handed agent must stand next to it, face it, and use `INTERACT` to start cooking. If holding something, place it on a nearby counter first.
4. Pick up a dish when soup will be needed soon.
5. Serve the soup by bringing the filled dish to the serving station.

Each agent may prefer different roles, but both agents can perform all actions.

--------------------------------------------------
Kitchen Layout
--------------------------------------------------

- The active layout described here is `large_room`.
- The room is a 7x7 grid.
- Coordinates use `(x, y)` with `(0, 0)` at the top-left.
- Cells can be floor (walkable), counter/wall, pot, serving station, onion dispenser, or dish dispenser.
- Each agent occupies one floor cell and cannot pass through counters or through the other agent.

`large_room` item locations and reliable interaction workspaces:
- Pot tile: `(3, 0)`; stand at `(3, 1)` and face NORTH to add onion, start cooking, or pick up soup.
- Left onion dispenser tile: `(0, 1)`; stand at `(1, 1)` and face WEST to pick up onion.
- Right onion dispenser tile: `(6, 1)`; stand at `(5, 1)` and face EAST to pick up onion.
- Dish dispenser tile: `(1, 6)`; stand at `(1, 5)` and face SOUTH to pick up dish.
- Serving station tile: `(5, 6)`; stand at `(5, 5)` and face SOUTH to serve soup.
- Initial player positions: `agent_0` at `(1, 5)`, `agent_1` at `(5, 1)`

--------------------------------------------------
Action Space
--------------------------------------------------

Discrete action space (6 actions):

0 : NORTH  (move up)
1 : SOUTH  (move down)
2 : EAST   (move right)
3 : WEST   (move left)
4 : STAY   (do nothing)
5 : INTERACT (pick up / put down / serve, depending on what is in front)

Interaction rule:
- `INTERACT` only affects the tile directly in front of the agent.
- To use an object, stand on the correct adjacent floor tile and face the object.
- If the object is adjacent but orientation is wrong, output the direction action that turns/faces the object first. Use `INTERACT` on the next step once facing correctly.

--------------------------------------------------
Observation Space
--------------------------------------------------

For `large_room`, the wrapper uses a compact observation vector of 38 elements.
Important feature in `large_room` observation layout:

  [0:4]   Orientation one-hot: [North, South, East, West]
  [4:7]   Held object: [onion, soup, dish]
            - All ZEROS when holding nothing. There is NO "none" indicator bit.
            - obs[4]=1 → holding onion
            - obs[5]=1 → holding soup
            - obs[6]=1 → holding dish
  [7:20]  Additional auxiliary features are present here, but you usually do not need them for policy design in `large_room`. Prefer using the fixed layout coordinates and interaction workspaces given above.
  [20:30] Pot 0 (nearest and only pot in `large_room`):
            [exists, is_empty, is_full, is_cooking, is_ready,
             num_onions, num_tomatoes, cook_time, dx, dy]
  [30:34] Wall adjacency: [North, South, East, West]
            - `1` means the adjacent tile in that direction is not walkable this step because it is a counter, pot, dispenser, serving station, or map boundary.
            - If your target object is directly adjacent, you may still need to output that direction once to face the object before using `INTERACT`.
  [34:36] (dx, dy) distance to teammate
  [36:38] Ego player grid position (x, y)

To check holding state correctly:
  holding_nothing = (obs[4] == 0 and obs[5] == 0 and obs[6] == 0)
  holding_onion   = (obs[4] == 1)
  holding_soup    = (obs[5] == 1)
  holding_dish    = (obs[6] == 1)

Pot indexing reminder:
  [20] pot0_exists
  [21] pot0_is_empty
  [22] pot0_is_full
  [23] pot0_is_cooking
  [24] pot0_is_ready
  [25] pot0_num_onions
  [26] pot0_num_tomatoes
  [27] pot0_cook_time
  [28] pot0_dx
  [29] pot0_dy

Policy-writing rules for this observation design:
- Prefer horizontal movement before vertical movement when both reduce distance. In `large_room`, this is usually safer because the important wall objects sit on the top and bottom rows.
- Use `obs[36:38]` ego position together with the fixed layout coordinates above to compute where you should go.
- Use `obs[0:4]` (orientation) together with the fixed object locations and interaction workspaces listed above.
- Treat `obs[4:7]` as a one-hot held-object vector with three possible held objects: onion, soup, dish. Do not infer "empty hands" unless all three bits are zero.
- Use `obs[30:34]` wall flags when moving.
- Use `obs[34:36]` to avoid crowding the teammate near pots, dispensers, and narrow hallways. If the teammate is blocking the path or workspace, `STAY` can be the correct action.
- Relative offsets use `(dx, dy)`: positive `dx` means east, negative `dx` west; positive `dy` means south, negative `dy` north.
- Counters exist ONLY as a last resort to free your hands so you can start cooking a full pot. They are NEVER a way to make progress or hand items to the teammate (see Coordination Requirement — counter hand-off roles are invalid and deadlock). A counter holds only ONE item, and the observation does not tell you whether a counter is already occupied, so a second drop onto the same counter silently fails and traps the agent.
  - Counters at `(2, 0)` and `(4, 0)`, reached from `(2, 1)` or `(4, 1)` while facing NORTH.
  - Bottom counters at `(2, 6)`, `(3, 6)`, `(4, 6)`, reached from row `y=5` while facing SOUTH.

Common failure cases to avoid:
- Getting stuck facing a wall or counter because the policy keeps selecting a blocked move such as walking directly toward the pot tile instead of toward a reachable interaction square.
- Getting stuck in congestion with the teammate by repeatedly pushing into the same lane, pot workspace, or dispenser tile.
- Fetching a dish too early when the pot is full but not cooking yet. In this setup, a full pot still needs a start-cooking `INTERACT` before soup can become ready.
- Trying to start a full pot while still holding an item. The agent must first place the held item on an empty counter or otherwise free its hands.
- Good policies explicitly handle both problems: check walls before moving, and add a detour or yield rule when the teammate is in the way.

--------------------------------------------------
Visual Representation
--------------------------------------------------

- The **ego agent** (agent_0) is shown in **blue**.
- The **teammate agent** (agent_1) is shown in **green**.

--------------------------------------------------
Coordination Requirement
--------------------------------------------------

The full soup pipeline is made of these REAL steps, each of which moves a real item to its real destination:
  (A) pick up an onion from a dispenser
  (B) put the onion into the pot
  (C) start cooking: empty-handed INTERACT on a full, not-yet-cooking pot
  (D) pick up a dish from the dish dispenser
  (E) pick up the finished soup into the dish and deliver it to the serving station

Every behavior specializes the ego in a subset of these real steps. A valid behavior must satisfy BOTH rules below.

RULE 1 — Make real progress, never "stage" on counters.
The ego's job must be genuine pipeline work: carrying a real item to its real destination (an onion INTO the pot, a dish TO the pot to scoop soup, soup TO the serving station). Do NOT design "stager", "prepper", "monitor", "coordinator", "runner", or "hand-off" roles whose plan is to fetch an item and drop it on a counter for the teammate to pick up later. Counter hand-offs are NOT progress: they score nothing, a counter holds only one item, and the observation cannot tell you a counter is already occupied — so the second drop silently fails and the agent gets stuck forever repeating INTERACT. If you cannot phrase the ego's role as "move real item X to its real destination Y", the behavior is invalid. Rephrase it as one of the real steps above.

RULE 2 — Omit at least one real step ENTIRELY so the teammate is required.
Choose at least one pipeline step that the ego's CODE NEVER performs under ANY observation, no matter how long the episode runs — e.g. never touches an onion dispenser, OR never starts the cook, OR never picks up a dish, OR never delivers to the serving station. The teammate must be needed to cover the omitted step(s). This is about the IMPLEMENTED CODE, not the wording. The single most common failure is a policy that *sounds* specialized ("onion courier", "dish helper") but whose code adds fallback branches that, given enough time, walk the pot through every remaining stage by itself. A role like "balances between tasks", "adapts to what the pot needs", or "fills in wherever help is needed" is a violation the instant the code path exists to finish a soup alone. Reducing how *often* the ego does a step is not enough — the code branch for the omitted step(s) must not exist at all.

The two rules work together: do your chosen steps FOR REAL and FULLY (Rule 1), and make sure the steps you did NOT choose are completely absent from your code (Rule 2).

--------------------------------------------------
Spatial Constraints for Policy Design
--------------------------------------------------

Critical interaction tiles that must stay clear when not actively used
(known as `_LARGE_ROOM_BLOCKING_POSITIONS`):
  (3, 1) — pot interaction tile
  (1, 1) — left onion dispenser interaction tile
  (5, 1) — right onion dispenser interaction tile
  (1, 5) — dish dispenser interaction tile
  (5, 5) — serving station interaction tile

If your policy is idle (waiting for the pot to cook, waiting for the teammate, etc.), do NOT park at any of the above tiles. Standing there blocks the cooperator's path and prevents it from completing its tasks.

Instead, move to one of the safe idle positions in the centre of the room
(known as `_LARGE_ROOM_IDLE_POSITIONS`):
  [3, 3], [2, 3], [4, 3], [3, 4]

These centre tiles are always walkable and never obstruct any key workspace, so they are the correct place to wait when not actively pursuing a task.

--------------------------------------------------
Policy Output Requirement
--------------------------------------------------

The policy must return **a single scalar action ID (integer)**.

Correct:
    return torch.tensor(4)  # STAY

Incorrect:
    return torch.tensor([4])
"""

REWARD_API_REFERENCE = """Reward function API (authoritative):
- Signature: `compute_reward(env, agent_index)`.
- `env` is the `OvercookedEnv` instance (from overcooked_ai_py.mdp.overcooked_env).
- `agent_index` is 0 (ego) or 1 (teammate).
- Accessible state:
  - `env.state`: the current OvercookedState.
    - `env.state.players`: list of PlayerState objects (one per agent).
    - `env.state.objects`: dict mapping (x,y) positions to ObjectState.
    - `env.state.timestep`: current step count.
  - `env.mdp`: the OvercookedGridworld instance.
    - `env.mdp.terrain_mtx`: 2D list of terrain characters.
    - `env.mdp.get_pot_states(state)`: dict of pot positions grouped by state.
    - `env.mdp.num_players`: number of agents (always 2 here).
  - `env.game_stats`: dict tracking cumulative events.
    - `env.game_stats["cumulative_sparse_rewards_by_agent"]`: array of cumulative sparse rewards per agent.
    - `env.game_stats["soup_pickup"]`: list of per-agent soup pickup events this episode.
    - `env.game_stats["dish_pickup"]`: list of per-agent dish pickup events.
    - `env.game_stats["useful_onion_pickup"]`: list of per-agent useful onion pickups.
  - Player attributes (`env.state.players[agent_index]`):
    - `player.position`: (x, y) tuple.
    - `player.orientation`: (dx, dy) direction tuple.
    - `player.held_object`: ObjectState or None (e.g., `held_object.name` is "onion", "dish", or "soup").
- Return value:
  - Return a scalar reward for this single agent at the current step (Python `float` recommended).
  - Keep the per-step reward within [-10, 10]. Values outside that range, and any NaN or inf, are clamped at runtime.
  - Keep dense per-step terms small, around |r| <= 1, and reserve larger values for rare events (order 1-5). The clamp bounds a single step, not the episode: a dense term of 1 per step already accumulates to hundreds over an episode, and returns that large make advantages explode, which drives the policy to NaN and kills the run. Aim for an episode return in the low tens.
  - Prefer normalized or tanh-shaped dense terms over raw distances or unbounded sums.
"""

REWARD_EXAMPLE = """
Behavior: reward agents for making progress toward soup delivery, with bonuses for picking up ingredients and cooking.
```python
def compute_reward(env, agent_index):
    reward = 0.0
    player = env.state.players[agent_index]

    # Reward for holding an onion (progress toward pot)
    if player.held_object is not None and player.held_object.name == "onion":
        reward += 0.1

    # Reward for holding a dish (progress toward serving)
    if player.held_object is not None and player.held_object.name == "dish":
        reward += 0.2

    # Reward for holding a ready soup (about to deliver)
    if player.held_object is not None and player.held_object.name == "soup":
        reward += 0.5

    # Large reward on delivery (tracked via cumulative sparse rewards delta)
    cumulative = env.game_stats["cumulative_sparse_rewards_by_agent"]
    if cumulative[agent_index] > 0:
        reward += 1.0

    return reward
```
"""

# ---------------------------------------------------------------------------- #
#                                 Action Type                                  #
# ---------------------------------------------------------------------------- #


class ActionType(BaseModel):
    """
    Action type for the Overcooked environment.

    The mapping of integer to action is:
    0: NORTH
    1: SOUTH
    2: EAST
    3: WEST
    4: STAY
    5: INTERACT

    Example: to stay, return ActionType(action=4).
    The action is a single scalar integer.
    """

    action: int = Field(
        ..., description="Integer representing the action to take (0–5)."
    )

    @classmethod
    def default_action(cls):
        """Default action (STAY)."""
        return cls(action=4)

    def to_tensor(self):
        """Convert action to a scalar torch tensor."""
        return torch.tensor(self.action, dtype=torch.int64)


# ---------------------------------------------------------------------------- #
#                            Internal Helpers                                  #
# ---------------------------------------------------------------------------- #


def _build_overcooked_group_map() -> tuple[dict, dict]:
    """Build group map with ego (agent_0) and teammate (agent_1)."""
    group_map = {"ego": ["agent_0"], "teammate": ["agent_1"]}
    trainable_group = {"ego": ["agent_0"]}
    return group_map, trainable_group


def _collect_step_state(td) -> dict:
    """Extract per-group observations/actions/rewards into canonical dict."""
    nested_keys = set(td.keys(include_nested=True))
    groups = [
        key
        for key in td.keys()
        if isinstance(key, str) and (key, "observation") in nested_keys
    ]

    result = {"ego_team": {}, "opponent_team": {}}
    ego_assigned = False
    teammate_idx = 0

    for group in groups:
        obs = td[group, "observation"]

        try:
            act = td[group, "action"]
        except KeyError:
            act = None

        try:
            rew = td["next", group, "reward"]
        except KeyError:
            try:
                rew = td[group, "reward"]
            except KeyError:
                rew = None

        obs_2d = obs.reshape(-1, obs.shape[-1])
        num_agents = obs_2d.shape[0]
        act_2d = act.reshape(num_agents, -1) if act is not None else None
        rew_2d = rew.reshape(num_agents, -1) if rew is not None else None

        for i in range(num_agents):
            action_val = None
            if act_2d is not None:
                action_item = act_2d[i].cpu().tolist()
                action_val = action_item[0] if len(action_item) == 1 else action_item

            reward_val = None
            if rew_2d is not None:
                reward_item = rew_2d[i].cpu().tolist()
                reward_val = reward_item[0] if len(reward_item) == 1 else reward_item

            payload = {
                "state": obs_2d[i].cpu().tolist(),
                "action": action_val,
                "reward": reward_val,
            }

            if not ego_assigned:
                canonical_name = "ego_agent"
                ego_assigned = True
                result["ego_team"][canonical_name] = payload
            elif group == "teammate" or not ego_assigned:
                canonical_name = f"teammate_{teammate_idx}"
                teammate_idx += 1
                result["ego_team"][canonical_name] = payload
            else:
                canonical_name = f"teammate_{teammate_idx}"
                teammate_idx += 1
                result["ego_team"][canonical_name] = payload

    return result


def _extract_step_group_rewards(
    rollout_td: TensorDictBase,
    group: str,
    step_idx: int,
    num_steps: int,
) -> list[float]:
    key = ("next", group, "reward")
    if key not in rollout_td.keys(include_nested=True):
        return []

    rew_t = rollout_td[key]
    if rew_t.ndim == 0:
        return [float(rew_t.item())]

    if rew_t.ndim >= 2 and int(rew_t.shape[0]) == int(num_steps):
        rew_step = rew_t[step_idx]
    elif rew_t.ndim >= 2 and int(rew_t.shape[1]) == int(num_steps):
        rew_step = rew_t[:, step_idx]
    elif rew_t.shape[0] > step_idx:
        rew_step = rew_t[step_idx]
    else:
        rew_step = rew_t

    return [float(x) for x in rew_step.reshape(-1).detach().cpu().tolist()]


def _attach_rewards_from_rollout(
    states: list[dict], rollout_td: TensorDictBase
) -> None:
    if not states:
        return

    nested_keys = set(rollout_td.keys(include_nested=True))
    top_groups = [k for k in rollout_td.keys() if isinstance(k, str)]
    reward_groups = [g for g in top_groups if ("next", g, "reward") in nested_keys]
    num_steps = len(states)

    for step_idx, step in enumerate(states):
        ego_names = []
        if "ego_agent" in step.get("ego_team", {}):
            ego_names.append("ego_agent")
        ego_names.extend(
            sorted(
                [k for k in step.get("ego_team", {}) if k.startswith("teammate_")],
                key=lambda x: int(x.split("_")[1]),
            )
        )

        ego_vals: list[float] = []
        for g in reward_groups:
            vals = _extract_step_group_rewards(
                rollout_td=rollout_td,
                group=g,
                step_idx=step_idx,
                num_steps=num_steps,
            )
            ego_vals.extend(vals)

        for name, val in zip(ego_names, ego_vals):
            step["ego_team"][name]["reward"] = float(val)


# ---------------------------------------------------------------------------- #
#                            Training & Rollout                                #
# ---------------------------------------------------------------------------- #


def train_overcooked_env(
    reward_fn,
    output_dir,
    rl_cfg_dir,
    task_yaml_name="overcooked.yaml",
    experiment_yaml_name="base_experiment.yaml",
    seed=0,
    callbacks=None,
    teammate_policy=None,
    run_name=None,
    prioritized=True,
    prioritized_beta=1.0,
    prioritized_warmup_batches=30,
    **kwargs,
):
    """Train Overcooked with optional scripted teammate and custom reward function."""
    if reward_fn is not None and hasattr(reward_fn, "__globals__"):
        reward_fn.__globals__["torch"] = torch

    callbacks = list(callbacks or [])

    original_step = OvercookedBenchWrapper._step

    try:
        task = OvercookedTask.OVERCOOKED.get_from_yaml(
            path=str(rl_cfg_dir / task_yaml_name)
        )

        group_map, trainable_group = _build_overcooked_group_map()
        task.config["group_map"] = group_map

        if teammate_policy is not None:
            from popgen.agents.callbacks import PrioritizedCallback, select_teammate_callback_cls

            is_population = isinstance(teammate_policy[0], list)
            callback_cls = select_teammate_callback_cls(is_population=is_population, prioritized=prioritized)
            callback_kwargs = {}
            if callback_cls is PrioritizedCallback:
                # Only the prioritized callback takes the sampling prior's knobs; the
                # uniform ones would reject them.
                callback_kwargs = {"beta": prioritized_beta, "warmup_batches": prioritized_warmup_batches}
            callbacks.append(
                callback_cls(
                    teammate_policies=teammate_policy,
                    trainable_group=trainable_group,
                    **callback_kwargs,
                )
            )

        if reward_fn is not None:
            desired_group_map = task.config.get("group_map")

            # The generated reward describes the EGO agent only, so override the
            # ego group and leave every other group on the environment's own
            # reward. This mirrors vmas_common, which overrides `world.agents[0]`
            # alone.
            #
            # Evaluating the ego's reward from the teammate's index was wasted
            # work (nothing trains on the teammate group -- `trainable_group` is
            # ego-only) and it corrupted `collection/reward/*`, which is the mean
            # over groups. It was also semantically backwards: a reward that
            # penalises the ego for fetching dishes, so the partner must, then
            # penalised the scripted partner for doing exactly its job.
            #
            # Note for cooperator training: `TeamRewardSumCallback` adds the
            # teammate group's reward into the ego's training reward. It is only
            # attached by CooperatorTrainingPipeline, which passes reward_fn=None,
            # so it never combines with this override today. If the two are ever
            # used together, the ego would train on `generated(ego) +
            # task_reward(teammate)` rather than the generated reward summed over
            # both agents.
            ego_group, ego_agent_names = next(iter(trainable_group.items()))
            ego_idxs = [int(name.split("_")[1]) for name in ego_agent_names]

            def custom_step(self, tensordict):
                out_td = original_step(self, tensordict)

                ego_rewards = []
                for idx in ego_idxs:
                    r = reward_fn(self.oc_env, idx)
                    if isinstance(r, torch.Tensor):
                        r = r.item()
                    ego_rewards.append(float(r))
                reward_tensor = torch.tensor(
                    ego_rewards, device=self.device, dtype=torch.float32
                ).unsqueeze(-1)

                if desired_group_map:
                    out_td[ego_group, "reward"] = reward_tensor
                else:
                    # Flat layout: replace only the ego rows, keeping the other
                    # agents' environment reward intact.
                    flat = out_td["agents", "reward"].clone()
                    for row, idx in enumerate(ego_idxs):
                        flat[idx] = reward_tensor[row]
                    out_td["agents", "reward"] = flat

                return out_td

            OvercookedBenchWrapper._step = custom_step

        base_rl_cfg_dir = Path(rl_cfg_dir)
        mode_cfg_dir = resolve_cfg_dir(base_rl_cfg_dir, teammate_policy)

        algorithm_config = load_algorithm_config(mode_cfg_dir)
        experiment_config = ExperimentConfig.get_from_yaml(
            path=str(mode_cfg_dir / experiment_yaml_name)
        )
        model_config = load_model_config(mode_cfg_dir)
        critic_model_config = load_model_config(mode_cfg_dir)

        apply_train_kwargs(kwargs, experiment_config, algorithm_config, model_config, critic_model_config)

        requested_devices = [
            str(getattr(experiment_config, "sampling_device", "")),
            str(getattr(experiment_config, "train_device", "")),
            str(getattr(experiment_config, "buffer_device", "")),
        ]
        if any(dev.startswith("cuda") for dev in requested_devices) and not torch.cuda.is_available():
            logging.warning(
                "CUDA devices requested in Overcooked experiment config but CUDA unavailable. "
                "Falling back to CPU."
            )
            experiment_config.sampling_device = "cpu"
            experiment_config.train_device = "cpu"
            if str(getattr(experiment_config, "buffer_device", "")).lower() != "disk":
                experiment_config.buffer_device = "cpu"

        experiment_config.save_folder = output_dir
        display_run_name = run_name.strip() if isinstance(run_name, str) and run_name.strip() else Path(output_dir).name
        experiment_config.wandb_extra_kwargs = {
            "name": display_run_name,
            "tags": ["overcooked"],
        }

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=seed,
            config=experiment_config,
            callbacks=callbacks,
        )
        experiment.run()
    finally:
        OvercookedBenchWrapper._step = original_step

    return experiment


def rollout_overcooked_env(
    policy,
    env,
    max_steps: int = 400,
    seed: int | None = None,
    **kwargs,
):
    """Rollout a policy in the Overcooked environment."""
    if not isinstance(env, EnvBase):
        raise TypeError(
            f"Expected EnvBase for rollout_overcooked_env, got {type(env).__name__}."
        )
    policy.eval()

    def rollout_policy(tensordict):
        out = policy(tensordict)
        if not isinstance(out, TensorDictBase):
            raise TypeError(
                f"Policy output must be TensorDictBase, got {type(out).__name__}."
            )
        return out

    apply_rollout_seed(env, seed=seed)

    states = []
    images = []

    def render_callback(env_inner, td):
        try:
            frame = env_inner.render(mode="rgb_array")
        except TypeError:
            try:
                frame = env_inner.render()
            except Exception:
                frame = None
        except Exception:
            frame = None
        if isinstance(frame, np.ndarray):
            images.append(frame)
        states.append(_collect_step_state(td))

    rollout_sig = inspect.signature(env.rollout)
    supported_args = set(rollout_sig.parameters.keys())
    rollout_kwargs = {k: v for k, v in kwargs.items() if k in supported_args}
    if "max_steps" in supported_args and "max_steps" not in rollout_kwargs:
        rollout_kwargs["max_steps"] = max_steps

    with torch.no_grad():
        rollout_td = env.rollout(
            policy=rollout_policy,
            callback=render_callback,
            auto_cast_to_device=True,
            break_when_any_done=True,
            **rollout_kwargs,
        )

    _attach_rewards_from_rollout(states=states, rollout_td=rollout_td)

    return states, images, None


def rollout_fn(policy, env=None, **kwargs):
    """Rollout a policy in the Overcooked environment."""
    if env is None:
        env = create_env(**kwargs)

    if "max_steps" not in kwargs:
        kwargs["max_steps"] = 400

    return rollout_overcooked_env(policy=policy, env=env, **kwargs)


def train_fn(reward_fn: Callable | None = None, output_dir: str | None = None, run_name: str | None = None, **kwargs):
    """Train agents in Overcooked with a custom reward function.

    Args:
        reward_fn: Callable with signature ``compute_reward(env, agent_index)``
            where ``env`` is the ``OvercookedEnv`` instance and ``agent_index``
            is 0 (ego) or 1 (teammate). Returns a scalar float reward.
        output_dir: Directory to save results.
        run_name: Display name for this training run.
    """
    teammate_policy = kwargs.pop("teammate_policy", None)
    if teammate_policy is None:
        teammate_policy = [ScriptedPolicy()]

    return train_overcooked_env(
        reward_fn=reward_fn,
        output_dir=output_dir,
        rl_cfg_dir=RL_CFG_DIR,
        run_name=run_name,
        teammate_policy=teammate_policy,
        **kwargs,
    )


def _round_obs_pair(raw_state: list[float], start: int) -> list[int]:
    return [
        int(round(float(raw_state[start]))),
        int(round(float(raw_state[start + 1]))),
    ]


def _delta_from_positions(
    player_position: list[int] | tuple[int, int],
    target_position: list[int] | tuple[int, int],
) -> list[int]:
    return [
        int(target_position[0] - player_position[0]),
        int(target_position[1] - player_position[1]),
    ]


def _nearest_delta_and_position(
    player_position: list[int] | tuple[int, int],
    target_positions: list[list[int]],
) -> tuple[list[int], list[int]]:
    best_target = min(
        target_positions,
        key=lambda target: abs(target[0] - player_position[0]) + abs(target[1] - player_position[1]),
    )
    return _delta_from_positions(player_position, best_target), list(best_target)


def _decode_held_object(raw_state: list[float]) -> str | None:
    if len(raw_state) == 24:
        held = raw_state[4:7]
        obj_names = ("onion", "soup", "dish")
    else:
        held = raw_state[4:8]
        obj_names = ("onion", "soup", "dish", "tomato")
    if max(float(v) for v in held) <= 0.5:
        return None
    held_idx = int(np.argmax(np.asarray(held, dtype=float)))
    return obj_names[held_idx]


def _decode_orientation(raw_state: list[float]) -> tuple[str, list[int]]:
    orientation_idx = int(np.argmax(np.asarray(raw_state[0:4], dtype=float)))
    direction = Direction.INDEX_TO_DIRECTION[orientation_idx]
    return _DIRECTION_NAMES[orientation_idx], [direction[0], direction[1]]


def _decode_action(action: object) -> tuple[int | None, str | None, str | None]:
    if action is None:
        return None, None, None
    try:
        action_idx = int(action)
    except (TypeError, ValueError):
        return None, None, None
    if not 0 <= action_idx < NUM_ACTIONS:
        return action_idx, None, None
    action_obj = Action.INDEX_TO_ACTION[action_idx]
    return action_idx, _ACTION_NAMES.get(action_idx), Action.ACTION_TO_CHAR.get(action_obj)


def _serialize_large_room_state(raw_state: list[float], action: object) -> dict:
    compact_v2 = len(raw_state) == 24
    player_position = _round_obs_pair(raw_state, 22) if compact_v2 else _round_obs_pair(raw_state, 36)
    orientation_name, facing_delta = _decode_orientation(raw_state)
    facing_tile = [
        player_position[0] + facing_delta[0],
        player_position[1] + facing_delta[1],
    ]
    held_object = _decode_held_object(raw_state)
    teammate_delta = _round_obs_pair(raw_state, 20) if compact_v2 else _round_obs_pair(raw_state, 34)
    teammate_position = [
        player_position[0] + teammate_delta[0],
        player_position[1] + teammate_delta[1],
    ]
    wall_start = 16 if compact_v2 else 30
    wall_flags = [bool(float(v) > 0.5) for v in raw_state[wall_start : wall_start + 4]]
    blocked_directions = [
        _DIRECTION_NAMES[idx] for idx, blocked in enumerate(wall_flags) if blocked
    ]
    wall_tiles = [
        {
            "direction": _DIRECTION_NAMES[idx],
            "tile": [
                player_position[0] + Direction.INDEX_TO_DIRECTION[idx][0],
                player_position[1] + Direction.INDEX_TO_DIRECTION[idx][1],
            ],
        }
        for idx, blocked in enumerate(wall_flags)
        if blocked
    ]
    if compact_v2:
        pot_exists_idx = 7
        pot_delta = _round_obs_pair(raw_state, 14)
        onion_delta, onion_position = _nearest_delta_and_position(
            player_position, _LARGE_ROOM_LAYOUT_INFO["onion_dispenser_locations"]
        )
        dish_delta, dish_position = _nearest_delta_and_position(
            player_position, _LARGE_ROOM_LAYOUT_INFO["dish_dispenser_locations"]
        )
        serving_delta, serving_position = _nearest_delta_and_position(
            player_position, _LARGE_ROOM_LAYOUT_INFO["serving_station_locations"]
        )
        empty_counter_delta, empty_counter_position = _nearest_delta_and_position(
            player_position, _LARGE_ROOM_LAYOUT_INFO["counter_locations"]
        )
        soup_delta = list(pot_delta)
        soup_position = _LARGE_ROOM_LAYOUT_INFO["pot_locations"][0]
        soup_num_onions = int(round(float(raw_state[12])))
        soup_num_tomatoes = 0
        pot_num_tomatoes = 0
        pot_cook_time_idx = 13
    else:
        pot_exists_idx = 20
        pot_delta = _round_obs_pair(raw_state, 28)
        onion_delta = _round_obs_pair(raw_state, 8)
        onion_position = [
            player_position[0] + onion_delta[0],
            player_position[1] + onion_delta[1],
        ]
        dish_delta = _round_obs_pair(raw_state, 10)
        dish_position = [
            player_position[0] + dish_delta[0],
            player_position[1] + dish_delta[1],
        ]
        soup_delta = _round_obs_pair(raw_state, 12)
        soup_position = [
            player_position[0] + soup_delta[0],
            player_position[1] + soup_delta[1],
        ]
        soup_num_onions = int(round(float(raw_state[14])))
        soup_num_tomatoes = int(round(float(raw_state[15])))
        serving_delta = _round_obs_pair(raw_state, 16)
        serving_position = [
            player_position[0] + serving_delta[0],
            player_position[1] + serving_delta[1],
        ]
        empty_counter_delta = _round_obs_pair(raw_state, 18)
        empty_counter_position = [
            player_position[0] + empty_counter_delta[0],
            player_position[1] + empty_counter_delta[1],
        ]
        pot_num_tomatoes = int(round(float(raw_state[26])))
        pot_cook_time_idx = 27
    action_idx, action_name, action_symbol = _decode_action(action)

    return {
        "env": "overcooked",
        "layout_name": _LARGE_ROOM_LAYOUT_INFO["layout_name"],
        "obs_dim": len(raw_state),
        "room_size": _LARGE_ROOM_LAYOUT_INFO["room_size"],
        "player_position": player_position,
        "orientation": orientation_name,
        "facing_direction": facing_delta,
        "facing_tile": facing_tile,
        "held_object": held_object,
        "holding_anything": held_object is not None,
        "teammate_delta": teammate_delta,
        "teammate_position": teammate_position,
        "teammate_distance": abs(teammate_delta[0]) + abs(teammate_delta[1]),
        "action_id": action_idx,
        "action_name": action_name,
        "action_symbol": action_symbol,
        "walls": {
            "north": wall_flags[0],
            "south": wall_flags[1],
            "east": wall_flags[2],
            "west": wall_flags[3],
        },
        "blocked_directions": blocked_directions,
        "adjacent_wall_tiles": wall_tiles,
        "pot_status": {
            "position": _LARGE_ROOM_LAYOUT_INFO["pot_locations"][0],
            "recommended_interaction_tile": _LARGE_ROOM_LAYOUT_INFO["pot_interaction_tiles"][0],
            "delta": pot_delta,
            "distance": abs(pot_delta[0]) + abs(pot_delta[1]),
            "exists": bool(float(raw_state[pot_exists_idx]) > 0.5),
            "is_empty": bool(float(raw_state[pot_exists_idx + 1]) > 0.5),
            "is_full": bool(float(raw_state[pot_exists_idx + 2]) > 0.5),
            "is_cooking": bool(float(raw_state[pot_exists_idx + 3]) > 0.5),
            "is_ready": bool(float(raw_state[pot_exists_idx + 4]) > 0.5),
            "num_onions": int(round(float(raw_state[pot_exists_idx + 5]))),
            "num_tomatoes": pot_num_tomatoes,
            "cook_time_remaining": int(round(float(raw_state[pot_cook_time_idx]))),
            "adjacent": abs(pot_delta[0]) + abs(pot_delta[1]) == 1,
        },
        "static_objects": {
            "onion_dispensers": _LARGE_ROOM_LAYOUT_INFO["onion_dispenser_locations"],
            "dish_dispensers": _LARGE_ROOM_LAYOUT_INFO["dish_dispenser_locations"],
            "serving_stations": _LARGE_ROOM_LAYOUT_INFO["serving_station_locations"],
            "counters": _LARGE_ROOM_LAYOUT_INFO["counter_locations"],
            "pots": _LARGE_ROOM_LAYOUT_INFO["pot_locations"],
            "tomato_dispensers": _LARGE_ROOM_LAYOUT_INFO["tomato_dispenser_locations"],
        },
        "closest_targets": {
            "onion_dispenser_delta": onion_delta,
            "onion_dispenser_position": onion_position,
            "dish_delta": dish_delta,
            "dish_dispenser_position": dish_position,
            "soup_delta": soup_delta,
            "soup_position": soup_position,
            "soup_num_onions": soup_num_onions,
            "soup_num_tomatoes": soup_num_tomatoes,
            "serving_station_delta": serving_delta,
            "serving_station_position": serving_position,
            "empty_counter_delta": empty_counter_delta,
            "empty_counter_position": empty_counter_position,
        },
    }


def serialize_state_fn(rollout_data: list[dict]) -> list[dict]:
    """Add structured overcooked-specific fields to each rollout step."""
    for step in rollout_data:
        for team_name in ("ego_team", "opponent_team"):
            for agent_state in step.get(team_name, {}).values():
                raw_state = agent_state.get("state")
                if isinstance(raw_state, list):
                    obs_len = len(raw_state)
                    action = agent_state.get("action")
                    if obs_len in {24, 38}:
                        agent_state["serialized_state"] = _serialize_large_room_state(
                            raw_state=raw_state,
                            action=action,
                        )
                        continue

                    action_idx, action_name, action_symbol = _decode_action(action)
                    px, py = (raw_state[-2], raw_state[-1]) if obs_len >= 2 else (None, None)
                    agent_state["serialized_state"] = {
                        "env": "overcooked",
                        "obs_dim": obs_len,
                        "player_position": [px, py] if px is not None else None,
                        "action_id": action_idx,
                        "action_name": action_name,
                        "action_symbol": action_symbol,
                    }
    return rollout_data


# ---------------------------------------------------------------------------- #
#                              Scripted Policy                                 #
# ---------------------------------------------------------------------------- #

# STAY is action index 4 in overcooked
_STAY_IDX = Action.ACTION_TO_INDEX[Action.STAY]


class ScriptedPolicy(nn.Module):
    """Observation-driven scripted teammate for Overcooked."""

    _OBS_DIM_MIN = 24
    _FULL_OBS_DIM = 96
    _LEGACY_COMPACT_OBS_DIM = 38
    _COMPACT_OBS_DIM = 24
    _ORIENTATION_SLICE = slice(0, 4)
    _ORIENTATIONS = (
        Direction.NORTH,
        Direction.SOUTH,
        Direction.EAST,
        Direction.WEST,
    )
    _ACTION_BY_DIRECTION = {
        Direction.NORTH: Action.ACTION_TO_INDEX[Direction.NORTH],
        Direction.SOUTH: Action.ACTION_TO_INDEX[Direction.SOUTH],
        Direction.EAST: Action.ACTION_TO_INDEX[Direction.EAST],
        Direction.WEST: Action.ACTION_TO_INDEX[Direction.WEST],
    }
    _INTERACT_IDX = Action.ACTION_TO_INDEX[Action.INTERACT]
    _FULL_LAYOUT = {
        "held": slice(4, 8),
        "obj_names": ("onion", "soup", "dish", "tomato"),
        "onion": slice(8, 10),
        "dish": slice(12, 14),
        "serving": slice(18, 20),
        "empty_counter": slice(20, 22),
        "pots": (slice(22, 32), slice(32, 42)),
        "walls": slice(42, 46),
        "teammate": slice(92, 94),
        "position": slice(94, 96),
    }
    _LEGACY_COMPACT_LAYOUT = {
        "held": slice(4, 8),
        "obj_names": ("onion", "soup", "dish", "tomato"),
        "onion": slice(8, 10),
        "dish": slice(10, 12),
        "serving": slice(16, 18),
        "empty_counter": slice(18, 20),
        "pots": (slice(20, 30),),
        "walls": slice(30, 34),
        "teammate": slice(34, 36),
        "position": slice(36, 38),
    }
    _COMPACT_LAYOUT = {
        "held": slice(4, 7),
        "obj_names": ("onion", "soup", "dish"),
        "pots": (slice(7, 16),),
        "walls": slice(16, 20),
        "teammate": slice(20, 22),
        "position": slice(22, 24),
        "fixed_large_room": True,
    }

    def __init__(self, env=None, *, yield_steps: int = 2, enable_detour: bool = True):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.yield_steps = max(1, int(yield_steps))
        self.enable_detour = enable_detour
        self._stuck_steps: list[int] = []
        self._last_actions: list[int] = []
        self._last_positions: list[tuple[int, int] | None] = []
        self._last_teammate_deltas: list[tuple[int, int] | None] = []

    @staticmethod
    def _round_delta(delta: torch.Tensor) -> tuple[int, int]:
        return int(round(float(delta[0].item()))), int(round(float(delta[1].item())))

    @staticmethod
    def _delta_sub(lhs: tuple[int, int], rhs: tuple[int, int]) -> tuple[int, int]:
        return lhs[0] - rhs[0], lhs[1] - rhs[1]

    @classmethod
    def _held_object_name(cls, obs_row: torch.Tensor) -> str | None:
        layout = cls._obs_layout(obs_row)
        if layout is None:
            return None
        held = obs_row[layout["held"]]
        if held.numel() == 0 or float(held.max().item()) <= 0.5:
            return None
        return layout["obj_names"][int(torch.argmax(held).item())]

    @classmethod
    def _obs_layout(cls, obs_row: torch.Tensor) -> dict | None:
        if obs_row.numel() >= cls._FULL_OBS_DIM:
            return cls._FULL_LAYOUT
        if obs_row.numel() >= cls._LEGACY_COMPACT_OBS_DIM:
            return cls._LEGACY_COMPACT_LAYOUT
        if obs_row.numel() >= cls._COMPACT_OBS_DIM:
            return cls._COMPACT_LAYOUT
        return None

    @classmethod
    def _decode_pot(cls, obs_row: torch.Tensor, pot_slice: slice) -> dict:
        pot = obs_row[pot_slice]
        data = {
            "exists": bool(float(pot[0].item()) > 0.5),
            "is_empty": bool(float(pot[1].item()) > 0.5),
            "is_full": bool(float(pot[2].item()) > 0.5),
            "is_cooking": bool(float(pot[3].item()) > 0.5),
            "is_ready": bool(float(pot[4].item()) > 0.5),
            "num_onions": int(round(float(pot[5].item()))),
        }
        if pot.numel() >= 10:
            data["delta"] = cls._round_delta(pot[8:10])
        else:
            data["delta"] = cls._round_delta(pot[7:9])
        return data

    @staticmethod
    def _fixed_target_delta(
        ego_position: tuple[int, int],
        target_positions: list[list[int]],
    ) -> tuple[int, int]:
        best_target = min(
            target_positions,
            key=lambda target: abs(target[0] - ego_position[0]) + abs(target[1] - ego_position[1]),
        )
        return best_target[0] - ego_position[0], best_target[1] - ego_position[1]

    @staticmethod
    def _manhattan(delta: tuple[int, int]) -> int:
        return abs(delta[0]) + abs(delta[1])

    @classmethod
    def _delta_to_direction(cls, delta: tuple[int, int]) -> tuple[int, int] | None:
        dx, dy = delta
        if cls._manhattan(delta) != 1:
            return None
        if dx == 1:
            return Direction.EAST
        if dx == -1:
            return Direction.WEST
        if dy == 1:
            return Direction.SOUTH
        if dy == -1:
            return Direction.NORTH
        return None

    @classmethod
    def _direction_delta(cls, action_idx: int) -> tuple[int, int] | None:
        for direction, idx in cls._ACTION_BY_DIRECTION.items():
            if idx == action_idx:
                return direction
        return None

    @classmethod
    def _fallback_movement(
        cls,
        delta: tuple[int, int],
        wall_flags: torch.Tensor,
    ) -> int:
        dx, dy = delta
        preferred_dirs: list[tuple[int, int]] = []
        if abs(dx) >= abs(dy):
            if dx > 0:
                preferred_dirs.append(Direction.EAST)
            elif dx < 0:
                preferred_dirs.append(Direction.WEST)
            if dy > 0:
                preferred_dirs.append(Direction.SOUTH)
            elif dy < 0:
                preferred_dirs.append(Direction.NORTH)
        else:
            if dy > 0:
                preferred_dirs.append(Direction.SOUTH)
            elif dy < 0:
                preferred_dirs.append(Direction.NORTH)
            if dx > 0:
                preferred_dirs.append(Direction.EAST)
            elif dx < 0:
                preferred_dirs.append(Direction.WEST)

        for direction in cls._ORIENTATIONS:
            if direction not in preferred_dirs:
                preferred_dirs.append(direction)

        for direction in preferred_dirs:
            direction_idx = Direction.DIRECTION_TO_INDEX[direction]
            if float(wall_flags[direction_idx].item()) <= 0.5:
                return cls._ACTION_BY_DIRECTION[direction]
        return _STAY_IDX

    @classmethod
    def _safe_action_available(cls, action_idx: int, wall_flags: torch.Tensor) -> bool:
        if action_idx == _STAY_IDX or action_idx == cls._INTERACT_IDX:
            return True
        direction = cls._direction_delta(action_idx)
        if direction is None:
            return False
        direction_idx = Direction.DIRECTION_TO_INDEX[direction]
        return float(wall_flags[direction_idx].item()) <= 0.5

    @classmethod
    def _action_toward(
        cls,
        delta: tuple[int, int],
        orientation_idx: int,
        wall_flags: torch.Tensor,
        *,
        interact_when_adjacent: bool,
    ) -> int:
        if delta == (0, 0):
            return _STAY_IDX

        target_dir = cls._delta_to_direction(delta)
        if target_dir is not None:
            if interact_when_adjacent and cls._ORIENTATIONS[orientation_idx] == target_dir:
                return cls._INTERACT_IDX
            return cls._ACTION_BY_DIRECTION[target_dir]

        return cls._fallback_movement(delta=delta, wall_flags=wall_flags)

    @classmethod
    def _greedy_decision(
        cls,
        *,
        held_obj: str | None,
        onion_delta: tuple[int, int],
        dish_delta: tuple[int, int],
        serving_delta: tuple[int, int],
        empty_counter_delta: tuple[int, int],
        orientation_idx: int,
        wall_flags: torch.Tensor,
        ready_pot: dict | None,
        cookable_pot: dict | None,
        start_cooking_pot: dict | None,
        ingredient_pot: dict | None,
    ) -> dict:
        if held_obj == "soup":
            return {
                "action": cls._action_toward(
                    serving_delta,
                    orientation_idx,
                    wall_flags,
                    interact_when_adjacent=True,
                ),
                "target_delta": serving_delta,
                "target_kind": "serving",
            }

        if held_obj == "dish":
            target_pot = ready_pot or cookable_pot
            if target_pot is None:
                return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}
            return {
                "action": cls._action_toward(
                    target_pot["delta"],
                    orientation_idx,
                    wall_flags,
                    interact_when_adjacent=ready_pot is not None,
                ),
                "target_delta": target_pot["delta"],
                "target_kind": "pot",
            }

        if held_obj in {"onion", "tomato"}:
            if ingredient_pot is not None:
                return {
                    "action": cls._action_toward(
                        ingredient_pot["delta"],
                        orientation_idx,
                        wall_flags,
                        interact_when_adjacent=True,
                    ),
                    "target_delta": ingredient_pot["delta"],
                    "target_kind": "pot",
                }
            if empty_counter_delta != (0, 0):
                return {
                    "action": cls._action_toward(
                        empty_counter_delta,
                        orientation_idx,
                        wall_flags,
                        interact_when_adjacent=True,
                    ),
                    "target_delta": empty_counter_delta,
                    "target_kind": "counter",
                }
            return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}

        if start_cooking_pot is not None:
            return {
                "action": cls._action_toward(
                    start_cooking_pot["delta"],
                    orientation_idx,
                    wall_flags,
                    interact_when_adjacent=True,
                ),
                "target_delta": start_cooking_pot["delta"],
                "target_kind": "pot",
            }

        if ready_pot is not None or cookable_pot is not None:
            return {
                "action": cls._action_toward(
                    dish_delta,
                    orientation_idx,
                    wall_flags,
                    interact_when_adjacent=True,
                ),
                "target_delta": dish_delta,
                "target_kind": "dish",
            }

        if ingredient_pot is not None:
            return {
                "action": cls._action_toward(
                    onion_delta,
                    orientation_idx,
                    wall_flags,
                    interact_when_adjacent=True,
                ),
                "target_delta": onion_delta,
                "target_kind": "onion",
            }

        return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}

    @classmethod
    def _select_best_pot(
        cls,
        pots: list[dict],
        *,
        require_ready: bool = False,
        allow_cooking: bool = False,
    ) -> dict | None:
        candidates = []
        for pot in pots:
            if not pot["exists"]:
                continue
            if require_ready and not pot["is_ready"]:
                continue
            if allow_cooking and not (pot["is_ready"] or pot["is_cooking"] or pot["is_full"]):
                continue
            if not require_ready and not allow_cooking:
                candidates.append(pot)
                continue
            candidates.append(pot)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda pot: (cls._manhattan(pot["delta"]), -pot["num_onions"]),
        )

    @classmethod
    def _select_ingredient_pot(cls, pots: list[dict]) -> dict | None:
        candidates = [
            pot
            for pot in pots
            if pot["exists"] and not pot["is_ready"] and not pot["is_cooking"] and not pot["is_full"]
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda pot: (
                -(pot["num_onions"] > 0),
                -pot["num_onions"],
                cls._manhattan(pot["delta"]),
            ),
        )

    @classmethod
    def _select_start_cooking_pot(cls, pots: list[dict]) -> dict | None:
        candidates = [
            pot
            for pot in pots
            if pot["exists"] and pot["is_full"] and not pot["is_cooking"] and not pot["is_ready"]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pot: cls._manhattan(pot["delta"]))

    @classmethod
    def _recovery_candidates(cls, previous_action: int) -> list[int]:
        previous_direction = cls._direction_delta(previous_action)
        if previous_direction in {Direction.EAST, Direction.WEST}:
            candidates = [Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST]
        elif previous_direction in {Direction.NORTH, Direction.SOUTH}:
            candidates = [Direction.EAST, Direction.WEST, Direction.SOUTH, Direction.NORTH]
        else:
            candidates = list(cls._ORIENTATIONS)
        return [cls._ACTION_BY_DIRECTION[direction] for direction in candidates]

    @classmethod
    def _select_recovery_action(
        cls,
        previous_action: int,
        preferred_action: int,
        wall_flags: torch.Tensor,
    ) -> int:
        if preferred_action != previous_action and cls._safe_action_available(preferred_action, wall_flags):
            return preferred_action

        for action_idx in cls._recovery_candidates(previous_action):
            if action_idx == previous_action:
                continue
            if cls._safe_action_available(action_idx, wall_flags):
                return action_idx
        return _STAY_IDX

    def _ensure_tracking(self, batch_size: int) -> None:
        if len(self._stuck_steps) != batch_size:
            self._stuck_steps = [0] * batch_size
            self._last_actions = [_STAY_IDX] * batch_size
            self._last_positions = [None] * batch_size
            self._last_teammate_deltas = [None] * batch_size

    def _choose_action(self, obs_row: torch.Tensor, slot_idx: int) -> int:
        if obs_row.numel() < self._OBS_DIM_MIN:
            return _STAY_IDX

        layout = self._obs_layout(obs_row)
        if layout is None:
            return _STAY_IDX

        orientation_idx = int(torch.argmax(obs_row[self._ORIENTATION_SLICE]).item())
        held_obj = self._held_object_name(obs_row)
        wall_flags = obs_row[layout["walls"]]
        teammate_delta = self._round_delta(obs_row[layout["teammate"]])
        ego_position = self._round_delta(obs_row[layout["position"]])
        if layout.get("fixed_large_room"):
            onion_delta = self._fixed_target_delta(
                ego_position, _LARGE_ROOM_LAYOUT_INFO["onion_dispenser_locations"]
            )
            dish_delta = self._fixed_target_delta(
                ego_position, _LARGE_ROOM_LAYOUT_INFO["dish_dispenser_locations"]
            )
            serving_delta = self._fixed_target_delta(
                ego_position, _LARGE_ROOM_LAYOUT_INFO["serving_station_locations"]
            )
            empty_counter_delta = self._fixed_target_delta(
                ego_position, _LARGE_ROOM_LAYOUT_INFO["counter_locations"]
            )
        else:
            onion_delta = self._round_delta(obs_row[layout["onion"]])
            dish_delta = self._round_delta(obs_row[layout["dish"]])
            serving_delta = self._round_delta(obs_row[layout["serving"]])
            empty_counter_delta = self._round_delta(obs_row[layout["empty_counter"]])
        pots = [self._decode_pot(obs_row, pot_slice) for pot_slice in layout["pots"]]

        previous_action = self._last_actions[slot_idx]
        previous_position = self._last_positions[slot_idx]
        previous_teammate_delta = self._last_teammate_deltas[slot_idx]
        if (
            previous_position is not None
            and previous_teammate_delta is not None
            and previous_action not in {_STAY_IDX, self._INTERACT_IDX}
            and ego_position == previous_position
            and teammate_delta == previous_teammate_delta
        ):
            self._stuck_steps[slot_idx] += 1
        else:
            self._stuck_steps[slot_idx] = 0

        ready_pot = self._select_best_pot(pots, require_ready=True)
        cookable_pot = self._select_best_pot(pots, allow_cooking=True)
        start_cooking_pot = self._select_start_cooking_pot(pots)
        ingredient_pot = self._select_ingredient_pot(pots)

        decision = self._greedy_decision(
            held_obj=held_obj,
            onion_delta=onion_delta,
            dish_delta=dish_delta,
            serving_delta=serving_delta,
            empty_counter_delta=empty_counter_delta,
            orientation_idx=orientation_idx,
            wall_flags=wall_flags,
            ready_pot=ready_pot,
            cookable_pot=cookable_pot,
            start_cooking_pot=start_cooking_pot,
            ingredient_pot=ingredient_pot,
        )
        action = decision["action"]
        if self.enable_detour and self._stuck_steps[slot_idx] >= self.yield_steps:
            action = self._select_recovery_action(
                previous_action=previous_action,
                preferred_action=action,
                wall_flags=wall_flags,
            )

        self._last_actions[slot_idx] = action
        self._last_positions[slot_idx] = ego_position
        self._last_teammate_deltas[slot_idx] = teammate_delta
        return action

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Return action indices for each observation in the batch.

        Args:
            obs_tensor: shape ``(batch, obs_dim)``

        Returns:
            Action tensor, shape ``(batch,)``.
        """
        if not isinstance(obs_tensor, torch.Tensor):
            raise TypeError(
                f"ScriptedPolicy expected torch.Tensor, got {type(obs_tensor).__name__}."
            )
        if obs_tensor.dim() != 2:
            raise ValueError(
                f"ScriptedPolicy expects shape (batch, obs_dim), got {tuple(obs_tensor.shape)}."
            )
        batch_size = obs_tensor.shape[0]
        self._ensure_tracking(batch_size)
        actions = torch.full(
            (batch_size,), _STAY_IDX, dtype=torch.long, device=obs_tensor.device
        )
        for b in range(batch_size):
            actions[b] = self._choose_action(obs_tensor[b], b)

        return actions


# ---------------------------------------------------------------------------- #
#                        Diverse Evaluation Behaviors                          #
# ---------------------------------------------------------------------------- #

# Interaction tiles that must stay clear for the cooperator to use freely.
_LARGE_ROOM_BLOCKING_POSITIONS: frozenset[tuple[int, int]] = frozenset({
    (3, 1),  # pot interaction tile
    (1, 1),  # left onion dispenser interaction tile
    (5, 1),  # right onion dispenser interaction tile
    (1, 5),  # dish dispenser interaction tile
    (5, 5),  # serving station interaction tile
})
# Safe floor tiles in the centre of the room used as idle targets.
_LARGE_ROOM_IDLE_POSITIONS: list[list[int]] = [[3, 3], [2, 3], [4, 3], [3, 4]]


class _NonBlockingEvaluationPolicy(ScriptedPolicy):
    """Shared base for evaluation policies.

    When the policy would STAY at a critical interaction tile (in front of the
    pot, dispensers, or serving station), it instead moves toward the nearest
    safe idle position so the cooperator's path stays unobstructed.
    """

    _BLOCKING_POSITIONS = _LARGE_ROOM_BLOCKING_POSITIONS
    _IDLE_POSITIONS = _LARGE_ROOM_IDLE_POSITIONS

    def _choose_action(self, obs_row: torch.Tensor, slot_idx: int) -> int:
        action = super()._choose_action(obs_row, slot_idx)
        if action != _STAY_IDX:
            return action
        layout = self._obs_layout(obs_row)
        if layout is None:
            return action
        ego_position = self._round_delta(obs_row[layout["position"]])
        if ego_position in self._BLOCKING_POSITIONS:
            idle_delta = self._fixed_target_delta(ego_position, self._IDLE_POSITIONS)
            wall_flags = obs_row[layout["walls"]]
            return self._fallback_movement(delta=idle_delta, wall_flags=wall_flags)
        return action


class IngredientFillerPolicy(_NonBlockingEvaluationPolicy):
    """Ingredient specialist: fetches onions and fills the pot only.

    Ignores dish pickup, cooking start, and serving entirely.
    STAYs once the pot is full, cooking, or ready.
    """

    @classmethod
    def _greedy_decision(
        cls,
        *,
        held_obj,
        onion_delta,
        dish_delta,
        serving_delta,
        empty_counter_delta,
        orientation_idx,
        wall_flags,
        ready_pot,
        cookable_pot,
        start_cooking_pot,
        ingredient_pot,
    ) -> dict:
        # Edge case: holding soup — serve it to free hands.
        if held_obj == "soup":
            return {
                "action": cls._action_toward(serving_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                "target_delta": serving_delta,
                "target_kind": "serving",
            }
        if held_obj in {"onion", "tomato"}:
            if ingredient_pot is not None:
                return {
                    "action": cls._action_toward(ingredient_pot["delta"], orientation_idx, wall_flags, interact_when_adjacent=True),
                    "target_delta": ingredient_pot["delta"],
                    "target_kind": "pot",
                }
            if empty_counter_delta != (0, 0):
                return {
                    "action": cls._action_toward(empty_counter_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                    "target_delta": empty_counter_delta,
                    "target_kind": "counter",
                }
            return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}
        if ingredient_pot is not None:
            return {
                "action": cls._action_toward(onion_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                "target_delta": onion_delta,
                "target_kind": "onion",
            }
        # Pot is full/cooking/ready — nothing left for this role.
        return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}


class SoupServerPolicy(_NonBlockingEvaluationPolicy):
    """Service specialist: fetches dish and delivers soup only.

    Ignores onion fetching and cooking start entirely.
    STAYs until the pot becomes cookable.
    """

    @classmethod
    def _greedy_decision(
        cls,
        *,
        held_obj,
        onion_delta,
        dish_delta,
        serving_delta,
        empty_counter_delta,
        orientation_idx,
        wall_flags,
        ready_pot,
        cookable_pot,
        start_cooking_pot,
        ingredient_pot,
    ) -> dict:
        if held_obj == "soup":
            return {
                "action": cls._action_toward(serving_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                "target_delta": serving_delta,
                "target_kind": "serving",
            }
        if held_obj == "dish":
            target_pot = ready_pot or cookable_pot
            if target_pot is None:
                return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}
            return {
                "action": cls._action_toward(
                    target_pot["delta"], orientation_idx, wall_flags,
                    interact_when_adjacent=ready_pot is not None,
                ),
                "target_delta": target_pot["delta"],
                "target_kind": "pot",
            }
        if ready_pot is not None or cookable_pot is not None:
            return {
                "action": cls._action_toward(dish_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                "target_delta": dish_delta,
                "target_kind": "dish",
            }
        return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}


class CookingManagerPolicy(_NonBlockingEvaluationPolicy):
    """Production manager: fills onions AND starts cooking; never touches dish or serving station.

    STAYs once the pot is cooking or ready, waiting for the next production cycle.
    """

    @classmethod
    def _greedy_decision(
        cls,
        *,
        held_obj,
        onion_delta,
        dish_delta,
        serving_delta,
        empty_counter_delta,
        orientation_idx,
        wall_flags,
        ready_pot,
        cookable_pot,
        start_cooking_pot,
        ingredient_pot,
    ) -> dict:
        # Edge case: holding soup — serve it to free hands.
        if held_obj == "soup":
            return {
                "action": cls._action_toward(serving_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                "target_delta": serving_delta,
                "target_kind": "serving",
            }
        if held_obj in {"onion", "tomato"}:
            if ingredient_pot is not None:
                return {
                    "action": cls._action_toward(ingredient_pot["delta"], orientation_idx, wall_flags, interact_when_adjacent=True),
                    "target_delta": ingredient_pot["delta"],
                    "target_kind": "pot",
                }
            if empty_counter_delta != (0, 0):
                return {
                    "action": cls._action_toward(empty_counter_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                    "target_delta": empty_counter_delta,
                    "target_kind": "counter",
                }
            return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}
        if start_cooking_pot is not None:
            return {
                "action": cls._action_toward(start_cooking_pot["delta"], orientation_idx, wall_flags, interact_when_adjacent=True),
                "target_delta": start_cooking_pot["delta"],
                "target_kind": "pot",
            }
        if ingredient_pot is not None:
            return {
                "action": cls._action_toward(onion_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                "target_delta": onion_delta,
                "target_kind": "onion",
            }
        # Pot is cooking or ready — nothing left for this role.
        return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}


class PatienceServerPolicy(_NonBlockingEvaluationPolicy):
    """Conservative server: only fetches dish once the pot is fully ready (not just cooking).

    While waiting for soup to finish, drifts toward the pot workspace, creating congestion
    near the pot. The cooperator must learn to route around this parked partner.
    """

    @classmethod
    def _greedy_decision(
        cls,
        *,
        held_obj,
        onion_delta,
        dish_delta,
        serving_delta,
        empty_counter_delta,
        orientation_idx,
        wall_flags,
        ready_pot,
        cookable_pot,
        start_cooking_pot,
        ingredient_pot,
    ) -> dict:
        if held_obj == "soup":
            return {
                "action": cls._action_toward(serving_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                "target_delta": serving_delta,
                "target_kind": "serving",
            }
        if held_obj == "dish":
            if ready_pot is not None:
                return {
                    "action": cls._action_toward(ready_pot["delta"], orientation_idx, wall_flags, interact_when_adjacent=True),
                    "target_delta": ready_pot["delta"],
                    "target_kind": "pot",
                }
            # Pot not ready yet — drift toward it without interacting.
            if cookable_pot is not None:
                return {
                    "action": cls._action_toward(cookable_pot["delta"], orientation_idx, wall_flags, interact_when_adjacent=False),
                    "target_delta": cookable_pot["delta"],
                    "target_kind": "pot",
                }
            return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}
        # Only fetch dish when soup is fully ready.
        if ready_pot is not None:
            return {
                "action": cls._action_toward(dish_delta, orientation_idx, wall_flags, interact_when_adjacent=True),
                "target_delta": dish_delta,
                "target_kind": "dish",
            }
        # Pot is cooking but not ready — drift toward it to wait nearby.
        if cookable_pot is not None:
            return {
                "action": cls._action_toward(cookable_pot["delta"], orientation_idx, wall_flags, interact_when_adjacent=False),
                "target_delta": cookable_pot["delta"],
                "target_kind": "pot",
            }
        return {"action": _STAY_IDX, "target_delta": (0, 0), "target_kind": None}


EVALUATION_BEHAVIOR_SET: list[dict] = [
    {
        "name": "ingredient_filler",
        "description": "Ingredient specialist: fetches onions and fills the pot only; ignores dish pickup, cooking start, and serving entirely.",
        "policy_cls": IngredientFillerPolicy,
    },
    {
        "name": "soup_server",
        "description": "Service specialist: fetches dish and delivers soup only; ignores onion fetching and cooking start entirely.",
        "policy_cls": SoupServerPolicy,
    },
    {
        "name": "cooking_manager",
        "description": "Production manager: fills onions and starts cooking; never fetches a dish or visits the serving station.",
        "policy_cls": CookingManagerPolicy,
    },
    {
        "name": "patience_server",
        "description": "Conservative server: only fetches a dish once the pot is fully ready; otherwise drifts to the pot workspace and waits, creating congestion near the pot.",
        "policy_cls": PatienceServerPolicy,
    },
]


def get_evaluation_scripted_policies() -> list[Callable]:
    """Return scripted policy constructors for evaluation behavior sweeps."""
    return [entry["policy_cls"] for entry in EVALUATION_BEHAVIOR_SET]


# ---------------------------------------------------------------------------- #
#                              Make env function                               #
# ---------------------------------------------------------------------------- #


def create_env(
    layout_name=None,
    horizon=None,
    num_envs=1,
    continuous_actions=False,
    seed=None,
    device="cpu",
    randomize_start_pos=True,
    **kwargs,
):
    """Create an OvercookedBenchWrapper via OvercookedTask.

    Args:
        layout_name: Optional Overcooked layout override. If omitted, use `rl_cfg/overcooked.yaml`.
        horizon: Optional episode-length override. If omitted, use `rl_cfg/overcooked.yaml`.
        num_envs: Passed to get_env_fun (ignored for non-vectorized env).
        seed: Optional random seed.
        device: Torch device string.
        randomize_start_pos: If True, agent start positions are sampled randomly each episode.

    Returns:
        An OvercookedBenchWrapper instance.
    """
    task = OvercookedTask.OVERCOOKED.get_from_yaml(
        path=str(RL_CFG_DIR / "overcooked.yaml")
    )
    group_map, _ = _build_overcooked_group_map()
    task.config["group_map"] = group_map
    # Allow callers to override layout/horizon
    if layout_name is not None:
        task.config["layout_name"] = layout_name
    if horizon is not None:
        task.config["horizon"] = horizon
    task.config["randomize_start_pos"] = randomize_start_pos
    return task.get_env_fun(
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
    )()
