# Point Mass Rendezvous (PMR)

## Description
In this scenario, agents must coordinate to navigate to the same landmark simultaneously. There are multiple landmarks available, representing different potential solutions. The agents must agree on which landmark to go to without explicit communication (unless added).

## Modes
The environment supports two modes for landmark distribution:
- **PMR-C (Circle)**: Landmarks are distributed evenly on the circumference of a circle with radius 1.0. All landmarks are equidistant from the origin (where agents spawn), making them equally optimal in terms of distance.
- **PMR-L (Line)**: Landmarks are placed on a line along the x-axis at intervals of 0.5. Closer landmarks are easier to reach.

## Arguments
- `n_agents` (int, default: 2): Number of agents.
- `n_landmarks` (int, default: 4): Number of landmarks.
- `mode` (str, default: "PMR-C"): Distribution mode, either "PMR-C" or "PMR-L".
- `landmark_radius` (float, default: 0.05): Radius of the landmarks.
- `agent_radius` (float, default: 0.05): Radius of the agents.
- `shared_reward` (bool, default: True): Whether reward is shared among agents.

## Observation Space
Each agent observes:
1.  Its own position (2 dims).
2.  Its own velocity (2 dims).
3.  Relative position to each landmark (2 * `n_landmarks` dims).
4.  Relative position to every other agent (2 * (`n_agents` - 1) dims).

## Action Space
Continuous action space controlling the agent's force/velocity in 2D.

## Reward Function
- The reward is **sparse** and **shared**.
- Agents receive a reward of `+1.0` if **all** agents are simultaneously within the radius of the **same** landmark.
- Otherwise, the reward is `0.0`.

## Termination (Done)
The episode terminates when any landmark is successfully captured (i.e., all agents are on it).
