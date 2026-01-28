# Behavior Generation Pipeline (Reward Function)

A multi-stage pipeline for generating, evaluating, and aligning cooperative multi-agent behaviors using Large Language Models (LLMs) to synthesize reward functions.

## TODO
Match format with behavior_policy experiment. This include redesign of pipeline.py for RL training, update of main.py

## Overview

This pipeline automatically generates diverse cooperative behaviors for multi-agent environments, synthesizes corresponding **reward functions** using LLMs, trains agents using Reinforcement Learning (RL), executes rollouts, and evaluates whether the trained policies align with the intended behaviors. The system iteratively refines reward functions until the trained agents successfully demonstrate the target behaviors.

## Features

- **Behavior Generation**: Two strategies for generating diverse cooperative behaviors
  - `in_context`: Context-based behavior generation
  - `stratified`: Stratified sampling across behavior dimensions
- **Reward Generation**: LLM-powered reward function generation for cooperative agents
- **RL Training**: Train agents using BenchMARL (MAPPO) with generated rewards
- **Automated Rollout**: Execute trained policies in simulation and capture trajectories
- **Evaluation**: Vision-language model evaluation of behavior alignment
- **Iterative Refinement**: Automatic retry with error feedback for both training/rollout failures and misaligned behaviors
- **Caching**: Save and reuse generated behaviors and checkpoints across runs

## Configuration

The pipeline uses Hydra for configuration management. Configuration files are located in `conf/`:

```yaml
# conf/config.yaml
strategy: "in_context"           # or "stratified"
llm_model: "gpt-4"               # LLM model identifier
max_alignment_retries: 3         # Max attempts to align policy with behavior
max_rollout_retries: 3           # Max attempts to fix rollout failures
output_dir: "./outputs"          # Base directory for outputs

env:
  name: "point_mass_rendezvous"
  env_path: "popgen.envs.point_mass_rendezvous.env"
```

### Configuration Parameters

- `strategy`: Behavior generation strategy (`in_context` or `stratified`)
- `llm_model`: Model identifier for LLM API calls
- `max_alignment_retries`: Maximum attempts to align a policy with its target behavior
- `max_rollout_retries`: Maximum attempts to fix syntax/runtime errors in generated policy
- `output_dir`: Root directory for pipeline outputs
- `env.name`: Environment identifier
- `env.env_path`: Python import path to environment module

## Usage

### Basic Usage

```bash
# Run with default configuration
python main.py

# Override configuration parameters
python main.py strategy=stratified llm_model=gpt-4 max_alignment_retries=5

# Use a different config file
python main.py --config-name=my_config
```

### Environment Requirements

Each environment module must provide:
- `ENVIRONMENT_DESCRIPTION`: Text description of the environment
- `OBSERVATION_FORMAT`: Format specification for agent observations
- `rollout_behavior`: Function that executes policies and returns trajectory data

## Pipeline Stages

### Stage 1: Behavior Generation

Generates a diverse set of cooperative behaviors based on the environment description.

**Outputs:**
- `behaviors.json`: List of generated behaviors with names and descriptions
- Cached and reused if file exists

**Example Behavior:**
```json
{
  "name": "Coordinated Approach",
  "description": "Both agents move directly toward the target while maintaining awareness of each other's position to avoid collisions."
}
```

### Stage 2: Reward Generation & Evaluation

For each behavior, iteratively:
1. **Generate Reward**: LLM generates Python code for `compute_reward(agent, world)`
2. **Train Agents**: Train policies using the generated reward function (BenchMARL/MAPPO)
3. **Run Rollout**: Execute trained policies in simulation
4. **Evaluate**: Vision-language model assesses behavior alignment
5. **Refine**: If misaligned, retry with feedback

**Outputs per behavior:**
```
behavior_<idx>/
  attempt_<n>/
    run_<k>/
      reward.py           # Generated reward function
      original_config/    # BenchMARL Configs
      checkpoints/        # RL Checkpoints
      rollout.json        # Trajectory data
      rollout.gif         # Visualization
      states/             # Individual state images
    evaluation.json       # Evaluation results
```

## Output Structure

```
outputs/
  <env_name>__<strategy>__<model>/
    behaviors.json
    behavior_0/
      attempt_0/
        rollout_0/
        rollout_1/
        ...
        evaluation.json
      attempt_1/
        ...
    behavior_1/
      ...
```

## Log Format

Logs use a structured prefix format for tracking progress:

```
[B=0 | A=1 | R=2] Message
 │     │     └─ Rollout attempt
 │     └─ Alignment attempt
 └─ Behavior index
```

**Example:**
```
[B=2 | A=0 | R=1] Generating policy...
[B=2 | A=0 | R=1] Running rollout...
[B=2 | A=0] Evaluation PASSED
```

## Error Handling

The pipeline includes robust error handling:

1. **Rollout Failures**: Automatically retries with error feedback to LLM
2. **Alignment Failures**: Retries policy generation with evaluation reasoning
3. **Generation Failures**: Graceful degradation with informative error messages

## Examples

### Generate behaviors for point mass environment

```bash
python main.py env.name=point_mass_rendezvous strategy=in_context
```

### Use stratified strategy with Claude

```bash
python main.py strategy=stratified llm_model=claude-3-opus-20240229
```

### Increase retry limits for complex behaviors

```bash
python main.py max_alignment_retries=5 max_rollout_retries=5
```

## Extending

### Adding a New Environment

1. Create environment module in `popgen/envs/<env_name>/`
2. Implement required exports:
   - `ENVIRONMENT_DESCRIPTION`
   - `OBSERVATION_FORMAT`
   - `rollout_behavior(policy_fn)`
3. Add configuration in `conf/`

### Adding a New Generation Strategy

1. Implement strategy class inheriting from base strategy
2. Override `generate()` method
3. Register in `main.py` strategy selection

## Troubleshooting

**Issue**: Behaviors file not regenerating
- **Solution**: Delete `outputs/<run_id>/behaviors.json` to force regeneration

**Issue**: LLM API errors
- **Solution**: Check `.env` file for valid API keys and model availability

**Issue**: Rollout failures persist
- **Solution**: Increase `max_rollout_retries` or simplify behavior descriptions
