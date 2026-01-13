# Behavior Generation Pipeline (Policy Code)

A multi-stage pipeline for generating, evaluating, and aligning cooperative multi-agent behaviors using Large Language Models (LLMs).

## Overview

This pipeline automatically generates diverse cooperative behaviors for multi-agent environments, synthesizes corresponding policies using LLMs, executes rollouts, and evaluates whether the policies align with the intended behaviors. The system iteratively refines policies until they successfully demonstrate the target behaviors.

## Features

- **Behavior Generation**: Two strategies for generating diverse cooperative behaviors
  - `in_context`: Context-based behavior generation
  - `stratified`: Stratified sampling across behavior dimensions
- **Policy Synthesis**: LLM-powered policy code generation for cooperative agents
- **Automated Rollout**: Execute policies in simulation and capture trajectories
- **Evaluation**: Vision-language model evaluation of behavior alignment
- **Iterative Refinement**: Automatic retry with error feedback for both rollout failures and misaligned behaviors
- **Caching**: Save and reuse generated behaviors across runs

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

### Stage 2: Policy Generation & Evaluation

For each behavior, iteratively:
1. **Generate Policy**: LLM generates Python code for agent policies
2. **Run Rollout**: Execute policies in simulation
3. **Evaluate**: Vision-language model assesses behavior alignment
4. **Refine**: If misaligned, retry with feedback

**Outputs per behavior:**
```
behavior_<idx>/
  attempt_<n>/
    rollout_<k>/
      policy.py           # Generated policy code
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
   - `rollout_behavior(agent0_policy_fn, agent1_policy_fn)`
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
