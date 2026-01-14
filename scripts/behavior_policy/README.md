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
- **Evaluation**: Vision-language model evaluation of behavior alignment using images and states
- **Iterative Refinement**: Automatic retry with feedback for misaligned behaviors
- **Caching**: Save and reuse generated behaviors across runs

## Configuration

The pipeline uses Hydra for configuration management. Configuration file is located in `cfg/config.yaml`:

```yaml
# Environment selection
env: football

# Behavior generation strategy
strategy: stratified  # Options: stratified, in_context

# LLM model selection (litellm format)
llm_model: openai/gpt-4o

# Pipeline settings
max_behaviors: 5
max_alignment_retries: 3
n_eval_rollouts: 1
n_workers: 1  # Number of parallel workers

# Output directory
output_dir: ./results
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `env` | Environment name (must exist in `popgen/envs/`) |
| `strategy` | Behavior generation strategy (`in_context` or `stratified`) |
| `llm_model` | Model identifier in litellm format (e.g., `openai/gpt-4o`, `anthropic/claude-3-opus`) |
| `max_alignment_retries` | Maximum attempts to align a policy with its target behavior |
| `n_eval_rollouts` | Number of rollouts to run for evaluation |
| `n_workers` | Number of parallel workers for behavior processing (default: 1) |
| `output_dir` | Root directory for pipeline outputs |

## Usage

### Basic Usage

```bash
# Run with default configuration
python main.py

# Override configuration parameters
python main.py env=point_mass_rendezvous strategy=in_context

# Change model
python main.py llm_model=anthropic/claude-3-opus

# Increase retry limits
python main.py max_alignment_retries=5
```

### Environment Requirements

Each environment module in `popgen/envs/<env_name>/` must provide:

| Export | Description |
|--------|-------------|
| `ENVIRONMENT_DESCRIPTION` | Text description of the environment and accessible variables |
| `OBSERVATION_FORMAT` | Format specification for agent observations |
| `EGO_VISUAL_DESCRIPTION` | Visual description of the ego agent for evaluation |
| `rollout_fn` | Function that executes policies and returns `(states, images, error)` |

See [Adding a New Environment](../../docs/add_env.md) for detailed instructions.

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
2. **Run Rollouts**: Execute policies in simulation (configurable number of rollouts)
3. **Evaluate**: Vision-language model assesses behavior alignment using images and states
4. **Refine**: If misaligned, retry with feedback

**Outputs per behavior:**
```
behavior_<idx>/
  attempt_<n>/
    policy.py             # Generated policy code
    evaluation.json       # Evaluation results with reasoning
    rollouts/
      rollout_0/
        rollout_data.jsonl  # Sampled trajectory states
        rollout.gif         # Visualization
      rollout_1/
        ...
```

## Output Structure

```
results/
  <env>__<strategy>__<model>/
    behaviors.json
    behavior_0/
      attempt_0/
        policy.py
        evaluation.json
        rollouts/
          rollout_0/
          ...
      attempt_1/
        ...
    behavior_1/
      ...
```

## Log Format

Logs use a structured prefix format for tracking progress:

```
[B=0 | A=1] Message
 │     └─ Alignment attempt index
 └─ Behavior index
```

**Example:**
```
[B=2 | A=0] Generating policy...
[B=2 | A=0] Running rollout (1/3)...
[B=2 | A=0] Evaluating rollout...
[B=2 | A=0] SUCCESS – Alignment succeeded on attempt 1
```

## Error Handling

The pipeline includes robust error handling:

1. **Rollout Failures**: Captured and reported in evaluation results
2. **Alignment Failures**: Retries policy generation with evaluation reasoning as feedback
3. **Generation Failures**: Graceful degradation with informative error messages

## Examples

### Generate behaviors for point mass environment

```bash
python main.py env=point_mass_rendezvous strategy=in_context
```

### Use stratified strategy with GPT-4o

```bash
python main.py strategy=stratified llm_model=openai/gpt-4o
```

### Run multiple evaluation rollouts

```bash
python main.py n_eval_rollouts=3 max_alignment_retries=5
```

### Run with parallel workers

```bash
python main.py n_workers=4
```

## Extending

### Adding a New Environment

See the detailed guide: [Adding a New Environment](../../docs/add_env.md)

Quick steps:
1. Create `popgen/envs/<env_name>/` folder
2. Implement `prompts.py` with `ENVIRONMENT_DESCRIPTION`, `OBSERVATION_FORMAT`, `EGO_VISUAL_DESCRIPTION`
3. Implement `rollout.py` with `rollout_fn()`
4. Create `__init__.py` exporting all components

### Adding a New Generation Strategy

1. Implement strategy class inheriting from base strategy
2. Override `generate()` method
3. Register in `main.py` strategy selection

## Troubleshooting

**Issue**: Behaviors file not regenerating
- **Solution**: Delete `results/<run_id>/behaviors.json` to force regeneration

**Issue**: LLM API errors
- **Solution**: Check `.env` file for valid API keys (e.g., `OPENAI_API_KEY`)

**Issue**: Alignment failures persist
- **Solution**: Increase `max_alignment_retries` or simplify behavior descriptions

**Issue**: Environment not found
- **Solution**: Ensure the environment exists in `popgen/envs/` and has proper `__init__.py`
