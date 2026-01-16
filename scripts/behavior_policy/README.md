# Behavior Generation Pipeline (Policy Code)

A multi-stage pipeline for generating, evaluating, and aligning cooperative multi-agent behaviors using LLMs.

## Overview

This pipeline generates diverse behaviors for a multi-agent environment, synthesizes Python policies with an LLM, runs rollouts, and evaluates alignment between intended and observed behavior. Misaligned policies are iteratively refined with feedback until they pass or the retry budget is exhausted.

## Features

- **Behavior Generation**: Two strategies
  - `in_context`: Context-based behavior generation
  - `stratified`: Stratified sampling across behavior dimensions
- **Policy Synthesis**: LLM-generated policy code for cooperative agents
- **Automated Rollouts**: Execute policies and capture trajectories
- **Evaluation**: Vision-language model evaluation using images and states
- **Iterative Refinement**: Retry with feedback for misaligned behaviors
- **Caching**: Save and reuse generated behaviors across runs

## Quickstart

From the repo root:

```bash
# Run with default configuration
python scripts/behavior_policy/main.py

# Override configuration parameters
python scripts/behavior_policy/main.py env=point_mass_rendezvous strategy=in_context

# Change model
python scripts/behavior_policy/main.py llm_model=anthropic/claude-3-opus

# Increase retry limits
python scripts/behavior_policy/main.py max_alignment_retries=5
```

## Configuration

The pipeline uses Hydra for configuration management. Default configuration is in `scripts/behavior_policy/cfg/config.yaml`:

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

## Environment Requirements

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

Generates a diverse set of cooperative behaviors from the environment description.

**Outputs:**
- `behaviors.json`: List of generated behaviors with names and descriptions
- Cached and reused if the file exists

**Example Behavior:**
```json
{
  "name": "Coordinated Approach",
  "description": "Both agents move directly toward the target while maintaining awareness of each other's position to avoid collisions."
}
```

### Stage 2: Policy Generation and Evaluation

For each behavior, iteratively:
1. **Generate Policy**: LLM generates Python policy code
2. **Run Rollouts**: Execute policies in simulation (configurable number of rollouts)
3. **Evaluate**: Vision-language model assesses alignment using images and states
4. **Refine**: If misaligned, retry with feedback

**Outputs per behavior:**
```
behavior_<idx>/
  attempt_<n>/
    policy.py              # Generated policy code
    evaluation.json        # Evaluation results with reasoning
    rollouts/
      rollout_0/
        rollout_data.jsonl  # Sampled trajectory states
        rollout.gif         # Visualization
      rollout_1/
        ...
```

## Output Structure

```
<output_dir>/
  <run_name>/
    summary.json
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

### Output Files (Examples)

- `summary.json`: Run-level rollup (counts, latencies, per-attempt success map).
  - Example fields: `n_behaviors`, `n_succeeded`, `n_attempts`, `avg_latencies`, `total_latencies`, `behavior_steps_to_success`, `attempt_summaries`.
- `evaluation.json`: Per-attempt evaluation metrics and reasoning from the judge model.
  - Example fields: `instance_metrics`, `overall_metrics`, `metadata`.
- `rollout_data.jsonl`: One JSON object per step with state features (positions, velocities, ball/goal info, step index).
  - Example keys: `blue_agent0_pos`, `blue_agent0_vel`, `red_agent0_pos`, `ball_pos`, `goal_pos`, `step`.

## Error Handling

The pipeline includes robust error handling:

1. **Rollout Failures**: Captured and reported in evaluation results
2. **Alignment Failures**: Retries policy generation with evaluation reasoning as feedback
3. **Generation Failures**: Graceful degradation with informative error messages

## Examples

```bash
python scripts/behavior_policy/main.py env=point_mass_rendezvous strategy=in_context
python scripts/behavior_policy/main.py strategy=stratified llm_model=openai/gpt-4o
python scripts/behavior_policy/main.py n_eval_rollouts=3 max_alignment_retries=5
python scripts/behavior_policy/main.py n_workers=4
```

## Troubleshooting

**Issue**: Behaviors file not regenerating
- **Solution**: Delete `results/<run_id>/behaviors.json` to force regeneration

**Issue**: LLM API errors
- **Solution**: Check `.env` for valid API keys (e.g., `OPENAI_API_KEY`)

**Issue**: Alignment failures persist
- **Solution**: Increase `max_alignment_retries` or simplify behavior descriptions

**Issue**: Environment not found
- **Solution**: Ensure the environment exists in `popgen/envs/` and has proper `__init__.py`
