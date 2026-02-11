# LLM Usage in PopGen

This document explains how Large Language Models (LLMs) are used throughout the PopGen framework, including configuration, supported providers, and usage patterns.

## Overview

PopGen uses [LiteLLM](https://github.com/BerriAI/litellm) as a unified interface for calling 100+ LLM providers (OpenAI, Anthropic, Google, Cohere, local models via vLLM, etc.). This allows seamless switching between different models and providers without changing your code.

## LiteLLM Basics

LiteLLM provides a unified API that translates calls to different LLM providers:

```python
import litellm

# OpenAI
response = litellm.completion(model="openai/gpt-4o", messages=[...])

# Anthropic
response = litellm.completion(model="anthropic/claude-3-5-sonnet-20241022", messages=[...])

# Local vLLM server
response = litellm.completion(
    model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    messages=[...]
)
```

## Configuration with `LitellmConfig`

PopGen centralizes all LLM configuration through the `LitellmConfig` class located in `popgen/utils/config.py`.

### Basic Configuration

```python
from popgen.utils.config import LitellmConfig

# Simple configuration with just model name
config = LitellmConfig(model_name="openai/gpt-4o-mini")

# Advanced configuration
config = LitellmConfig(
    model_name="anthropic/claude-3-5-sonnet-20241022",
    api_base=None,  # Optional custom API base URL
    generation_kwargs={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2000
    }
)
```

### Configuration Parameters

- **`model_name`** (str, required): Model identifier following LiteLLM's naming convention
  - OpenAI: `"openai/gpt-4o"`, `"openai/gpt-4o-mini"`
  - Anthropic: `"anthropic/claude-3-5-sonnet-20241022"`
  - Local vLLM: `"hosted_vllm/model-name"`

- **`api_base`** (str, optional): Custom API base URL for self-hosted models
  - Automatically set to `"http://localhost:8000/v1"` for `hosted_vllm` models if not specified

- **`generation_kwargs`** (dict, optional): Additional parameters passed to `litellm.completion()`
  - Common options: `temperature`, `top_p`, `max_tokens`, `response_format`

### Using the Configuration

The `LitellmConfig` provides a helper method to get all completion kwargs:

```python
config = LitellmConfig(
    model_name="openai/gpt-4o",
    generation_kwargs={"temperature": 0.7}
)

# Get kwargs for litellm.completion()
kwargs = config.get_completion_kwargs()
# Returns: {"temperature": 0.7}

# Override with additional kwargs
kwargs = config.get_completion_kwargs(overrides={"max_tokens": 500})
# Returns: {"temperature": 0.7, "max_tokens": 500}
```

## LLM Usage in PopGen

PopGen uses LLMs in two main areas: **Generators** and **Evaluators**.

### 1. Generators

Generators use LLMs to create diverse content (behaviors, policies, rewards, trajectories).

#### Code Generators

Generate executable Python code for policies or reward functions:

```python
from popgen.population.generation import PolicyCodeGenerator, RewardFnGenerator
from popgen.utils.config import LitellmConfig

# Create configuration
config = LitellmConfig(model_name="openai/gpt-4o")

# Policy generator
policy_gen = PolicyCodeGenerator(model_config=config)
policy = policy_gen.generate(
    behavior="Move aggressively toward the goal",
    observation_format="obs is a tensor of shape (10,)",
    action_format="action is a tensor of shape (2,)"
)
print(policy.code)  # Generated Python code
print(policy.reasoning)  # LLM's reasoning

# Reward generator
reward_gen = RewardFnGenerator(model_config=config)
reward = reward_gen.generate(
    behavior="Stay close to teammates",
    environment_description="Multi-agent navigation task",
    observation_format="...",
    reward_format="..."
)
```

#### Trajectory Generator

Generate complete action sequences:

```python
from popgen.population.generation import TrajectoryGenerator

config = LitellmConfig(model_name="openai/gpt-4o-mini")
traj_gen = TrajectoryGenerator(model_config=config)

trajectory = traj_gen.generate(
    behavior="Circle around the target",
    environment_description="2D navigation",
    action_description="2D force vector",
    action_type="Continuous",
    rollout_length=100,
    initial_observation={"position": [0, 0], "velocity": [0, 0]}
)
print(trajectory.actions)  # List of actions
print(trajectory.reasoning)  # High-level strategy
```

#### Diversity Strategies

Generate diverse sets of behaviors:

```python
from popgen.population.generation.diversity import (
    InContextStrategy,
    StratifiedStrategy,
    VerbalizedSamplingStrategy
)
from popgen.population.types import Behavior

config = LitellmConfig(
    model_name="openai/gpt-4o",
    generation_kwargs={"temperature": 0.8}  # Higher temp for diversity
)

# In-context learning strategy
in_context = InContextStrategy(
    model_config=config,
    output_type=Behavior
)
behaviors = in_context.generate(
    input_prompt="Generate diverse soccer behaviors",
    num_items=10
)

# Stratified strategy (property-based partitioning)
stratified = StratifiedStrategy(
    model_config=config,
    output_type=Behavior,
    candidate_upsampling=1.5,
    num_workers=4
)
behaviors = stratified.generate(
    input_prompt="Generate diverse soccer behaviors",
    num_items=16
)

# Verbalized sampling (probability-aware generation)
verbalized = VerbalizedSamplingStrategy(
    model_config=config,
    output_type=Behavior,
    probability_tuning=0.2,  # Limit high-probability responses
    do_cot=True  # Include chain-of-thought reasoning
)
behaviors = verbalized.generate(
    input_prompt="Generate diverse soccer behaviors",
    num_items=8
)
```

### 2. Evaluators

Evaluators use LLMs as judges to assess quality, adherence, and diversity.

#### Behavior Quality Evaluator

Evaluates behaviors across three rubrics:

```python
from popgen.population.evaluation import BehaviorQualityEvaluator

config = LitellmConfig(model_name="openai/gpt-4o")
evaluator = BehaviorQualityEvaluator(model_config=config, num_workers=8)

result = evaluator.evaluate(
    inputs=[
        {
            "behavior": "Aggressive forward: constantly push toward goal",
            "environment_description": "Soccer simulation with 11v11 agents"
        }
    ]
)

print(result.overall_metrics)
# {
#     "relevance": {"mean": 4.5, "std": 0.2, ...},
#     "realism": {"mean": 4.0, "std": 0.3, ...},
#     "specificity": {"mean": 3.8, "std": 0.4, ...}
# }
```

#### Behavior Adherence Evaluator

Evaluates if agents follow their intended behaviors:

```python
from popgen.population.evaluation import BehaviorAdherenceEvaluator

config = LitellmConfig(model_name="openai/gpt-4o")  # Vision model for images
evaluator = BehaviorAdherenceEvaluator(model_config=config, num_workers=8)

result = evaluator.evaluate(
    inputs=[
        {
            "behavior": "Stay in defensive position",
            "environment_description": "...",  # ENVIRONMENT_DESCRIPTION from env module
            "rollout_states": [...],  # List of state dicts
            "rollout_images": [...]   # Optional: list of image arrays
        }
    ]
)

print(result.overall_metrics["adherence"])
# {"mean": 0.85, "std": 0.12, ...}
```

#### Diversity Discriminator

Evaluates diversity between texts:

```python
from popgen.population.evaluation import DiversityDiscriminator

config = LitellmConfig(model_name="openai/gpt-4o-mini")
evaluator = DiversityDiscriminator(model_config=config, num_workers=4)

behaviors = [
    "Aggressive forward player",
    "Defensive midfielder",
    "Ball-hogging striker",
    # ...
]

result = evaluator.evaluate(inputs=behaviors)
print(result.overall_metrics["diversity_score"])
# {"mean": 3.2, "std": 0.8, "min": 1, "max": 5}
```

## Robust Response Parsing

PopGen includes robust parsing utilities in `popgen/utils/parsing.py` that handle various LLM response formats:

```python
from popgen.utils.parsing import parse_llm_response

# Tries multiple strategies:
# 1. Direct JSON validation
# 2. Extract from markdown code blocks: ```json {...} ```
# 3. Find any JSON object in the content

result = parse_llm_response(
    response=llm_response,
    model_class=CodeSnippet,
    postprocess_fn=lambda x: {"code": clean_code(x.code)}
)
```

## Model Selection Guide

### For Generation Tasks

- **High Quality**: `"openai/gpt-4o"`, `"anthropic/claude-3-5-sonnet-20241022"`
- **Fast & Cheap**: `"openai/gpt-4o-mini"`, `"anthropic/claude-3-5-haiku-20241022"`
- **Local/Self-hosted**: `"hosted_vllm/meta-llama/Llama-3.1-70B-Instruct"`

### For Evaluation Tasks

- **Vision Required**: `"openai/gpt-4o"`, `"anthropic/claude-3-5-sonnet-20241022"`
- **Text Only**: `"openai/gpt-4o-mini"` (cheaper, faster)

### Temperature Guidelines

- **Code/Structured Output**: 0.0 - 0.3 (deterministic, focused)
- **Diverse Behaviors**: 0.7 - 1.0 (creative, varied)
- **Evaluation**: 0.0 - 0.2 (consistent, reliable)

## Configuration in Scripts

### Hydra-based Scripts

Most scripts use Hydra for configuration. Define models in YAML:

```yaml
# config.yaml
llm_model: "openai/gpt-4o-mini"
judge_llm_model: "openai/gpt-4o"

llm_temperature: 0.7
llm_top_p: 0.9
```

Then in your script:

```python
from popgen.utils.config import LitellmConfig

# For generation
gen_config = LitellmConfig(
    model_name=cfg.llm_model,
    generation_kwargs={
        "temperature": cfg.get("llm_temperature", 0.7),
        "top_p": cfg.get("llm_top_p", 0.9)
    }
)
generator = PolicyCodeGenerator(model_config=gen_config)

# For evaluation (separate config)
eval_config = LitellmConfig(model_name=cfg.judge_llm_model)
evaluator = BehaviorAdherenceEvaluator(model_config=eval_config)
```

### Environment Variables

LiteLLM automatically reads API keys from environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

## Using Local Models with vLLM

### 1. Start vLLM Server

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4
```

### 2. Configure PopGen

```python
config = LitellmConfig(
    model_name="hosted_vllm/meta-llama/Llama-3.1-70B-Instruct",
    api_base="http://localhost:8000/v1",
    generation_kwargs={"temperature": 0.7}
)
```

Note: `api_base` is automatically set to `"http://localhost:8000/v1"` for `hosted_vllm` models if not specified.

## Best Practices

1. **Reuse Configs**: Create one `LitellmConfig` and reuse it across multiple components:
   ```python
   config = LitellmConfig(model_name="openai/gpt-4o-mini")
   gen = PolicyCodeGenerator(model_config=config)
   eval = BehaviorQualityEvaluator(model_config=config)
   ```

2. **Separate Configs for Different Roles**: Use different models for generation vs evaluation:
   ```python
   gen_config = LitellmConfig(model_name="openai/gpt-4o-mini")  # Cheaper for generation
   eval_config = LitellmConfig(model_name="openai/gpt-4o")      # Better for judging
   ```

3. **Handle Errors Gracefully**: LLM calls can fail. Use try-except or rely on PopGen's built-in error handling:
   ```python
   # PopGen evaluators return error info in results
   result = evaluator.compute_instance_metric(input_data)
   if result["error"] is not None:
       print(f"Evaluation failed: {result['error']}")
   ```

4. **Monitor Costs**: Track your LLM usage, especially when generating large populations:
   ```python
   # Use cheaper models for initial exploration
   config = LitellmConfig(model_name="openai/gpt-4o-mini")

   # Switch to better models for final runs
   # config = LitellmConfig(model_name="openai/gpt-4o")
   ```

## Common Issues

### Issue: "API key not found"
**Solution**: Set the appropriate environment variable:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Issue: "Model not found" with vLLM
**Solution**: Ensure vLLM server is running and the model name matches:
```python
# Check your vLLM server logs for the exact model name
config = LitellmConfig(
    model_name="hosted_vllm/exact-model-name-from-vllm",
    api_base="http://localhost:8000/v1"
)
```

### Issue: Inconsistent outputs
**Solution**: Lower the temperature:
```python
config = LitellmConfig(
    model_name="openai/gpt-4o",
    generation_kwargs={"temperature": 0.0}  # Deterministic
)
```

### Issue: Response parsing fails
**Solution**: PopGen's `parse_llm_response()` has fallback strategies, but you can add custom postprocessing:
```python
result = parse_llm_response(
    response,
    model_class=MyModel,
    postprocess_fn=lambda x: {"field": clean_value(x.field)}
)
```

## References

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Supported LLM Providers](https://docs.litellm.ai/docs/providers)
- [vLLM Documentation](https://docs.vllm.ai/)
- PopGen Config: [`popgen/utils/config.py`](../popgen/utils/config.py)
- PopGen Parsing: [`popgen/utils/parsing.py`](../popgen/utils/parsing.py)
