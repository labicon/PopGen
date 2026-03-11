from typing import Any

import litellm

from popgen.utils.config import LitellmConfig

FEEDBACK_SYSTEM_PROMPT = """You are an advisor for a code generation pipeline.
Your task is to diagnose why a generated code fails to represent its target behavior in rollouts and provide actionable feedback to improve the next code attempt.

You will be given:
- The intended behavior
- The environment description
- The code attempt (type: {generation_type})
- Evaluation results from multiple rollout executions.

Focus on the most important failure patterns and give specific, actionable suggestions to fix the code.
"""

FEEDBACK_PROMPT_TEMPLATE = """Code generation type: {generation_type}

Inputs
--------------

## Behavior
{behavior}

## Environment Description
{environment_description}

## Code Attempt
```{code_attempt}
```

## Rollout Evaluation Results
{metrics_str}

--------------

Analyze the results and provide feedback for the next code attempt.
"""


def gather_feedback(
    instance_metrics: list[dict[str, Any]],
    code_attempt: str,
    behavior: str,
    environment_description: str,
    generation_type: str,
    advisor_config: LitellmConfig | None = None,
    negative_only: bool = False,
) -> str:
    """Gather feedback for the next attempt, optionally using an advisor model."""
    if negative_only:
        instance_metrics = [m for m in instance_metrics if m["adherence"] < 1.0]

    metrics_str = "\n".join(f"- [Rollout #{i}] Success={m['adherence'] == 1}\n  Reason: {m['reasoning']}\n" for i, m in enumerate(instance_metrics))

    if advisor_config is None:
        return "Feedback from rollouts:\n" + metrics_str

    if generation_type == "PolicyCode":
        generation_type_str = "Policy Code: Generate code that implements the policy that satisfies the behavior."
    elif generation_type == "RewardFn":
        generation_type_str = "Reward Function: Generate a reward function that is used to train a policy to satisfy the behavior."
    else:
        raise ValueError(f"Unsupported generation type: {generation_type}")

    system_prompt = FEEDBACK_SYSTEM_PROMPT.format(generation_type=generation_type_str)
    prompt = FEEDBACK_PROMPT_TEMPLATE.format(
        generation_type=generation_type_str,
        behavior=behavior,
        environment_description=environment_description,
        code_attempt=code_attempt,
        metrics_str=metrics_str,
    )

    response = litellm.completion(
        model=advisor_config.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        **advisor_config.get_completion_kwargs(),
    )
    return response.choices[0].message.content.strip()
