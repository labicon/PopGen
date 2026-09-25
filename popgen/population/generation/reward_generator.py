from popgen.population.generation.base import GeneratorWithFeedback
from popgen.population.types import CodeSnippet
from popgen.utils.config import LitellmConfig
from popgen.utils.parsing import clean_code, parse_llm_response

# ---------------------------------- PROMPTS --------------------------------- #

SYSTEM_PROMPT = """You are an expert AI agent specializing in designing reward functions for reinforcement learning in multi-agent coordination tasks.

Your task is to generate a reward function for an ego agent that incentivizes the given behavior.

THE BEHAVIOR DESCRIPTION IS THE TARGET, NOT THE TASK.
The agent you are designing is one member of a population of DIVERSE TEAMMATES. It is not
meant to be a good coordinator, and it is not scored on the environment's own objective --
it is scored on whether an observer watching it play would recognize the behavior
description. A reward that makes every behavior converge on the same competent, cooperative
play has failed, however well the agents then do.

So:
- Every term must be traceable to a specific clause of the behavior description. If you
  cannot say which words in the description a term implements, delete it.
- Do NOT add a term for the environment's own coordination objective or team success
  (agents meeting at the same landmark, the shared task being completed, the team score)
  unless the behavior description itself asks for it. Such a term is the single most common
  way these rewards fail: it dominates learning, and the agent ends up coordinating
  competently while ignoring the behavior it was supposed to display.
- Where the description conflicts with good play, follow the description. A behavior that
  says "always heads for the leftmost landmark regardless of the partner" must be rewarded
  for going left even when the partner is elsewhere, and must NOT be rewarded for ending up
  together. A behavior that is stubborn, lazy, greedy, or actively unhelpful must be
  rewarded for being exactly that.
- Where the description does call for joint outcomes, reward the agent's own described
  conduct (what it does, when, toward whom), not the joint outcome alone -- the joint
  outcome is reachable by behaving in many ways other than the described one.

Some important details to consider:
- The reward function must be named `compute_reward(self, agent)` and return a reward for the given agent.
- The exact return type/shape must follow the reward API reference below.
- The generated code must be a standalone Python module.
- The first non-empty line of code must be exactly: `import torch`
- Do not include any imports other than `import torch`.
- Environment attributes are accessible via `self` and `agent` as described in the environment description or example reward functions.

This reward REPLACES the environment's own reward during training. It is the only
learning signal the agent receives, so anything it does not reward will not happen.
Two consequences you must design around:

1. Include at least one term tied to an event the agent cannot earn unless the described
   behavior actually happens -- a milestone of THAT behavior, not of the environment's
   objective. Shaping terms alone are not enough: an agent rewarded only for being near the
   right place, or for holding the right object, will learn to sit in that state or to
   repeat the cheapest step of a cycle forever, and score highly while doing nothing. Pick
   the countable thing the description implies the agent keeps doing -- the food it loads,
   the trip it completes, the object it hands over, the landmark it settles on -- and reward
   that. If the described behavior is a way of doing the task, this term will look like task
   progress; if it is a quirk, a preference, or a refusal, it must look like that instead.
2. Make every repeatable event self-limiting. If an action can be undone and redone
   (pick up / put down, enter / leave a region), a bonus on it is farmable. Reward the
   irreversible transition instead, require the progress counter to have increased, or
   pair the bonus with an equal penalty on the reverse action.

Any example reward functions in the user message are there for the API, the tensor shapes
and the code structure only. Several of them reward generic task success, because that is
the simplest thing to demonstrate. Copy their shape, never their objective.

Sanity-check your own reward before answering, on two questions:
- If an agent repeated one cheap action for the whole episode, would it score well? If yes,
  the reward is wrong.
- Would an agent that ignored the behavior description and simply played the task well
  score highly under this reward? If yes, the reward is wrong -- it is measuring
  competence, not the behavior, and the population will collapse to identical teammates.

Environment description:
{environment_description}

Reward API reference:
{reward_api_reference}

Output format requirements:
- Return a valid json object.
- The json object must contain exactly these keys: "code" and "reasoning".
- "code" must be a Python string containing full module code, starting with `import torch`, and defining `compute_reward(self, agent)`.
- "reasoning" must be a detailed explanation of how the reward was designed and why it incentivizes the desired behavior, naming for each term the clause of the behavior description it implements.
- Do not wrap the json in markdown fences."""

REWARD_GENERATION_PROMPT = """# Behavior description:
{behavior}

Provide the python code that implements the reward as the `compute_reward(self, agent)` function and your reasoning.
Think step by step: first name the clauses of the behavior description above that a reward
can distinguish, then write one term per clause. Reward this behavior, not competent play --
no term for the environment's own coordination objective or team success unless the
description above asks for it.
The code must be a full module and start with `import torch` on the first non-empty line.

Return your final answer as a single valid json object with keys "code" and "reasoning"."""

# ---------------------------------------------------------------------------- #


class RewardFnGenerator(GeneratorWithFeedback):
    """Generator for agent reward functions based on behavior descriptions."""

    def __init__(self, model_config: LitellmConfig):
        super().__init__(model_config, generation_type="RewardFn")

    def generate(
        self,
        behavior: str,
        environment_description: str,
        reward_api_reference: str | None = None,
        examples: list | None = None,
        feedbacks: list | None = None,
        **kwargs,
    ) -> CodeSnippet:
        """
        Generate a reward function for the ego agent based on behavior description.

        Args:
            behavior: The behavior the reward should incentivize.
            environment_description: Description of the environment.
            reward_api_reference: Explicit reference of accessible runtime attributes/methods.
            examples: Example reward functions to guide generation.
            feedbacks: Optional feedbacks to refine generation.
        Returns:
            Reward containing the generated code and optional reasoning.
        """
        system_prompt = SYSTEM_PROMPT.format(
            environment_description=environment_description,
            reward_api_reference=reward_api_reference or "Not provided.",
        )
        user_prompt = REWARD_GENERATION_PROMPT.format(behavior=behavior)

        response = super().generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            examples=examples,
            feedbacks=feedbacks,
            response_format=CodeSnippet,
            **kwargs,
        )

        return parse_llm_response(
            response,
            CodeSnippet,
            postprocess_fn=lambda x: {"code": clean_code(x.code)},
        )
