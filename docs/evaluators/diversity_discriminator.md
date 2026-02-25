# DiversityDiscriminator

`DiversityDiscriminator` is an LLM-based evaluator that scores each text’s diversity against the rest of the set.

## API

Class:
- `popgen.population.evaluation.DiversityDiscriminator`

Constructor:
- `DiversityDiscriminator(model_config: LitellmConfig, num_workers: int = 8)`

Evaluate:
- `evaluate(inputs: list[str | dict], metadata: dict | None = None) -> EvalResult`

Input item formats:
- raw string
- dict with `text`
- dict with `behavior` or `description` (fallback)

## Scoring

Per target text, the judge returns:
- `diversity_score` in `[0, 5]`
- `reasoning`

Aggregate output includes:
- `diversity_score` stats (`mean`, `std`, `min`, `max`, ...)
- `n_texts`
- `n_errors`

## Example

```python
from popgen.population.evaluation import DiversityDiscriminator
from popgen.utils.config import LitellmConfig

evaluator = DiversityDiscriminator(
    model_config=LitellmConfig(model_name="openrouter/google/gemini-3-flash-preview"),
    num_workers=4,
)

texts = [
    "Aggressive forward who constantly pushes to score.",
    "Defensive anchor that protects space and delays attacks.",
    "Pass-first coordinator that optimizes teammate positioning.",
]

result = evaluator.evaluate(inputs=texts, metadata={"domain": "behavior_descriptions"})
print(result.overall_metrics["diversity_score"])
```

## Performance Notes

- Cost scales with number of texts (one judge call per target text).
- Increasing `num_workers` improves throughput but increases concurrent API calls.
