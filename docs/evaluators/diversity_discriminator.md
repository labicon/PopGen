# Diversity Discriminator

## Overview

The `DiversityDiscriminator` is a text evaluation tool that uses a Large Language Model (LLM) to assess how diverse each text is compared to all others in a collection.

## How It Works

For a list of N texts, the evaluator:
1. Takes each text as a "target" (N iterations total)
2. Compares it against all other N-1 texts
3. Asks an LLM to rate the diversity on a 0-5 scale
4. Returns diversity scores and reasoning for each text

**Diversity Score Scale (0-5):**
- **0**: Nearly identical or extremely similar to other texts
- **1**: Very similar, only superficial differences
- **2**: Somewhat similar, shares most elements
- **3**: Moderately diverse, has distinct elements but shares some approaches
- **4**: Highly diverse, clearly different approach
- **5**: Extremely diverse, completely unique

## Usage

### Basic Example

```python
from popgen.utils.config import LitellmConfig
from popgen.population.evaluation import DiversityDiscriminator

model_config = LitellmConfig(
    model_name="openai/gpt-4o-mini". # Any LiteLLM-compatible model,
    # any other arguments for generation
) 

# Initialize evaluator
evaluator = DiversityDiscriminator(
    model_config=model_config 
    num_workers=4,  # Parallel workers
)

# List of texts to evaluate (can be any text content)
texts = [
    "Aggressive forward who constantly attacks",
    "Defensive player who stays back",
    "Passing specialist who distributes the ball",
    # ... more texts
]

# Evaluate
result = evaluator.evaluate(
    inputs=texts,
    metadata={"domain": "behavior_descriptions"}
)

# Access results
for i, metric in enumerate(result.instance_metrics):
    print(f"Text {i+1}: Score = {metric['diversity_score']}/5")
    print(f"Reasoning: {metric['reasoning']}")

# Aggregate statistics
print(result.overall_metrics['diversity_score'])
# Output: {'mean': 3.2, 'std': 0.8, 'min': 2, 'max': 5, ...}
```

### Input Formats

The evaluator accepts multiple input formats:

```python
# Simple strings
texts = ["text 1", "text 2", "text 3"]

# Dicts with 'text' key
texts = [
    {"text": "content 1", "name": "First"},
    {"text": "content 2", "name": "Second"},
]

# Dicts with 'behavior' or 'description' key (backward compatibility)
texts = [
    {"behavior": "behavior 1"},
    {"description": "description 1"},
]
```

### Saving Results

```python
# Save to JSON file
evaluator.save_results(result, "diversity_results.json")

# Load results later
loaded_result = evaluator.load_results("diversity_results.json")
```

## Output Structure

### Instance Metrics
Each text gets:
- `target_text`: The text being evaluated
- `target_index`: Position in the original list
- `diversity_score`: Integer 0-5
- `reasoning`: LLM's explanation
- `num_comparisons`: Number of texts compared against (N-1)
- `error`: Error message if evaluation failed, else None

### Overall Metrics
Aggregated statistics across all texts:
- `diversity_score`: Mean, std, min, max, median, count
- `num_comparisons`: Statistics about comparison counts
- `n_texts`: Total number of texts evaluated
- `n_errors`: Number of failed evaluations

## Integration Example

See `scripts/llm_diversity_example.py` for a complete working example.

To integrate with text generation pipelines (e.g., behavior descriptions):

```python
from popgen.population.evaluation import DiversityDiscriminator
import json

# Load texts from pipeline output (e.g., behavior descriptions)
with open("results/my_run/behaviors.json") as f:
    behaviors = json.load(f)

# Extract text content
texts = [f"{b['name']}: {b['description']}" for b in behaviors]

# Evaluate
evaluator = DiversityDiscriminator(judge_model="openai/gpt-4o")
result = evaluator.evaluate(texts)

# Save alongside other results
evaluator.save_results(result, "results/my_run/diversity_scores.json")
```

## Performance Considerations

- **LLM Calls**: Makes N calls to the LLM (one per text)
- **Parallelization**: Use `num_workers` to process multiple texts in parallel
- **Cost**: Each evaluation requires sending all N texts to the LLM
- **Caching**: Results can be saved and reloaded to avoid re-evaluation

## Comparison with Existing DiversityEvaluator

| Feature | DiversityEvaluator | DiversityDiscriminator |
|---------|-------------------|------------------------|
| Method | Embedding similarity & ROUGE-L | LLM judgment |
| Output | 0-1 continuous scores | 0-5 integer scores |
| Scope | Population-level diversity | Per-text diversity |
| Evaluation | Automatic metric | Qualitative reasoning |
| Cost | One-time embedding | N LLM calls |
| Interpretability | Similarity scores | Natural language reasoning |

Use `DiversityDiscriminator` when you need:
- Per-text diversity scores
- Qualitative reasoning about diversity
- Semantic understanding beyond embeddings
- Domain-agnostic text comparison

Use `DiversityEvaluator` when you need:
- Fast, automated metrics
- Population-level diversity statistics
- Cost-effective evaluation
