# PopGen

PopGen generates and evaluates behavior-conditioned agents for multi-agent environments.

Current active pipeline is single-behavior, per-run generation/evaluation under:
- `scripts/population/` (general harness)
- `experiments/b2t/` (experiment harness)

## Setup

Recommended:

```bash
uv sync
```

Alternative editable install:

```bash
pip install -e .
pip install -e deps/VectorizedMultiAgentSimulator
pip install -e deps/BenchMARL
```

Add provider keys in `.env` at repo root (copy from `.env.example` if present).

## Main Workflows

### Population harness

```bash
cd scripts/population
./run_pipeline.sh env=lbf generation=policy_code
```

Key scripts:
- `generate_behaviors.py`
- `generate_agents.py`
- `gather_results.py`

### B2T harness

```bash
cd experiments/b2t
bash run.sh --env lbf --output-dir ./results
```

## Output Layout (Current)

Both harnesses produce per-experiment outputs with:

```text
<result_path>/
  agents/
    behavior_<i>/
      <run_timestamp>/
        attempt_<j>/...
  metadata/
    <timestamp>/
      behaviors.json
      summary.json
```

`gather_results.py` supports this layout and legacy layouts.

## Documentation

- Environment integration: `docs/add_env.md`
- LLM configuration/usage: `docs/llm_usage.md`
- Diversity discriminator evaluator: `docs/evaluators/diversity_discriminator.md`
