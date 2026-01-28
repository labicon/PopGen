# PopGen

## Quick Start

### Install
> We recommand using [`uv`](https://docs.astral.sh/uv/). In which case you can skip this and simply run `uv sync`

```bash
pip install -e .
pip install -e deps/VectorizedMultiAgentSimulator
pip install -e deps/BenchMARL
```

### Setup API Keys
Create a `.env` file at root with your API keys. You can copy the `.env.example`.

## Examples
- **Diverse Teammate Generation (Policy Code)**: `scripts/behavior_policy`
- **Diverse Teammate Generation (Reward Function)**: `scripts/reward_generation`