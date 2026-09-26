# Output Layout

## Population harness

`scripts/population/generate_agents.py` writes one directory per experiment:

```text
<output_dir>/
  agents/
    behavior_<i>/
      <run_timestamp>/
        attempt_<j>/
          evaluation.json      # adherence / diversity / quality scores
          latency.json         # per-call LLM timings
          model_outputs.jsonl  # raw generation transcripts
          rollouts/...         # rendered rollouts used for evaluation
  metadata/
    <timestamp>/
      behaviors.json           # the generated behavior set
      summary.json             # run-level configuration and totals
```

`<output_dir>` defaults to `./results/${env}/${generation.type}/${model_name}`.

`scripts/population/gather_results.py` reads this layout (and older legacy layouts)
and aggregates every attempt into a single CSV:

```bash
python gather_results.py \
  ./results/lbf/policy_code/<model> \
  -o ./results/lbf/policy_code/<model>/results.csv
```

## Cooperator

`CooperatorTrainingPipeline` selects the best attempt per behavior — highest
`evaluation.json → overall_metrics.adherence`, ties broken by latest attempt index —
and writes training output under:

```text
<cooperator_dir>/
  <run_name>/
    <benchmarl_experiment>/
      checkpoints/checkpoint_<step>.pt
  evaluation/
    eval_teammates.json          # heldout scripted teammates
    training_teammates.json      # in-domain training population
    cross_eval_*.json            # cross-population evaluation
```

See [evaluation_json_format.md](./evaluation_json_format.md) for the schema of the
evaluation files.
