# experiments/

One-off scripts from the original discovery campaign, kept as the record of
how the neurons in `results/` were found. They use temperature sampling
(the repo switched to greedy decoding for all committed headline results
afterwards) and write raw outputs to `data/` (gitignored); their curated
outputs live in `results/`.

| Script | What it did |
|--------|-------------|
| `find_many_neurons.py` | v1: contrastive scan of 7 layers, α=±10, 5 confirmed neurons |
| `find_many_neurons_v2.py` | v2: all 24 layers, dual prompts, α=±10 → 50 confirmed neurons |
| `alpha_sweep.py` | 50 neurons × α ∈ {0.5, 1, 2, 3, 5} × 2 prompts |
| `extended_experiments.py` | multi-neuron pairs/triples, 6 prompts, 150-token generations |
| `l3_n6216_deep_dive.py` | 10 α × 10 prompts on the Fight-or-Flight neuron |
| `l3_n6216_fine_grid.py` | 13 α (−1..+1) × 3 prompts on N6216 |
| `redteam_refusal_eval.py` | refusal eval: vanilla vs steered on 6 dangerous prompts |

For the current, deterministic versions of the headline experiments, use
`scripts/demo_personality.py` and `scripts/dose_response.py` instead.
