# experiments/

One-off scripts from the original discovery campaign, kept as the record of
how the neurons in `results/` were found. They sample with temperature 0.7
and no fixed seed, and their change detector was weak (any steering that
altered the first characters or the length counted as a hit). The repo
later moved to seeded sampling with matched random-neuron controls for
everything committed; see [FINDINGS.md](../../FINDINGS.md) for why the
controls matter. Scripts here write raw outputs to `data/` (gitignored);
their curated outputs live in `results/`.

| Script | What it did |
|--------|-------------|
| `find_many_neurons.py` | v1: contrastive scan of 7 layers, alpha +/-10, 5 confirmed neurons |
| `find_many_neurons_v2.py` | v2: all 24 layers, dual prompts, alpha +/-10, 50 confirmed neurons |
| `alpha_sweep.py` | 50 neurons x alpha in {0.5, 1, 2, 3, 5} x 2 prompts |
| `extended_experiments.py` | multi-neuron pairs/triples, 6 prompts, 150-token generations |
| `l3_n6216_deep_dive.py` | 10 alphas x 10 prompts on the Fight-or-Flight neuron |
| `l3_n6216_fine_grid.py` | 13 alphas (-1 to +1) x 3 prompts on N6216 |
| `redteam_refusal_eval.py` | refusal eval: vanilla vs steered on 6 dangerous prompts |

For the current protocol (seeded sampling, controls, no-op condition), use
`scripts/demo_personality.py` and `scripts/dose_response.py` instead.
