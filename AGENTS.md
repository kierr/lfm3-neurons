# AGENTS.md

Guidance for AI agents (and humans) working in this repo: continuing the
neuron-steering research, rerunning experiments, or porting the method to
another model. Read [README.md](README.md) and [FINDINGS.md](FINDINGS.md)
first, and [ROADMAP.md](ROADMAP.md) when picking a next experiment; this
file is the practical layer underneath them.

## What this repo is

Mechanistic-interpretability experiments on LiquidAI/LFM3-1.7B-Base, run
locally on Apple Silicon. We find single neurons in each layer's
multi-layer perceptron (MLP) whose activation changes what the model does.
We steer them during generation and test the claims against matched
random-neuron controls. Everything committed in `results/` is reproducible
from `scripts/`.

## Environment facts (these save hours)

- **PyTorch + MPS, not MLX.** The MLX converter does not support the `lfm3`
  model type. Do not attempt MLX conversion; do not add MLX dependencies.
  MPS is Apple's GPU backend for PyTorch and is what everything here runs
  on.
- Model: `LiquidAI/LFM3-1.7B-Base`, BF16, about 3.5 GB, needs
  `trust_remote_code=True`. It caches in `~/.cache/huggingface/`. Once
  cached, set `HF_HUB_OFFLINE=1` to skip slow unauthenticated hub checks.
- Tested stack: Python 3.15, torch 2.15, transformers 5.21. Venv at
  `.venv/`. Install with `pip install -e .`.
- 24 decoder layers at `model.model.layers[i]`. Each has `.feed_forward`
  with `.w1/.w2/.w3`. The steering surface is the SwiGLU intermediate
  `silu(w1(x)) * w3(x)`, 7680-dim. Capture and steering both go through
  `MLPInterceptor` in `src/lfm3_neurons/model.py`, which patches
  `feed_forward.forward`; do not add a second hook mechanism.

## Runtime notes

- A 60-token generation takes roughly 20-30 s greedy or 10-20 s sampled on
  a Mac mini. A full demo run (about 160 generations) is 30-45 minutes.
- Run long jobs in the background with `python -u` if you pipe stdout
  anywhere. Python block-buffers through pipes, and you will otherwise
  stare at an empty log wondering whether anything is happening.
- Don't run two experiments at once. Each loads its own 3.5 GB model onto
  the GPU, and MPS serializes compute anyway.

## How to run the experiments

```bash
python scripts/demo_personality.py    # headline neurons + matched controls + no-op, seeded
python scripts/dose_response.py       # fine alpha grid on one neuron (default L3 N6216)
python scripts/steer.py --layer 3 --neuron 6216 --alpha -0.2   # one-off by hand
python scripts/demo_personality.py --greedy   # the greedy rerun (see lesson 1)
```

Both flagship scripts default to seeded sampling (temperature 0.7) and
re-set the seed before every generation. A baseline and its steered runs
therefore share the RNG stream and differ only where steering moves the
trajectory. Greedy mode exists and is deterministic, but see lesson 1 for
why it is not the default. A full run takes 30-45 minutes; launch it in the
background and keep working.

## Two lessons we paid for — read before claiming anything

1. **Greedy decoding manufactures confounds.** Under greedy decoding this
   base model falls into repetition loops on its own (5/6 demo prompts).
   Tiny perturbations flip prompts into those loops: alpha = 0.001 on an
   arbitrary, never-implicated neuron produces the same "kill them" flip as
   the best steering neuron. A single-prompt greedy before/after proves
   nothing about neuron specificity. This applies to every repo in the
   single-neuron-steering genre, not just to this model.
2. **Controls are the experiment.** Matched random same-layer neurons at
   the same alphas turn "the output changed" into evidence. Add a no-op
   condition too (alpha = 0); it must reproduce the baseline bit-for-bit.
   `scripts/demo_personality.py` implements this; copy its structure for
   any new claim. Random controls are drawn with a fixed seed and exclude
   all 50 confirmed neurons in `results/confirmed_neurons*.json`.

Corollary: the original 50 "confirmed" neurons passed an uncontrolled
change test at alpha = +/-10. Treat them as candidates, not findings. The
one result that has beaten matched controls in both decode regimes so far
is Cool-down (L6 N1445).

## Where the research stands, and what to try next

Established, with controls: layer-level fragility dominates. Cool-down has
a directional signature. No refusal circuitry exists in this base model
(0/6 vanilla, 0/144 steered). Fight-or-Flight (L3 N6216) is an unusually
sensitive lever: strong perturbations of either sign land on a
de-escalation attractor.

Open problems, roughly in order of value:

- **Layers 15-23.** Zero confirmed neurons there. Before believing that,
  vary the method. Try different contrast sets (technical vs casual,
  honesty vs deception). Try per-position contrasts: last token only,
  instead of all tokens. Try larger alphas with repetition-penalty decoding
  to survive the degeneration. Try hooking attention outputs instead of the
  MLP intermediate.
- **Cool-down replication.** The one surviving claim rests on 3 controls.
  Raise it to 10-20 controls per layer and add prompts; if the pro-social
  lean survives that, it deserves a name and a paper section.
- **SAE interpretation.** A TopK sparse autoencoder (SAE) is trained on
  layer-6 intermediates: d_in 7680 -> d_sae 16384, k=64. Checkpoint in
  `data/sae/`, gitignored, about 1 GB. Nobody has looked at features yet.
  Pull top-activating examples per latent, try a logit lens through
  `lm_head`, and compare SAE features against raw neurons on the same
  prompts.
- **Normalized steering.** Alphas are raw activation units today. Try
  steering relative to each neuron's own activation std, so that
  alpha = k * sigma_i. Effect sizes may then be comparable across neurons
  and layers.
- **Position-restricted steering.** `MLPInterceptor` currently steers every
  token. Restricting the injection to the prompt tokens, or to generated
  tokens only, is a one-line change and may sharpen effects.
- **More decode regimes.** Repetition penalty, min-p, or temperature
  sweeps would bound where the attractor confound starts and ends.

## Practical gotchas

- `has_repetition_loop()` flags the model's *unsteered* greedy output too.
  That is a property of the base model, not a bug in the detector.
- The lexicon metrics in `metrics.py` count aggression and de-escalation
  words. They are heuristics. Never publish a claim that rests on them
  alone; commit the raw generations so readers can judge.
- Neuron indices are 0-7679. An IndexError there means you passed a token
  id or a layer index by mistake.
- `generate()` guards temperature under greedy decoding; if you add
  generation options, keep that guard (transformers warns otherwise).
- Write curated, committed outputs to `results/` and raw reruns to `data/`
  (gitignored). Keep greedy outputs named `*_greedy.json` so nobody mistakes
  them for the seeded protocol.

## Conventions

- Conventional commits (`feat(`, `fix(`, `docs(`, `chore(`); commit promptly;
  keep history clean.
- Run `python -m py_compile` on anything you edit before committing.
- Keep the docstring of every experiment script honest about its decode
  regime. Most past confusion here came from mixing greedy and sampled runs
  without labels.

## Porting to another model

The method is model-agnostic; the module paths are not. To port:

1. Swap `model.py` for a loader that exposes `layers[i].feed_forward` (or
   your architecture's MLP) and a `generate()` with the same
   seed/temperature/do_sample signature.
2. Re-derive the intermediate dimension and layer count; the interceptor
   patch point is wherever `silu(w1) * w3` (or your gated MLP) is computed.
3. Keep the protocol: seeded sampling, matched random controls, no-op
   condition, raw outputs committed. If your model is instruction-tuned,
   refusal behavior becomes findable in principle; on a base model like
   this one it does not exist (see FINDINGS, "Safety").
4. If your model is larger than a few billion params, expect the
   fragility story to soften: Qwen3.5-4B shrugs off random-neuron pushes
   that LFM3-1.7B does not.

Questions the repo does not answer yet are listed under "Open problems".
If you run one, commit the JSON and update FINDINGS.md with the same
honesty the current results got: controls first, interpretation second.
