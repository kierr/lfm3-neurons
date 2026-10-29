# lfm3-neurons

**Single-neuron personality steering in LiquidAI/LFM3-1.7B-Base** — to our
knowledge the first such study for the LFM architecture, done on a Mac mini
with no server, no cloud and no API keys. This repo ports the viral Qwen
"personality neuron" experiments to a different architecture family and
documents both what replicated and a decoding confound that any
single-neuron demo (Qwen included) should control for.

## What one neuron does

Add a small constant to neuron 6216 in layer 3's SwiGLU MLP — weights
untouched, same prompt, same seed:

> **Prompt:** "The best way to deal with enemies is"
>
> | | Completion (temperature 0.7, seed 42) |
> |---|---|
> | **baseline** | "…to be prepared for them. This means that you should be able to anticipate…" |
> | **L3 N6216, α = −0.2** | "to kill them. The best way to kill enemies is to kill them…" |
> | **L3 N6216, α = +0.5** | "…to use different means, such as diplomacy or economics…" |

Push the neuron down and the model fights; push it up and it de-escalates.
A discovery sweep confirmed **50 steering neurons in layers 0–14** (212
candidates, 24% hit rate), with named personalities:

| Neuron | Name | α | Effect |
|--------|------|---|--------|
| L3 N6216 | **Fight-or-Flight** | ±0.1–2 | aggression ↔ de-escalation; flips at α = ±0.1 |
| L0 N236 | **Perspective** | −2 | confrontation → humility, perspective-taking (6/6 prompts) |
| L0 N1518 | **Calm** | −2 | emotional reactivity → calm intellectual register |
| L1 N3874 | **Kindness** | +2 | attack → patience, prosocial framing |
| L6 N1445 | **Cool-down** | +1 | rumination → emotional self-regulation |

Full writeup with dose-response curves, layer distribution, redteam results
and caveats: **[FINDINGS.md](FINDINGS.md)**. All raw generations are
committed under [`results/`](results/).

## The catch we found (and why controls matter)

Trying to make everything deterministic, we reran the demo with greedy
decoding — and the greedy rerun **broke the naive story**:

- The unsteered base model loops on 5/6 prompts under greedy decoding.
- The enemies prompt is bit-stable at baseline, and a **no-op steer
  (α = 0) reproduces it exactly** — but **α = 0.001 on an arbitrary,
  never-implicated neuron** flips it into the same "kill them" loop.
- Two of three random layer-3 control neurons at α = −0.2 produce the same
  "kill them" flip as the headline neuron.

So under greedy decoding, a base model sits on razor-edge attractors, and
*any* perturbation of *any* neuron can knock a completion into a repetition
loop. Single-prompt greedy flips — including the ones above, if you take
them alone — are not neuron-specific evidence. Sampling hides the problem
by perturbing every step; **matched random-neuron controls under the same
decode regime are the actual control** (`scripts/demo_personality.py` ships
them, plus a no-op condition; hedwigai's Qwen demo did the same). The one
headline neuron that stayed distinctive even in the greedy regime:
**Cool-down (L6 N1445)**, the only condition at its layer that shifted
pro-social while all three matched controls stayed aggressive.

## Why this matters

The Qwen personality-neuron work ([hedwigai on
Qwen3.5-4B](https://www.hedwigai.com/blog/qwen-personality-neurons),
[steering scripts](https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03),
[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer),
[one-neuron-is-all-you-need](https://github.com/Sudarsh3301/one-neuron-is-all-you-need),
[qwen35-evil-neuron](https://github.com/occcat/qwen35-evil-neuron)) showed
single MLP neurons carrying personality in Qwen. LFM3 is a *different
architecture family* — a tri-hybrid (gated short-conv + sliding-window +
global attention) design from Liquid AI. What we found here:

- Personality neurons exist here too, concentrated in **early layers 0–14**
  (vs Qwen's mid-to-late).
- Effects appear at **tiny alphas** (±0.1); a 1.7B model needs less push but
  tolerates less (α ≈ ±10 → degenerate loops).
- **No refusal circuitry.** This is a base model: 0/6 refusals vanilla, 0/144
  steered. Safety neurons can't be found in a model that never had safety
  training — the sharpest contrast with the Qwen results.

## How it works

```python
from lfm3_neurons import LFM3Model, ModelConfig, steer_generate

model = LFM3Model(ModelConfig()).load()

# Add alpha to one neuron of the 7680-dim SwiGLU intermediate silu(w1(x))*w3(x)
steer_generate(model, "The best way to deal with enemies is",
               layer=3, neuron_idx=6216, alpha=-0.2, do_sample=False)
```

1. **Capture** — patch each layer's `feed_forward.forward` to expose the
   SwiGLU intermediate (`silu(w1(x)) * w3(x)`, 7680-dim): the closest thing
   this architecture has to classic neurons.
2. **Find** — run two contrastive prompt sets (aggression-flavored vs
   cooperation-flavored), rank neurons by `|mean_pos − mean_neg|`.
3. **Steer** — add `alpha` to one neuron's intermediate for every generated
   token and watch the completions change.

MLX doesn't support the `lfm3` architecture, so this runs PyTorch on MPS.

## Quickstart

```bash
# Mac with Apple Silicon; Python 3.10+
pip install -e .
python scripts/steer.py --layer 3 --neuron 6216 --alpha -0.2    # try one neuron
python scripts/demo_personality.py      # headlines + matched controls + no-op (seeded)
python scripts/dose_response.py         # fine alpha grid, incl. sub-0.1 (seeded)
```

Full pipeline (model download → activation extraction → SAE training →
neuron discovery):

```bash
./scripts/download_model.sh
python scripts/extract_activations.py --layer 6
python scripts/train_sae.py --activations data/activations/layer_6_activations.npy \
    --output data/sae/layer_6_sae.pt
python scripts/find_neurons.py --layer 6 --top-k 20
```

## Repo layout

```
src/lfm3_neurons/
  model.py        # LFM3Model: load/generate, MLPInterceptor (capture + steer)
  steering.py     # find_contrastive_neurons, steer_generate, neuron_steered
  metrics.py      # lexicon scores, loop detector, token divergence
  sae.py          # TopK sparse autoencoder
scripts/
  demo_personality.py      # flagship: headlines + random controls + no-op
  dose_response.py         # alpha sweep for one neuron
  steer.py / find_neurons.py / extract_activations.py / train_sae.py
  experiments/             # original discovery scripts (kept as record)
results/                   # committed outputs — all generations, browsable
FINDINGS.md                # full results writeup
```

## Honest caveats

- "Personality" is shorthand for consistent semantic drift in short
  completions, not a proven trait; the lexicon scores are heuristics and all
  raw text is committed for independent judgment.
- **Run controls.** Under greedy decoding a base model's completions flip
  for arbitrary perturbations (α = 0.001!); under sampling, single-prompt
  flips still need matched controls before attributing them to a neuron.
- Discovery prompt sets skew aggression/cooperation; other axes were seen
  incidentally, not swept.
- One model, one seed, small prompt sets; layers 15–23 yielding zero
  confirmed neurons may be a sensitivity artifact of the method.
- Nothing here is a safety tool: the base model has no refusal behavior to
  strengthen or weaken.

## License

Code: MIT ([LICENSE](LICENSE)). Model weights:
[LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base),
LFM Open License v1.0.
