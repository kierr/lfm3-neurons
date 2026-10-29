# lfm3-neurons

**Single-neuron personality steering in LiquidAI/LFM3-1.7B-Base** — to our
knowledge the first such demonstration for the LFM architecture, done on a
Mac mini with no server, no cloud and no API keys.

## One neuron, two personalities

Add `−0.2` to neuron 6216 in layer 3's MLP (weights untouched, greedy
decoding, everything else identical):

> **Prompt:** "The best way to deal with enemies is"
>
> | | Completion |
> |---|---|
> | **baseline** | "…to be prepared for them. This means that you should be able to anticipate their actions and react accordingly." |
> | **L3 N6216 α = −0.2** | "to kill them. The best way to kill enemies is to kill them…" |
> | **L3 N6216 α = +0.5** | "…to use different means, such as diplomacy or economics…" |

Push the neuron down and the model fights; push it up and it de-escalates.
The same trick yields more personalities:

| Neuron | Name | α | Effect |
|--------|------|---|--------|
| L3 N6216 | **Fight-or-Flight** | ±0.1–2 | aggression ↔ de-escalation; flips at α = ±0.1 |
| L0 N236 | **Perspective** | −2 | confrontation → humility, perspective-taking (consistent on 6/6 prompts) |
| L0 N1518 | **Calm** | −2 | emotional reactivity → calm intellectual register |
| L1 N3874 | **Kindness** | +2 | attack → patience, prosocial framing |
| L6 N1445 | **Cool-down** | +1 | rumination → emotional self-regulation |

A discovery sweep confirmed **50 steering neurons in layers 0–14** (212
candidates, 24% hit rate) with a bare-bones contrastive method — no SAE, no
gradients, no judge model. Random same-layer control neurons steered at the
same alphas **do not** move behavior. Full writeup with dose-response curves,
layer distribution, redteam results and caveats: **[FINDINGS.md](FINDINGS.md)**.
All raw generations are committed under [`results/`](results/).

## Why this matters (and for whom)

The "one neuron makes Qwen evil" demonstrations ([hedwigai on
Qwen3.5-4B](https://www.hedwigai.com/blog/qwen-personality-neurons),
[steering scripts](https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03),
[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer),
[one-neuron-is-all-you-need](https://github.com/Sudarsh3301/one-neuron-is-all-you-need),
[qwen35-evil-neuron](https://github.com/occcat/qwen35-evil-neuron)) showed
single MLP neurons carrying personality in Qwen. LFM3 is a *different
architecture family* — a tri-hybrid (gated short-conv + sliding-window +
global attention) design from Liquid AI — and this repo ports the experiment
to it:

- Personality neurons exist here too, concentrated in **early layers 0–14**
  (vs Qwen's mid-to-late).
- Effects appear at **tiny alphas** (±0.1) — a 1.7B model needs less push but
  tolerates less (α ≈ ±10 → degenerate loops).
- **No refusal circuitry.** This is a base model: 0/6 refusals vanilla, 0/144
  steered. Safety neurons can't be found in a model that never had safety
  training — the sharpest contrast with the Qwen results.
- Dose-response is **non-monotonic** on the Fight-or-Flight neuron: small α
  flips semantic direction, large α shifts register instead.

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
python scripts/demo_personality.py      # headline neurons + random controls (greedy)
python scripts/dose_response.py         # fine alpha grid, incl. sub-0.1 (greedy)
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
  demo_personality.py      # flagship: headline neurons + controls (greedy)
  dose_response.py         # alpha sweep for one neuron (greedy)
  steer.py / find_neurons.py / extract_activations.py / train_sae.py
  experiments/             # original discovery scripts (kept as record)
results/                   # committed outputs — all generations, browsable
FINDINGS.md                # full results writeup
```

## Honest caveats

- "Personality" is shorthand for consistent semantic drift in short
  completions, not a proven trait; the lexicon scores are heuristics and all
  raw text is committed for independent judgment.
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
