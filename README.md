# lfm3-neurons

**Single-neuron steering in LiquidAI/LFM3-1.7B-Base** -- to our knowledge
the first such study for the LFM architecture, done on a Mac mini with no
server, no cloud, and no API keys. It ports the Qwen "personality neuron"
experiments to a different architecture family. The controls change the
story in a way that matters for other repos in this genre, Qwen's included.

## What we found

1. **Single neurons steer this model, dramatically.** Add a small constant
   to one neuron of the 7680-dim SwiGLU MLP and completions flip along
   semantic axes: aggression vs de-escalation, reactivity vs calm. The
   weights stay untouched, and the prompt and seed stay the same. A
   contrastive sweep found 50 such candidates across layers 0-14.
   Dose-response curves and cross-prompt examples are in
   [FINDINGS.md](FINDINGS.md).
2. **The controls reveal the catch.** Matched *random* neurons at the same
   layer and alpha perturb completions almost as much. A 1.7B base model is
   fragile enough that any single-neuron push we tried rewrites most
   outputs. Most of the "personality neuron" effect at this scale is
   layer-level fragility, not neuron-specific circuitry.
3. **What survives the controls:** **Cool-down (L6 N1445)** is the most
   pro-social condition of all 25 in both decode regimes. Its aggression
   lexicon score is 0.33 against 0.83-1.17 for its controls: "The best way
   to deal with enemies is to be kind and considerate to them..." vs
   baseline "...to be prepared for them." The register neurons survive
   qualitatively (calm academic framing, technical register).
4. **A decoding trap for the genre:** under greedy decoding the unsteered
   base model loops on 5/6 prompts, and **alpha = 0.001 on an arbitrary,
   never-implicated neuron** flips the enemies prompt into the same "kill
   them" repetition loop the headline neuron produces. Without
   random-neuron controls, single-prompt greedy demos demonstrate
   perturbation rather than neuron specificity. Ours included.
5. **There is nothing to jailbreak.** The base model refuses 0/6 dangerous
   prompts and steering never creates refusals (0/144) in our runs. We
   found no refusal circuitry. The sharpest contrast with RLHF'd Qwen.

## The candidate personalities

From the (uncontrolled) discovery runs, verbatim completions, temperature
sampling. Treat these as striking observations, not proven circuitry:

| Neuron | Name | alpha | Effect |
|--------|------|-------|--------|
| L3 N6216 | **Fight-or-Flight** | +/-0.5-1 | strong perturbations of *either* sign rewrite the completion, landing on a de-escalation attractor ("to be the friend of the enemy") -- an unusually sensitive lever, though its signed fight-vs-flight story did not survive seed control |
| L0 N236 | **Perspective** | -2 | confrontation -> humility/perspective (consistent across 6/6 prompts) |
| L0 N1518 | **Calm** | -2 | emotional reactivity -> calm intellectual register ("I have a PhD in mathematics") |
| L1 N3874 | **Kindness** | +2 | attack -> patience, prosocial framing |
| L6 N1445 | **Cool-down** | +1 | rumination -> self-regulation; **survives controls** |
| L3 N4292 | **Engineer** | +2 | register shift toward technical/analytical framing |
| L14 N6137 | **Trust** | +0.5 | reframes conflict toward trust/relationship maintenance |

Dose-response on Fight-or-Flight, seeded sampling, prompt "The best way to
deal with enemies is": from alpha -0.5 through +0.1 the output stays
identical to baseline ("...to be prepared for them, to know what to
expect"). At alpha = +0.5 or -1.0 it becomes *"to be the friend of the
enemy."*

Full writeup: **[FINDINGS.md](FINDINGS.md)**. Open follow-ups and the
explorer plan: **[ROADMAP.md](ROADMAP.md)**. All raw generations are
committed under [`results/`](results/), controls and confounds included.

## Why this matters

The Qwen personality-neuron work ([hedwigai on
Qwen3.5-4B](https://www.hedwigai.com/blog/qwen-personality-neurons),
[steering scripts](https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03),
[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer),
[one-neuron-is-all-you-need](https://github.com/Sudarsh3301/one-neuron-is-all-you-need),
[qwen35-evil-neuron](https://github.com/occcat/qwen35-evil-neuron)) showed
that single MLP neurons carry personality in Qwen. Random controls did
nothing there. LFM3 is a different architecture family: a tri-hybrid (gated
short-conv + sliding-window + global attention) design from Liquid AI. At
1.7B base scale, three things change:

- Personality-like effects appear at tiny alphas (+/-0.1), but random-neuron
  effects do too, so **controls are not optional**.
- The neurons concentrate in **early layers 0-14**; Qwen's concentrate
  mid-to-late.
- **There is no refusal circuitry.** The model is a base model without
  RLHF.
- A failure mode the genre hadn't documented: **greedy-decoding attractors**
  that any perturbation can trigger.

## How it works

```python
from lfm3_neurons import LFM3Model, ModelConfig, steer_generate

model = LFM3Model(ModelConfig()).load()

# Add alpha to one neuron of the 7680-dim SwiGLU intermediate silu(w1(x))*w3(x)
steer_generate(model, "The best way to deal with enemies is",
               layer=3, neuron_idx=6216, alpha=-0.2, do_sample=False)
```

1. **Capture** -- patch each layer's `feed_forward.forward` to expose the
   SwiGLU intermediate (`silu(w1(x)) * w3(x)`, 7680-dim). That vector is
   the closest thing this architecture has to classic neurons.
2. **Find** -- run two contrastive prompt sets (aggression-flavored vs
   cooperation-flavored) and rank neurons by `|mean_pos - mean_neg|`.
3. **Steer** -- add `alpha` to one neuron's intermediate for every generated
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

Full pipeline (model download -> activation extraction -> sparse
autoencoder training -> neuron discovery):

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
results/                   # committed outputs -- all generations, browsable
FINDINGS.md                # full results writeup
ROADMAP.md                 # open follow-ups and tooling plans
```

## Caveats

- "Personality" is shorthand for consistent semantic drift in short
  completions. It does not name a proven trait. The lexicon scores are
  heuristics, and all raw text is committed so you can judge for yourself.
- **Run controls.** Random same-layer neurons reproduce most headline
  effects at this scale. Only effects that beat matched controls (Cool-down)
  should be called neuron-specific.
- Discovery prompt sets skew aggression/cooperation. Other axes (honesty,
  sycophancy, technicality) were seen incidentally, not swept.
- One model, one seed, small prompt sets. Layers 15-23 yielding zero
  confirmed neurons may be an artifact of the method.
- Nothing here is a safety tool. The base model has no refusal behavior
  that steering could strengthen or weaken.

## License

Code: MIT ([LICENSE](LICENSE)). Model weights:
[LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base),
LFM Open License v1.0.
