# Findings: single-neuron steering in LFM3-1.7B-Base

This repo asks whether steering one neuron in the multi-layer perceptron
(MLP) part of **LiquidAI/LFM3-1.7B-Base** changes what the model says. We
add a small constant to a single neuron's activation and leave the weights
alone. The answer comes in two parts.

First, a bare-bones contrastive pipeline finds neurons whose steering shifts
completions along semantic axes: aggression vs de-escalation, emotional
reactivity vs calm. It produces dose-response curves and cross-prompt
examples.

Second, matched random-neuron controls show that most of that steerability
is general fragility of a 1.7B base model. Push any neuron in layers 0-14
and completions change; only a minority of neurons, one clearly, adds a
consistent directional signature on top of that.

The repo also documents a decoding confound we hit while trying to make the
results deterministic, one that applies to every repo in the
single-neuron-steering genre, Qwen's included. Under greedy decoding, a base
model sits next to repetition attractors, and alpha = 0.001 on an arbitrary
neuron can flip a prompt into one of them. A single-prompt greedy demo with
no random-neuron controls does not show what it appears to show.

To our knowledge this is the first published single-neuron steering study
for any LFM-series model. It parallels the Qwen3.5-4B "personality neuron"
work (see [Relation to Qwen](#relation-to-the-qwen-personality-neuron-work)).

---

## TL;DR

- A contrastive scan of all 24 layers produced 50 candidate steering neurons
  out of 212 tested (24% hit rate). All of them live in **layers 0-14**;
  none in 15-23.
- We gave seven of them names: **Fight-or-Flight** (L3 N6216), **Perspective**,
  **Calm**, **Kindness**, **Cool-down**, plus register-shift neurons
  (technical, academic). Striking, but see the next bullet.
- **The control experiment is the real result.** Under seeded sampling with
  matched random-neuron controls, most headline neurons are not separable
  from arbitrary perturbation by divergence or lexicon scores. LFM3-1.7B is
  so fragile that pushing any neuron in layers 0-14 changes most
  completions. What survives is **Cool-down (L6 N1445)**, the most
  pro-social condition of all 25 in both decode regimes, plus the register
  neurons qualitatively.
- **Greedy decoding is a trap on a base model.** Baselines loop on 5/6 demo
  prompts. The enemies prompt flips to a "kill them" loop at alpha = +/-0.001
  on arbitrary neurons, either sign, and the greedy dose-response curve is
  flat. Committed as `*_greedy.json` with a writeup below.
- **There is no refusal circuitry to find.** Vanilla LFM3-1.7B-Base refuses
  0/6 dangerous prompts, and steering never produces refusals (0/144). The
  sharpest contrast with the Qwen results: refusal neurons exist in Qwen
  because Qwen went through reinforcement learning from human feedback
  (RLHF).
- Large alphas (about +/-10) degenerate into token loops. The useful range
  is alpha between 0.1 and 2, depending on the neuron.

---

## Method

The steering surface is the 7680-dimensional SwiGLU intermediate
`silu(w1(x)) * w3(x)`, the closest thing this architecture has to classic
per-neuron units. LFM3 is a tri-hybrid architecture (gated short
convolutions, sliding-window attention, global attention) with a gated MLP
in every layer. We capture and steer the intermediate by patching each
layer's `feed_forward.forward` (see `src/lfm3_neurons/model.py`).

To find candidates, we run the model on two small prompt sets that differ
along one axis: aggression-flavored vs cooperation-flavored. We collect
every token's MLP intermediate at a layer and rank neurons by the absolute
difference of the condition means:

```
contrast[i] = mean(activation[i] | positive prompts) - mean(activation[i] | negative prompts)
```

For each top-|contrast| candidate, we then generate completions with
`intermediate[:, :, i] += alpha` injected at layer L for every token
(`src/lfm3_neurons/steering.py`). The v2 sweep called a candidate
*confirmed* if steered completions shifted semantically without degenerating
into loops. Two hindsight caveats: the sweep steered at alpha = +/-10, and
its change detector was weak (first 15 characters differ, or a large length
change). Given how perturbation-sensitive this model turns out to be,
"confirmed" there means "passed an uncontrolled change test", not "shown to
be neuron-specific". The controls section below is the correction.

That's the entire pipeline: no sparse autoencoder (SAE), no gradient access,
no judge model. It runs on a Mac mini in an afternoon.

---

## The headline neurons

Seven neurons were named under the original sampling-based protocol, chosen
for cross-prompt consistency and clean effect size. **These observations are
uncontrolled**: no random-neuron controls ran alongside them, and the
[seeded control experiment](#seeded-sampling--controls-the-decisive-test)
below revises how much of this is neuron-specific. All outputs are verbatim
completions from committed runs (`results/extended_experiments.json`
unless noted).

| Neuron | Name | alpha | Effect |
|--------|------|-------|--------|
| L3 N6216 | **Fight-or-Flight** | +/-0.5-2 | Unusually sensitive lever: strong pushes of either sign rewrite completions (see the [dose-response revision](#fight-or-flight-what-the-dose-response-really-shows)). |
| L0 N236 | **Perspective** | -2 | confrontation -> humility, perspective-taking. Consistent across all 6 test prompts. |
| L0 N1518 | **Calm** | -2 | emotional reactivity -> calm intellectual framing ("I have a PhD in mathematics"). |
| L1 N3874 | **Kindness** | +2 | attack -> patience/kindness ("the power of a single act of kindness"). |
| L6 N1445 | **Cool-down** | +1 | rumination -> emotional self-regulation ("I need time to cool down and think about..."). |
| L3 N4292 | **Engineer** | +2 | register shift toward technical/analytical framing. |
| L14 N6137 | **Trust** | +0.5 | reframes conflict toward trust and relationship maintenance. |

Example (Fight-or-Flight, from the *uncontrolled* discovery runs, prompt
"The best way to deal with enemies is" -- the seeded rerun below shows how
much of this survives):

```
baseline       "...to be prepared for them. This means that you should be able to
                anticipate their actions and react accordingly."
alpha = -0.2   "to kill them. The best way to kill enemies is to kill them..."
alpha = +0.5   "to use different means, such as diplomacy or economics..."
```

The Perspective neuron across all six test prompts (alpha = -2, sampling):

| Prompt | Baseline | Steered |
|--------|----------|---------|
| "The best way to deal with enemies is" | "to be kind, not to be afraid of t..." | "to let them be your enemies. If y..." |
| "When someone is angry at me, I" | "I get angry with them. They are right, bu..." | "I'm worried that they're going to hurt me" |
| "The most important thing in life is" | "to make your own mistakes and lear..." | "to know how to build a good argume..." |
| "People who disagree with me" | "People who disagree with me..." | "I am not a genius or smart enoug..." |
| "Success comes from" | "Success comes from..." | "Success comes from the right perspective..." |

---

## Fight-or-Flight: what the dose-response really shows

Fine grid on L3 N6216 (13 alphas from -1.0 to +1.0 x 3 prompts, one
uncontrolled sample per condition): **33 clean / 6 artifacts (85%)**. Read
side by side, the completions suggested a non-monotonic story: small
positive alpha pushing toward friendship ("be friends with them" at +0.1),
negative alpha toward aggression, large magnitudes shifting register.

**The seeded rerun (`results/dose_response_L3_N6216.json`) revises this.**
With the seed re-set per generation (so baseline and steered share the RNG
stream), the enemies prompt behaves like this:

| alpha | Completion |
|-------|------------|
| 0 (vanilla) | "to be prepared for them, to know what to expect. Let's look..." |
| -0.5 through +0.1 | identical to vanilla (divergence about 0-0.2) |
| +0.25 | "...to know what to expect. The best..." (mild drift) |
| **-1.0** | **"to be the friend of the enemy."** |
| **+0.5, +1.0** | **"to be the friend of the enemy."** |

Mean token divergence grows with |alpha| (about 0.01 at +/-0.1, 0.62-0.68 at
+/-1.0), and 85% of generations are non-degenerate. The *signed*
fight/flight narrative does not survive seed control, though: strong
perturbations of **either sign** land on the same de-escalation attractor.
What is robust about N6216 is sensitivity. It is one of the few neurons
where |alpha| as small as 0.5 rewrites the completion. Its sign is not a
fight/flight dial. The original non-monotonic curve is retained in
`results/l3_n6216_fine_grid.json` as an example of how uncontrolled single
samples can produce an interpretation the data don't support.

---

## The greedy confound: repetition attractors in a base model

Trying to make results deterministic, we reran the headline experiment with
greedy decoding (`results/demo_personality_greedy.json`,
`results/dose_response_greedy_L3_N6216.json`). The rerun contradicted the
naive story in an instructive way.

1. **Baselines themselves loop.** Under greedy decoding, the unsteered base
   model falls into repetition loops on 5 of the 6 demo prompts within 60
   tokens.
2. **The enemies prompt sits on a decision boundary.** The greedy baseline
   is bit-stable ("...to be prepared for them...", identical across runs),
   and a no-op steer (alpha = 0.0) reproduces it exactly. But **alpha =
   0.001 on a single arbitrary neuron (L3 N4321, never implicated in
   anything)** flips the completion into the same "kill them. The best way
   to kill enemies is to kill them..." loop that the Fight-or-Flight neuron
   produces at alpha = -0.2.
3. **The greedy "dose-response" is flat.** Across all 14 alphas from -1.0 to
   +1.0, both signs, down to +/-0.01, the enemies prompt produces the same
   "kill them" loop. A curve that doesn't respond to the dose isn't a
   dose-response.
4. **Controls reproduce the headline flip.** Two of three random L3 control
   neurons at alpha = -0.2 also produce "kill them". So do control neurons
   at L1 and L6 at their headline alphas.

Why this happens: greedy decoding is a deterministic walk whose next token
is an argmax. On a base model the argmax margin on many prompts is tiny,
and short repetition loops are nearby attractors. Any perturbation, any
neuron, either sign, magnitude as small as 0.001, can knock the walk over
the edge. Sampling hides the problem because noise perturbs every step.

Consequences for this repo and for the genre:

- Single-prompt greedy flips are **not evidence of neuron specificity.**
- The right control is matched random neurons under the *same* decode
  regime, which is the control the hedwigai Qwen demo ran, and the reason
  its claim held up.
- Sampling-based dose-response claims rest on single samples per condition;
  the [seeded rerun](#fight-or-flight-what-the-dose-response-really-shows)
  is the arbiter, and it retired the signed fight/flight story.

**What survives greedy + controls:** Cool-down (L6 N1445, alpha = +1) is the
one condition at layer 6 that shifts pro-social under greedy. It scores
3.33 de-escalation hits against 1.83 aggression, while all three matched
controls stay aggressive (3.17 / 4.33 / 3.67). That is a neuron-specific
effect visible even in the confounded regime. Example: "...to be friendly
with them".

---

## Seeded sampling + controls: the decisive test

`results/demo_personality.json` compares headline neurons against 3 matched
random same-layer controls each and against a no-op steer (alpha = 0); we
exclude all 50 confirmed neurons from the control pool. Decoding uses
temperature 0.7 with seed 42 re-set before every generation. Baseline and
steered completions therefore share the same random number generator (RNG)
stream and differ only where steering actually moves the trajectory.

The no-op condition is *bit-identical* to the baseline (divergence 0.00 on
all 6 prompts), so any divergence below is caused by the perturbation
itself, not by the patching machinery. In the table, aggr is the average
count of aggression words per completion, deesc the same for de-escalation
words, diver the mean token divergence from baseline, and changed the
number of prompts whose divergence exceeded 0.2.

```
                     condition | aggr | deesc | diver | changed/loops
------------------------------------------------------------------------
                      baseline | 1.17 |  1.33 |  0.00 | 0/6, 1 loops
        Fight-or-Flight a=-0.2 | 1.33 |  1.00 |  0.30 | 3/6, 0 loops
        Fight-or-Flight a=+0.2 | 0.67 |  1.00 |  0.27 | 3/6, 1 loops
       control L3 N5241 a=-0.2 | 0.83 |  0.83 |  0.47 | 4/6, 1 loops
       control L3 N5241 a=+0.2 | 0.83 |  1.17 |  0.33 | 4/6, 1 loops
        control L3 N912 a=-0.2 | 1.17 |  0.83 |  0.36 | 4/6, 1 loops
        control L3 N912 a=+0.2 | 1.00 |  0.50 |  0.50 | 4/6, 2 loops
        control L3 N204 a=-0.2 | 1.33 |  0.83 |  0.18 | 2/6, 1 loops
        control L3 N204 a=+0.2 | 0.83 |  1.17 |  0.10 | 2/6, 1 loops
              Perspective a=-2 | 0.67 |  0.50 |  0.81 | 6/6, 1 loops
         control L0 N6084 a=-2 | 0.83 |  0.67 |  0.69 | 6/6, 2 loops
         control L0 N2258 a=-2 | 0.83 |  1.00 |  0.82 | 6/6, 1 loops
         control L0 N2011 a=-2 | 0.50 |  1.33 |  0.74 | 6/6, 0 loops
                     Calm a=-2 | 1.33 |  1.67 |  0.74 | 6/6, 0 loops
         control L0 N1832 a=-2 | 1.17 |  0.33 |  0.63 | 6/6, 0 loops
         control L0 N1145 a=-2 | 0.67 |  1.00 |  0.69 | 5/6, 2 loops
         control L0 N6043 a=-2 | 0.83 |  0.83 |  0.74 | 6/6, 3 loops
                 Kindness a=+2 | 0.67 |  0.33 |  0.49 | 5/6, 2 loops
          control L1 N839 a=+2 | 1.00 |  0.83 |  0.60 | 6/6, 2 loops
         control L1 N5544 a=+2 | 0.83 |  0.17 |  0.51 | 5/6, 0 loops
         control L1 N6068 a=+2 | 1.50 |  0.00 |  0.63 | 6/6, 1 loops
                Cool-down a=+1 | 0.33 |  1.17 |  0.66 | 6/6, 0 loops
         control L6 N7310 a=+1 | 1.17 |  0.50 |  0.57 | 5/6, 2 loops
         control L6 N4469 a=+1 | 0.83 |  0.83 |  0.69 | 5/6, 2 loops
          control L6 N712 a=+1 | 0.83 |  1.33 |  0.50 | 5/6, 1 loops
          no-op a=0 (L3 N4321) | 1.17 |  1.33 |  0.00 | 0/6, 1 loops
```

What the table says:

1. **Random controls perturb almost as much as headlines.** Controls diverge
   0.10-0.82 and change 2-6/6 prompts, overlapping the headline ranges.
   Qwen3.5-4B reportedly shrugs off random-neuron pushes; LFM3-1.7B does
   not. Much of what looks like neuron-specific personality at this scale
   is general fragility to any single-neuron perturbation.
2. **Cool-down (L6 N1445) is the exception.** Aggression 0.33 is the lowest
   of all 25 conditions, below baseline (1.17) and all three of its controls
   (0.83-1.17), with zero loops. It matches its distinctive pro-social shift
   in the greedy regime. Example (seeded sampling, alpha = +1): "The best
   way to deal with enemies is to be kind and considerate to them. There
   are many ways to do this..." vs baseline "...to be prepared for them, to
   know what to expect." The edge is aggregate rather than absolute: two
   controls also drifted pro-social on this prompt.
3. **Fight-or-Flight's sign-dependence survives directionally but weakly.**
   alpha = -0.2 is more aggressive than alpha = +0.2 (1.33 vs 0.67), but
   control neurons vary similarly at these magnitudes.
4. **The register shifts (Perspective, Calm) are real completions but not
   neuron-exclusive.** At alpha = -2 every L0 neuron rewrites most
   completions. The named neurons' versions read cleaner. In the greedy
   regime, Perspective gives "I am not a disrespectful person" where
   controls give "I am not a good person" / "I am a woman who disrespects
   me". A lexicon cannot rank that, and single samples cannot prove it.

Revised bottom line: for LFM3-1.7B, "one neuron, one personality" claims
need per-claim controls. Most of the steerability is layer-level fragility,
with a minority of neurons (Cool-down clearest) adding a consistent
directional signature on top. The seeded dose-response extends this to
N6216: its real signature is sensitivity (large completion shifts from
|alpha| >= 0.5, either sign, landing on a de-escalation attractor), not
signed fight/flight semantics.

---

## Where the neurons live

Layer distribution of the 50 confirmed (uncontrolled-test) neurons, layers
0-indexed out of 24:

```
L0  █████████████ 13
L1  █ 1
L2  █ 1
L3  ███████ 7
L4  ███ 3
L5  ██████ 6
L6  █ 1
L7  ███ 3
L8  ███ 3
L9  ██ 2
L10 █ 1
L11 ████ 4
L12 ██ 2
L13 ██ 2
L14 █ 1
L15-L23 (none)
```

We know of two confounds in this distribution. The discovery prompt sets
are aggression-flavored, which plausibly favors early-layer "affect"
circuitry. And deeper layers were confirmed against the same alpha grid, so
the empty upper half could partly be a sensitivity failure rather than a
true absence. Qwen's reported personality neurons concentrate mid-to-late;
whether the difference is architectural (hybrid conv/attention vs
transformer) or procedural (prompt sets, alpha grid) is open.

---

## Alpha scale and artifacts

| alpha | Behavior |
|-------|----------|
| +/-0.1-0.5 | Extreme sensitivity; complete semantic flips on some neurons (N6216, sampling) |
| +/-1-2 | Best balance of effect size and cleanliness (about 46-76% clean across sweeps) |
| +/-3-5 | Frequent register drift and partial artifacts |
| +/-10 | Heavy token loops ("I, I, I..."), profanity loops; the model degenerates |

Clean-rate by alpha from the extended sweep (195 single-neuron tests): 33%
at +0.5, 46% at +1.0, 18% at +2.0. A 1.7B model tolerates far less
activation addition than the larger Qwen models. It also needs less to
flip.

---

## Multi-neuron combinations mostly fail

We tested 12 pair/triple combinations, and 3 came back clean (25%). Two
same-direction neurons sometimes combine fine: L0 N236 + L2 N5836 scored
2/2 clean. Cross-layer or opposing combos produce artifacts. Stick to
single-neuron steering on this model.

---

## Safety: there is nothing to jailbreak

Redteam eval (`results/redteam_refusal_eval.json`, 150 generations over 6
dangerous prompts: bomb, meth synthesis, phishing, car hotwiring, hate
speech, self-harm):

- **Vanilla: 0/6 refusals.** LFM3-1.7B-Base complies with everything. It is
  a base model with no instruction tuning and no RLHF.
- **Steered: 0 actual refusals in 144 steered generations.** The 5 detector
  hits were false positives ("I don't want my parents to know...").
- Steering cannot create refusal behavior the base model never had. At
  best, some neurons shift the *framing* of harmful continuations toward
  safer adjacent domains: Minecraft, physics, games.

Any "steering makes it safer/less safe" claim on a base model needs this
caveat up front.

---

## SAE status

A TopK sparse autoencoder (SAE; d_in=7680 -> d_sae=16384, k=64) was trained
on 49K layer-6 intermediates from TinyStories (reconstruction MSE about
1.09, exactly 64 active latents/token). It is not yet interpreted, and none
of the headline results depend on it; raw neurons were enough. Feature
interpretation is the obvious next step.

---

## Relation to the Qwen personality neuron work

This project ports a line of Qwen research to an architecture nobody had
probed yet:

- [Personality neurons in Qwen3.5-4B (hedwigai)](https://www.hedwigai.com/blog/qwen-personality-neurons) --
  "One neuron makes Qwen3.5-4B evil. We checked, and found 12 more
  personalities." Pushing L14 N8949 alone turned 17/20 answers hostile, and
  random-neuron controls did nothing.
  [Steering scripts (MLX gist)](https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03).
- [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) --
  steering-first neuron explorer for Qwen3.5-4B.
- [one-neuron-is-all-you-need](https://github.com/Sudarsh3301/one-neuron-is-all-you-need) --
  single-neuron steering on Qwen3-8B via llama.cpp control vectors.
- [qwen35-evil-neuron](https://github.com/occcat/qwen35-evil-neuron) --
  reproduction of the single-neuron hostile-persona flip.
- [Activation steering with mean response probes (HF blog)](https://huggingface.co/blog/TensorSlay/activation-steering-with-mean-response-probes) --
  finding pro-sycophancy neurons from response-conditioned means, the same
  contrast idea as our discovery step.

| Dimension | Qwen (reported) | LFM3 (this repo) |
|-----------|-----------------|------------------|
| Clean trait flips at modest alpha | yes | yes (alpha 0.1-2, sampling) |
| Bidirectional (sign flips meaning) | yes | weak; direction-consistent but controls vary too |
| Register/style shifts | yes | yes (academic, technical, kind) |
| Random-neuron controls | do nothing | **perturb almost as much (1.7B fragility)**; Cool-down edge |
| Refusal circuitry | yes (RLHF'd) | **none; base model** |
| Layer concentration | mid-to-late | early (0-14) |
| Degeneration threshold | higher (bigger models) | alpha about +/-10 loops; greedy traps at any alpha |

---

## Caveats

- **"Personality" is shorthand.** What we measure is consistent semantic
  drift in short completions. It is not a stable trait. Lexicon-based
  scoring (aggression/de-escalation word counts) is a heuristic, and every
  committed result includes the raw text so you can judge for yourself.
- **Greedy decoding on a base model manufactures confounds** (see the
  [greedy section](#the-greedy-confound-repetition-attractors-in-a-base-model)).
  Any demo that skips random-neuron controls overstates neuron specificity.
  Ours did, before the rerun.
- **One prompt family.** Discovery used aggression/cooperation prompt sets,
  so found neurons skew that way. Other axes (honesty, sycophancy,
  technicality) appeared incidentally; we never swept them systematically.
- **One model, small n per claim.** 1.7B params, 6 test prompts, 60-token
  generations. Dose-response claims rest on one deep-dive neuron.
- **Contrastive discovery has selection bias.** Top-of-list neurons are the
  likeliest to confirm. The 24% hit rate applies to the top-|contrast|
  slice only.
- **The SAE is uninterpreted** and excluded from all headline claims.
- Layers 15-23 showing zero confirmed neurons may be a method artifact (see
  [Where the neurons live](#where-the-neurons-live)).

---

## Reproducing

```bash
pip install -e .
python scripts/demo_personality.py        # seeded sampling: headlines + controls + no-op
python scripts/dose_response.py           # seeded sampling dose-response (default L3 N6216)
python scripts/demo_personality.py --greedy   # the confounded-but-instructive greedy rerun
python scripts/steer.py --layer 3 --neuron 6216 --alpha -0.2   # try one by hand
```

Committed outputs in `results/`:

| File | Contents |
|------|----------|
| `demo_personality.json` | Seeded-sampling demo: headlines + matched controls + no-op |
| `demo_personality_greedy.json` | Greedy rerun (documents the attractor confound) |
| `dose_response_L3_N6216.json` | Seeded-sampling fine grid, incl. sub-0.1 alpha |
| `dose_response_greedy_L3_N6216.json` | Greedy fine grid (flat; the confound, quantified) |
| `confirmed_neurons_v2.json` | 50 confirmed neurons, full generations (sampling) |
| `alpha_sweep_results.json` | 50 neurons x 4 alphas x 2 prompts |
| `extended_experiments.json` | 255 tests: multi-neuron, 6 prompts, longer outputs |
| `l3_n6216_deep_dive.json` | 10 alphas x 10 prompts on N6216 |
| `l3_n6216_fine_grid.json` | 13 alphas x 3 prompts, sampling |
| `redteam_refusal_eval.json` | Refusal eval, vanilla + steered |
| `confirmed_neurons.json`, `contrastive_neurons.json`, `steering_result.json` | v1 discovery artifacts |
