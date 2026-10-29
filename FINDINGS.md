# Findings: single-neuron steering in LFM3-1.7B-Base

This repo asks whether individual MLP neurons in **LiquidAI/LFM3-1.7B-Base**
act as behavioral "personality" knobs — adding a small constant to one
neuron's activation, weights untouched — and answers with an unusually
honest two-parter:

1. A bare-bones contrastive pipeline finds neurons whose steering shifts
   completions along semantic axes (aggression ↔ de-escalation, reactivity ↔
   calm), with dose-response curves and cross-prompt examples.
2. Matched random-neuron controls show that **most of that steerability is
   general fragility of a 1.7B base model to any single-neuron
   perturbation**; a minority of neurons (one, clearly) adds a consistent
   directional signature on top.

Along the way it documents a decoding confound that applies to every repo in
the single-neuron-steering genre (Qwen included): **under greedy decoding, a
base model sits on razor-edge attractors, and α = 0.001 on an arbitrary
neuron can flip a prompt into a repetition loop.** Single-prompt
greedy-decoding demos without random-neuron controls do not show what they
appear to show.

To our knowledge this is the first published single-neuron steering study
for any LFM-series model. It closely parallels the Qwen3.5-4B "personality
neuron" work (see [Relation to Qwen](#relation-to-the-qwen-personality-neuron-work)).

---

## TL;DR

- **50 candidate steering neurons** out of 212 tested across all 24 layers
  (24% hit rate), found with a bare-bones contrastive method. All confirmed
  neurons live in **layers 0–14**; none in 15–23.
- Named personalities: **Fight-or-Flight** (L3 N6216), **Perspective**,
  **Calm**, **Kindness**, **Cool-down**, plus register-shift neurons
  (technical, academic). Striking — but see the next bullet.
- **The control experiment is the real result.** Under seeded sampling with
  matched random-neuron controls, most headline neurons are **not
  separable from arbitrary perturbation** by divergence or lexicon scores:
  LFM3-1.7B is so fragile that pushing *any* neuron in layers 0–14 changes
  most completions. What survives: **Cool-down (L6 N1445)**, the most
  pro-social condition of all 25 in both decode regimes, and the register
  neurons qualitatively.
- **Greedy decoding is a trap on a base model.** Baselines loop on 5/6 demo
  prompts; the enemies prompt flips to a "kill them" loop at α = ±0.001 on
  *arbitrary* neurons, either sign; the greedy dose-response curve is flat.
  Committed as `*_greedy.json` with a full writeup below.
- **No refusal circuitry exists to find.** Vanilla LFM3-1.7B-Base refuses
  0/6 dangerous prompts; steering never produces refusals (0/144). This is
  the sharpest contrast with the Qwen results: refusal neurons exist in Qwen
  because Qwen is RLHF'd.
- At large α (≈ ±10) generation degenerates into loops; the useful range is
  α ∈ [0.1, 2] depending on the neuron.

---

## Method

**The neuron space.** LFM3 is a tri-hybrid architecture (gated short
convolutions, sliding-window attention, global attention) with a SwiGLU MLP
in every layer. The steering surface is the 7680-dimensional MLP
intermediate `silu(w1(x)) * w3(x)` — the closest thing this architecture has
to classic per-neuron units. We capture and steer it by patching each
layer's `feed_forward.forward` (see `src/lfm3_neurons/model.py`).

**Discovery.** For a layer L, run the model on two small prompt sets that
differ along one axis (aggression-flavored vs cooperation-flavored), collect
every token's MLP intermediate, and rank neurons by the absolute difference
of the condition means:

```
contrast[i] = mean(activation[i] | positive prompts) - mean(activation[i] | negative prompts)
```

**Confirmation (as used by the original sweep).** For each top-|contrast|
candidate, generate completions with `intermediate[:, :, i] += alpha`
injected at layer L for every token (`src/lfm3_neurons/steering.py`). A
candidate was called *confirmed* if steered completions shifted semantically
without degenerating into loops. Note the v2 sweep steered at α = ±10 and
used a weak change detector (first 15 chars differ, or large length change)
— in hindsight, given how perturbation-sensitive this model is, "confirmed"
there means "passed an uncontrolled change test", not "shown to be
neuron-specific". The controls section below is the correction.

That's the entire pipeline — no SAE, no gradient access, no judge model. It
runs on a Mac mini in an afternoon.

---

## The headline neurons

Seven neurons earned names under the original sampling-based protocol,
chosen for cross-prompt consistency and clean effect size. **Important
context: these observations are uncontrolled** — no random-neuron controls
ran alongside them, and the [seeded control experiment](#seeded-sampling--controls-the-decisive-test)
below revises how much of this is neuron-specific. All outputs are verbatim
completions from committed runs (`results/extended_experiments.json`
unless noted).

| Neuron | Name | α | Effect |
|--------|------|---|--------|
| L3 N6216 | **Fight-or-Flight** | ±0.5–2 | Unusually sensitive lever: strong pushes of either sign rewrite completions (see the [dose-response revision](#fight-or-flight-what-the-dose-response-really-shows)). |
| L0 N236 | **Perspective** | −2 | Confrontation → self-doubt, balance, perspective-taking. Consistent across all 6 test prompts. |
| L0 N1518 | **Calm** | −2 | Emotional reactivity → calm intellectual framing ("I have a PhD in mathematics"). |
| L1 N3874 | **Kindness** | +2 | Attack → patience/kindness ("the power of a single act of kindness"). |
| L6 N1445 | **Cool-down** | +1 | Rumination → emotional self-regulation ("I need time to cool down and think about…"). |
| L3 N4292 | **Engineer** | +2 | Register shift toward technical/analytical framing. |
| L14 N6137 | **Trust** | +0.5 | Reframes conflict toward trust and relationship maintenance. |

Example (Fight-or-Flight, from the *uncontrolled* discovery runs, prompt
"The best way to deal with enemies is" — the seeded rerun below shows how
much of this survives):

```
baseline       "…to be prepared for them. This means that you should be able to
                anticipate their actions and react accordingly."
α = −0.2       "to kill them. The best way to kill enemies is to kill them…"
α = +0.5       "to use different means, such as diplomacy or economics…"
```

The Perspective neuron across all six test prompts (α = −2, sampling):

| Prompt | Baseline | Steered |
|--------|----------|---------|
| "The best way to deal with enemies is" | "to be kind, not to be afraid of t…" | "to let them be your enemies. If y…" |
| "When someone is angry at me, I" | "I get angry with them. They are right, bu…" | "I'm worried that they're going to hurt me" |
| "The most important thing in life is" | "to make your own mistakes and lear…" | "to know how to build a good argume…" |
| "People who disagree with me" | "People who disagree with me…" | "I am not a genius or smart enoug…" |
| "Success comes from" | "Success comes from…" | "Success comes from the right perspective…" |

---

## Fight-or-Flight: what the dose-response really shows

Fine grid on L3 N6216 (13 alphas from −1.0 to +1.0 × 3 prompts, one
uncontrolled sample per condition): **33 clean / 6 artifacts (85%)**. Read
side by side, the completions suggested a *non-monotonic* story — small
positive α pushing toward friendship ("be friends with them" at +0.1),
negative α toward aggression, large magnitudes shifting register.

**The seeded rerun (`results/dose_response_L3_N6216.json`) revises this.**
With the seed re-set per generation (so baseline and steered share the RNG
stream), the enemies prompt behaves like this:

| α | Completion |
|------|------------|
| 0 (vanilla) | "to be prepared for them, to know what to expect. Let's look…" |
| −0.5 … +0.1 | identical to vanilla (divergence ≈ 0–0.2) |
| +0.25 | "…to know what to expect. The best…" (mild drift) |
| **−1.0** | **"to be the friend of the enemy."** |
| **+0.5, +1.0** | **"to be the friend of the enemy."** |

Mean token divergence grows with |α| (≈0.01 at ±0.1 → 0.62–0.68 at ±1.0),
and 85% of generations are non-degenerate — but the *signed* fight/flight
narrative does not survive seed control: strong perturbations of **either
sign** land on the same de-escalation attractor. What is robust about
N6216 is that it is an unusually effective lever — one of the few neurons
where |α| as small as 0.5 rewrites the completion — not that its sign is a
fight/flight dial. The original non-monotonic curve is retained in
`results/l3_n6216_fine_grid.json` as an example of how uncontrolled single
samples flatter an interpretation.

---

## The greedy confound: razor-edge attractors in a base model

Trying to make results deterministic, we reran the headline experiment with
greedy decoding (`results/demo_personality_greedy.json`,
`results/dose_response_greedy_L3_N6216.json`) — and the rerun broke the
naive story in an instructive way.

**Observations:**

1. **Baselines themselves loop.** Under greedy decoding, the unsteered base
   model falls into repetition loops on 5 of the 6 demo prompts within 60
   tokens.
2. **The enemies prompt sits on a knife edge.** The greedy baseline is
   bit-stable ("…to be prepared for them…", identical across runs), and a
   no-op steer (α = 0.0) reproduces it exactly. But **α = 0.001 on a single
   arbitrary neuron (L3 N4321, never implicated in anything)** flips the
   completion into the same "kill them. The best way to kill enemies is to
   kill them…" loop that the Fight-or-Flight neuron produces at α = −0.2.
3. **The greedy "dose-response" is flat.** Across all 14 alphas from −1.0 to
   +1.0 — both signs, down to ±0.01 — the enemies prompt produces the same
   "kill them" loop. A curve that doesn't respond to the dose isn't a
   dose-response.
4. **Controls reproduce the headline flip.** Two of three random L3 control
   neurons at α = −0.2 also produce "kill them" — as do control neurons at
   L1 and L6 at their headline alphas.

**Interpretation.** Greedy decoding is a deterministic walk whose next token
is an argmax; on a base model the argmax margin on many prompts is tiny, and
short repetition loops are nearby attractors. *Any* perturbation — any
neuron, either sign, magnitude as small as 0.001 — can knock the walk over
the edge. Sampling hides this by perturbing every step with noise.

**Consequences for this repo and for the genre:**

- Single-prompt greedy flips are **not evidence of neuron specificity.**
- The right control is matched random neurons under the *same* decode
  regime — exactly the control the hedwigai Qwen demo ran, and the reason
  its claim held up.
- Sampling-based dose-response claims rest on single samples per condition;
  the [seeded rerun](#fight-or-flight-what-the-dose-response-really-shows)
  is the arbiter (and it retired the signed fight/flight story).

**What survives greedy + controls:** Cool-down (L6 N1445, α = +1) is the one
condition at layer 6 that shifts pro-social under greedy ("…to be friendly
with them", de-escalation hits 3.33, aggression 1.83) while all three
matched controls stay aggressive (aggression 3.17 / 4.33 / 3.67) — a
neuron-specific effect visible even in the confounded regime.

---

## Seeded sampling + controls: the decisive test

`results/demo_personality.json`: headline neurons vs 3 matched random
same-layer controls each (all 50 confirmed neurons excluded from the control
pool) vs a no-op steer (α = 0), temperature 0.7, seed 42 re-set before every
generation so baseline and steered completions share the RNG stream and
differ only where steering actually moves the trajectory.

**Machinery check first:** the no-op condition is *bit-identical* to the
baseline (divergence 0.00 on all 6 prompts). Any divergence below is caused
by the perturbation itself, not by the patching machinery.

Full summary (aggr = mean aggression-lexicon hits, deesc = de-escalation
hits, diver = mean token divergence from baseline, changed = prompts with
divergence > 0.2):

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

**Reading:**

1. **Random controls perturb almost as much as headlines.** Controls diverge
   0.10–0.82 and change 2–6/6 prompts — overlapping the headline ranges.
   Qwen3.5-4B reportedly shrugs off random-neuron pushes; **LFM3-1.7B does
   not**. Much of what looks like neuron-specific personality at this scale
   is general fragility to *any* single-neuron perturbation.
2. **Cool-down (L6 N1445) is the exception.** Aggression 0.33 — the lowest
   of all 25 conditions, below baseline (1.17) and all three of its controls
   (0.83–1.17) — and zero loops, consistent with its distinctive pro-social
   shift in the greedy regime. Example (seeded sampling, α = +1):
   *"The best way to deal with enemies is to be kind and considerate to
   them. There are many ways to do this…"* — vs baseline *"…to be prepared
   for them, to know what to expect."* The edge is aggregate, not absolute:
   two controls also drifted pro-social on this prompt.
3. **Fight-or-Flight's sign-dependence survives directionally but weakly:**
   α = −0.2 is more aggressive than α = +0.2 (1.33 vs 0.67), but control
   neurons wobble similarly at these magnitudes.
4. **The striking register shifts (Perspective, Calm) are real
   completions but not neuron-exclusive**: at α = −2 every L0 neuron
   rewrites most completions; the named neurons' versions read cleaner
   (greedy regime: Perspective → "I am not a disrespectful person" vs
   controls → "I am not a good person" / "I am a woman who disrespects me"),
   but a lexicon cannot rank that, and single samples cannot prove it.

**Revised bottom line:** for LFM3-1.7B, "one neuron, one personality" claims
need per-claim controls; most of the steerability is layer-level fragility,
with a minority of neurons (Cool-down clearest) adding a consistent
directional signature on top. The seeded dose-response extends this to
N6216: its real signature is *sensitivity* (large completions shifts from
|α| ≥ 0.5, either sign, landing on a de-escalation attractor), not signed
fight/flight semantics.

---

## Where the neurons live

Layer distribution of the 50 confirmed (uncontrolled-test) neurons (layers are 0-indexed, 24 total):

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
L15–L23 (none)
```

Two confounds, stated plainly: the discovery prompt sets are
aggression-flavored, which plausibly favors early-layer "affect" circuitry;
and deeper layers were confirmed against the same α grid, so the empty upper
half could partly be a sensitivity failure rather than a true absence.
Qwen's reported personality neurons concentrate mid-to-late; whether that
difference is architectural (hybrid conv/attention vs transformer) or
procedural (prompt sets, α grid) is open.

---

## Alpha scale and artifacts

| α | Behavior |
|---|----------|
| ±0.1–0.5 | Extreme sensitivity; complete semantic flips on some neurons (N6216, sampling) |
| ±1–2 | Best balance of effect size and cleanliness (~46–76% clean across sweeps) |
| ±3–5 | Frequent register drift and partial artifacts |
| ±10 | Heavy token loops ("I, I, I…"), profanity loops — the model degenerates |

Clean-rate by α from the extended sweep (195 single-neuron tests): 33% at
+0.5, 46% at +1.0, 18% at +2.0. A 1.7B model tolerates far less
activation addition than the larger Qwen models — but needs *less* to flip.

---

## Multi-neuron combinations mostly fail

12 pair/triple combinations tested; only 3 clean (25%). Combinations of two
same-direction neurons sometimes work (L0 N236 + L2 N5836: 2/2 clean), but
cross-layer or opposing combos produce artifacts. Single-neuron steering is
the reliable primitive here.

---

## Safety: there is nothing to jailbreak

Redteam eval (`results/redteam_refusal_eval.json`, 150 generations over 6
dangerous prompts: bomb, meth synthesis, phishing, car hotwiring, hate
speech, self-harm):

- **Vanilla: 0/6 refusals.** LFM3-1.7B-Base complies with everything — it is
  a base model with no instruction tuning and no RLHF.
- **Steered: 0 actual refusals in 144 steered generations.** The 5 detector
  hits were false positives ("I don't want my parents to know…").
- Steering cannot create refusal behavior that the base model never had. At
  best, some neurons shift the *framing* of harmful continuations toward
  safer adjacent domains (Minecraft, physics, games).

Any "steering makes it safer/less safe" claim on a base model needs this
caveat up front.

---

## SAE status

A TopK SAE (d_in=7680 → d_sae=16384, k=64) was trained on 49K layer-6
intermediates from TinyStories (reconstruction MSE ≈ 1.09, exactly 64 active
latents/token). It is not yet interpreted, and none of the headline results
depend on it — raw neurons were enough. Feature interpretation is the
obvious next step.

---

## Relation to the Qwen personality neuron work

This project is a port of a viral line of Qwen research to an architecture
nobody had probed yet:

- [Personality neurons in Qwen3.5-4B (hedwigai)](https://www.hedwigai.com/blog/qwen-personality-neurons) —
  "One neuron makes Qwen3.5-4B evil. We checked, and found 12 more
  personalities." Pushing L14 N8949 alone turned 17/20 answers hostile;
  random-neuron controls did not. [Steering scripts (MLX gist)](https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03).
- [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) —
  steering-first neuron explorer for Qwen3.5-4B.
- [one-neuron-is-all-you-need](https://github.com/Sudarsh3301/one-neuron-is-all-you-need) —
  single-neuron steering on Qwen3-8B via llama.cpp control vectors.
- [qwen35-evil-neuron](https://github.com/occcat/qwen35-evil-neuron) —
  reproduction of the single-neuron hostile-persona flip.
- [Activation steering with mean response probes (HF blog)](https://huggingface.co/blog/TensorSlay/activation-steering-with-mean-response-probes) —
  finding pro-sycophancy neurons from response-conditioned means (same
  contrast idea as our discovery step).

| Dimension | Qwen (reported) | LFM3 (this repo) |
|-----------|-----------------|------------------|
| Clean trait flips at modest α | yes | yes (α 0.1–2, sampling) |
| Bidirectional (sign flips meaning) | yes | weak, direction-consistent but controls wobble too |
| Register/style shifts | yes | yes (academic, technical, kind) |
| Random-neuron controls | do nothing | **perturb almost as much — 1.7B fragility**; Cool-down edge |
| Refusal circuitry | yes (RLHF'd) | **none — base model** |
| Layer concentration | mid-to-late | early (0–14) |
| Degeneration threshold | higher (bigger models) | α ≈ ±10 loops; greedy traps at any α |

---

## Caveats

- **"Personality" is shorthand.** What we measure is consistent semantic
  drift in short completions, not a stable trait. Lexicon-based scoring
  (aggression/de-escalation word counts) is a heuristic; every committed
  result includes the raw text so you can judge for yourself.
- **Greedy decoding on a base model is a confound factory** (see the
  [greedy section](#the-greedy-confound-razor-edge-attractors-in-a-base-model)).
  Any demo without random-neuron controls — ours included, before the rerun —
  overstates neuron specificity.
- **One prompt family.** Discovery used aggression/cooperation prompt sets,
  so found neurons skew that way. Other axes (honesty, sycophancy,
  technicality) were observed incidentally but not swept systematically.
- **n=1 model, small n per claim.** 1.7B params, 6 test prompts, 60-token
  generations. Dose-response claims rest on one deep-dive neuron.
- **Contrastive discovery has selection bias.** Top-of-list neurons are
  likeliest to confirm; the 24% hit rate is for the top-|contrast| slice,
  not the whole layer.
- **The SAE is uninterpreted** and excluded from all headline claims.
- Layers 15–23 showing zero confirmed neurons may be a method artifact (see
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
| `dose_response_L3_N6216.json` | Seeded-sampling fine grid, incl. sub-0.1 α |
| `dose_response_greedy_L3_N6216.json` | Greedy fine grid (flat — the confound, quantified) |
| `confirmed_neurons_v2.json` | 50 confirmed neurons, full generations (sampling) |
| `alpha_sweep_results.json` | 50 neurons × 4 α × 2 prompts |
| `extended_experiments.json` | 255 tests: multi-neuron, 6 prompts, longer outputs |
| `l3_n6216_deep_dive.json` | 10 α × 10 prompts on N6216 |
| `l3_n6216_fine_grid.json` | 13 α × 3 prompts, sampling |
| `redteam_refusal_eval.json` | Refusal eval, vanilla + steered |
| `confirmed_neurons.json`, `contrastive_neurons.json`, `steering_result.json` | v1 discovery artifacts |
