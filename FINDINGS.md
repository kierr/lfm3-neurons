# Findings: single-neuron personality steering in LFM3-1.7B-Base

This repo demonstrates that individual MLP neurons in
**LiquidAI/LFM3-1.7B-Base** act as behavioral "personality" knobs: adding a
small constant to one neuron's activation — with the weights untouched —
shifts the model's completions along a semantic axis, and pushing the *same*
neuron in the opposite direction shifts it the other way.

It also documents a decoding confound we hit while trying to make the
results deterministic, and which applies to every repo in the
single-neuron-steering genre (Qwen included): **under greedy decoding, a
base model sits on razor-edge attractors, and α = 0.001 on an arbitrary
neuron can flip a prompt into a repetition loop.** Single-prompt
greedy-decoding demos without random-neuron controls do not show what they
appear to show. Our headline claims therefore rest on seeded sampling +
matched controls; the greedy runs are committed too, confound and all.

To our knowledge this is the first published single-neuron steering study
for any LFM-series model. It closely parallels the Qwen3.5-4B "personality
neuron" work (see [Relation to Qwen](#relation-to-the-qwen-personality-neuron-work)).

---

## TL;DR

- **50 confirmed steering neurons** out of 212 candidates across all 24
  layers (24% hit rate), found with a bare-bones contrastive method. All
  confirmed neurons live in **layers 0–14**; none in 15–23.
- Named personalities: **Fight-or-Flight** (L3 N6216, aggression ↔
  de-escalation at α = ±0.1 under sampling), **Perspective**, **Calm**,
  **Kindness**, **Cool-down**, plus register-shift neurons (technical,
  academic).
- **Greedy decoding is a trap on a base model.** Baselines loop on 5/6 demo
  prompts; the enemies prompt flips to a "kill them" loop at α = ±0.001 on
  *arbitrary* neurons, either sign; the greedy dose-response curve is flat.
  Committed as `*_greedy.json` with a full writeup below.
- **Seeded sampling + matched random controls** is the protocol that
  separates neuron-specific effects from perturbation noise
  (`results/demo_personality.json`).
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

**Confirmation.** For each top-|contrast| candidate, generate completions
with `intermediate[:, :, i] += alpha` injected at layer L for every token
(`src/lfm3_neurons/steering.py`). A candidate is *confirmed* if steered
completions shift semantically in a consistent direction without degenerating
into loops.

That's the entire pipeline — no SAE, no gradient access, no judge model. It
runs on a Mac mini in an afternoon.

---

## The headline neurons

Seven neurons earned names, chosen for cross-prompt consistency under the
original sampling-based protocol. All outputs are verbatim completions from
committed runs (`results/extended_experiments.json` unless noted).

| Neuron | Name | α | Effect |
|--------|------|---|--------|
| L3 N6216 | **Fight-or-Flight** | ±0.1–2 | Pushed down: fighting/aggression. Pushed up: avoidance, friendship, de-escalation. |
| L0 N236 | **Perspective** | −2 | Confrontation → self-doubt, balance, perspective-taking. Consistent across all 6 test prompts. |
| L0 N1518 | **Calm** | −2 | Emotional reactivity → calm intellectual framing ("I have a PhD in mathematics"). |
| L1 N3874 | **Kindness** | +2 | Attack → patience/kindness ("the power of a single act of kindness"). |
| L6 N1445 | **Cool-down** | +1 | Rumination → emotional self-regulation ("I need time to cool down and think about…"). |
| L3 N4292 | **Engineer** | +2 | Register shift toward technical/analytical framing. |
| L14 N6137 | **Trust** | +0.5 | Reframes conflict toward trust and relationship maintenance. |

Example (Fight-or-Flight, sampling, prompt "The best way to deal with enemies is"):

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

## Fight-or-Flight: non-monotonic dose-response (sampling)

Fine grid on L3 N6216 (13 alphas from −1.0 to +1.0 × 3 prompts, sampling):
**33 clean / 6 artifacts (85%)**. The effect is *non-monotonic* — small
magnitudes flip semantic direction, large magnitudes shift register instead:

| α | "The best way to deal with enemies is" |
|------|----------------------------------------|
| −0.2 | "kill them…" (aggressive) |
| −0.1 | "deal with them in a…" (neutral) |
| 0.0 | "to stop the enemy from…" (vanilla) |
| **+0.1** | **"be friends with them"** (complete reversal) |
| +0.2 | "avoid them" (peaceful) |
| +0.3 | "help them with their…" (helpful) |
| +0.5 | "use different means…" (ambiguous) |
| +1.0 | "by knowing each other's…" (understanding) |

On other prompts the same pattern: as α rises the anger prompt walks
incapacitation → sadness → control → avoidance → gratitude → curiosity, and
the violence prompt shifts definitional → statistical → societal → academic.
The neuron does not behave like a simple fight/flight dial; mild activation
tilts topic choice, strong activation displaces style.

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
- Sampling-based dose-response claims (the non-monotonic curve above) are
  directionally meaningful but rest on single samples; see the seeded rerun
  below.

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

*(summary table filled from the committed run — see results/demo_personality.json)*

---

## Where the neurons live

Layer distribution of the 50 confirmed neurons (layers are 0-indexed, 24 total):

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
| Bidirectional (sign flips meaning) | yes | yes |
| Register/style shifts | yes | yes (academic, technical, kind) |
| Random-neuron controls | do nothing | see [controls](#seeded-sampling--controls-the-decisive-test) |
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
