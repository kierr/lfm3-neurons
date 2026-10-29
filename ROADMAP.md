# Roadmap

Status first, so this list doesn't overstate what's open. Already shipped:
contrastive discovery across all 24 layers (50 candidates, 7 named). Also
shipped: the seeded-sampling protocol with matched controls and a no-op
condition, dose-response curves on N6216 in both decode regimes, the
greedy-attractor documentation, the refusal eval, and a
trained-but-uninterpreted sparse autoencoder (SAE). Everything committed
under `results/` is reproducible from `scripts/`.

Priorities are rough. If you pick something up, see `AGENTS.md` for the
environment facts and the controls-first protocol, then commit your JSON
and update `FINDINGS.md`.

## P1. Tighten the claims we already made

**Cool-down hardening (the top pick).** The one result that beat matched
controls in both decode regimes rests on 3 controls. The seeded-sampling
dose-response already runs per neuron (`scripts/dose_response.py --layer
--neuron`). The cheap version: run the same script on Cool-down plus 10-20
random layer-6 controls, at alpha +/-0.2 and +/-1.0, with more prompts.
Then report the distribution of the aggression gap between headline and
controls. If the pro-social lean survives that, it deserves stronger
language in FINDINGS.

**Enrichment curves.** We know qualitatively that headline neurons perturb
about as much as controls. The quantitative statement is a pair of curves.
For each alpha, plot P(completion changed | headline neuron) against
P(completion changed | control). This needs one script over existing JSON,
plus maybe one new sweep. It would also show at which alpha, if any,
headline neurons start separating from controls.

**Layers 15-23.** Zero confirmed neurons in the upper half. Before
believing it: different contrast sets (technical vs casual, honesty vs
deception), per-position contrasts (last token only), larger alphas with
repetition-penalty decoding, and hooking attention outputs instead of the
MLP intermediate.

## P2. New axes, same controls protocol

- **Honesty and sycophancy.** The Qwen trend's classic axes. New contrast
  prompt sets, then the same find -> steer -> controls pipeline. On a base
  model these may not exist; a clean negative result with controls is
  publishable too.
- **Position-restricted steering.** `MLPInterceptor` currently steers every
  token. Restricting injection to prompt tokens, or to generated tokens
  only, is a small change and may sharpen effects.
- **Activation-normalized doses.** Alphas are raw units today. Dosing as
  alpha = k * sigma_i (each neuron's own activation std) makes effect
  sizes comparable across neurons and layers. neuron-explorer does this
  relative to the median residual norm; per-neuron sigma is the simpler
  version.

## P3. Tooling: an explorer for LFM3

The Qwen projects set the bar here. [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)
sweeps whole layers, saves every rollout with logprobs, computes per-neuron
delta-logprob and KL stats, and serves a 3D web map with live steered chat.
[one-neuron-is-all-you-need](https://github.com/Sudarsh3301/one-neuron-is-all-you-need)
ranks all 344K neurons by a persona-direction score, judges outputs with a
rubric and bootstrap confidence intervals, and ships a static HTML explorer
plus a 3D map. We want the LFM3 version, sized for MPS (Apple's GPU
backend). In order:

1. **Logit lens per neuron (cheap, do first).** A neuron's write vector is
   one column of `w2`. Project columns through the final norm and
   `lm_head` to get the tokens each neuron promotes or suppresses. That is
   7680 columns per layer against one 2048 x 128K matrix, which takes
   seconds on MPS and no generation. one-neuron-is-all-you-need's `lens.py`
   is the template.
2. **Delta-logprob screening.** For every neuron in a layer, run one
   forward pass with the steering patch on a fixed prompt set. Record
   next-token logprob shifts and KL from baseline. That is about 7680
   forwards per layer, an overnight MPS run when batched. It ranks neurons
   by causal effect without generating anything.
3. **Static HTML explorer.** A neuron picker, a dose slider, and the
   committed rollouts, rendered from JSON. No server, works on GitHub
   Pages. The dose-response and demo JSONs already have the right shape.
4. **Rollout sweeps as JSONL.** Sampled subsets of neurons x doses x
   prompts, saved with token ids and logprobs; neuron-explorer's run
   format is the model. Full-layer sweeps at 10-20 s per generation are
   not MPS-realistic for all 7680 neurons. Subsets plus the screening pass
   above cover the gap.
5. **LLM judge with a rubric.** Replace the lexicon heuristics with a
   fixed six-label rubric and bootstrap confidence intervals on labeled
   outputs. The lexicons stay, labeled as heuristics.
6. **3D map.** Neurons as points, layers as planes, colored by logit-lens
   or screening scores, clickable into the HTML explorer. Last, and only
   if 1-4 land.

## P4. SAE interpretation

The TopK SAE (d_in 7680 -> d_sae 16384, k=64) sits uninterpreted in
`data/sae/`. Pull top-activating examples per latent, steer with latents
instead of raw neurons, run the same controls, and compare how often each
one steers completions beyond what matched controls do. The logit lens
through `lm_head` applies to decoder directions too.

## P5. Infra and community

- **Notebook walkthrough.** One notebook: load model, capture, steer one
  neuron, run a mini control comparison. The fastest way for a newcomer to
  trust the method.
- **Release a small activation sample.** The 1.4 GB full tensor stays
  gitignored, but a 50K-token subset cast to float16 is about 300 MB.
  Share it as a Hugging Face dataset so people can work without
  re-extracting.
- **Cross-model results.** The porting guide is in AGENTS.md. A second
  architecture (LFM2, or a small Qwen run through the same controls
  protocol) would test whether the fragility finding tracks scale or model
  family.
