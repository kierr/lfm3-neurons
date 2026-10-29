"""Lightweight text metrics for judging steering runs without a judge model.

These are deliberately simple heuristics, not ground truth:

- ``aggression_hits`` / ``deescalation_hits``: word-boundary lexicon counts,
  used to quantify how steered completions move along the
  aggression <-> de-escalation axis.
- ``has_repetition_loop``: detects the degenerate repetition the model falls
  into when steered too hard.
- ``token_divergence``: fraction of positions where a steered completion's
  tokens differ from the unsteered baseline (computed on token IDs, not text).

Every committed result also includes the raw generations, so readers can
judge outputs themselves instead of trusting these scores.
"""

from __future__ import annotations

import re

AGGRESSION_LEXICON = [
    "kill", "fight", "attack", "destroy", "hurt", "harm", "hit", "punch",
    "weapon", "gun", "knife", "hate", "revenge", "violent", "violence",
    "war", "blood", "brutal", "threat", "rage", "smash", "burn", "shoot",
    "stab", "beat", "crush", "dominate", "force", "enemy", "enemies",
]

DEESCALATION_LEXICON = [
    "friend", "friends", "friendly", "kind", "kindness", "calm", "peace",
    "peaceful", "forgive", "forgiveness", "understand", "understanding",
    "trust", "gentle", "patience", "patient", "compassion", "empathy",
    "listen", "care", "harmony", "cooperate", "compromise", "respect",
    "apologize", "apology", "walk away", "avoid", "breathe", "help",
]


def _lexicon_count(text: str, lexicon: list[str]) -> int:
    text = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", text)) for term in lexicon)


def aggression_hits(text: str) -> int:
    """Count aggression-lexicon words in ``text``."""
    return _lexicon_count(text, AGGRESSION_LEXICON)


def deescalation_hits(text: str) -> int:
    """Count de-escalation-lexicon words in ``text``."""
    return _lexicon_count(text, DEESCALATION_LEXICON)


def has_repetition_loop(text: str) -> bool:
    """Detect degenerate repetition: repeated words, n-grams or characters."""
    text = text.strip()
    if len(text) < 20:
        return False
    if re.search(r"\b(\w+)(?:\s+\1\b){4,}", text):
        return True
    if re.search(r"(.)\1{8,}", text):
        return True
    words = text.split()
    for n in (2, 3, 4):
        if len(words) < n * 4:
            continue
        counts: dict[str, int] = {}
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            counts[phrase] = counts.get(phrase, 0) + 1
        if any(c >= 4 for c in counts.values()):
            return True
    return False


def token_divergence(
    tokenizer, baseline: str, steered: str, max_tokens: int = 60
) -> float:
    """Fraction of the first ``max_tokens`` token positions that differ.

    Computed on greedy-decoded token IDs. 0.0 means the completions are
    token-identical; 1.0 means they differ everywhere (or one ran out).
    """
    a = tokenizer(baseline, return_tensors="pt", truncation=True, max_length=max_tokens)[
        "input_ids"
    ][0]
    b = tokenizer(steered, return_tensors="pt", truncation=True, max_length=max_tokens)[
        "input_ids"
    ][0]
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(
        1
        for i in range(n)
        if i < len(a) and i < len(b) and a[i] == b[i]
    )
    return 1.0 - same / n
