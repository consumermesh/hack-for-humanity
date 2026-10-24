"""Text normalisation and lightweight matching used by ingest and evaluation."""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^a-z0-9/ ]+")
_WS = re.compile(r"\s+")

# Abbreviations clinicians use in notes but rarely say aloud.
_ABBREV = {
    "llq": "left lower quadrant",
    "rlq": "right lower quadrant",
    "luq": "left upper quadrant",
    "ruq": "right upper quadrant",
    "hx": "history",
    "pmh": "past medical history",
    "dh": "drug history",
    "sh": "social history",
    "imp": "impression",
    "dx": "diagnosis",
    "sx": "symptoms",
}
_NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_STOP = {"of", "the", "a", "an", "and", "to", "in", "on", "for", "with", "is", "was", "has", "had"}


def expand_clinical(text: str) -> str:
    """Spell out note shorthand so '3/7 hx of diarrhea' can match speech."""
    text = re.sub(r"\b(\d+)\s*/\s*7\b", r"\1 day", text)
    for short, long in _ABBREV.items():
        text = re.sub(rf"\b{short}\b", long, text)
    for word, digit in _NUM_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return text


def normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation, and collapse whitespace.

    Keeps digits and '/' so clinically meaningful forms like '3/7' survive.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("-", " ")
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def shingles(tokens_: list[str], n: int = 3) -> set[tuple[str, ...]]:
    if len(tokens_) < n:
        return {tuple(tokens_)} if tokens_ else set()
    return {tuple(tokens_[i : i + n]) for i in range(len(tokens_) - n + 1)}


def containment(needle: str, haystack: str, n: int = 3, threshold: float = 0.6) -> float:
    """Fraction of `needle`'s n-grams that occur in `haystack`.

    Robust to small ASR spelling differences and filler words, unlike exact
    substring matching. Returns 1.0 for an exact normalised substring.
    """
    needle_norm, hay_norm = expand_clinical(normalize(needle)), expand_clinical(normalize(haystack))
    if not needle_norm:
        return 0.0
    if needle_norm in hay_norm:
        return 1.0
    needle_sh = shingles(needle_norm.split(), n)
    if not needle_sh:
        return 0.0
    hay_sh = shingles(hay_norm.split(), n)
    return len(needle_sh & hay_sh) / len(needle_sh)


def token_coverage(needle: str, haystack: str) -> float:
    """Share of the needle's content words present anywhere in the haystack.

    Order-insensitive and tolerant of paraphrases ("LLQ pain" vs "left lower
    abdominal pain"), which strict n-gram matching misses.
    """
    needle_tokens = [
        t for t in expand_clinical(normalize(needle)).split() if t not in _STOP
    ]
    if not needle_tokens:
        return 0.0
    hay_tokens = set(expand_clinical(normalize(haystack)).split())
    return sum(1 for t in needle_tokens if t in hay_tokens) / len(needle_tokens)


def is_supported(quote: str, source: str, threshold: float = 0.6) -> bool:
    return bool(normalize(quote)) and containment(quote, source, threshold=threshold) >= threshold


def split_into_items(note: str) -> list[str]:
    """Split a free-text gold note into candidate facts (lines / sentences)."""
    lines: list[str] = []
    for chunk in re.split(r"[\n;]+", note):
        chunk = chunk.strip(" \t-*")
        if not chunk:
            continue
        lines.extend(s for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip())
    return lines
