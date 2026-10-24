"""Gemma-assisted, gated repair of medical terms mangled by ASR.

The model proposes edits; a deterministic gate decides whether to apply them.
A correction is only accepted when the changed words are phonetically close
(same Soundex) or orthographically close, and no content is added or removed.
That blocks the classic failure mode where an LLM "fixes" a term into a
different diagnosis.
"""
from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field

from .schema import Utterance

SYSTEM = """You are a medical transcription editor. The transcript comes from \
automatic speech recognition and may mis-hear medical words.

You may ONLY correct medical terms, drug names, anatomy, abbreviations, and \
lab values that were mis-heard. Rules:
1. The corrected word must sound like what was transcribed (a homophone or \
close ASR error), e.g. "diary" -> "diarrhea", "microgynan" -> "microgynon".
2. Never change the meaning, never add or remove information, never rewrite \
general English, and never substitute a different diagnosis.
3. `original` must be an exact substring of that line, character for character.
4. If nothing is clearly wrong, return an empty edits list.
Return ONLY JSON: {"edits":[{"line":0,"original":"...","corrected":"...","reason":"..."}]}"""

USER_TEMPLATE = """Transcript lines (line number: speaker: text):
{lines}

Return the JSON object."""

_SOUNDEX = {
    **{c: "1" for c in "bfpv"},
    **{c: "2" for c in "cgjkqsxz"},
    **{c: "3" for c in "dt"},
    "l": "4",
    **{c: "5" for c in "mn"},
    "r": "6",
}


def soundex(word: str) -> str:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    first = word[0].upper()
    last = _SOUNDEX.get(word[0], "")
    digits: list[str] = []
    for ch in word[1:]:
        code = _SOUNDEX.get(ch, "")
        if code and code != last:
            digits.append(code)
        last = code
    return (first + "".join(digits) + "000")[:4]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similar_words(a: str, b: str) -> bool:
    a_norm, b_norm = re.sub(r"[^a-z0-9]", "", a.lower()), re.sub(r"[^a-z0-9]", "", b.lower())
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    if soundex(a_norm) == soundex(b_norm):
        return True
    ratio = 1 - levenshtein(a_norm, b_norm) / max(len(a_norm), len(b_norm))
    return ratio >= 0.6


@dataclass
class Edit:
    line: int
    original: str
    corrected: str
    reason: str = ""
    accepted: bool = False
    rejection: str = ""


@dataclass
class CorrectionResult:
    utterances: list[Utterance]
    edits: list[Edit] = field(default_factory=list)
    raw: str = ""
    latency_s: float = 0.0
    model: str = ""

    @property
    def accepted(self) -> list[Edit]:
        return [e for e in self.edits if e.accepted]

    @property
    def rejected(self) -> list[Edit]:
        return [e for e in self.edits if not e.accepted]


def build_lines(utterances: list[Utterance]) -> str:
    return "\n".join(
        f'{i}: {u.speaker.capitalize()}: {u.text}' for i, u in enumerate(utterances)
    )


def check_edit(original: str, corrected: str) -> tuple[bool, str]:
    if original == corrected:
        return False, "no change"
    a_words, b_words = original.split(), corrected.split()
    if abs(len(a_words) - len(b_words)) > 2:
        return False, "too many words changed"
    matcher = difflib.SequenceMatcher(None, a_words, b_words)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("insert", "delete"):
            return False, f"{tag} introduces or drops content"
        for x, y in zip(a_words[i1:i2], b_words[j1:j2]):
            if not similar_words(x, y):
                return False, f"'{x}' -> '{y}' is not a phonetic/orthographic match"
            changed += 1
    if not changed:
        return False, "no changed words"
    return True, ""


def apply_edits(utterances: list[Utterance], edits: list[Edit]) -> list[Utterance]:
    out = [u.model_copy() for u in utterances]
    for edit in edits:
        if not (0 <= edit.line < len(out)):
            edit.rejection = "line out of range"
            continue
        target = out[edit.line].text
        if edit.original.lower() not in target.lower():
            edit.rejection = "original not found in line"
            continue
        ok, reason = check_edit(edit.original, edit.corrected)
        if not ok:
            edit.rejection = reason
            continue
        idx = target.lower().index(edit.original.lower())
        out[edit.line].text = target[:idx] + edit.corrected + target[idx + len(edit.original) :]
        edit.accepted = True
    return out


def parse_edits(raw: str) -> list[Edit]:
    from .gemma import parse_json_object

    obj = parse_json_object(raw)
    edits = []
    for item in obj.get("edits") or []:
        try:
            edits.append(
                Edit(
                    line=int(item["line"]),
                    original=str(item["original"]),
                    corrected=str(item["corrected"]),
                    reason=str(item.get("reason", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return edits


def correct_transcript(client, utterances: list[Utterance]) -> CorrectionResult:
    started = time.perf_counter()
    raw = client.chat(SYSTEM, USER_TEMPLATE.format(lines=build_lines(utterances)))
    edits = parse_edits(raw)
    corrected = apply_edits(utterances, edits)
    return CorrectionResult(
        utterances=corrected,
        edits=edits,
        raw=raw,
        latency_s=time.perf_counter() - started,
        model=client.config.model if getattr(client, "config", None) else "",
    )
