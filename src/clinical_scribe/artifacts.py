"""Helpers for run artifacts (ASR, corrected transcripts, notes)."""
from __future__ import annotations

import re


def slug(encounter_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", encounter_id).strip("-")


def transcript_for_prompt(payload: dict) -> str:
    """Prefer speaker-labelled utterances; fall back to plain text."""
    utterances = payload.get("utterances") or []
    if utterances:
        return "\n".join(
            f"{str(u.get('speaker', 'unknown')).capitalize()}: {u.get('text', '')}"
            for u in utterances
            if u.get("text")
        )
    return payload.get("text", "")
