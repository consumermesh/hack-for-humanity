"""Shared data models for encounters, transcripts, and extracted notes.

The `ClinicalNote` model is both the validation target for Gemma's JSON output
and the contract the evaluation harness scores against.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Section = Literal[
    "chief_complaint",
    "hpi",
    "ros",
    "pmh",
    "medication",
    "allergy",
    "social",
    "family",
    "assessment",
    "plan",
    "follow_up",
]

Speaker = Literal["patient", "clinician", "unknown"]


class Fact(BaseModel):
    """A single clinically meaningful statement extracted from the transcript."""

    section: Section
    text: str = Field(description="Concise clinical phrasing of the fact.")
    evidence: str = Field(
        default="",
        description="Verbatim transcript quote that supports the fact. Empty if not stated.",
    )
    speaker: Speaker = "unknown"


class ClinicalNote(BaseModel):
    chief_complaint: str = ""
    facts: list[Fact] = Field(default_factory=list)
    note_text: str = Field(
        default="",
        description="Clinician-style note (HPI / PMH / DH / SH / Imp / Plan) for note-level scoring.",
    )

    def facts_by_section(self) -> dict[str, list[Fact]]:
        out: dict[str, list[Fact]] = {}
        for fact in self.facts:
            out.setdefault(fact.section, []).append(fact)
        return out


class Utterance(BaseModel):
    start: float
    end: float
    text: str
    speaker: Speaker = "unknown"


class Encounter(BaseModel):
    """One audio recording plus optional references, in a source-agnostic shape."""

    id: str
    source: Literal["primock57", "imerit"]
    language: str = "en-GB"
    audio: list[str] = Field(default_factory=list)
    reference_transcript: list[Utterance] = Field(default_factory=list)
    reference_note: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    def transcript_text(self) -> str:
        """Speaker-labelled transcript for prompting."""
        return "\n".join(
            f"{u.speaker.capitalize()}: {u.text}" for u in self.reference_transcript
        )

    def transcript_plain(self) -> str:
        """Unlabelled transcript, comparable to raw ASR output for WER."""
        return " ".join(u.text for u in self.reference_transcript)
