"""Prompts for Gemma clinical-note extraction."""
from __future__ import annotations

SECTIONS = [
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

SYSTEM = """You are a careful clinical documentation assistant. You convert a \
verbatim transcript of a clinician-patient consultation into a structured note.

Rules:
1. Use ONLY information explicitly stated in the transcript. Never invent a \
symptom, diagnosis, medication, dose, duration, or history.
2. Preserve negatives and uncertainty: "no blood in stool", "denies fever", \
"possible", "cannot rule out" must stay negative/uncertain.
3. Every fact must include `evidence`: an exact, verbatim quote from the \
transcript (copy the words, do not paraphrase). If no quote supports it, omit \
the fact.
4. Attribute `speaker` as patient or clinician when the transcript labels turns; \
otherwise use unknown.
5. Do not promote a patient's guess ("maybe it's my gallbladder") into a \
clinician assessment.
6. If a section was not discussed, leave it out. Do not write "not discussed".
7. `note_text` must read like a clinician's note using these headings when \
relevant: HPI, PMH, DH, SH, Imp, Plan. Be concise and use clinical phrasing.
8. Return a single JSON object and nothing else."""

USER_TEMPLATE = """### Transcript
{transcript}

### Output JSON schema
Return exactly this shape:
{{
  "chief_complaint": "one short phrase, or empty string if not stated",
  "facts": [
    {{
      "section": one of {sections},
      "text": "concise clinical statement",
      "evidence": "verbatim transcript quote",
      "speaker": "patient" | "clinician" | "unknown"
    }}
  ],
  "note_text": "clinician-style note with HPI/PMH/DH/SH/Imp/Plan headings"
}}

Extract all clinically relevant facts: symptoms with onset/duration/severity/\
radiation, negatives, past medical history, medications and doses, allergies, \
social and family history, examination findings, impression, and plan/follow-up.

### Example
Transcript:
Clinician: What brings you in?
Patient: I've had a sore throat for three days and a fever since last night.
Clinician: Any cough?
Patient: No cough.

Output:
{{"chief_complaint": "sore throat for 3 days", "facts": [
 {{"section": "chief_complaint", "text": "sore throat for 3 days", "evidence": "I've had a sore throat for three days", "speaker": "patient"}},
 {{"section": "hpi", "text": "fever since last night", "evidence": "a fever since last night", "speaker": "patient"}},
 {{"section": "ros", "text": "no cough", "evidence": "No cough.", "speaker": "patient"}}
], "note_text": "HPI: 3/7 sore throat with fever since last night. No cough.\\nImp: viral pharyngitis.\\nPlan: symptomatic care."}}

Now extract the note for the transcript above."""


def build_user_prompt(transcript: str) -> str:
    return USER_TEMPLATE.format(transcript=transcript.strip(), sections=SECTIONS)


REPAIR_TEMPLATE = """Your previous response was not valid for the required schema.

Error: {error}

Previous response:
{previous}

Return ONLY a corrected JSON object matching the schema. No markdown fences, no \
commentary."""
