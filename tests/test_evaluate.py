from clinical_scribe.evaluate import (
    asr_metrics,
    highlight_recall,
    rouge_l,
    score_encounter,
)
from clinical_scribe.schema import ClinicalNote, Encounter, Fact, Utterance


def _encounter() -> Encounter:
    return Encounter(
        id="primock57/day1_consultation01",
        source="primock57",
        reference_transcript=[
            Utterance(start=0, end=5, text="I have had a bad headache for three days", speaker="patient"),
            Utterance(start=6, end=9, text="Any vomiting?", speaker="clinician"),
            Utterance(start=10, end=12, text="No vomiting", speaker="patient"),
        ],
        reference_note={
            "presenting_complaint": "bad headache for three days",
            "note": "HPI: 3/7 headache. No vomiting. Imp: migraine.",
            "highlights": ["3/7 headache", "No vomiting", "migraine"],
        },
    )


def test_wer_zero_for_identical_text():
    enc = _encounter()
    wer, cer = asr_metrics(enc.transcript_plain(), enc.transcript_plain())
    assert wer == 0.0 and cer == 0.0


def test_highlight_recall_and_rouge():
    generated = "HPI: 3/7 headache. No vomiting. Imp: migraine."
    assert highlight_recall(["3/7 headache", "migraine"], generated) == 1.0
    assert rouge_l("HPI: 3/7 headache. No vomiting. Imp: migraine.", generated) > 0.9


def test_score_encounter_grounding_flags_unsupported_fact():
    enc = _encounter()
    note = ClinicalNote(
        chief_complaint="bad headache for three days",
        facts=[
            Fact(section="hpi", text="headache for 3 days", evidence="bad headache for three days"),
            Fact(section="pmh", text="diabetes", evidence="I have diabetes"),
        ],
        note_text="HPI: 3/7 headache. PMH: diabetes.",
    )
    m = score_encounter(enc, enc.transcript_text(), note)
    assert m.n_facts == 2
    assert m.n_unsupported == 1
    assert 0.4 < m.grounding_rate < 0.6
    assert m.chief_complaint_f1 and m.chief_complaint_f1 > 0.8
