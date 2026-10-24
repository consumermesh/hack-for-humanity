from clinical_scribe.correct import (
    Edit,
    apply_edits,
    check_edit,
    parse_edits,
    soundex,
)
from clinical_scribe.schema import Utterance


def test_soundex_matches_common_asr_confusions():
    assert soundex("diary") == soundex("diarrhea")
    assert soundex("exema") == soundex("eczema")
    assert soundex("microgynan") == soundex("microgynon")
    assert soundex("headache") != soundex("migraine")


def test_gate_accepts_phonetic_fix_and_rejects_diagnosis_swap():
    assert check_edit("diary", "diarrhea")[0]
    assert not check_edit("headache", "migraine")[0]
    assert not check_edit("no vomiting", "vomiting")[0]  # dropping content
    assert not check_edit("mild pain", "mild pain and fever")[0]  # adding content


def test_apply_edits_is_line_scoped():
    utts = [
        Utterance(start=0, end=1, text="I have had diary for 3 days", speaker="patient"),
        Utterance(start=1, end=2, text="Any vomiting? No vomiting.", speaker="clinician"),
    ]
    edits = [Edit(line=0, original="diary", corrected="diarrhea")]
    out = apply_edits(utts, edits)
    assert out[0].text == "I have had diarrhea for 3 days"
    assert out[1].text == utts[1].text
    assert edits[0].accepted


def test_apply_edits_rejects_missing_original():
    utts = [Utterance(start=0, end=1, text="hello", speaker="unknown")]
    edits = [Edit(line=0, original="diary", corrected="diarrhea")]
    apply_edits(utts, edits)
    assert not edits[0].accepted
    assert "not found" in edits[0].rejection


def test_parse_edits_from_model_output():
    raw = '```json\n{"edits":[{"line":1,"original":"diary","corrected":"diarrhea","reason":"ASR"}]}\n```'
    edits = parse_edits(raw)
    assert len(edits) == 1
    assert edits[0].line == 1
