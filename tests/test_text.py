import pytest

from clinical_scribe.text import containment, normalize, is_supported


def test_normalize_keeps_clinical_forms():
    assert normalize("3/7 hx of diarrhea, no blood!") == "3/7 hx of diarrhea no blood"
    assert normalize("Follow-up") == "follow up"


def test_containment_exact_and_fuzzy():
    transcript = "Patient: I have had a bad headache for three days now."
    assert containment("bad headache for three days", transcript) == 1.0
    # small ASR-style variation still counts
    assert containment("a bad headache for 3 days", transcript) >= 0.3
    assert containment("chest pain", transcript) < 0.6


def test_is_supported_rejects_empty():
    assert not is_supported("", "anything")
    assert is_supported("bad headache", "I have a bad headache")


def test_user_prompt_renders():
    from clinical_scribe.prompts import build_user_prompt

    prompt = build_user_prompt("Patient: I have a headache")
    assert "I have a headache" in prompt
    assert '"chief_complaint"' in prompt


def test_whisper_language_mapping():
    from clinical_scribe.asr import whisper_language

    assert whisper_language("en-GB") == "en"
    assert whisper_language("UK English") == "en"
    assert whisper_language("French") == "fr"
    assert whisper_language(None) is None
