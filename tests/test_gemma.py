import pytest

from clinical_scribe.gemma import GemmaClient, parse_json_object
from clinical_scribe.schema import ClinicalNote


def test_parse_fenced_json_with_prose():
    raw = '''Here is the note:
```json
{"chief_complaint": "cough", "facts": [], "note_text": "HPI: cough."}
```
Hope that helps!'''
    obj = parse_json_object(raw)
    note = ClinicalNote.model_validate(obj)
    assert note.chief_complaint == "cough"
    assert note.note_text == "HPI: cough."


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_parse_drops_empty_facts():
    raw = '{"chief_complaint": "", "facts": [{"section": "hpi", "text": "  ", "evidence": ""}], "note_text": ""}'
    note = GemmaClient._parse(raw)
    assert note.facts == []


def test_invalid_section_raises():
    raw = '{"chief_complaint": "", "facts": [{"section": "vitals", "text": "BP 120/80"}]}'
    with pytest.raises(Exception):
        GemmaClient._parse(raw)


def test_google_backend_parses_parts(monkeypatch):
    from clinical_scribe.gemma import GemmaClient, GemmaConfig

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"candidates": [{"content": {"parts": [{"text": '{"chief_complaint": "cough"'}]}}]}

    calls = {}

    def fake_post(url, params=None, json=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        return FakeResponse()

    monkeypatch.setattr("clinical_scribe.gemma.requests.post", fake_post)
    client = GemmaClient(GemmaConfig(api="google", api_key="k", model="gemma-3-27b-it"))
    out = client.chat("sys", "user")
    assert out.startswith("{")
    assert "gemma-3-27b-it:generateContent" in calls["url"]
    assert calls["params"]["key"] == "k"


def test_google_backend_requires_key():
    import pytest as _pytest

    from clinical_scribe.gemma import GemmaClient, GemmaConfig

    with _pytest.raises(ValueError):
        GemmaClient(GemmaConfig(api="google", api_key="")).chat("s", "u")
