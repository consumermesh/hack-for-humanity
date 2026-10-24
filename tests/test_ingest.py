import json
from pathlib import Path

from clinical_scribe.ingest import build_primock

TG = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 12
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "{tier}"
        xmin = 0
        xmax = 12
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 5
            text = "{first}"
        intervals [2]:
            xmin = 7
            xmax = 11
            text = "{second}"
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_build_primock_merges_tracks_in_time_order(tmp_path):
    repo = tmp_path / "primock"
    _write(repo / "notes" / "day1_consultation01.json", json.dumps({
        "day": 1,
        "consultation": 1,
        "presenting_complaint": "headache",
        "note": "HPI: headache 3/7. Imp: migraine.",
        "highlights": ["headache", "migraine"],
    }))
    _write(repo / "transcripts" / "day1_consultation01_doctor.TextGrid", TG.format(
        tier="Doctor", first="Good morning", second="Any vomiting"))
    _write(repo / "transcripts" / "day1_consultation01_patient.TextGrid", TG.format(
        tier="Patient", first="I have a headache", second="No vomiting"))

    audio = tmp_path / "audio"
    for role in ("doctor", "patient"):
        _write(audio / "audio" / f"day1_consultation01_{role}.wav", "")

    [enc] = build_primock(repo, audio)
    assert enc.id == "primock57/day1_consultation01"
    assert len(enc.audio) == 2
    assert [u.speaker for u in enc.reference_transcript][:2] == ["clinician", "patient"]
    assert [u.text for u in enc.reference_transcript][:4] == [
        "Good morning", "I have a headache", "Any vomiting", "No vomiting",
    ]
    assert enc.reference_note["highlights"] == ["headache", "migraine"]


def test_build_primock_without_audio_dir(tmp_path):
    repo = tmp_path / "primock"
    _write(repo / "notes" / "day1_consultation01.json", json.dumps({"day": 1, "consultation": 1}))
    _write(repo / "transcripts" / "day1_consultation01_doctor.TextGrid", TG.format(
        tier="Doctor", first="Hello", second=""))
    _write(repo / "transcripts" / "day1_consultation01_patient.TextGrid", TG.format(
        tier="Patient", first="Hi", second=""))

    [enc] = build_primock(repo, repo)
    assert enc.audio == []
