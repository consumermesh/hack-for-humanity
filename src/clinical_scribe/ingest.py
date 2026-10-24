"""Normalise PriMock57 and the iMerit ambient-scribe dataset into one manifest."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Iterable

from .schema import Encounter, Utterance

_TAG_RE = re.compile(r"<UNSURE>|</UNSURE>|<UNIN/>|<INAUDIBLE_SPEECH/>|<INAUDIBLE/>")


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


def parse_textgrid(path: str | Path) -> list[Utterance]:
    import textgrid

    tg = textgrid.TextGrid()
    tg.read(str(path))
    utts: list[Utterance] = []
    for tier in tg.tiers:
        for interval in tier.intervals:
            text = _strip_tags(interval.mark)
            if text:
                utts.append(
                    Utterance(start=float(interval.minTime), end=float(interval.maxTime), text=text)
                )
    return utts


def build_primock(notes_dir: str | Path, audio_dir: str | Path) -> list[Encounter]:
    """Build encounters from the PriMock57 repo layout.

    notes_dir: .../primock57 (contains notes/, transcripts/, consultation_checklists/)
    audio_dir: .../primock57_audio (contains audio/*.wav) or the same repo if LFS pulled
    """
    notes_dir, audio_dir = Path(notes_dir), Path(audio_dir)
    notes_path, tg_path = notes_dir / "notes", notes_dir / "transcripts"

    encounters: list[Encounter] = []
    for note_file in sorted(notes_path.glob("*.json")):
        note = json.loads(note_file.read_text())
        cid = note_file.stem  # e.g. day1_consultation01
        doctor_utts = parse_textgrid(tg_path / f"{cid}_doctor.TextGrid")
        patient_utts = parse_textgrid(tg_path / f"{cid}_patient.TextGrid")
        for u in doctor_utts:
            u.speaker = "clinician"
        for u in patient_utts:
            u.speaker = "patient"
        utterances = sorted(doctor_utts + patient_utts, key=lambda u: u.start)

        audio = [
            str(audio_dir / "audio" / f"{cid}_{role}.wav")
            for role in ("doctor", "patient")
            if (audio_dir / "audio" / f"{cid}_{role}.wav").exists()
        ]
        if not audio:
            audio = [
                str(audio_dir / f"{cid}_{role}.wav")
                for role in ("doctor", "patient")
                if (audio_dir / f"{cid}_{role}.wav").exists()
            ]

        encounters.append(
            Encounter(
                id=f"primock57/{cid}",
                source="primock57",
                language="en-GB",
                audio=audio,
                reference_transcript=utterances,
                reference_note=note,
                metadata={
                    "day": note.get("day"),
                    "consultation": note.get("consultation"),
                    "specialty": "primary care",
                    "doctor_track": str(audio_dir / "audio" / f"{cid}_doctor.wav"),
                    "patient_track": str(audio_dir / "audio" / f"{cid}_patient.wav"),
                },
            )
        )
    return encounters


_QUOTE_CHARS = "\"'\u201c\u201d\u201e\u2018\u2019\u00ab\u00bb"


def _clean_imerit_line(line: str) -> str:
    return line.strip().strip(_QUOTE_CHARS).strip()


def _imerit_transcript(path: Path) -> list[Utterance]:
    """Parse an iMerit transcript: quoted, unlabelled, alternating turns.

    The files have no speaker labels, so clinician/patient alternation is
    inferred (flagged in encounter metadata) rather than treated as gold.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    utts: list[Utterance] = []
    for line in raw.splitlines():
        text = _clean_imerit_line(line)
        if not text or text.lower().startswith(("transcripted line", "transcribed line")):
            continue
        speaker = "clinician" if len(utts) % 2 == 0 else "patient"
        utts.append(Utterance(start=0.0, end=0.0, text=text, speaker=speaker))
    return utts


def _html_to_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</p\s*>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in value.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _iter_leaves(nodes, prefix: str = ""):
    for node in nodes:
        title = (node.get("title") or "").strip()
        path = f"{prefix}{title}".strip()
        children = node.get("classifications") or []
        if children:
            yield from _iter_leaves(children, path + " - ")
        elif isinstance(node.get("answer"), str) and node["answer"].strip():
            yield path, _html_to_text(node["answer"])


_NOTE_SKIP = {"Name", "Gender", "Consulting Physician", "Consultation Type", "Reason for Visit"}


def _imerit_notes(path: Path) -> dict[int, dict]:
    """Parse iMerit's AnnoHub export into {encounter_number: note dict}."""
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    entries = data if isinstance(data, list) else [data]
    notes: dict[int, dict] = {}
    for entry in entries:
        m = re.search(r"encounter\s*(\d+)", entry.get("externalId", ""), re.I)
        if not m:
            continue
        fields = list(_iter_leaves((entry.get("task") or {}).get("classifications") or []))
        section_map = dict(fields)
        note_lines = []
        for key, value in fields:
            if key in _NOTE_SKIP:
                continue
            note_lines.append(f"{key}: {value}")
        notes[int(m.group(1))] = {
            "presenting_complaint": section_map.get("Reason for Visit", ""),
            "note": "\n".join(note_lines),
            "sections": section_map,
            "consultation_type": section_map.get("Consultation Type", ""),
            "highlights": [],
        }
    return notes


def build_imerit(root: str | Path) -> list[Encounter]:
    """Build encounters from the iMerit Multilingual Ambient Scribe layout.

    root/<Language>/audio/*.mp3 + transcripts/*.txt + summaries/*.json
    """
    root = Path(root)
    base = next((p for p in root.iterdir() if p.is_dir() and p.name.startswith("iMerit")), root)

    encounters: list[Encounter] = []
    for lang_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        lang = lang_dir.name
        notes: dict[int, dict] = {}
        for summary_file in (lang_dir / "summaries").glob("*.json"):
            notes.update(_imerit_notes(summary_file))

        token = lang.split()[0]
        audio_files = sorted(
            (lang_dir / "audio").glob("*.mp3"),
            key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1))
            if re.search(r"(\d+)$", p.stem)
            else 0,
        )
        for audio in audio_files:
            m = re.search(r"encounter[_\s-]*(\d+)", audio.stem, re.I)
            if not m:
                continue
            num = int(m.group(1))
            transcript = lang_dir / "transcripts" / f"Encounter {num}_{token}.txt"
            if not transcript.exists():
                candidates = list((lang_dir / "transcripts").glob(f"*Encounter {num}_*"))
                transcript = candidates[0] if candidates else None

            encounters.append(
                Encounter(
                    id=f"imerit/{lang}/encounter-{num}",
                    source="imerit",
                    language=lang,
                    audio=[str(audio)],
                    reference_transcript=_imerit_transcript(transcript) if transcript else [],
                    reference_note=notes.get(num, {}),
                    metadata={
                        "specialty": "ambient scribe",
                        "scenario": num,
                        "speaker_attribution": "inferred-alternating",
                        "consultation_type": notes.get(num, {}).get("consultation_type", ""),
                    },
                )
            )
    return encounters


def write_jsonl(encounters: Iterable[Encounter], out: str | Path) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for enc in encounters:
            f.write(enc.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Encounter]:
    with Path(path).open(encoding="utf-8") as f:
        return [Encounter.model_validate_json(line) for line in f if line.strip()]
