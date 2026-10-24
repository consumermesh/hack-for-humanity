"""Metrics: transcription quality, note-level similarity, fact coverage, grounding.

No single number is trustworthy for clinical notes. We report four families:
  * ASR: WER / CER against PriMock's manual transcript
  * Coverage: recall of the clinician's highlighted facts
  * Grounding: share of extracted facts whose evidence quote is in the transcript
  * Note similarity: ROUGE-L against the clinician-written note
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import jiwer

from .artifacts import slug
from .schema import ClinicalNote, Encounter, Utterance
from .text import containment, normalize, token_coverage


@dataclass
class EncounterMetrics:
    id: str
    n_ref_utterances: int = 0
    n_asr_segments: int = 0
    wer: float | None = None
    cer: float | None = None
    rouge_l: float | None = None
    highlight_recall: float | None = None
    n_highlights: int = 0
    grounding_rate: float | None = None
    n_facts: int = 0
    n_unsupported: int = 0
    chief_complaint_f1: float | None = None
    speaker_accuracy: float | None = None
    n_edits: int = 0
    n_edits_accepted: int = 0
    error: str = ""


def _token_f1(ref: str, hyp: str) -> float:
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r or not h:
        return 0.0
    common = {}
    for t in r:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in h:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    p, rec = overlap / len(h), overlap / len(r)
    return 2 * p * rec / (p + rec)


def asr_metrics(reference: str, hypothesis: str) -> tuple[float | None, float | None]:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref or not hyp:
        return None, None
    return float(jiwer.wer(ref, hyp)), float(jiwer.cer(ref, hyp))


def rouge_l(reference: str, hypothesis: str) -> float | None:
    if not reference.strip() or not hypothesis.strip():
        return None
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, hypothesis)["rougeL"].fmeasure)


def highlight_recall(highlights: list[str], generated: str, threshold: float = 0.6) -> float | None:
    if not highlights:
        return None
    def matched(h: str) -> bool:
        return (
            containment(h, generated, threshold=threshold) >= threshold
            or token_coverage(h, generated) >= 0.75
        )

    hits = sum(1 for h in highlights if matched(h))
    return hits / len(highlights)


def grounding(note: ClinicalNote, transcript: str, threshold: float = 0.6) -> tuple[float | None, int]:
    if not note.facts:
        return None, 0
    supported = sum(
        1 for f in note.facts if f.evidence and containment(f.evidence, transcript, threshold=threshold) >= threshold
    )
    return supported / len(note.facts), len(note.facts) - supported


def speaker_accuracy(predicted: list[dict], gold: list[Utterance]) -> float | None:
    """Frame-level speaker attribution accuracy against PriMock track labels."""
    gold = [g for g in gold if g.speaker in ("clinician", "patient")]
    labelled = [p for p in predicted if p.get("speaker") in ("clinician", "patient")]
    if not gold or not labelled:
        return None
    correct = total = 0
    for p in labelled:
        ps, pe = float(p.get("start", 0)), float(p.get("end", 0))
        best, best_overlap = None, 0.0
        for g in gold:
            overlap = min(pe, g.end) - max(ps, g.start)
            if overlap > best_overlap:
                best, best_overlap = g, overlap
        if best is None:
            continue
        total += 1
        correct += p.get("speaker") == best.speaker
    return correct / total if total else None


def score_encounter(
    encounter: Encounter,
    asr_text: str,
    note: ClinicalNote | None,
    *,
    transcript_for_grounding: str | None = None,
    asr_payload: dict | None = None,
) -> EncounterMetrics:
    m = EncounterMetrics(id=encounter.id, n_ref_utterances=len(encounter.reference_transcript))
    gold_note = (
        encounter.reference_note.get("note")
        or encounter.reference_note.get("summary")
        or ""
    )
    if asr_text:
        m.wer, m.cer = asr_metrics(encounter.transcript_plain(), asr_text)
    if asr_payload:
        m.speaker_accuracy = speaker_accuracy(asr_payload.get("utterances") or [], encounter.reference_transcript)
        m.n_edits = len(asr_payload.get("edits") or [])
        m.n_edits_accepted = sum(1 for e in asr_payload.get("edits") or [] if e.get("accepted"))
    if note is not None:
        m.n_facts = len(note.facts)
        generated = " ".join([note.note_text] + [f.text for f in note.facts])
        m.rouge_l = rouge_l(gold_note, note.note_text)
        highlights = encounter.reference_note.get("highlights") or []
        m.n_highlights = len(highlights)
        m.highlight_recall = highlight_recall(highlights, generated)
        m.grounding_rate, m.n_unsupported = grounding(
            note, transcript_for_grounding or asr_text or encounter.transcript_plain()
        )
        gold_cc = encounter.reference_note.get("presenting_complaint") or ""
        m.chief_complaint_f1 = _token_f1(gold_cc, note.chief_complaint) if gold_cc else None
    return m


def load_artifacts(
    encounter: Encounter, asr_dir: Path, notes_dir: Path
) -> tuple[str, ClinicalNote | None, str, dict]:
    key = slug(encounter.id)
    asr_text, note, error, payload = "", None, "", {}
    asr_file = asr_dir / f"{key}.json"
    if asr_file.exists():
        payload = json.loads(asr_file.read_text())
        asr_text = payload.get("text", "")
    note_file = notes_dir / f"{key}.json"
    if note_file.exists():
        payload = json.loads(note_file.read_text())
        try:
            note = ClinicalNote.model_validate(payload["note"])
        except Exception as e:  # corrupt output should not kill the report
            error = f"invalid note: {e}"
    return asr_text, note, error, payload


def evaluate_all(
    encounters: list[Encounter],
    asr_dir: str | Path,
    notes_dir: str | Path,
    grounding_source: str = "asr",
) -> list[EncounterMetrics]:
    asr_dir, notes_dir = Path(asr_dir), Path(notes_dir)
    rows = []
    for enc in encounters:
        asr_text, note, error, payload = load_artifacts(enc, asr_dir, notes_dir)
        grounding_text = enc.transcript_text() if grounding_source == "reference" else None
        m = score_encounter(
            enc, asr_text, note, asr_payload=payload, transcript_for_grounding=grounding_text
        )
        m.error = error
        rows.append(m)
    return rows


def summarize(rows: list[EncounterMetrics]) -> dict:
    def mean(values):
        vals = [v for v in values if v is not None]
        return statistics.fmean(vals) if vals else None

    return {
        "n": len(rows),
        "n_with_asr": sum(1 for r in rows if r.wer is not None),
        "n_with_notes": sum(1 for r in rows if r.rouge_l is not None),
        "wer_mean": mean([r.wer for r in rows]),
        "cer_mean": mean([r.cer for r in rows]),
        "rouge_l_mean": mean([r.rouge_l for r in rows]),
        "highlight_recall_mean": mean([r.highlight_recall for r in rows]),
        "grounding_rate_mean": mean([r.grounding_rate for r in rows]),
        "facts_total": sum(r.n_facts for r in rows),
        "unsupported_total": sum(r.n_unsupported for r in rows),
        "chiefly_complaint_f1_mean": mean([r.chief_complaint_f1 for r in rows]),
        "speaker_accuracy_mean": mean([r.speaker_accuracy for r in rows]),
        "edits_total": sum(r.n_edits for r in rows),
        "edits_accepted": sum(r.n_edits_accepted for r in rows),
    }


def write_report(rows: list[EncounterMetrics], out_dir: str | Path, title: str = "Evaluation report") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_encounter.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    s = summarize(rows)

    def fmt(v, pct=False):
        if v is None:
            return "n/a"
        return f"{v * 100:.1f}%" if pct else f"{v:.3f}"

    lines = [
        f"# {title}",
        "",
        f"Encounters scored: **{s['n']}** (ASR: {s['n_with_asr']}, notes: {s['n_with_notes']})",
        "",
        "| Metric | Mean |",
        "|---|---|",
        f"| WER | {fmt(s['wer_mean'], pct=True)} |",
        f"| CER | {fmt(s['cer_mean'], pct=True)} |",
        f"| ROUGE-L vs gold note | {fmt(s['rouge_l_mean'])} |",
        f"| Gold highlight recall | {fmt(s['highlight_recall_mean'], pct=True)} |",
        f"| Evidence grounding | {fmt(s['grounding_rate_mean'], pct=True)} |",
        f"| Chief-complaint token F1 | {fmt(s['chiefly_complaint_f1_mean'])} |",
        f"| Speaker attribution | {fmt(s['speaker_accuracy_mean'], pct=True)} |",
        f"| Transcript edits | {s['edits_total']} proposed, {s['edits_accepted']} accepted |",
        f"| Extracted facts | {s['facts_total']} ({s['unsupported_total']} unsupported) |",
        "",
    ]

    worst = [r for r in rows if r.highlight_recall is not None]
    if worst:
        lines += ["## Weakest encounters by highlight recall", "", "| Encounter | Recall | WER |", "|---|---|---|"]
        for r in sorted(worst, key=lambda x: x.highlight_recall)[:10]:
            lines.append(
                f"| {r.id} | {fmt(r.highlight_recall, pct=True)} | {fmt(r.wer, pct=True)} |"
            )
        lines.append("")

    lines += [
        "## Reading this",
        "",
        "- WER/CER use PriMock's manual transcript as reference; overlapped speech makes it a hard, realistic test.",
        "- Highlight recall only covers facts the clinicians chose to highlight (a small, noisy subset).",
        "- Low grounding with high highlight recall means the model is right for the wrong reasons; inspect quotes.",
        "- ROUGE-L is style-sensitive. Treat it as a smoke signal, not clinical validity.",
        "",
    ]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
