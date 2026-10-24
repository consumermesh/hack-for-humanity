"""Build the single-encounter demo page: audio + transcript + Gemma note.

Usage:
    python scripts/build_demo.py --encounter primock57/day1_consultation01
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from clinical_scribe.artifacts import slug
from clinical_scribe.evaluate import load_artifacts, score_encounter
from clinical_scribe.ingest import read_jsonl
from clinical_scribe.schema import ClinicalNote

ROOT = Path(__file__).resolve().parents[1]


def build_one(encounter_id: str, index: int, asr_dir: str, notes_dir: str, corrected_dir: str) -> dict:
    encounters = read_jsonl(ROOT / "data/processed/primock57.jsonl")
    enc = next(e for e in encounters if e.id == encounter_id)
    key = slug(encounter_id)

    asr = json.loads((ROOT / asr_dir / f"{key}.json").read_text())
    note_payload = json.loads((ROOT / notes_dir / f"{key}.json").read_text())
    note = ClinicalNote.model_validate(note_payload["note"])

    corrected = None
    corrected_utterances: list[dict] = []
    corrected_path = ROOT / corrected_dir / f"{key}.json"
    if corrected_path.exists():
        corrected = json.loads(corrected_path.read_text())
        corrected_utterances = corrected.get("utterances", [])
        # Older ASR artifacts predate the utterances field; project accepted
        # edits onto the ASR segments so the corrected view stays readable.
        if len(corrected_utterances) <= 1 and asr.get("segments"):
            corrected_utterances = [dict(seg) for seg in asr["segments"]]
            for edit in corrected.get("edits", []):
                if not edit.get("accepted"):
                    continue
                for seg in corrected_utterances:
                    lower = seg["text"].lower()
                    needle = edit["original"].lower()
                    if needle in lower:
                        i = lower.index(needle)
                        seg["text"] = seg["text"][:i] + edit["corrected"] + seg["text"][i + len(needle):]
                        break

    asr_text = asr.get("text", "")
    metrics = score_encounter(enc, asr_text, note)

    audio_dir = ROOT / "demo/audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    mixed = next((ROOT / "data/processed/audio").glob(f"{key}-*.wav"))
    audio_name = f"encounter-{index}.wav"
    audio_target = audio_dir / audio_name
    if not audio_target.exists() or audio_target.stat().st_size != mixed.stat().st_size:
        shutil.copyfile(mixed, audio_target)

    return {
        "id": enc.id,
        "label": (enc.reference_note.get("presenting_complaint") or enc.id)[:70],
        "language": enc.language,
        "audio": f"audio/{audio_name}",
        "source": enc.source,
        "model": note_payload.get("model", "gemma-4-31b-it"),
        "asr_model": asr.get("model", "base"),
        "duration_s": asr.get("duration_s"),
        "asr_text": asr_text,
        "asr_segments": asr.get("segments", []),
        "asr_utterances": asr.get("utterances", []),
        "gold_transcript": [u.model_dump() for u in enc.reference_transcript],
        "note": note.model_dump(),
        "gold_note": enc.reference_note.get("note", ""),
        "gold_highlights": enc.reference_note.get("highlights", []),
        "metrics": {
            "wer": metrics.wer,
            "grounding": metrics.grounding_rate,
            "highlight_recall": metrics.highlight_recall,
            "rouge_l": metrics.rouge_l,
            "facts": metrics.n_facts,
            "unsupported": metrics.n_unsupported,
        },
        "corrections": (corrected or {}).get("edits", []),
        "corrected_utterances": corrected_utterances,
    }


def build(encounter_ids: list[str], asr_dir: str, notes_dir: str, corrected_dir: str, out_dir: str) -> Path:
    demos = [build_one(eid, i + 1, asr_dir, notes_dir, corrected_dir) for i, eid in enumerate(encounter_ids)]
    payload = json.dumps(demos, ensure_ascii=False).replace("</", "<\\/")
    page = ROOT / out_dir / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    return page


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Clinical scribe demo</title>
<style>
  :root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; --card:#fff;
          --accent:#0d9488; --accent-soft:#ccfbf1; --warn:#b45309; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         color:var(--ink); background:var(--bg); }
  header { padding:20px 28px 14px; background:var(--card); border-bottom:1px solid var(--line); }
  .head-row { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap; }
  .picker { margin-left:auto; font:13px/1.4 inherit; padding:7px 10px; border:1px solid var(--line);
            border-radius:9px; background:#fff; color:var(--ink); max-width:420px; }
  h1 { margin:0 0 4px; font-size:20px; letter-spacing:-.01em; }
  .sub { color:var(--muted); font-size:13px; }
  .chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
  .chip { background:var(--accent-soft); color:#115e59; border-radius:999px; padding:3px 10px;
          font-size:12px; font-weight:600; }
  .chip.muted { background:#f1f5f9; color:var(--muted); }
  .wrap { max-width:1280px; margin:0 auto; padding:20px 24px 40px; }
  .player { position:sticky; top:0; z-index:5; background:var(--bg); padding:12px 0 10px;
            border-bottom:1px solid var(--line); }
  audio { width:100%; }
  .cols { display:grid; grid-template-columns:1.05fr 1fr; gap:20px; margin-top:18px; }
  @media (max-width:900px){ .cols { grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
  .card h2 { margin:0; padding:12px 16px; font-size:13px; text-transform:uppercase;
             letter-spacing:.06em; color:var(--muted); border-bottom:1px solid var(--line); }
  .tabs { display:flex; gap:4px; padding:8px 10px; border-bottom:1px solid var(--line); }
  .tab { border:0; background:transparent; padding:6px 10px; border-radius:8px; cursor:pointer;
         font-size:13px; color:var(--muted); }
  .tab.active { background:#f1f5f9; color:var(--ink); font-weight:600; }
  .scroll { max-height:62vh; overflow:auto; padding:10px 14px; }
  .turn { display:flex; gap:10px; padding:7px 4px; border-radius:8px; cursor:pointer; }
  .turn:hover { background:#f8fafc; }
  .turn.active { background:#f0fdfa; }
  .who { flex:0 0 74px; font-size:11px; font-weight:700; text-transform:uppercase;
         letter-spacing:.05em; padding-top:2px; }
  .who.clinician { color:var(--accent); } .who.patient { color:#7c3aed; }
  .who.unknown { color:var(--muted); }
  .time { flex:0 0 42px; font-size:11px; color:var(--muted); padding-top:2px; font-variant-numeric:tabular-nums; }
  .txt { flex:1; }
  .section { margin:0 0 16px; }
  .section h3 { margin:0 0 6px; font-size:11px; text-transform:uppercase; letter-spacing:.06em;
                color:var(--muted); }
  .fact { padding:6px 0 6px 10px; border-left:2px solid var(--accent-soft); margin-bottom:6px; }
  .fact .ev { display:block; color:var(--muted); font-size:12px; font-style:italic; margin-top:2px; }
  .fact.bad { border-left-color:#fca5a5; }
  pre.note { white-space:pre-wrap; font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;
             background:#f8fafc; border-radius:8px; padding:12px; margin:0; }
  .edit { display:flex; gap:8px; align-items:baseline; font-size:13px; padding:5px 0;
          border-bottom:1px dashed var(--line); }
  .old { color:#b91c1c; text-decoration:line-through; }
  .new { color:#047857; font-weight:600; }
  .note-meta { padding:10px 14px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); }
  details summary { cursor:pointer; color:var(--muted); font-size:12px; }
  .empty { color:var(--muted); font-size:13px; padding:14px; }
  .nav { display:flex; align-items:center; gap:10px; padding:10px 24px; background:var(--card);
         border-bottom:1px solid var(--line); position:sticky; top:0; z-index:20; }
  .brand { font-weight:700; letter-spacing:-.01em; }
  .nav .spacer { flex:1; }
  .btn { border:1px solid var(--line); background:#fff; padding:7px 12px; border-radius:9px;
         font:13px/1 inherit; cursor:pointer; color:var(--ink); }
  .btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  .flow-hero { max-width:980px; margin:0 auto; padding:34px 24px 70px; }
  .flow-hero h1 { font-size:28px; margin:0 0 6px; }
  .lede { color:var(--muted); max-width:720px; margin:0 0 18px; }
  .callout { background:#f0fdfa; border:1px solid #99f6e4; color:#115e59; border-radius:10px;
             padding:10px 14px; font-size:13px; margin:0 0 26px; }
  .flow { position:relative; padding-left:30px; }
  .flow::before { content:''; position:absolute; left:9px; top:8px; bottom:8px; width:2px;
                  background:linear-gradient(#99f6e4,#cbd5e1); }
  .stage { position:relative; margin-bottom:14px; }
  .stage::before { content:''; position:absolute; left:-26px; top:18px; width:10px; height:10px;
                   border-radius:50%; background:var(--accent); border:2px solid #fff;
                   box-shadow:0 0 0 1px var(--accent); }
  .box { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line);
         border-radius:10px; padding:12px 15px; }
  .box h3 { margin:0 0 3px; font-size:14px; display:flex; align-items:center; gap:8px; }
  .box p { margin:0; color:var(--muted); font-size:13px; }
  .box code { background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:12px; }
  .kind { font-size:10px; text-transform:uppercase; letter-spacing:.08em; font-weight:700;
          color:#fff; background:var(--muted); padding:2px 7px; border-radius:999px; }
  .b-input { border-left-color:#6366f1; } .b-input .kind { background:#6366f1; }
  .b-model { border-left-color:var(--accent); } .b-model .kind { background:var(--accent); }
  .b-check { border-left-color:#f59e0b; } .b-check .kind { background:#d97706; }
  .b-out   { border-left-color:#16a34a; } .b-out .kind { background:#16a34a; }
  .side { margin:8px 0 0 14px; padding:8px 0 0 14px; border-left:2px dashed #cbd5e1;
          display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .side .box { padding:7px 11px; background:#fff7ed; border-color:#fed7aa; border-left:0; }
  .side .box p { color:#9a3412; font-size:12px; }
  .branch { display:flex; gap:12px; margin:10px 0 0 14px; flex-wrap:wrap; }
  .branch .box { flex:1; min-width:220px; }
  .branch-label { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
                  margin:0 0 4px; font-weight:700; }
  .gemma-badge { background:linear-gradient(135deg,#4f46e5,#0d9488); color:#fff; font-size:10px;
                 font-weight:700; letter-spacing:.06em; text-transform:uppercase; padding:2px 8px;
                 border-radius:999px; }
  .gemma-summary { display:flex; align-items:center; gap:10px; flex-wrap:wrap; background:#eef2ff;
                   border:1px solid #c7d2fe; border-radius:10px; padding:10px 14px; font-size:13px;
                   color:#3730a3; margin:0 0 22px; }
  .gemma-summary .btn { margin-left:auto; }
  .stage.gemma .box { border-color:#a5b4fc; box-shadow:0 0 0 3px #eef2ff; }
  .stage.gemma::before { background:#4f46e5; box-shadow:0 0 0 1px #4f46e5; }
  .flow.focus-gemma .stage:not(.gemma) { opacity:.25; filter:grayscale(.5); }
  .flow.focus-gemma .stage.gemma .box { box-shadow:0 0 0 3px #c7d2fe, 0 8px 24px rgba(79,70,229,.15); }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin-top:28px; color:var(--muted); font-size:12px; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .dot { width:9px; height:9px; border-radius:3px; display:inline-block; }
</style>
</head>
<body>
<nav class="nav">
  <span class="brand">Clinical scribe</span>
  <span class="spacer"></span>
  <button class="btn" id="to-flow">Pipeline</button>
  <button class="btn primary" id="to-demo">Encounter demo →</button>
</nav>
<main class="flow-hero" id="flow-view">
  <h1>How a consultation becomes a clinical note</h1>
  <p class="lede">Every stage is a separate, inspectable artifact — so a wrong note can be
     traced to the audio, the transcript, a rejected edit, or a missing evidence quote.</p>
  <div class="callout"><b>Why not send audio straight to Gemma?</b> Gemma's native audio input is
     capped at 30 seconds per clip, and a consultation is 7–15 minutes. Chunking would lose the
     speaker turns and cross-turn context that clinical extraction depends on. Gemma runs on the
     full, speaker-labelled transcript instead.</div>

  <div class="gemma-summary">
    <span class="gemma-badge">Gemma 4</span>
    <span>Gemma is used in <b>2 of 12 stages</b>: gated term repair and clinical-note extraction.
      Everything else is preprocessing, Whisper ASR, or deterministic checks.</span>
    <button class="btn" id="gemma-toggle" aria-pressed="false">Highlight Gemma stages</button>
  </div>

  <div class="flow" id="flow">
    <div class="stage"><div class="box b-input"><h3><span class="kind">input</span> Consultation audio</h3>
      <p>PriMock57 doctor + patient WAV tracks (or a room mic in production). Simulated encounter, no PHI.</p></div></div>

    <div class="stage"><div class="box b-input"><h3><span class="kind">data</span> Preprocess</h3>
      <p>Decode, resample to 16 kHz mono, sum the two tracks to simulate one room microphone.</p></div></div>

    <div class="stage"><div class="box b-model"><h3><span class="kind">model</span> Diarization <em>(optional)</em></h3>
      <p>VAD → ECAPA speaker embeddings → agglomerative clustering → speaker blocks. Needed for
         doctor/patient attribution; question density maps the clusters to clinician vs patient.</p></div></div>

    <div class="stage"><div class="box b-model"><h3><span class="kind">model</span> ASR</h3>
      <p>Whisper <code>large-v3</code> on GPU (<code>base</code> on CPU), one pass per speaker block.
         Output: transcript text, timestamps, speaker labels.</p></div>
      <div class="side"><div class="box"><h3>WER / CER</h3><p>vs PriMock's manual transcript</p></div>
        <div class="box"><h3>Highlight recall</h3><p>did the medical terms survive?</p></div></div></div>

    <div class="stage"><div class="box b-input"><h3><span class="kind">artifact</span> Transcript</h3>
      <p>Text + timestamps + speaker labels + model metadata, saved as JSON per encounter.</p></div></div>

    <div class="stage gemma"><div class="box b-model"><h3><span class="kind">model</span> Gated term repair <span class="gemma-badge">Gemma</span></h3>
      <p>Gemma proposes edits as <code>{line, original, corrected, reason}</code> — medical terms only,
         never a rewrite or a new diagnosis.</p></div></div>

    <div class="stage"><div class="box b-check"><h3><span class="kind">gate</span> Correction gate</h3>
      <p>Apply only if the original is an exact substring, no content is added or dropped, and every
         changed word shares a <b>Soundex</b> code or ≥0.6 orthographic similarity.</p></div>
      <div class="branch">
        <div class="box"><p class="branch-label">accepted</p><p><code>diary → diarrhea</code> (both D600) —
           corrected transcript + audit log.</p></div>
        <div class="box"><p class="branch-label">rejected</p><p><code>headache → migraine</code> —
           kept verbatim, rejection reason logged.</p></div>
      </div></div>

    <div class="stage gemma"><div class="box b-model"><h3><span class="kind">model</span> Gemma extraction <span class="gemma-badge">Gemma</span></h3>
      <p>Temperature 0, strict JSON schema: chief complaint, facts by section, verbatim evidence quote
         per fact, SOAP-style note. No fact without a quote.</p></div></div>

    <div class="stage"><div class="box b-check"><h3><span class="kind">gate</span> Schema validation</h3>
      <p>Pydantic validates the JSON; one repair retry with the validation error if it fails.</p></div></div>

    <div class="stage"><div class="box b-input"><h3><span class="kind">artifact</span> ClinicalNote</h3>
      <p>Structured facts, sections, SOAP text, evidence quotes — and the raw model response for audit.</p></div>
      <div class="side"><div class="box"><h3>Gold note</h3><p>ROUGE-L, chief-complaint F1</p></div></div></div>

    <div class="stage"><div class="box b-check"><h3><span class="kind">check</span> Grounding &amp; scoring</h3>
      <p>Every evidence quote must appear in the transcript (n-gram containment). Counts unsupported
         facts, clinician-highlight recall, and writes the per-encounter report.</p></div></div>

    <div class="stage"><div class="box b-out"><h3><span class="kind">output</span> Note for clinician review</h3>
      <p>Facts, SOAP note, edit log, and metrics — plus this demo UI with clickable audio and quotes.</p></div></div>
  </div>

  <div class="legend">
    <span><i class="dot" style="background:#6366f1"></i> input / artifact</span>
    <span><i class="dot" style="background:#0d9488"></i> model stage</span>
    <span><i class="dot" style="background:#d97706"></i> deterministic gate</span>
    <span><i class="dot" style="background:#16a34a"></i> output</span>
    <span><i class="dot" style="background:#fed7aa"></i> evaluation</span>
    <span><i class="dot" style="background:#4f46e5"></i> Gemma call</span>
  </div>
</main>
<section id="demo-view" hidden>
<header>
  <div class="head-row">
    <div>
      <h1>Ambient clinical scribe</h1>
      <div class="sub" id="sub"></div>
    </div>
    <select id="picker" class="picker" aria-label="Encounter"></select>
  </div>
  <div class="chips" id="chips"></div>
</header>
<div class="wrap">
  <div class="player"><audio id="audio" controls preload="metadata"></audio></div>
  <div class="cols">
    <div class="card">
      <h2>Transcript</h2>
      <div class="tabs">
        <button class="tab active" data-pane="asr">ASR output</button>
        <button class="tab" data-pane="corrected">Corrected</button>
        <button class="tab" data-pane="gold">Clinician transcript</button>
      </div>
      <div class="scroll" id="transcript"></div>
    </div>
    <div class="card">
      <h2>Extracted note</h2>
      <div class="tabs">
        <button class="tab active" data-pane="facts">Structured facts</button>
        <button class="tab" data-pane="soap">SOAP note</button>
        <button class="tab" data-pane="goldnote">Clinician gold note</button>
      </div>
      <div class="scroll" id="note"></div>
      <div class="note-meta" id="meta"></div>
    </div>
  </div>

  <div class="card" style="margin-top:20px" id="corrections-card">
    <h2>Correction gate (medical terms only)</h2>
    <div class="scroll" id="corrections" style="max-height:30vh"></div>
  </div>

  <div class="card" style="margin-top:20px">
    <h2>How to read this</h2>
    <div class="note-meta" style="border:0">
      Two encounters — switch with the dropdown. Audio is simulated PriMock57 (no PHI).
      The transcript is Whisper
      <b>base</b> on CPU, so it contains real ASR errors ("diary" for "diarrhea").
      Gemma 4 31B extracts facts with verbatim evidence quotes; grounding checks each
      quote against the transcript. A hallucinated fact cannot hide behind fluent prose.
      WER and grounding are computed against PriMock's manual transcript and clinician note.
    </div>
  </div>
</div>
</section>
<script>
const DEMOS = __DATA__;
let D = DEMOS[0];
const audio = document.getElementById('audio');
const picker = document.getElementById('picker');

const fmt = s => {
  s = Math.max(0, Math.floor(s || 0));
  return Math.floor(s/60) + ':' + String(s%60).padStart(2,'0');
};
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const pct = v => v == null ? 'n/a' : Math.round(v*100) + '%';

DEMOS.forEach((d, i) => {
  const opt = document.createElement('option');
  opt.value = String(i);
  opt.textContent = `${i+1}. ${d.label}`;
  picker.appendChild(opt);
});

function setHeader() {
  document.getElementById('sub').textContent =
    `${D.id} · ${D.language} · ${fmt(D.duration_s)} · ASR: Whisper ${D.asr_model} · extraction: ${D.model}`;
  const chips = document.getElementById('chips');
  chips.innerHTML = '';
  [
    ['WER', pct(D.metrics.wer)],
    ['Evidence grounding', pct(D.metrics.grounding)],
    ['Clinician-highlight recall', pct(D.metrics.highlight_recall)],
    ['Facts', D.metrics.facts + (D.metrics.unsupported ? ` (${D.metrics.unsupported} unsupported)` : '')],
  ].forEach(([k,v]) => {
    const c = el('span','chip'); c.textContent = `${k}: ${v}`; chips.appendChild(c);
  });
  if (!D.corrections.length) {
    const c = el('span','chip muted'); c.textContent = 'no transcript corrections proposed'; chips.appendChild(c);
  }
}

function turnRow(u, labelled) {
  const row = el('div','turn');
  row.appendChild(el('div','who ' + (labelled ? (u.speaker||'unknown') : 'unknown'),
                     labelled ? (u.speaker||'unknown').slice(0,8) : ''));
  row.appendChild(el('div','time', fmt(u.start)));
  row.appendChild(el('div','txt', u.text));
  row.addEventListener('click', () => { audio.currentTime = u.start; audio.play(); });
  row.dataset.start = u.start;
  return row;
}

function renderTranscript(pane) {
  const box = document.getElementById('transcript');
  box.innerHTML = '';
  if (pane === 'gold') {
    D.gold_transcript.forEach(u => box.appendChild(turnRow(u, true)));
  } else if (pane === 'corrected' && D.corrected_utterances.length) {
    D.corrected_utterances.forEach(u => box.appendChild(turnRow(u, false)));
  } else {
    (D.asr_utterances.length ? D.asr_utterances : D.asr_segments)
      .forEach(u => box.appendChild(turnRow(u, false)));
  }
}

function renderFacts() {
  const box = document.getElementById('note');
  box.innerHTML = '';
  const by = {};
  D.note.facts.forEach(f => { (by[f.section] = by[f.section] || []).push(f); });
  const order = ['chief_complaint','hpi','ros','pmh','medication','allergy','social','family','assessment','plan','follow_up'];
  order.filter(sec => by[sec]).forEach(sec => {
    const block = el('div','section');
    block.appendChild(el('h3', null, sec.replace('_',' ')));
    by[sec].forEach(f => {
      const item = el('div','fact');
      item.appendChild(el('div', null, f.text));
      if (f.evidence) item.appendChild(el('span','ev', '“' + f.evidence + '”'));
      block.appendChild(item);
    });
    box.appendChild(block);
  });
}

function renderSoap() {
  const box = document.getElementById('note');
  box.innerHTML = '';
  box.appendChild(el('pre','note', D.note.note_text));
}

function renderGold() {
  const box = document.getElementById('note');
  box.innerHTML = '';
  box.appendChild(el('pre','note', D.gold_note || 'no gold note'));
}

function renderCorrections() {
  const box = document.getElementById('corrections');
  box.innerHTML = '';
  if (!D.corrections.length) {
    box.appendChild(el('div','empty',
      'No transcript edits were applied by the editor pass for this encounter. ' +
      'Gemma still normalized mis-heard terms inside the note while keeping the verbatim ASR quote as evidence:'));
    const cc = D.note.facts.find(f => f.section === 'chief_complaint');
    if (cc) {
      const row = el('div','edit');
      row.appendChild(el('span','old', cc.evidence || ''));
      row.appendChild(el('span','new', cc.text));
      row.appendChild(el('span','chip','note-level'));
      box.appendChild(row);
    }
    return;
  }
  D.corrections.forEach(e => {
    const row = el('div','edit');
    row.appendChild(el('span','old', e.original));
    row.appendChild(el('span','new', e.corrected));
    row.appendChild(el('span','chip' + (e.accepted ? '' : ' muted'), e.accepted ? 'accepted' : 'rejected'));
    const why = e.reason || e.rejection || '';
    if (why) row.appendChild(el('span','empty', why));
    box.appendChild(row);
  });
}

function render(which, pane) {
  if (which === 'transcript') return renderTranscript(pane);
  if (pane === 'facts') return renderFacts();
  if (pane === 'soap') return renderSoap();
  return renderGold();
}

function resetTabs() {
  document.querySelectorAll('.tab').forEach(b => {
    b.classList.toggle('active', b.dataset.pane === 'asr' || b.dataset.pane === 'facts');
  });
}

function init(d) {
  D = d;
  audio.pause();
  audio.src = d.audio;
  audio.load();
  picker.value = String(DEMOS.indexOf(d));
  setHeader();
  resetTabs();
  renderTranscript('asr');
  renderFacts();
  renderCorrections();
  document.getElementById('meta').textContent =
    `${D.note.facts.length} facts · each with a verbatim evidence quote · model ${D.model}`;
}

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const group = btn.closest('.card');
    group.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
    render(group.querySelector('#transcript') ? 'transcript' : 'note', btn.dataset.pane);
  });
});

audio.addEventListener('timeupdate', () => {
  const t = audio.currentTime;
  document.querySelectorAll('#transcript .turn').forEach(row => {
    row.classList.toggle('active', t >= +row.dataset.start && t < +row.dataset.start + 8);
  });
});

picker.addEventListener('change', () => init(DEMOS[+picker.value]));

const flowView = document.getElementById('flow-view');
const demoView = document.getElementById('demo-view');
function show(view) {
  const isDemo = view === 'demo';
  demoView.hidden = !isDemo;
  flowView.hidden = isDemo;
  window.scrollTo(0, 0);
}
const gemmaToggle = document.getElementById('gemma-toggle');
gemmaToggle.addEventListener('click', () => {
  const on = document.getElementById('flow').classList.toggle('focus-gemma');
  gemmaToggle.textContent = on ? 'Show all stages' : 'Highlight Gemma stages';
  gemmaToggle.setAttribute('aria-pressed', String(on));
});
document.getElementById('to-demo').addEventListener('click', () => show('demo'));
document.getElementById('to-flow').addEventListener('click', () => show('flow'));

init(D);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encounters",
        nargs="+",
        default=["primock57/day1_consultation01", "primock57/day1_consultation02"],
    )
    parser.add_argument("--asr-dir", default="runs/asr")
    parser.add_argument("--notes-dir", default="runs/notes-asr")
    parser.add_argument("--corrected-dir", default="runs/corrected")
    parser.add_argument("--out", default="demo")
    args = parser.parse_args()
    page = build(args.encounters, args.asr_dir, args.notes_dir, args.corrected_dir, args.out)
    print("wrote", page)
