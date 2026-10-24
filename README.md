# Clinical scribe: audio → diarized transcript → Gemma clinical note

A reproducible pipeline for turning consultation audio into a structured
clinical note, built around evaluation rather than a demo. It ships with the two
Kaggle datasets that pair **encounter audio** with **clinical notes**.

```
audio ──► diarization ──► ASR ─────────────► gated Gemma repair ──► Gemma note
            (speechbrain/     (Whisper large-v3 /     (medical terms only,      (JSON +
             pyannote)         MedASR)                  phonetic gate)           evidence)
                                     │                                           │
                                     └────────────► evaluation ◄────────────────┘
                                        WER · speaker accuracy · highlight recall
                                        evidence grounding · ROUGE-L · edit audit
```

## Why not feed audio straight to Gemma

Gemma's native audio input is capped at **30 s per clip** (16 kHz mono, 25
tokens/s on Gemma 4, 6.25 on Gemma 3n — [audio docs](https://ai.google.dev/gemma/docs/capabilities/audio)).
A 10-minute consultation would need ~20 disjoint clips, losing exactly the
speaker turns and cross-turn context clinical extraction depends on. Gemma does
the part it is good at: constrained, evidence-linked extraction from a full,
speaker-labelled transcript.

## Chosen configuration

| Stage | Default | Why |
|---|---|---|
| Diarization | `speechbrain/spkrec-ecapa-voxceleb` (public) or `pyannote/speaker-diarization-3.1` (gated) | Notes need doctor/patient attribution |
| ASR | Whisper `large-v3` on GPU (`float16`), `base` on CPU fallback | English-only scope; best open general accuracy |
| Medical ASR alt. | `google/medasr` | 105M dictation model, 5.2–6.9% WER on medical dictation, but conversational domain gap |
| Term repair | Gemma, gated | Fixes `diary → diarrhea` without letting the model change diagnoses |
| Extraction | Gemma 4 / 3 27B (or any Gemma served locally) | Schema-validated JSON with verbatim evidence |

## Datasets

| Dataset | Contents | License | Role |
|---|---|---|---|
| [PriMock57 audio](https://www.kaggle.com/datasets/enanko/primock57data) + [notes](https://www.kaggle.com/datasets/zeyadahmed2005/primock57-dataset) | 57 mock primary-care consultations (7–15 min), separate doctor/patient WAVs, utterance TextGrids, clinician notes + `highlights` | CC BY 4.0 | Accuracy benchmark |
| [iMerit Multilingual Ambient Scribe](https://www.kaggle.com/datasets/imeritinc/multilingual-ambient-scribe-dataset) | 199 encounters, 5 languages (UK 42, FR 48, DE 42, ES 35, NL 32), MP3 + transcripts + structured summaries | CC BY-NC-SA 4.0 | Scale / multilingual |

Caveats found while preparing the data:

- The `zeyadahmed2005` Kaggle copy stores WAVs as **133-byte Git-LFS pointers**.
  Audio must come from `enanko/primock57data` (or the upstream
  [babylonhealth/primock57](https://github.com/babylonhealth/primock57)).
- iMerit has ~48 scenarios translated across languages, so **split by
  `metadata.scenario`, not by file**, or the same conversation leaks across
  train/test.
- iMerit transcripts have no speaker labels; alternation is inferred and flagged
  as `speaker_attribution: inferred-alternating`.
- Both are simulated/mock encounters — no PHI. iMerit is non-commercial.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . pytest
export HF_HOME="$PWD/data/cache/huggingface"   # this sandbox's ~/.cache is read-only

bash scripts/download_data.sh

.venv/bin/python -m clinical_scribe ingest \
  --primock-notes data/raw/primock57_notes \
  --primock-audio data/raw/primock57_audio \
  --imerit data/raw/imerit \
  --out data/processed
```

### Chosen path: diarized large-v3 + Gemma

```bash
# On a GPU box this auto-selects large-v3/float16; --device cuda to force it.
.venv/bin/python -m clinical_scribe transcribe \
  --manifest data/processed/primock57.jsonl \
  --diarize --num-speakers 2 --model large-v3 --device auto

# Gated repair of mis-heard medical terms (needs a Gemma endpoint or GGUF)
.venv/bin/python -m clinical_scribe correct \
  --manifest data/processed/primock57.jsonl \
  --asr-dir runs/asr --out runs/corrected \
  --api ollama --model gemma3:27b

.venv/bin/python -m clinical_scribe extract \
  --manifest data/processed/primock57.jsonl \
  --asr-dir runs/corrected --out runs/notes \
  --api ollama --model gemma3:27b --check

.venv/bin/python -m clinical_scribe evaluate \
  --manifest data/processed/primock57.jsonl \
  --asr-dir runs/corrected --notes-dir runs/notes
```

### Google-hosted Gemma (Gemini API)

```bash
# put the key in .env (gitignored) so it never lands in shell history
echo 'GOOGLE_API_KEY=...' >> .env
.venv/bin/python -m clinical_scribe extract --api google --model gemma-3-27b-it --check
```

`--check` calls the models list, so run it first to confirm the exact Gemma model
id available to your key. The native `generateContent` endpoint is used (not the
OpenAI shim), with `temperature=0` and `maxOutputTokens=4096`.

Any OpenAI-compatible server works too:

```bash
GEMMA_API=openai GEMMA_BASE_URL=http://localhost:8000/v1 GEMMA_MODEL=google/gemma-3-27b-it \
  .venv/bin/python -m clinical_scribe extract --manifest data/processed/primock57.jsonl
```

CPU-only / no endpoint: `--api llamacpp --model-path data/models/gemma4-e2b/google_gemma-4-E2B-it-Q4_K_M.gguf`
runs a quantized Gemma in-process. E2B is a plumbing smoke test, not an accuracy claim.
`scripts/run_local_gemma.sh` waits for the GGUF and runs the whole local chain.

### Ablations that matter

```bash
# Oracle: how much does ASR cost the note? (gold transcript -> Gemma)
.venv/bin/python -m clinical_scribe extract --transcript reference --limit 10

# No diarization: single-stream ASR
.venv/bin/python -m clinical_scribe transcribe --limit 10 --model large-v3

# No term repair: extract directly from runs/asr
```

## Pipeline modules

| Stage | Module | What happens |
|---|---|---|
| Ingest | `ingest.py` | PriMock TextGrids + iMerit AnnoHub exports → common `Encounter` JSONL |
| Audio | `audio.py` | Doctor/patient tracks summed into one 16 kHz mono WAV (room-mic simulation) |
| Diarize | `diarize.py` | VAD → ECAPA embeddings → agglomerative clustering → speaker blocks; question-density maps clusters to clinician/patient |
| ASR | `asr.py` | Whisper per speaker block (`faster-whisper`), or `MedASRASR` |
| Correct | `correct.py` | Gemma proposes term edits; a Soundex/Levenshtein gate accepts or rejects each one |
| Extract | `gemma.py`, `prompts.py` | Schema-validated `ClinicalNote` with verbatim `evidence`; one repair retry |
| Evaluate | `evaluate.py` | WER/CER, speaker accuracy, highlight recall, grounding, ROUGE-L, edit audit |

### The correction gate

The model can only fix a word, never rewrite. A correction is applied only when:

1. `original` is an exact substring of the cited line,
2. no words are added or dropped (beyond 2),
3. every changed word pair shares a **Soundex** code or ≥0.6 orthographic
   similarity.

So `diary → diarrhea` (both `D600`) and `microgynan → microgynon` pass, while
`headache → migraine` and `no vomiting → vomiting` are rejected and logged.
Rejected edits stay in the artifact for audit.

## Evaluation

| Metric | Meaning | Watch out |
|---|---|---|
| WER / CER | ASR vs PriMock manual transcript | Overlapped speech makes it harsher than clean dictation |
| Speaker accuracy | Predicted vs track-derived speaker per utterance | PriMock's two tracks are the ground truth |
| Gold highlight recall | Share of clinician-highlighted facts present | Sparse, inconsistent labels |
| Evidence grounding | Share of facts whose quote is in the transcript | Catches invented facts, not omissions |
| ROUGE-L | Note text vs clinician note | Style-sensitive; smoke signal only |
| Transcript edits | Proposed vs accepted term repairs | An accepted wrong edit is a silent accuracy regression |

No single number means "clinically accurate". Report **grounding** and
**highlight recall** together, and hand-audit negation, laterality, medication
dose, and duration.

## Measured so far (CPU, Whisper `base` int8, 3 encounters)

| Metric | Value |
|---|---|
| WER | 24.2% |
| CER | 16.1% |
| Clinician-highlight term recall in ASR text | 8/16 (50%) |
| Speed | ~44 s per ~9-min consultation (~10× realtime) |

The term failures are the point: `diarrhea → "diary"`, plus missed `LLQ pain`,
`eczema`, `emollients`, `photophobia`, `microgynon`. That is why the chosen
stack is large-v3 + MedASR benchmarking + gated repair, not `base` alone.

## Known gaps

- Diarization is evaluated on 2-speaker primary care only; overlapping speech
  and >2 speakers are untested.
- The role heuristic (question density) can swap roles in unusual encounters;
  the raw cluster labels are kept in the artifact.
- Long transcripts are sent whole; hour-long encounters need map-reduce.
- PriMock57 has 57 encounters — use leave-one-day-out, not a random 80/20.

## Demo page

```bash
.venv/bin/python scripts/build_demo.py            # two PriMock encounters
python3 -m http.server 8765 -d demo               # open http://127.0.0.1:8765
```

Single page: audio player, ASR/corrected/gold transcripts, structured facts with
verbatim evidence, SOAP note, and the correction gate.

The generated `demo/` folder is gitignored: it contains PriMock57 audio and
transcripts, which come from Kaggle and are not redistributed here. Build it
locally after downloading the data.

## Tests

```bash
.venv/bin/python -m pytest -q
```
