"""Command line entry point: ingest -> transcribe -> correct -> extract -> evaluate."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from . import __version__
from .artifacts import slug, transcript_for_prompt
from .asr import FasterWhisperASR, MedASRASR, whisper_language
from .audio import decode, prepare_audio
from .correct import correct_transcript
from .diarize import PyannoteDiarizer, SpeechBrainDiarizer, transcribe_diarized
from .evaluate import evaluate_all, summarize, write_report
from .gemma import GemmaClient, GemmaConfig
from .ingest import build_imerit, build_primock, read_jsonl, write_jsonl
from .schema import Encounter, Utterance


def cmd_ingest(args) -> int:
    out = Path(args.out)
    total = 0
    if args.primock_notes:
        encounters = build_primock(args.primock_notes, args.primock_audio or args.primock_notes)
        n = write_jsonl(encounters, out / "primock57.jsonl")
        print(f"primock57: {n} encounters -> {out / 'primock57.jsonl'}")
        total += n
    if args.imerit:
        encounters = build_imerit(args.imerit)
        n = write_jsonl(encounters, out / "imerit.jsonl")
        print(f"imerit: {n} encounters -> {out / 'imerit.jsonl'}")
        total += n
    if not total:
        print("nothing to ingest: pass --primock-notes and/or --imerit", file=sys.stderr)
        return 2
    return 0


def _load_manifest(path: str, limit: int | None) -> list[Encounter]:
    encounters = read_jsonl(path)
    return encounters[:limit] if limit else encounters


def cmd_inspect(args) -> int:
    encounters = _load_manifest(args.manifest, args.limit)
    print(f"{len(encounters)} encounters from {args.manifest}")
    by_source: dict[str, int] = {}
    missing_audio = 0
    with_note = 0
    for enc in encounters:
        by_source[enc.source] = by_source.get(enc.source, 0) + 1
        missing_audio += sum(1 for a in enc.audio if not Path(a).exists())
        with_note += bool(enc.reference_note)
    print("sources:", by_source)
    print(f"audio files missing: {missing_audio}")
    print(f"with reference note: {with_note}")
    sample = encounters[0]
    print("\nsample:", sample.id)
    print("audio:", sample.audio)
    print("transcript utterances:", len(sample.reference_transcript))
    print("first 2:", [u.text for u in sample.reference_transcript[:2]])
    print("note keys:", list(sample.reference_note))
    return 0


def _default_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def _default_compute(device: str, compute_type: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def cmd_transcribe(args) -> int:
    encounters = _load_manifest(args.manifest, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = _default_device(args.device)
    if args.backend == "medasr":
        asr = MedASRASR(args.model or "google/medasr")
        model_name = asr.model_id
    else:
        model_name = args.model or ("large-v3" if device == "cuda" else "base")
        asr = FasterWhisperASR(
            model=model_name,
            device=device,
            compute_type=_default_compute(device, args.compute_type),
            language=args.language,
        )

    diarizer = None
    if args.diarize:
        if args.diarizer == "pyannote":
            diarizer = PyannoteDiarizer(token=args.hf_token)
        else:
            diarizer = SpeechBrainDiarizer(savedir=args.speechbrain_dir, device=device)

    print(f"ASR: {model_name} device={device} diarize={args.diarize}")
    for enc in tqdm(encounters, desc="transcribe"):
        target = out / f"{slug(enc.id)}.json"
        if target.exists() and not args.force:
            continue
        if not enc.audio:
            print(f"skip {enc.id}: no audio", file=sys.stderr)
            continue
        try:
            started = time.perf_counter()
            audio_path = prepare_audio(enc.id, enc.audio, args.audio_cache)
            audio, sr = decode(audio_path)
            language = args.language or whisper_language(enc.language)

            if diarizer is not None:
                utterances = transcribe_diarized(
                    audio, sr, asr, diarizer, num_speakers=args.num_speakers, language=language
                )
                text = " ".join(u.text for u in utterances)
                segments = [{"start": u.start, "end": u.end, "text": u.text} for u in utterances]
            else:
                result = asr.transcribe_array(audio, sr, language)
                utterances = [
                    Utterance(start=s.start, end=s.end, text=s.text, speaker="unknown")
                    for s in result.segments
                ]
                text = result.text
                segments = [s.__dict__ for s in result.segments]

            payload = {
                "id": enc.id,
                "text": text,
                "utterances": [u.model_dump() for u in utterances],
                "segments": segments,
                "language": language,
                "duration_s": len(audio) / sr,
                "latency_s": time.perf_counter() - started,
                "model": model_name,
                "diarized": diarizer is not None,
                "audio": audio_path,
            }
            payload["rtf"] = payload["latency_s"] / payload["duration_s"]
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"FAILED {enc.id}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


def _client(args) -> GemmaClient:
    return GemmaClient(
        GemmaConfig(
            base_url=args.base_url,
            model=args.model,
            api=args.api,
            model_path=getattr(args, "model_path", ""),
            api_key=getattr(args, "api_key", ""),
            timeout=args.timeout,
            max_attempts=getattr(args, "max_attempts", 2),
        )
    )


def _load_utterances(payload: dict) -> list[Utterance]:
    utts = payload.get("utterances") or []
    if utts:
        return [Utterance.model_validate(u) for u in utts]
    text = payload.get("text", "")
    return [Utterance(start=0.0, end=0.0, text=text, speaker="unknown")] if text else []


def cmd_correct(args) -> int:
    encounters = _load_manifest(args.manifest, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = _client(args)
    if args.check:
        health = client.health()
        print("gemma endpoint:", health)
        if not health.get("ok"):
            return 1

    for enc in tqdm(encounters, desc="correct"):
        target = out / f"{slug(enc.id)}.json"
        if target.exists() and not args.force:
            continue
        source = Path(args.asr_dir) / f"{slug(enc.id)}.json"
        if not source.exists():
            print(f"skip {enc.id}: no ASR output", file=sys.stderr)
            continue
        payload = json.loads(source.read_text())
        utterances = _load_utterances(payload)
        if not utterances:
            continue
        try:
            result = correct_transcript(client, utterances)
            target.write_text(
                json.dumps(
                    {
                        "id": enc.id,
                        "text": " ".join(u.text for u in result.utterances),
                        "utterances": [u.model_dump() for u in result.utterances],
                        "edits": [asdict(e) for e in result.edits],
                        "accepted": len(result.accepted),
                        "rejected": len(result.rejected),
                        "raw": result.raw,
                        "latency_s": result.latency_s,
                        "model": result.model,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"FAILED {enc.id}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


def cmd_extract(args) -> int:
    encounters = _load_manifest(args.manifest, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = _client(args)
    if args.check:
        health = client.health()
        print("gemma endpoint:", health)
        if not health.get("ok"):
            return 1

    for enc in tqdm(encounters, desc="extract"):
        target = out / f"{slug(enc.id)}.json"
        if target.exists() and not args.force:
            continue
        if args.transcript == "reference":
            transcript = enc.transcript_text()
        else:
            asr_file = Path(args.asr_dir) / f"{slug(enc.id)}.json"
            if not asr_file.exists():
                print(f"skip {enc.id}: no transcript artifact", file=sys.stderr)
                continue
            transcript = transcript_for_prompt(json.loads(asr_file.read_text()))
        if not transcript.strip():
            print(f"skip {enc.id}: empty transcript", file=sys.stderr)
            continue
        try:
            result = client.extract_note(transcript)
            target.write_text(
                json.dumps(
                    {
                        "id": enc.id,
                        "note": result.note.model_dump(),
                        "raw": result.raw,
                        "attempts": result.attempts,
                        "latency_s": result.latency_s,
                        "model": result.model,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"FAILED {enc.id}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


def cmd_evaluate(args) -> int:
    encounters = _load_manifest(args.manifest, args.limit)
    rows = evaluate_all(encounters, args.asr_dir, args.notes_dir, args.grounding)
    report = write_report(rows, args.out, title=args.title)
    s = summarize(rows)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, indent=2))
    print(f"report: {report}")
    return 0


def _add_model_args(p: argparse.ArgumentParser, attempts: bool = False) -> None:
    p.add_argument("--base-url", default=GemmaConfig().base_url)
    p.add_argument("--model", default=GemmaConfig().model)
    p.add_argument(
        "--api",
        choices=["google", "ollama", "openai", "llamacpp"],
        default=GemmaConfig().api,
        help="google: Gemini API (Gemma hosted); ollama/openai: local server; llamacpp: in-process GGUF",
    )
    p.add_argument("--api-key", default=GemmaConfig().api_key, help="or set GOOGLE_API_KEY")
    p.add_argument("--model-path", default=GemmaConfig().model_path, help="GGUF path for --api llamacpp")
    p.add_argument("--timeout", type=float, default=900.0)
    if attempts:
        p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--check", action="store_true", help="ping the endpoint before running")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clinical-scribe", description=__doc__)
    p.add_argument("--version", action="version", version=f"clinical-scribe {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="normalise raw datasets into JSONL manifests")
    ing.add_argument("--primock-notes", help="PriMock57 repo root (notes/transcripts)")
    ing.add_argument("--primock-audio", help="folder containing audio/*.wav")
    ing.add_argument("--imerit", help="iMerit dataset root")
    ing.add_argument("--out", default="data/processed")
    ing.set_defaults(func=cmd_ingest)

    ins = sub.add_parser("inspect", help="print manifest statistics")
    ins.add_argument("--manifest", required=True)
    ins.add_argument("--limit", type=int)
    ins.set_defaults(func=cmd_inspect)

    tr = sub.add_parser("transcribe", help="run ASR (optionally diarized) over the manifest")
    tr.add_argument("--manifest", required=True)
    tr.add_argument("--out", default="runs/asr")
    tr.add_argument("--backend", choices=["faster-whisper", "medasr"], default="faster-whisper")
    tr.add_argument("--model", help="whisper size (large-v3, small.en, ...) or MedASR id")
    tr.add_argument("--device", default="auto", help="auto | cpu | cuda")
    tr.add_argument("--compute-type", default="auto", help="auto | int8 | float16")
    tr.add_argument("--language", help="force a language; default auto per encounter")
    tr.add_argument("--audio-cache", default="data/processed/audio")
    tr.add_argument("--diarize", action="store_true", help="speaker-label the transcript")
    tr.add_argument("--diarizer", choices=["speechbrain", "pyannote"], default="speechbrain")
    tr.add_argument("--num-speakers", type=int, default=2)
    tr.add_argument("--hf-token", help="Hugging Face token for the gated pyannote model")
    tr.add_argument("--speechbrain-dir", default="data/models/speechbrain")
    tr.add_argument("--limit", type=int)
    tr.add_argument("--force", action="store_true")
    tr.set_defaults(func=cmd_transcribe)

    co = sub.add_parser("correct", help="gated Gemma repair of mis-heard medical terms")
    co.add_argument("--manifest", required=True)
    co.add_argument("--asr-dir", default="runs/asr")
    co.add_argument("--out", default="runs/corrected")
    _add_model_args(co)
    co.set_defaults(func=cmd_correct, max_attempts=2)

    ex = sub.add_parser("extract", help="run Gemma extraction over transcripts")
    ex.add_argument("--manifest", required=True)
    ex.add_argument("--asr-dir", default="runs/asr")
    ex.add_argument("--out", default="runs/notes")
    ex.add_argument(
        "--transcript",
        choices=["asr", "reference"],
        default="asr",
        help="asr: extract from ASR output; reference: oracle run on the gold transcript",
    )
    _add_model_args(ex, attempts=True)
    ex.set_defaults(func=cmd_extract)

    ev = sub.add_parser("evaluate", help="score ASR + notes and write a report")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--asr-dir", default="runs/asr")
    ev.add_argument("--notes-dir", default="runs/notes")
    ev.add_argument("--out", default="runs/eval")
    ev.add_argument("--title", default="Clinical note extraction evaluation")
    ev.add_argument(
        "--grounding",
        choices=["asr", "reference"],
        default="asr",
        help="check evidence quotes against ASR text or the gold transcript (use reference for oracle runs)",
    )
    ev.add_argument("--limit", type=int)
    ev.set_defaults(func=cmd_evaluate)
    return p


def load_env(path: str = ".env") -> None:
    """Minimal .env loader so the API key never has to be pasted on the CLI."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
