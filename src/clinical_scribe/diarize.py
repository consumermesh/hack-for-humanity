"""Speaker diarization backends and speaker-attributed transcription.

Two backends:
  * SpeechBrainDiarizer: public ECAPA embeddings + agglomerative clustering, CPU-friendly.
  * PyannoteDiarizer: pyannote/speaker-diarization-3.1 (gated; needs an HF token).

We transcribe each diarized speaker block separately. That keeps speaker
attribution reliable (Whisper otherwise merges a question and its answer into
one segment) at the cost of more, shorter ASR calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .schema import Utterance


@dataclass
class Turn:
    start: float
    end: float
    cluster: int


class Diarizer(Protocol):
    def diarize(self, audio: np.ndarray, sr: int, num_speakers: int = 2) -> list[Turn]: ...


def _merge_windows(windows: list[tuple[float, float, int]], max_gap: float = 0.6) -> list[Turn]:
    turns: list[Turn] = []
    for start, end, label in sorted(windows):
        if turns and turns[-1].cluster == label and start - turns[-1].end <= max_gap:
            turns[-1].end = max(turns[-1].end, end)
        else:
            turns.append(Turn(start=start, end=end, cluster=label))
    return turns


class SpeechBrainDiarizer:
    """ECAPA speaker embeddings over VAD windows, clustered into N speakers."""

    def __init__(self, savedir: str = "data/models/speechbrain", device: str = "cpu"):
        self.savedir = savedir
        self.device = device
        self._encoder = None

    def _load(self):
        if self._encoder is None:
            from speechbrain.inference.speaker import EncoderClassifier

            self._encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.savedir,
                run_opts={"device": self.device},
            )
        return self._encoder

    def diarize(self, audio: np.ndarray, sr: int, num_speakers: int = 2) -> list[Turn]:
        import torch
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        from sklearn.cluster import AgglomerativeClustering

        regions = get_speech_timestamps(
            audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=150), sampling_rate=sr
        )
        win, hop = 1.5, 0.75
        spans: list[tuple[float, float]] = []
        for region in regions:
            start, end = region["start"] / sr, region["end"] / sr
            n = max(1, math.ceil((end - start - win) / hop) + 1)
            for i in range(n):
                s = start + i * hop
                e = min(s + win, end)
                if e - s >= 0.5:
                    spans.append((s, e))
        if not spans:
            return []

        encoder = self._load()
        embeddings: list[np.ndarray] = []
        batch = []
        for s, e in spans:
            batch.append(torch.from_numpy(audio[int(s * sr) : int(e * sr)]).unsqueeze(0))
            if len(batch) == 16:
                embeddings.extend(self._embed(encoder, batch))
                batch = []
        if batch:
            embeddings.extend(self._embed(encoder, batch))
        matrix = np.vstack(embeddings)
        k = max(1, min(num_speakers, len(spans)))
        if k == 1:
            labels = np.zeros(len(spans), dtype=int)
        else:
            labels = AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ).fit_predict(matrix)

        # Majority smoothing over five windows kills isolated flips.
        smoothed = labels.copy()
        for i in range(len(labels)):
            window = labels[max(0, i - 2) : i + 3]
            smoothed[i] = np.bincount(window).argmax()

        return _merge_windows(
            [(s, e, int(label)) for (s, e), label in zip(spans, smoothed)]
        )

    @staticmethod
    def _embed(encoder, batch: list) -> list[np.ndarray]:
        import torch

        with torch.no_grad():
            out = encoder.encode_batch(batch)
        return [row.squeeze().cpu().numpy() for row in out]


class PyannoteDiarizer:
    def __init__(self, model: str = "pyannote/speaker-diarization-3.1", token: str | None = None):
        self.model = model
        self.token = token

    def diarize(self, audio: np.ndarray, sr: int, num_speakers: int = 2) -> list[Turn]:
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(self.model, token=self.token)
        result = pipeline(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr},
            num_speakers=num_speakers,
        )
        names: dict[str, int] = {}
        turns: list[Turn] = []
        for segment, _, label in result.itertracks(yield_label=True):
            turns.append(Turn(segment.start, segment.end, names.setdefault(label, len(names))))
        return _merge_windows([(t.start, t.end, t.cluster) for t in turns], max_gap=0.2)


@dataclass
class _Block:
    start: float
    end: float
    cluster: int
    text: str


def assign_roles(blocks: list[_Block]) -> dict[int, str]:
    """Map anonymous clusters to clinician/patient using question density.

    Clinicians ask most of the questions in a primary-care visit. Falls back to
    whoever speaks first when the rates are close.
    """
    stats: dict[int, list[int]] = {}
    for b in blocks:
        q, n = stats.get(b.cluster, [0, 0])
        stats[b.cluster] = [q + b.text.count("?"), n + 1]
    if len(stats) < 2:
        return {next(iter(stats), 0): "clinician"}
    rates = {k: q / max(n, 1) for k, (q, n) in stats.items()}
    ranked = sorted(rates, key=rates.get, reverse=True)
    if abs(rates[ranked[0]] - rates[ranked[1]]) < 0.05:
        clinician = blocks[0].cluster
    else:
        clinician = ranked[0]
    return {k: ("clinician" if k == clinician else "patient") for k in stats}


def transcribe_diarized(
    audio: np.ndarray,
    sr: int,
    asr,
    diarizer: Diarizer,
    num_speakers: int = 2,
    language: str | None = None,
) -> list[Utterance]:
    """Diarize, transcribe each merged speaker block, and label the result."""
    turns = [t for t in diarizer.diarize(audio, sr, num_speakers) if t.end - t.start >= 0.3]
    if not turns:
        result = asr.transcribe_array(audio, sr, language)
        return [Utterance(start=s.start, end=s.end, text=s.text, speaker="unknown") for s in result.segments]

    blocks: list[_Block] = []
    for turn in turns:
        clip = audio[int(turn.start * sr) : int(turn.end * sr)]
        if len(clip) < int(0.2 * sr):
            continue
        result = asr.transcribe_array(clip, sr, language, vad=False)
        text = result.text.strip()
        if not text:
            continue
        if blocks and blocks[-1].cluster == turn.cluster and turn.start - blocks[-1].end < 0.4:
            blocks[-1].end = turn.end
            blocks[-1].text += " " + text
        else:
            blocks.append(_Block(start=turn.start, end=turn.end, cluster=turn.cluster, text=text))

    roles = assign_roles(blocks)
    labelled = [
        Utterance(start=b.start, end=b.end, text=b.text, speaker=roles.get(b.cluster, "unknown"))
        for b in blocks
    ]
    merged: list[Utterance] = []
    for u in labelled:
        if merged and merged[-1].speaker == u.speaker and u.start - merged[-1].end < 0.4:
            merged[-1].end = u.end
            merged[-1].text += " " + u.text
        else:
            merged.append(u)
    return merged
