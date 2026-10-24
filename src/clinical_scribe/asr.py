"""ASR backends. faster-whisper is the default because it runs on CPU without torch."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str


_WHISPER_LANGS = {
    "en-gb": "en",
    "en-us": "en",
    "uk english": "en",
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "dutch": "nl",
}


def whisper_language(label: str | None) -> str | None:
    """Map dataset language labels (e.g. 'UK English', 'en-GB') to Whisper codes."""
    if not label:
        return None
    label = label.strip().lower()
    if label in _WHISPER_LANGS:
        return _WHISPER_LANGS[label]
    if len(label) == 2:
        return label
    return None


@dataclass
class ASRResult:
    segments: list[ASRSegment]
    text: str
    language: str | None = None
    duration_s: float | None = None
    latency_s: float = 0.0
    model: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float | None:
        if not self.duration_s:
            return None
        return self.latency_s / self.duration_s


class FasterWhisperASR:
    """Thin wrapper so the pipeline is not coupled to a specific ASR library."""

    def __init__(
        self,
        model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 5,
    ):
        self.model_name = model
        self.language = language
        self.beam_size = beam_size
        self._model = None
        self._device = device
        self._compute_type = compute_type

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> ASRResult:
        from .audio import decode

        audio, sr = decode(audio_path)
        return self.transcribe_array(audio, sr, language)

    def transcribe_array(
        self, audio: np.ndarray, sr: int = 16000, language: str | None = None, vad: bool = True
    ) -> ASRResult:
        model = self._load()
        started = time.perf_counter()
        segments_iter, info = model.transcribe(
            audio,
            language=language or self.language,
            beam_size=self.beam_size,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 500} if vad else None,
            condition_on_previous_text=False,
        )
        segments = [
            ASRSegment(start=float(s.start), end=float(s.end), text=s.text.strip())
            for s in segments_iter
            if s.text.strip()
        ]
        latency = time.perf_counter() - started
        return ASRResult(
            segments=segments,
            text=" ".join(s.text for s in segments),
            language=getattr(info, "language", None),
            duration_s=len(audio) / sr if sr else None,
            latency_s=latency,
            model=self.model_name,
        )


class MedASRASR:
    """Adapter for google/medasr (gated; requires torch + transformers>=5).

    MedASR is dictation-trained, so treat it as a candidate to benchmark on
    conversational PriMock57 audio, not as the default.
    """

    def __init__(self, model_id: str = "google/medasr", device: str | None = None):
        self.model_id = model_id
        self._device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._pipe = pipeline(
                "automatic-speech-recognition", model=self.model_id, device=device
            )
        return self._pipe

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> ASRResult:
        from .audio import decode

        audio, sr = decode(audio_path)
        return self.transcribe_array(audio, sr, language)

    def transcribe_array(
        self, audio: np.ndarray, sr: int = 16000, language: str | None = None, vad: bool = True
    ) -> ASRResult:
        started = time.perf_counter()
        out = self._load()(audio, chunk_length_s=20, stride_length_s=2, sampling_rate=sr)
        latency = time.perf_counter() - started
        text = out["text"].strip() if isinstance(out, dict) else str(out)
        return ASRResult(
            segments=[ASRSegment(start=0.0, end=len(audio) / sr, text=text)],
            text=text,
            duration_s=len(audio) / sr,
            latency_s=latency,
            model=self.model_id,
        )
