"""Audio loading and mixing.

PriMock57 ships separate doctor and patient tracks. Real ambient scribes get a
single mixed room mic, so we mix the tracks for a realistic ASR input.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def decode(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-supported file to mono float32 at TARGET_SR."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        data, sr = _decode_with_av(path)
    mono = data.mean(axis=1)
    if sr != TARGET_SR:
        mono = _resample_linear(mono, sr, TARGET_SR)
    return mono, TARGET_SR


def _decode_with_av(path: str | Path) -> tuple[np.ndarray, int]:
    import av  # provided by faster-whisper

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SR)
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                frames.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            frames.append(out.to_ndarray().reshape(-1))
    if not frames:
        raise ValueError(f"no audio decoded from {path}")
    return np.concatenate(frames).astype(np.float32), TARGET_SR


def _resample_linear(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target or len(x) == 0:
        return x
    n = int(round(len(x) * target / sr))
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    xq = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


def duration_s(path: str | Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        audio, sr = decode(path)
        return len(audio) / sr


def mix_tracks(paths: list[str | Path], out_path: str | Path) -> str:
    """Sum tracks into one mono 16 kHz wav (clipped) and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracks = [decode(p)[0] for p in paths]
    length = max(len(t) for t in tracks)
    mixed = np.zeros(length, dtype=np.float32)
    for t in tracks:
        mixed[: len(t)] += t
    np.clip(mixed, -1.0, 1.0, out=mixed)
    sf.write(str(out_path), mixed, TARGET_SR, subtype="PCM_16")
    return str(out_path)


def prepare_audio(encounter_id: str, audio_paths: list[str], cache_dir: str | Path) -> str:
    """Return a single audio path for the encounter, mixing tracks if needed."""
    if len(audio_paths) == 1:
        return audio_paths[0]
    digest = hashlib.sha1("|".join(sorted(map(str, audio_paths))).encode()).hexdigest()[:10]
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in encounter_id).strip("-")
    out = Path(cache_dir) / f"{slug}-{digest}.wav"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    return mix_tracks(audio_paths, out)
