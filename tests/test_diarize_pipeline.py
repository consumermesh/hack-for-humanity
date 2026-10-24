import numpy as np

from clinical_scribe.asr import ASRResult, ASRSegment
from clinical_scribe.diarize import Turn, transcribe_diarized


class FakeDiarizer:
    def diarize(self, audio, sr, num_speakers=2):
        return [Turn(0.0, 2.0, 0), Turn(2.0, 4.0, 1), Turn(4.0, 6.0, 0)]


class FakeASR:
    def __init__(self):
        self.calls = 0

    def transcribe_array(self, audio, sr=16000, language=None, vad=True):
        self.calls += 1
        # First and third blocks are the clinician (questions), second is the patient.
        text = "How long have you had this?" if self.calls % 2 == 1 else "Three days now."
        return ASRResult(segments=[ASRSegment(0.0, 1.0, text)], text=text, duration_s=1.0)


def test_transcribe_diarized_labels_and_merges_blocks():
    audio = np.zeros(16000 * 6, dtype=np.float32)
    utts = transcribe_diarized(audio, 16000, FakeASR(), FakeDiarizer(), num_speakers=2)
    assert [u.speaker for u in utts] == ["clinician", "patient", "clinician"]
    assert utts[0].start == 0.0 and utts[1].start == 2.0
    assert "How long" in utts[0].text and "Three days" in utts[1].text


def test_transcribe_diarized_falls_back_when_no_turns():
    class EmptyDiarizer:
        def diarize(self, audio, sr, num_speakers=2):
            return []

    audio = np.zeros(16000 * 2, dtype=np.float32)
    utts = transcribe_diarized(audio, 16000, FakeASR(), EmptyDiarizer())
    assert [u.speaker for u in utts] == ["unknown"]
