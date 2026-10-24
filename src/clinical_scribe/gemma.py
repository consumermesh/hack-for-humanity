"""Gemma client (Google Gemini API, Ollama, OpenAI-compatible, llama.cpp)."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import requests

from .prompts import REPAIR_TEMPLATE, SYSTEM, build_user_prompt
from .schema import ClinicalNote, Fact


@dataclass
class GemmaConfig:
    base_url: str = field(default_factory=lambda: os.getenv("GEMMA_BASE_URL", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: os.getenv("GEMMA_MODEL", "gemma3:27b"))
    api: str = field(default_factory=lambda: os.getenv("GEMMA_API", "ollama"))  # google | ollama | openai | llamacpp
    api_key: str = field(
        default_factory=lambda: os.getenv("GEMMA_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    )
    model_path: str = field(default_factory=lambda: os.getenv("GEMMA_MODEL_PATH", ""))
    temperature: float = 0.0
    timeout: float = 900.0
    max_attempts: int = 2


@dataclass
class ExtractionResult:
    note: ClinicalNote
    raw: str
    attempts: int
    latency_s: float
    model: str


class GemmaClient:
    def __init__(self, config: GemmaConfig | None = None):
        self.config = config or GemmaConfig()

    # -- transport ---------------------------------------------------------
    def chat(self, system: str, user: str) -> str:
        if self.config.api == "google":
            return self._chat_google(system, user)
        if self.config.api == "openai":
            return self._chat_openai(system, user)
        if self.config.api == "llamacpp":
            return self._chat_llamacpp(system, user)
        return self._chat_ollama(system, user)

    def _chat_google(self, system: str, user: str) -> str:
        """Native Gemini API `generateContent` (hosts Gemma model ids)."""
        if not self.config.api_key:
            raise ValueError("Google API key missing: set GOOGLE_API_KEY (or GEMINI_API_KEY)")
        base = self.config.base_url.rstrip("/")
        if "11434" in base or base in ("", "http://127.0.0.1"):
            base = "https://generativelanguage.googleapis.com"
        url = f"{base}/v1beta/models/{self.config.model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": 8192,
            },
        }
        r = requests.post(url, params={"key": self.config.api_key}, json=payload, timeout=self.config.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"google api {r.status_code}: {r.text[:500]}")
        data = r.json()
        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise RuntimeError(f"prompt blocked: {feedback['blockReason']}")
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"no candidates: {json.dumps(data)[:300]}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        # Gemma 4 streams its reasoning as parts flagged thought=true; only the
        # unflagged parts are the answer we want to parse as JSON.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if not text:
            reason = candidates[0].get("finishReason", "unknown")
            thoughts = sum(1 for p in parts if p.get("thought"))
            raise ValueError(
                f"empty google response (finishReason={reason}, thought_parts={thoughts})"
            )
        return text

    def _chat_llamacpp(self, system: str, user: str) -> str:
        """In-process GGUF inference: no server, useful on CPU-only machines."""
        if getattr(self, "_llm", None) is None:
            from llama_cpp import Llama

            path = self.config.model_path or self.config.model
            if not os.path.exists(path):
                raise FileNotFoundError(f"GGUF model not found: {path}")
            self._llm = Llama(model_path=path, n_ctx=8192, n_threads=os.cpu_count(), verbose=False)
        out = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature,
            max_tokens=2048,
        )
        return out["choices"][0]["message"]["content"]

    def _chat_ollama(self, system: str, user: str) -> str:
        url = self.config.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.config.temperature, "num_ctx": 8192},
        }
        r = requests.post(url, json=payload, timeout=self.config.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _chat_openai(self, system: str, user: str) -> str:
        base = self.config.base_url.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=self.config.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def health(self) -> dict:
        if self.config.api == "google":
            if not self.config.api_key:
                return {"ok": False, "error": "missing GOOGLE_API_KEY", "model": self.config.model}
            base = self.config.base_url.rstrip("/")
            if "11434" in base:
                base = "https://generativelanguage.googleapis.com"
            try:
                # A models.list call succeeds even on a zero-credit project, so
                # probe an actual (tiny) generation instead.
                r = requests.post(
                    f"{base}/v1beta/models/{self.config.model}:generateContent",
                    params={"key": self.config.api_key},
                    json={
                        "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                        "generationConfig": {"maxOutputTokens": 1},
                    },
                    timeout=60,
                )
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                err = (body.get("error") or {})
                return {
                    "ok": r.ok,
                    "status": r.status_code,
                    "model": self.config.model,
                    "error": err.get("status", "") if err else "",
                }
            except Exception as e:  # pragma: no cover - network dependent
                return {"ok": False, "error": str(e), "model": self.config.model}
        if self.config.api == "llamacpp":
            path = self.config.model_path or self.config.model
            return {"ok": os.path.exists(path), "model": os.path.basename(path), "exists": os.path.exists(path)}
        try:
            if self.config.api == "openai":
                r = requests.get(self.config.base_url.rstrip("/") + "/v1/models", timeout=10)
            else:
                r = requests.get(self.config.base_url.rstrip("/") + "/api/tags", timeout=10)
            return {"ok": r.ok, "status": r.status_code, "model": self.config.model}
        except Exception as e:  # pragma: no cover - network dependent
            return {"ok": False, "error": str(e), "model": self.config.model}

    # -- extraction --------------------------------------------------------
    def extract_note(self, transcript: str) -> ExtractionResult:
        started = time.perf_counter()
        user = build_user_prompt(transcript)
        raw = ""
        last_error = ""
        for attempt in range(1, self.config.max_attempts + 1):
            if attempt == 1:
                raw = self.chat(SYSTEM, user)
            else:
                raw = self.chat(
                    SYSTEM,
                    REPAIR_TEMPLATE.format(error=last_error, previous=raw[:6000]),
                )
            try:
                note = self._parse(raw)
                return ExtractionResult(
                    note=note,
                    raw=raw,
                    attempts=attempt,
                    latency_s=time.perf_counter() - started,
                    model=self.config.model,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
        raise ValueError(f"Gemma did not return a valid note after {self.config.max_attempts} attempts: {last_error}")

    @staticmethod
    def _parse(raw: str) -> ClinicalNote:
        obj = parse_json_object(raw)
        note = ClinicalNote.model_validate(obj)
        clean_facts = [
            Fact(
                section=f.section,
                text=f.text.strip(),
                evidence=f.evidence.strip(),
                speaker=f.speaker,
            )
            for f in note.facts
            if f.text.strip()
        ]
        return note.model_copy(update={"facts": clean_facts})


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a model response.

    Handles markdown fences, leading prose, and trailing commentary.
    """
    text = _FENCE_RE.sub("", text.strip())
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj
