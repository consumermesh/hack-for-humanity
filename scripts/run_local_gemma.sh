#!/usr/bin/env bash
# Local, serverless Gemma path: wait for the GGUF, then repair + extract + score
# on the first PriMock encounter. Run from the repo root.
set -euo pipefail

GGUF_DIR="${1:-data/models/gemma4-e2b}"
MANIFEST="data/processed/primock57.jsonl"

echo "waiting for a GGUF in $GGUF_DIR ..."
while ! ls "$GGUF_DIR"/*.gguf >/dev/null 2>&1; do sleep 30; done
GGUF="$(ls "$GGUF_DIR"/*.gguf | head -1)"
echo "using $GGUF"

.venv/bin/python -m clinical_scribe correct \
  --manifest "$MANIFEST" --limit 1 --force \
  --api llamacpp --model-path "$GGUF"

.venv/bin/python -m clinical_scribe extract \
  --manifest "$MANIFEST" --limit 1 --force \
  --api llamacpp --model-path "$GGUF" \
  --asr-dir runs/corrected --out runs/notes

.venv/bin/python -m clinical_scribe evaluate \
  --manifest "$MANIFEST" --limit 1 \
  --asr-dir runs/corrected --notes-dir runs/notes \
  --out runs/eval-local-gemma
