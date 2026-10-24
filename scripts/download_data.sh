#!/usr/bin/env bash
# Download the two datasets used by this repo. Requires ~/.kaggle credentials
# (access_token or kaggle.json) with the Kaggle API enabled.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="${KAGGLE:-$ROOT/.venv/bin/kaggle}"
mkdir -p "$ROOT/data/raw"

# PriMock57: audio tracks + TextGrids (~1 GB download, ~2 GB unzipped)
"$KAGGLE" datasets download enanko/primock57data \
  -p "$ROOT/data/raw/primock57_audio" --unzip

# PriMock57: notes, checklists, human-eval data (~4 MB)
"$KAGGLE" datasets download zeyadahmed2005/primock57-dataset \
  -p "$ROOT/data/raw/primock57_notes" --unzip

# iMerit multilingual ambient scribe: 199 audio + transcripts + summaries (~1.4 GB)
"$KAGGLE" datasets download imeritinc/multilingual-ambient-scribe-dataset \
  -p "$ROOT/data/raw/imerit" --unzip

echo "done. next:"
echo "  $ROOT/.venv/bin/python -m clinical_scribe ingest \\"
echo "    --primock-notes data/raw/primock57_notes \\"
echo "    --primock-audio data/raw/primock57_audio \\"
echo "    --imerit data/raw/imerit --out data/processed"
