#!/usr/bin/env bash
set -euo pipefail
AGRI_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AGRI_MODULE_DIR/.."
export HF_DATASETS_CACHE="$AGRI_MODULE_DIR/.cache/hf-datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "$AGRI_MODULE_DIR/.venv-train/bin/python" -u -m agri_vla.full_research "$@"
