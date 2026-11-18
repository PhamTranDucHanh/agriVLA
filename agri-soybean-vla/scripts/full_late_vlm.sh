#!/usr/bin/env bash
set -euo pipefail
AGRI_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_DATASETS_CACHE="$AGRI_MODULE_DIR/.cache/hf-datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$AGRI_MODULE_DIR/.."
exec "$AGRI_MODULE_DIR/.venv-train/bin/python" -u -m agri_vla.late_vlm_full run "$@"
