#!/usr/bin/env bash
set -euo pipefail
AGRI_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$AGRI_REPO_ROOT"
AGRI_MODE="${1:-plan}"
AGRI_TAG="${2:-$(date +%Y%m%d-%H%M%S)}"
[[ "$AGRI_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run tag' >&2; exit 2; }
AGRI_PY="$AGRI_REPO_ROOT/agri-soybean-vla/.venv-train/bin/python"
export HF_DATASETS_CACHE="$AGRI_REPO_ROOT/agri-soybean-vla/.cache/hf-datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
case "$AGRI_MODE" in
  plan|smoke) AGRI_STEPS=2 ;;
  100) AGRI_STEPS=100 ;;
  300) AGRI_STEPS=300 ;;
  *) echo 'Usage: bash late_vlm_v1.sh {plan|smoke|100|300} RUN_TAG' >&2; exit 2 ;;
esac
for AGRI_ARM in control candidate; do
  AGRI_BASE="$AGRI_REPO_ROOT/agri-soybean-vla/outputs/small-cases/latekv-$AGRI_TAG-$AGRI_ARM"
  AGRI_ARGS=(--arm "$AGRI_ARM" --steps "$AGRI_STEPS" --output "$AGRI_BASE-s$AGRI_STEPS")
  if [[ "$AGRI_MODE" == 100 ]]; then
    AGRI_ARGS+=(--resume-checkpoint "$AGRI_BASE-s2/checkpoints/000002")
  elif [[ "$AGRI_MODE" == 300 ]]; then
    AGRI_ARGS+=(--resume-checkpoint "$AGRI_BASE-s100/checkpoints/000100")
  fi
  if [[ "$AGRI_MODE" != plan ]]; then AGRI_ARGS+=(--execute); fi
  "$AGRI_PY" -u -m agri_vla.late_vlm_experiment "${AGRI_ARGS[@]}"
done
