#!/usr/bin/env bash
set -euo pipefail

MODULE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$MODULE_DIR/.venv-train/bin/python"
CHECKPOINT="$MODULE_DIR/outputs/train/full-control-research-run1-to12000/full-control-research-run1-to12000-phase-009000/checkpoints/009000/pretrained_model"
DEVICE="${AGRI_DEVICE:-cuda}"
PORT="${AGRI_PORT:-8001}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Thiếu môi trường train: $PYTHON" >&2
  exit 1
fi
if [[ ! -s "$CHECKPOINT/adapter_model.safetensors" || ! -f "$CHECKPOINT/late_vlm.json" ]]; then
  echo "Thiếu checkpoint 9.000: $CHECKPOINT" >&2
  exit 1
fi

cd "$MODULE_DIR/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "Kiểm trạng thái: GET http://127.0.0.1:$PORT/api/status"
echo "Gọi model: POST http://127.0.0.1:$PORT/api/infer"
exec "$PYTHON" -u -m agri_vla.late_vlm_full web \
  --checkpoint "$CHECKPOINT" --device "$DEVICE" --port "$PORT"
