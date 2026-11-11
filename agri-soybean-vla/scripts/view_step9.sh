#!/usr/bin/env bash
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8780}"

case "$PORT" in
  ''|*[!0-9]*) echo "Usage: bash agri-soybean-vla/scripts/view_step9.sh [port]" >&2; exit 2 ;;
esac

for artifact in \
  outputs/eval/step9-final-val-full-20261109/metrics.json \
  outputs/eval/step9-final-val-full-20261109/predictions.jsonl \
  outputs/eval/step9-final-test-full-20261109/metrics.json \
  outputs/eval/step9-final-test-full-20261109/predictions.jsonl \
  outputs/eval/step9-baselines-20261109/metrics.json
do
  if [[ ! -f "$MODULE_DIR/$artifact" ]]; then
    echo "Missing Step 9 artifact: $MODULE_DIR/$artifact" >&2
    exit 2
  fi
done

echo "Step 9 viewer: http://127.0.0.1:${PORT}/src/agri_vla/web/step9.html"
echo "Press Ctrl+C to stop."
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$MODULE_DIR"
