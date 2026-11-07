#!/usr/bin/env bash
set -euo pipefail
AGRI_MODULE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AGRI_BROWSER_PY="$AGRI_MODULE_DIR/.venv/bin/python"
if [[ ! -x "$AGRI_BROWSER_PY" ]]; then
  echo "Thiếu môi trường data: $AGRI_BROWSER_PY" >&2
  echo "Chạy: bash \"$AGRI_MODULE_DIR/scripts/setup.sh\" --mode data" >&2
  exit 1
fi
# Current, versioned wording/explicit-override dataset. Legacy v0 remains
# available through: python -m agri_vla.cli browse --manifests .../v0
exec "$AGRI_BROWSER_PY" -m agri_vla.cli browse-revision "$@"
