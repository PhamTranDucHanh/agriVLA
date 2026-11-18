#!/usr/bin/env bash
set -euo pipefail

MODULE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AGRI_PORT="${AGRI_PORT:-8000}"
echo "Mở web demo: http://127.0.0.1:$AGRI_PORT"
exec bash "$MODULE_DIR/deploy.sh"
