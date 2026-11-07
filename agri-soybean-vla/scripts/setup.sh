#!/usr/bin/env bash
# Isolated bootstrap only: no raw-data or model-weight download in either mode.
set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$MODULE_ROOT/.venv"
VENDOR="$MODULE_ROOT/.vendor/lerobot"
LEROBOT_COMMIT=7e241bd630a3719a56157a497ce5d08f244784f1
MODE=data
PYTHON_BIN=python3.12
UV_BIN=uv
MANAGED=false
DRY_RUN=false

usage() {
    cat <<'HELP'
Usage: bash scripts/setup.sh [--mode data|train] [--python /path/to/python3.12]
                            [--managed-python] [--uv /path/to/uv] [--dry-run]
data: install only the package, small data dependencies and development tools.
train: install pinned LeRobot + smolvla,peft,training and CUDA wheels in .venv-train.
data remains in .venv; neither mode changes the demo environment.
--managed-python: use uv to obtain Python 3.12 locally if it is missing.
--dry-run: print commands without creating directories, downloading or installing.
No mode downloads raw datasets or model weights. Existing environments are kept.
HELP
}

die() { printf '%s\n' "$*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode|--python|--uv)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            case "$1" in
                --mode) MODE="$2" ;;
                --python) PYTHON_BIN="$2" ;;
                --uv) UV_BIN="$2" ;;
            esac
            shift 2 ;;
        --managed-python) MANAGED=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
[[ "$MODE" == data || "$MODE" == train ]] || die "--mode must be data or train"
if [[ "$MODE" == train ]]; then VENV="$MODULE_ROOT/.venv-train"; fi
if $MANAGED && [[ "$PYTHON_BIN" != python3.12 ]]; then
    die "Choose either --managed-python or --python, not both"
fi

run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if ! $DRY_RUN; then "$@"; fi
}

printf 'Project: %s\nMode: %s\nLeRobot pin: %s\n' "$MODULE_ROOT" "$MODE" "$LEROBOT_COMMIT"
[[ ! -L "$VENV" ]] || die "Refusing a symlinked environment; use an environment owned by this module"
export PIP_CACHE_DIR="$MODULE_ROOT/.cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
if ! $DRY_RUN; then mkdir -p "$MODULE_ROOT/.cache/tmp-$MODE"; fi
export TMPDIR="$MODULE_ROOT/.cache/tmp-$MODE"
if [[ ! -x "$VENV/bin/python" ]]; then
    [[ ! -e "$VENV" ]] || die "Existing .venv has no Python; inspect it manually, it will not be deleted"
    if $MANAGED; then
        if ! $DRY_RUN; then command -v "$UV_BIN" >/dev/null || die "Install uv or pass --uv /path/to/uv"; fi
        export UV_CACHE_DIR="$MODULE_ROOT/.cache/uv"
        export UV_PYTHON_INSTALL_DIR="$MODULE_ROOT/.tools/python"
        export UV_PYTHON_BIN_DIR="$MODULE_ROOT/.tools/bin"
        run "$UV_BIN" venv --python 3.12 --managed-python --seed "$VENV"
    else
        if ! $DRY_RUN; then
            command -v "$PYTHON_BIN" >/dev/null || die "Python 3.12 missing: use --python /path/to/python3.12 or --managed-python --uv /path/to/uv"
            "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 is required")'
        fi
        run "$PYTHON_BIN" -m venv "$VENV"
    fi
fi
run "$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.prefix != sys.base_prefix else "An isolated Python 3.12 environment is required")'
run "$VENV/bin/python" -m pip install --require-hashes -r "$MODULE_ROOT/requirements-data.lock.txt"
run "$VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$MODULE_ROOT"

if [[ "$MODE" == train ]]; then
    if [[ -e "$VENDOR" ]]; then
        if ! $DRY_RUN; then
            actual_commit="$(git -C "$VENDOR" rev-parse HEAD)"
            [[ "$actual_commit" == "$LEROBOT_COMMIT" ]] || die "Vendor commit differs from $LEROBOT_COMMIT; checkout left untouched"
        fi
    else
        run mkdir -p "$MODULE_ROOT/.vendor"
        run env GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch v0.6.1 https://github.com/huggingface/lerobot.git "$VENDOR"
    fi
    run "$VENV/bin/python" -c 'import subprocess, sys; actual = subprocess.check_output(["git", "-C", sys.argv[1], "rev-parse", "HEAD"], text=True).strip(); sys.exit(0 if actual == sys.argv[2] else "LeRobot commit mismatch")' "$VENDOR" "$LEROBOT_COMMIT"
    run "$VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.1' 'torchvision==0.22.1'
    if [[ -f "$MODULE_ROOT/requirements-train.lock.txt" ]]; then
        run "$VENV/bin/python" -m pip install -r "$MODULE_ROOT/requirements-train.lock.txt"
    fi
    run "$VENV/bin/python" -m pip install --no-build-isolation -c "$MODULE_ROOT/requirements-train.constraints.txt" -e "$VENDOR[smolvla,peft,training]"
fi
run "$VENV/bin/python" -m pip check
run "$VENV/bin/python" -m agri_vla.cli preflight --mode "$MODE"
if ! $DRY_RUN; then
    mkdir -p "$MODULE_ROOT/reports/environment"
    snapshot="$(mktemp "$MODULE_ROOT/reports/environment/${MODE}-freeze-XXXXXXXX.txt")"
    "$VENV/bin/python" -m pip freeze --all > "$snapshot"
    printf 'Environment snapshot (not a portable lock): %s\n' "$snapshot"
fi
printf 'Setup commands complete. No dataset/model weights downloaded; no training performed.\n'
