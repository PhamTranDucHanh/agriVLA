#!/usr/bin/env bash

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "======================================"
echo " agriVLA - Light Demo Installer"
echo "======================================"

cd "$ROOT"

# --------------------------------------------------
# 1. Check Python
# --------------------------------------------------

echo
echo "[1/6] Checking Python..."

python3 --version

# --------------------------------------------------
# 2. Create virtual environment
# --------------------------------------------------

echo
echo "[2/6] Creating .venv..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
else
    echo ".venv already exists."
fi

source .venv/bin/activate

# --------------------------------------------------
# 3. Upgrade pip
# --------------------------------------------------

echo
echo "[3/6] Upgrading pip..."

python -m pip install --upgrade pip setuptools wheel

# --------------------------------------------------
# 4. Install PyTorch CUDA 12.6
# --------------------------------------------------

echo
echo "[4/6] Installing PyTorch..."

pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126

# --------------------------------------------------
# 5. Install LeRobot
# --------------------------------------------------

echo
echo "[5/6] Installing LeRobot 0.6.1..."

if [ ! -d "lerobot" ]; then
    git clone \
        --branch v0.6.1 \
        --depth 1 \
        https://github.com/huggingface/lerobot.git
fi

# Install LeRobot + PEFT support
pip install -e "./lerobot[peft]"

# Explicitly ensure PEFT is installed
pip install peft

# Other demo dependencies
pip install -r requirements.txt

# Verify required Python packages
python - <<'PY'
import torch
import peft

print("PyTorch:", torch.__version__)
print("PEFT:", peft.__version__)
print("PEFT import: OK")
PY

# --------------------------------------------------
# 6. GPU check
# --------------------------------------------------

echo
echo "[6/6] Checking CUDA..."

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA GPU not detected.")
PY

echo
echo "======================================"
echo " Installation completed"
echo "======================================"
echo
echo "Run:"
echo "  source .venv/bin/activate"
echo "  python scripts/demo_light_vla.py"
