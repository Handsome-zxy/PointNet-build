#!/usr/bin/env bash
# Create the Python environment for PointNet and fetch ModelNet40.
# Usage: bash setup_env.sh
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

if [ ! -d .venv ]; then
    echo ">>> creating .venv (Python $(python3 -V 2>&1 | awk '{print $2}'))"
    python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip -i "$PIP_INDEX"
.venv/bin/pip install -r requirements.txt -i "$PIP_INDEX"

# provider.py downloads ModelNet40 automatically on first import. Do it here so
# failures surface during setup instead of hanging inside training.
.venv/bin/python -c "import provider; print('ModelNet40 ready at', provider.DATA_DIR)"

echo
echo "Environment ready. Example:"
echo "  source .venv/bin/activate"
echo "  python train.py --gpu 1 --num_point 1024 --batch_size 32"
