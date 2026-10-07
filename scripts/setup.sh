#!/usr/bin/env bash
set -Eeuo pipefail

# CPU/meta validation used Python 3.10 and PyTorch 2.6.0. Select the torch wheel
# index appropriate for your CUDA driver; this script does not install a driver.
VENV_PATH="${1:-.venv}"
python3 -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install torch==2.6.0 \
  --index-url "${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu124}"
"$VENV_PATH/bin/python" -m pip install -e '.[dev]'
echo "Environment ready: $VENV_PATH (install '.[features]' for raw video/text encoding)"
