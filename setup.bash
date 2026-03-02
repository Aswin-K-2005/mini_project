#!/usr/bin/env bash
set -euo pipefail

# NeonGuard safe local setup: isolated venv, no system-wide installs.
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[1/5] Creating virtual environment in ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[2/5] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[3/5] Installing core dependencies"
pip install \
  numpy \
  opencv-python \
  pyaudio \
  sentence-transformers \
  mediapipe \
  faster-whisper \
  nudenet

echo "[4/5] Optional production routing dependency"
pip install pyvirtualcam || true

echo "[5/5] Verifying imports"
python - <<'PY'
mods = [
    'numpy',
    'cv2',
    'pyaudio',
    'sentence_transformers',
    'mediapipe',
    'faster_whisper',
    'nudenet',
]
for m in mods:
    try:
        __import__(m)
        print(f"OK: {m}")
    except Exception as e:
        print(f"WARN: {m} -> {e}")
print("Setup complete. Activate with: source .venv/bin/activate")
PY
