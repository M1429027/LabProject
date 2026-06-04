#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PATH="${VENV_PATH:-/home/yp8700/amass/.venv}"
MODEL_DIR="${REPO_ROOT}/learning/karate_selfcal/checkpoints/hrnet_w32_coco_256x192"
MMCV_WHL_INDEX="https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html"
HRNET_CONFIG_NAME="td-hm_hrnet-w32_8xb64-210e_coco-256x192"

echo "[karate_selfcal] repo root: ${REPO_ROOT}"
echo "[karate_selfcal] venv path: ${VENV_PATH}"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Virtual environment not found at ${VENV_PATH}" >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"

python - <<'PY'
import torch
print("[karate_selfcal] torch:", torch.__version__)
print("[karate_selfcal] cuda_available:", torch.cuda.is_available())
print("[karate_selfcal] cuda_device_count:", torch.cuda.device_count())
PY

pip install -U openmim
mim install mmengine
pip install 'mmcv>=2.0.1,<2.2.0' -f "${MMCV_WHL_INDEX}"
mim install "mmdet>=3.1.0,<3.3.0"
mim install "mmpose>=1.3.0,<1.4.0"

mkdir -p "${MODEL_DIR}"
mim download mmpose --config "${HRNET_CONFIG_NAME}" --dest "${MODEL_DIR}"

echo
echo "[karate_selfcal] install complete."
echo "[karate_selfcal] downloaded files:"
find "${MODEL_DIR}" -maxdepth 1 -type f | sort
echo
echo "[karate_selfcal] next config file:"
echo "  ${REPO_ROOT}/learning/karate_selfcal/configs/detection_hrnet_w32.yaml"
