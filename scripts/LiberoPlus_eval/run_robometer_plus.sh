#!/usr/bin/env bash
# Complete EMERGE agent: seven Plus dimensions, fifty distinct tasks each.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ROBOMETER_URL="${ROBOMETER_URL:-http://127.0.0.1:8100}"
export ROBOMETER_MODEL_PATH="${ROBOMETER_MODEL_PATH:-${REPO_ROOT}/checkpoints/robometer-4b-fft-libero}"
exec "${EMERGE_PYTHON:-python}" -u scripts/LiberoPlus_eval/eval_libero_plus_agent.py \
    --policy-backend wam --driver-config dev/libero_plus_robometer_eval.json \
    --dimension all --count "${PLUS_TASKS_PER_DIMENSION:-50}" --trials-per-task 1 \
    --workers "${EVAL_WORKERS:-2}" --episode-timeout-s "${EVAL_EPISODE_TIMEOUT:-1800}" \
    --record-video --robometer-url "${ROBOMETER_URL}" \
    --robometer-model-path "${ROBOMETER_MODEL_PATH}" \
    --output-dir "${EVAL_OUTPUT_DIR:-artifacts/emerge_robometer_plus_350}" "$@"
