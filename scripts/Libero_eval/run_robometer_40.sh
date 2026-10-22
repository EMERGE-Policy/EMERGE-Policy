#!/usr/bin/env bash
# Full EMERGE agent: four ten-task suites, one official initial state per task.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ROBOMETER_URL="${ROBOMETER_URL:-http://127.0.0.1:8100}"
export ROBOMETER_MODEL_PATH="${ROBOMETER_MODEL_PATH:-${REPO_ROOT}/checkpoints/robometer-4b-fft-libero}"
exec "${EMERGE_PYTHON:-python}" -u scripts/Libero_eval/eval_libero_agent.py \
    --policy-backend wam --driver-config dev/libero_robometer_eval.json \
    --suite all --task-ids all --trials-per-task 1 \
    --workers "${EVAL_WORKERS:-2}" --episode-timeout-s "${EVAL_EPISODE_TIMEOUT:-1800}" \
    --watchdog-ready-timeout-s 300 --record-video \
    --robometer-url "${ROBOMETER_URL}" --robometer-model-path "${ROBOMETER_MODEL_PATH}" \
    --output-dir "${EVAL_OUTPUT_DIR:-artifacts/emerge_robometer_40}" "$@"
