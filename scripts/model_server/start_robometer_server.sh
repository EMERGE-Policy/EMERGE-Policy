#!/usr/bin/env bash
# Start the UNMODIFIED official Robometer server in its separate environment.
set -euo pipefail
ROBOMETER_REVISION=352d160389daa964788de1ec933d1925f3a6de4f
ROBOMETER_REPO="${ROBOMETER_REPO:?Set ROBOMETER_REPO to the official checkout directory}"
ROBOMETER_MODEL_PATH="${ROBOMETER_MODEL_PATH:-jesbu1/robometer-4b-fft-libero}"
ROBOMETER_PYTHON="${ROBOMETER_PYTHON:-${ROBOMETER_REPO}/.venv/bin/python}"
ROBOMETER_GPU="${ROBOMETER_GPU:-3}"
ROBOMETER_PORT="${ROBOMETER_PORT:-8100}"
ROBOMETER_HOST="${ROBOMETER_HOST:-127.0.0.1}"
actual_revision="$(git -C "${ROBOMETER_REPO}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${ROBOMETER_REVISION}" ]]; then
    echo "Use the verified official Robometer revision ${ROBOMETER_REVISION}; got ${actual_revision}" >&2
    exit 1
fi
if [[ ! -x "${ROBOMETER_PYTHON}" ]]; then
    echo "Robometer Python not found: ${ROBOMETER_PYTHON}. Install official dependencies first." >&2
    exit 1
fi
cd "${ROBOMETER_REPO}"
export CUDA_VISIBLE_DEVICES="${ROBOMETER_GPU}"
exec "${ROBOMETER_PYTHON}" -m robometer.evals.eval_server \
    "model_path=${ROBOMETER_MODEL_PATH}" "server_url=${ROBOMETER_HOST}" \
    "server_port=${ROBOMETER_PORT}" num_gpus=1 max_workers=1 batch_size=4 "$@"
