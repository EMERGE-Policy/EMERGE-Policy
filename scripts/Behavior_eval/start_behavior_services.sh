#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

export OPENPI_SERVER_MODE="behavior_b1k"
export OPENPI_ROOT="${OPENPI_ROOT:-${WORKSPACE_ROOT}/openpi}"
export OPENPI_PYTHON="${OPENPI_PYTHON:-${OPENPI_ROOT}/.venv/bin/python}"
export OPENPI_PORT="${OPENPI_PORT:-8011}"
export OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT:-${WORKSPACE_ROOT}/checkpoints/pi05TurningOnRadio/pi05_turn_on_the_radio}"
export OPENPI_GPU="${OPENPI_GPU:-0}"
export SAM3_GPU="${SAM3_GPU:-0}"

exec bash "${REPO_ROOT}/scripts/model_server/start_external_model_servers.sh" \
    --services openpi,sam3
