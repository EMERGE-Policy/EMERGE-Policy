#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

export OPENPI_SERVER_MODE="behavior_champion"
export OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/behavior-1k-solution}"
export OPENPI_PYTHON="${OPENPI_PYTHON:-${OPENPI_ROOT}/.venv/bin/python}"
export OPENPI_PORT="${OPENPI_PORT:-8011}"
export OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT:-${WORKSPACE_ROOT}/checkpoints/behavior_submission_hfmirror/checkpoint_2}"
export OPENPI_CHAMPION_MAPPING="${OPENPI_CHAMPION_MAPPING:-${OPENPI_ROOT}/task_checkpoint_mapping.json}"
export OPENPI_CHAMPION_POLICY_CONFIG="${OPENPI_CHAMPION_POLICY_CONFIG:-pi_behavior_b1k_fast}"
export OPENPI_GPU="${OPENPI_GPU:-0}"
export SAM3_GPU="${SAM3_GPU:-0}"
BEHAVIOR_SERVICES="${BEHAVIOR_SERVICES:-openpi,sam3}"

exec bash "${REPO_ROOT}/scripts/model_server/start_external_model_servers.sh" \
    --services "${BEHAVIOR_SERVICES}"
