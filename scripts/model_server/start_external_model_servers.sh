#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDA_BIN="${CONDA_BIN:-conda}"
OPENPI_ENV="${OPENPI_ENV:-pi05_server}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"
PERCEPTION_ENV="${PERCEPTION_ENV:-EmergePolicy}"
WAM_ENV="${WAM_ENV:-cosmos-policy}"
WAM_PYTHON="${WAM_PYTHON:-}"
PORT_CHECK_PYTHON="${PORT_CHECK_PYTHON:-}"

SERVICES="${SERVICES:-openpi,vggt,sam3}"

OPENPI_PORT="${OPENPI_PORT:-8000}"
VGGT_PORT="${VGGT_PORT:-8001}"
SAM3_PORT="${SAM3_PORT:-8002}"
WAM_PORT="${WAM_PORT:-8003}"

OPENPI_GPU="${OPENPI_GPU:-0}"
VGGT_GPU="${VGGT_GPU:-1}"
SAM3_GPU="${SAM3_GPU:-2}"
WAM_GPU="${WAM_GPU:-0}"

OPENPI_CONFIG="${OPENPI_CONFIG:-pi05_libero}"
OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT:-${REPO_ROOT}/checkpoints/pi05_libero}"
OPENPI_MAX_BATCH_SIZE="${OPENPI_MAX_BATCH_SIZE:-4}"
OPENPI_BATCH_WAIT_MS="${OPENPI_BATCH_WAIT_MS:-10}"
VGGT_MAX_BATCH_SIZE="${VGGT_MAX_BATCH_SIZE:-4}"
VGGT_BATCH_WAIT_MS="${VGGT_BATCH_WAIT_MS:-0}"
SAM3_MAX_BATCH_SIZE="${SAM3_MAX_BATCH_SIZE:-4}"
SAM3_BATCH_WAIT_MS="${SAM3_BATCH_WAIT_MS:-0}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-${REPO_ROOT}/checkpoints/vggt/model.pt}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-${REPO_ROOT}/checkpoints/sam3/model.pt}"
WAM_POLICY_CHECKPOINT="${WAM_POLICY_CHECKPOINT:-${REPO_ROOT}/checkpoints/cosmos-policy/Cosmos-Policy-LIBERO-Predict2-2B.pt}"
WAM_BASE_MODEL_DIR="${WAM_BASE_MODEL_DIR:-${REPO_ROOT}/checkpoints/cosmos-policy/Cosmos-Predict2-2B-Video2World}"
WAM_DATASET_STATS="${WAM_DATASET_STATS:-${REPO_ROOT}/checkpoints/cosmos-policy/libero_dataset_statistics.json}"
WAM_T5_EMBEDDINGS="${WAM_T5_EMBEDDINGS:-${REPO_ROOT}/checkpoints/cosmos-policy/libero_t5_embeddings.pkl}"
WAM_CONFIG_FILE="${WAM_CONFIG_FILE:-cosmos_policy/config/config.py}"
WAM_PYTHONPATH="${WAM_PYTHONPATH:-${REPO_ROOT}/third_party/cosmos-policy}"
WAM_EMBEDDING_MODE="${WAM_EMBEDDING_MODE:-cache-only}"
WAM_GENERATED_T5_CACHE="${WAM_GENERATED_T5_CACHE:-${REPO_ROOT}/artifacts/t5_embeddings/generated.pkl}"
T5_MODEL_NAME_OR_PATH="${T5_MODEL_NAME_OR_PATH:-google-t5/t5-11b}"
T5_MODEL_REVISION="${T5_MODEL_REVISION:-main}"
T5_CACHE_DIR="${T5_CACHE_DIR:-${HF_HOME:-}}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PIDS=()
CLEANUP_STARTED=0
RED="\033[1;31m"
GREEN="\033[1;32m"
RESET="\033[0m"

usage() {
    cat <<'EOF'
Usage: scripts/model_server/start_external_model_servers.sh [--services LIST]

Start selected external model services.

Command-line options:
  --services LIST   Comma-separated services to start. Accepted values:
                    openpi, vggt, sam3, cosmos, all
                    Aliases: vla=openpi, wam=cosmos
                    Overrides the SERVICES environment variable.
  -h, --help        Show this help message and exit.

Service selection and runtime:
  SERVICES                      Services used without --services
                                (default: openpi,vggt,sam3)
  CONDA_BIN                     Conda executable (default: conda)
  PORT_CHECK_PYTHON             Python executable used for port checks
                                (default: auto-detect python3 or python)

OpenPI:
  OPENPI_ENV                    Conda environment (default: pi05_server)
  OPENPI_PYTHON                 Direct Python executable; bypasses OPENPI_ENV
                                (default: empty)
  OPENPI_PORT                   Service port (default: 8000)
  OPENPI_GPU                    CUDA_VISIBLE_DEVICES (default: 0)
  OPENPI_CONFIG                 Training config name (default: pi05_libero)
  OPENPI_CHECKPOINT             Checkpoint directory
                                (default: <repo>/checkpoints/pi05_libero)
  OPENPI_MAX_BATCH_SIZE         Maximum requests per batch (default: 4)
  OPENPI_BATCH_WAIT_MS          Maximum batch collection time in ms
                                (default: 10)

VGGT and SAM3 runtime:
  PERCEPTION_ENV                Shared Conda environment
                                (default: EmergePolicy)

VGGT:
  VGGT_PORT                     Service port (default: 8001)
  VGGT_GPU                      CUDA_VISIBLE_DEVICES (default: 1)
  VGGT_CHECKPOINT               Model file
                                (default: <repo>/checkpoints/vggt/model.pt)
  VGGT_MAX_BATCH_SIZE           Maximum requests per batch (default: 4)
  VGGT_BATCH_WAIT_MS            Maximum batch collection time in ms
                                (default: 0)

SAM3:
  SAM3_PORT                     Service port (default: 8002)
  SAM3_GPU                      CUDA_VISIBLE_DEVICES (default: 2)
  SAM3_CHECKPOINT               Model file
                                (default: <repo>/checkpoints/sam3/model.pt)
  SAM3_MAX_BATCH_SIZE           Maximum requests per batch (default: 4)
  SAM3_BATCH_WAIT_MS            Maximum batch collection time in ms
                                (default: 0)

Cosmos Policy WAM:
  WAM_ENV                       Conda environment (default: cosmos-policy)
  WAM_PYTHON                    Direct Python executable; bypasses WAM_ENV
                                (default: empty)
  WAM_PORT                      Service port (default: 8003)
  WAM_GPU                       CUDA_VISIBLE_DEVICES (default: 0)
  WAM_POLICY_CHECKPOINT         Policy checkpoint
                                (default: <repo>/checkpoints/cosmos-policy/
                                Cosmos-Policy-LIBERO-Predict2-2B.pt)
  WAM_BASE_MODEL_DIR            Cosmos Predict2 base-model directory
                                (default: <repo>/checkpoints/cosmos-policy/
                                Cosmos-Predict2-2B-Video2World)
  WAM_DATASET_STATS             LIBERO dataset statistics file
                                (default: <repo>/checkpoints/cosmos-policy/
                                libero_dataset_statistics.json)
  WAM_T5_EMBEDDINGS             Pre-generated T5 embedding cache
                                (default: <repo>/checkpoints/cosmos-policy/
                                libero_t5_embeddings.pkl)
  WAM_CONFIG_FILE               Cosmos Policy config, relative to
                                WAM_PYTHONPATH when not absolute
                                (default: cosmos_policy/config/config.py)
  WAM_PYTHONPATH                Cosmos Policy source directory
                                (default: <repo>/third_party/cosmos-policy)
  WAM_EMBEDDING_MODE            T5 embedding mode (default: cache-only)
  WAM_GENERATED_T5_CACHE        Writable generated-embedding cache
                                (default: <repo>/artifacts/t5_embeddings/
                                generated.pkl)
  T5_MODEL_NAME_OR_PATH         T5 model name or path
                                (default: google-t5/t5-11b)
  T5_MODEL_REVISION             T5 model revision (default: main)
  T5_CACHE_DIR                  Hugging Face/T5 cache directory
                                (default: HF_HOME when set, otherwise empty)

JAX:
  XLA_PYTHON_CLIENT_PREALLOCATE Allow JAX GPU-memory preallocation
                                (default: false)

Examples:
  bash scripts/model_server/start_external_model_servers.sh
  bash scripts/model_server/start_external_model_servers.sh --services cosmos
  OPENPI_MAX_BATCH_SIZE=8 VGGT_MAX_BATCH_SIZE=4 \
    bash scripts/model_server/start_external_model_servers.sh \
      --services openpi,vggt
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --services)
            if [[ "$#" -lt 2 ]]; then
                echo "--services requires a comma-separated value" >&2
                exit 2
            fi
            SERVICES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

normalize_services() {
    local raw
    raw="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    raw="${raw//[[:space:]]/}"
    if [[ -z "${raw}" ]]; then
        echo "At least one service must be selected" >&2
        return 2
    fi
    if [[ ",${raw}," == *,all,* ]]; then
        echo "openpi,vggt,sam3,cosmos"
        return
    fi

    local normalized=()
    local seen=","
    local service
    IFS=',' read -r -a requested <<< "${raw}"
    for service in "${requested[@]}"; do
        [[ "${service}" == "vla" ]] && service="openpi"
        [[ "${service}" == "wam" ]] && service="cosmos"
        case "${service}" in
            openpi|vggt|sam3|cosmos) ;;
            *)
                echo "Unknown service: ${service}" >&2
                return 2
                ;;
        esac
        if [[ "${seen}" != *,${service},* ]]; then
            normalized+=("${service}")
            seen="${seen}${service},"
        fi
    done
    local joined
    joined="$(IFS=','; echo "${normalized[*]}")"
    echo "${joined}"
}

SERVICES="$(normalize_services "${SERVICES}")"

service_enabled() {
    [[ ",${SERVICES}," == *,"$1",* ]]
}

require_path() {
    local name="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        echo -e "${RED}✗ ${name} not found: ${path}${RESET}" >&2
        return 1
    fi
}

cleanup() {
    if [[ "${CLEANUP_STARTED}" -eq 1 ]]; then
        return
    fi
    CLEANUP_STARTED=1
    trap - EXIT INT TERM HUP QUIT

    if [[ "${#PIDS[@]}" -eq 0 ]]; then
        return
    fi

    echo "Stopping external model servers..."
    for pid in "${PIDS[@]}"; do
        kill -TERM -- "-${pid}" "${pid}" 2>/dev/null || true
    done

    for _attempt in {1..50}; do
        local any_running=0
        for pid in "${PIDS[@]}"; do
            if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
                any_running=1
                break
            fi
        done
        if [[ "${any_running}" -eq 0 ]]; then
            break
        fi
        sleep 0.1
    done

    for pid in "${PIDS[@]}"; do
        if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
            echo "Force stopping process group ${pid}..."
            kill -KILL -- "-${pid}" "${pid}" 2>/dev/null || true
        fi
    done
    wait "${PIDS[@]}" 2>/dev/null || true
    echo "External model servers stopped."
}

trap cleanup EXIT
trap 'exit 130' INT TERM HUP QUIT

check_port() {
    local name="$1"
    local port="$2"
    local python_bin="${PORT_CHECK_PYTHON}"

    if [[ -z "${python_bin}" ]]; then
        python_bin="$(command -v python3 || command -v python || true)"
    fi
    if [[ -z "${python_bin}" && -n "${WAM_PYTHON}" ]]; then
        python_bin="${WAM_PYTHON}"
    fi
    if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
        echo -e "${RED}✗ Cannot check ${name} port: set PORT_CHECK_PYTHON to an executable Python${RESET}" >&2
        return 1
    fi

    if ! "${python_bin}" -c '
import socket
import sys

sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", int(sys.argv[1])))
sock.close()
' "${port}" 2>/dev/null; then
        echo -e "${RED}✗ ${name} port ${port} is already in use${RESET}" >&2
        return 1
    fi
    echo -e "${GREEN}✓ ${name} port ${port} is available${RESET}"
}

start_server() {
    local name="$1"
    local conda_env="$2"
    local visible_devices="$3"
    shift 3

    echo "Starting ${name} (conda=${conda_env}, CUDA_VISIBLE_DEVICES=${visible_devices})"
    CUDA_VISIBLE_DEVICES="${visible_devices}" setsid \
        "${CONDA_BIN}" run --no-capture-output -n "${conda_env}" \
        "$@" &
    PIDS+=("$!")
}

start_server_with_python() {
    local name="$1"
    local python_bin="$2"
    local visible_devices="$3"
    shift 3

    if [[ ! -x "${python_bin}" ]]; then
        echo -e "${RED}✗ ${name} Python is not executable: ${python_bin}${RESET}" >&2
        return 1
    fi
    echo "Starting ${name} (python=${python_bin}, CUDA_VISIBLE_DEVICES=${visible_devices})"
    CUDA_VISIBLE_DEVICES="${visible_devices}" setsid "$@" &
    PIDS+=("$!")
}

start_server_in_directory() {
    local name="$1"
    local conda_env="$2"
    local visible_devices="$3"
    local working_directory="$4"
    shift 4

    echo "Starting ${name} (conda=${conda_env}, CUDA_VISIBLE_DEVICES=${visible_devices}, cwd=${working_directory})"
    (
        cd "${working_directory}"
        CUDA_VISIBLE_DEVICES="${visible_devices}" exec setsid \
            "${CONDA_BIN}" run --no-capture-output -n "${conda_env}" "$@"
    ) &
    PIDS+=("$!")
}

start_server_with_python_in_directory() {
    local name="$1"
    local python_bin="$2"
    local visible_devices="$3"
    local working_directory="$4"
    shift 4

    if [[ ! -x "${python_bin}" ]]; then
        echo -e "${RED}✗ ${name} Python is not executable: ${python_bin}${RESET}" >&2
        return 1
    fi
    echo "Starting ${name} (python=${python_bin}, CUDA_VISIBLE_DEVICES=${visible_devices}, cwd=${working_directory})"
    (
        cd "${working_directory}"
        CUDA_VISIBLE_DEVICES="${visible_devices}" exec setsid "$@"
    ) &
    PIDS+=("$!")
}

wait_for_server_exit() {
    local pid
    while true; do
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                set +e
                wait "${pid}"
                local exit_code=$?
                set -e
                return "${exit_code}"
            fi
        done
        sleep 0.2
    done
}

cd "${REPO_ROOT}"

echo "Selected services: ${SERVICES}"

if service_enabled openpi; then
    check_port "OpenPI" "${OPENPI_PORT}"
fi
if service_enabled vggt; then
    check_port "VGGT" "${VGGT_PORT}"
fi
if service_enabled sam3; then
    check_port "SAM3" "${SAM3_PORT}"
fi
if service_enabled cosmos; then
    check_port "Cosmos WAM" "${WAM_PORT}"
    require_path "WAM policy checkpoint" "${WAM_POLICY_CHECKPOINT}"
    require_path "WAM base model directory" "${WAM_BASE_MODEL_DIR}"
    require_path "WAM dataset statistics" "${WAM_DATASET_STATS}"
    require_path "WAM T5 embeddings" "${WAM_T5_EMBEDDINGS}"
    if [[ "${WAM_CONFIG_FILE}" = /* ]]; then
        require_path "WAM config file" "${WAM_CONFIG_FILE}"
    else
        require_path "WAM config file" "${WAM_PYTHONPATH}/${WAM_CONFIG_FILE}"
    fi
fi

if service_enabled openpi; then
    openpi_command=(
        "${OPENPI_PYTHON:-python}" -m external_model_server.openpi_batch_server
        --config-name "${OPENPI_CONFIG}"
        --checkpoint-dir "${OPENPI_CHECKPOINT}"
        --port "${OPENPI_PORT}"
        --max-batch-size "${OPENPI_MAX_BATCH_SIZE}"
        --batch-wait-ms "${OPENPI_BATCH_WAIT_MS}"
    )
    if [[ -n "${OPENPI_PYTHON}" ]]; then
        start_server_with_python \
            "OpenPI server :${OPENPI_PORT}" \
            "${OPENPI_PYTHON}" \
            "${OPENPI_GPU}" \
            "${openpi_command[@]}"
    else
        start_server \
            "OpenPI server :${OPENPI_PORT}" \
            "${OPENPI_ENV}" \
            "${OPENPI_GPU}" \
            "${openpi_command[@]}"
    fi
fi

if service_enabled vggt; then
    start_server \
        "VGGT server :${VGGT_PORT}" \
        "${PERCEPTION_ENV}" \
        "${VGGT_GPU}" \
        python -m external_model_server.vggt_server \
        --model-path "${VGGT_CHECKPOINT}" \
        --port "${VGGT_PORT}" \
        --max-batch-size "${VGGT_MAX_BATCH_SIZE}" \
        --batch-wait-ms "${VGGT_BATCH_WAIT_MS}"
fi

if service_enabled sam3; then
    start_server \
        "SAM3 server :${SAM3_PORT}" \
        "${PERCEPTION_ENV}" \
        "${SAM3_GPU}" \
        python -m external_model_server.sam3_server \
        --model-path "${SAM3_CHECKPOINT}" \
        --port "${SAM3_PORT}" \
        --max-batch-size "${SAM3_MAX_BATCH_SIZE}" \
        --batch-wait-ms "${SAM3_BATCH_WAIT_MS}"
fi

if service_enabled cosmos; then
    wam_command=(
        env "PYTHONPATH=${REPO_ROOT}:${WAM_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
        "${WAM_PYTHON:-python}" -m external_model_server.cosmos_policy_server
        --policy-checkpoint "${WAM_POLICY_CHECKPOINT}"
        --base-model-dir "${WAM_BASE_MODEL_DIR}"
        --dataset-stats "${WAM_DATASET_STATS}"
        --t5-embeddings "${WAM_T5_EMBEDDINGS}"
        --embedding-mode "${WAM_EMBEDDING_MODE}"
        --generated-t5-cache "${WAM_GENERATED_T5_CACHE}"
        --t5-model-name-or-path "${T5_MODEL_NAME_OR_PATH}"
        --t5-revision "${T5_MODEL_REVISION}"
        --config-file "${WAM_CONFIG_FILE}"
        --host 127.0.0.1
        --port "${WAM_PORT}"
    )
    if [[ -n "${T5_CACHE_DIR}" ]]; then
        wam_command+=(--t5-cache-dir "${T5_CACHE_DIR}")
    fi
    if [[ -n "${WAM_PYTHON}" ]]; then
        start_server_with_python_in_directory \
            "Cosmos WAM server :${WAM_PORT}" \
            "${WAM_PYTHON}" \
            "${WAM_GPU}" \
            "${WAM_PYTHONPATH}" \
            "${wam_command[@]}"
    else
        start_server_in_directory \
            "Cosmos WAM server :${WAM_PORT}" \
            "${WAM_ENV}" \
            "${WAM_GPU}" \
            "${WAM_PYTHONPATH}" \
            "${wam_command[@]}"
    fi
fi

service_enabled openpi && echo "OpenPI health   : http://localhost:${OPENPI_PORT}/healthz"
service_enabled vggt && echo "VGGT health     : http://localhost:${VGGT_PORT}/healthz"
service_enabled sam3 && echo "SAM3 health     : http://localhost:${SAM3_PORT}/healthz"
service_enabled cosmos && echo "Cosmos WAM health: http://localhost:${WAM_PORT}/healthz"
echo "Press Ctrl+C to stop the selected servers."

set +e
wait_for_server_exit
STATUS=$?
set -e
exit "${STATUS}"
