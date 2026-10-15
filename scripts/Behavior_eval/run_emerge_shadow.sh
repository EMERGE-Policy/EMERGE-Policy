#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

TASK_NAME="${1:-}"
INSTANCE_INDEX="${2:-0}"
UPSTREAM_PORT="${OPENPI_PORT:-8011}"
GATEWAY_PORT="${EMERGE_GATEWAY_PORT:-8000}"
BEHAVIOR_ROOT="${REPO_ROOT}/third_party/behavior-1k"
BEHAVIOR_PYTHON="${WORKSPACE_ROOT}/envs/behavior/bin/python"
EMERGE_PYTHON="${EMERGE_PYTHON:-/home/embodied/miniconda3/envs/phyagentos/bin/python}"
ROBOT_CONFIG="${REPO_ROOT}/dev/behavior_r1pro_openpi.yaml"
DRIVER_CONFIG="${REPO_ROOT}/dev/behavior_agent_eval.json"
OUTPUT_DIR="${WORKSPACE_ROOT}/outputs/Emerge/${TASK_NAME}/shadow"

if [[ -z "${TASK_NAME}" ]]; then
    echo "用法: bash $0 <task_name> [instance_index]" >&2
    exit 2
fi

if ! [[ "${INSTANCE_INDEX}" =~ ^[0-9]+$ ]] || (( INSTANCE_INDEX > 9 )); then
    echo "instance_index 必须是 public_test 的 0-9。" >&2
    exit 2
fi

INSTANCE_ID=$((301 + INSTANCE_INDEX))
OBSERVATION_WORKSPACE="${OUTPUT_DIR}/workspace/shadow_${INSTANCE_ID}"
GATEWAY_LOG="${OUTPUT_DIR}/logs/shadow_gateway_${INSTANCE_ID}.log"
EVALUATOR_LOG="${OUTPUT_DIR}/logs/shadow_evaluator_${INSTANCE_ID}.log"
RESULT_PATH="${OUTPUT_DIR}/json/${TASK_NAME}_${INSTANCE_ID}_0.json"

for REQUIRED in "${BEHAVIOR_PYTHON}" "${EMERGE_PYTHON}"; do
    if [[ ! -x "${REQUIRED}" ]]; then
        echo "Python 不存在或不可执行: ${REQUIRED}" >&2
        exit 1
    fi
done
for REQUIRED in "${BEHAVIOR_ROOT}" "${ROBOT_CONFIG}" "${DRIVER_CONFIG}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "缺少必需路径: ${REQUIRED}" >&2
        exit 1
    fi
done

if [[ -s "${RESULT_PATH}" ]]; then
    echo "该实例已有官方结果，拒绝覆盖: ${RESULT_PATH}" >&2
    exit 1
fi

if ! curl --noproxy '*' --silent --show-error --fail \
    "http://127.0.0.1:${UPSTREAM_PORT}/healthz" >/dev/null; then
    echo "冠军策略服务未就绪: http://127.0.0.1:${UPSTREAM_PORT}/healthz" >&2
    echo "请先运行 scripts/Behavior_eval/start_behavior_services.sh。" >&2
    exit 1
fi

if curl --noproxy '*' --silent --fail \
    "http://127.0.0.1:${GATEWAY_PORT}/healthz" >/dev/null 2>&1; then
    echo "网关端口 ${GATEWAY_PORT} 已被占用。" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OBSERVATION_WORKSPACE}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${EMERGE_PYTHON}" -m robot.behavior.gateway \
        --upstream-url "ws://127.0.0.1:${UPSTREAM_PORT}" \
        --listen-host 127.0.0.1 \
        --listen-port "${GATEWAY_PORT}" \
        --observation-workspace "${OBSERVATION_WORKSPACE}" \
        --observation-publish-interval 5 \
        --camera-calibration-config "${DRIVER_CONFIG}" \
        >"${GATEWAY_LOG}" 2>&1 &
GATEWAY_PID=$!

cleanup() {
    if kill -0 "${GATEWAY_PID}" 2>/dev/null; then
        kill "${GATEWAY_PID}" 2>/dev/null || true
        wait "${GATEWAY_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
    if curl --noproxy '*' --silent --fail \
        "http://127.0.0.1:${GATEWAY_PORT}/healthz" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then
        echo "EMERGE shadow gateway 启动失败，日志: ${GATEWAY_LOG}" >&2
        exit 1
    fi
    sleep 0.5
done

if ! curl --noproxy '*' --silent --fail \
    "http://127.0.0.1:${GATEWAY_PORT}/healthz" >/dev/null 2>&1; then
    echo "EMERGE shadow gateway 启动超时，日志: ${GATEWAY_LOG}" >&2
    exit 1
fi

echo "任务: ${TASK_NAME}"
echo "实例: public_test index ${INSTANCE_INDEX}（instance ${INSTANCE_ID}）"
echo "模式: EMERGE shadow（动作不修改，冠军策略仍拥有控制权）"
echo "输出目录: ${OUTPUT_DIR}"

(
    cd "${BEHAVIOR_ROOT}" || exit 1
    env \
        -u ISAAC_PATH \
        -u ISAACSIM_PATH \
        -u EXP_PATH \
        -u CARB_APP_PATH \
        OMNI_KIT_ACCEPT_EULA=YES \
        "${BEHAVIOR_PYTHON}" -m omnigibson.eval.eval \
            --task-name "${TASK_NAME}" \
            --mode public_test \
            --host 127.0.0.1 \
            --port "${GATEWAY_PORT}" \
            --instance-indices "${INSTANCE_INDEX}" \
            --num-envs 1 \
            --num-rollouts 1 \
            --replay-action-chunk-size 0 \
            --robot-config "${ROBOT_CONFIG}" \
            --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
            --output-dir "${OUTPUT_DIR}" \
            --video-fps 30 \
            --headless \
            --write-video
) 2>&1 | tee "${EVALUATOR_LOG}"

STATUS=${PIPESTATUS[0]}
if [[ ${STATUS} -ne 0 || ! -s "${RESULT_PATH}" ]]; then
    echo "shadow 评测失败，日志: ${EVALUATOR_LOG}" >&2
    exit 1
fi

echo "shadow 评测完成: ${RESULT_PATH}"
